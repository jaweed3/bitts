# BitJETS Architecture

BitJETS is a 1.58-bit quantized Text-to-Speech system based on the JETS architecture with BitNet b1.58 quantization applied to all 1D convolutional layers. The core research contribution is demonstrating that extreme ternary weight quantization (`{-1, 0, 1}`) can produce intelligible speech while reducing model size by ~5× compared to FP32.

---

## 1. Full TTS Pipeline

```mermaid
flowchart LR
    A[/"Text\n'hello world'"/] --> B["text_to_sequence\n(char → int)"]
    B --> C["Embedding\n[vocab → 256]"]
    C --> D["BitEncoder\n4× BitConvBlock"]
    D --> E["VarianceAdaptor\nDuration Predictor\n+ Length Regulator"]
    E --> F["BitDecoder\n4× BitConvBlock"]
    F --> G["Linear Projection\n[192 → 80]"]
    G --> H[/"Mel Spectrogram\n[T × 80]"/]
    H --> I["HiFi-GAN Vocoder\n(FP32)"]
    I --> J[/"Audio Waveform\n22050 Hz"/]

    style D fill:#e74c3c,color:#fff
    style F fill:#e74c3c,color:#fff
    style E fill:#e67e22,color:#fff
    style I fill:#27ae60,color:#fff
```

> **Red** = 1.58-bit quantized layers. **Green** = full-precision vocoder (intentionally kept FP32 — quantizing the final waveform generation layer causes severe audio degradation per paper Section 3.2).

---

## 2. BitConvBlock — The Core Quantized Unit

Every encoder and decoder layer is a `BitConvBlock`. This is where the compression happens.

```mermaid
flowchart TD
    IN["Input x\n[B, T, C]"] --> LN["LayerNorm\nnormalize over C dim\n(Sub-Layer Normalization)"]
    LN --> TR1["Transpose\n[B, T, C] → [B, C, T]"]
    TR1 --> BC["BitConv1d\n(see below)"]
    BC --> TR2["Transpose\n[B, C, T] → [B, T, C]"]
    TR2 --> OUT["Output\n[B, T, C]"]
    IN -->|"residual +"| ADD(("+"))
    TR2 --> ADD
    ADD --> RES["After residual add\n(in Encoder/Decoder)"]
```

The `LayerNorm → Transpose → BitConv1d → Transpose` pattern is the **SubLN** (Sub-Layer Normalization) design from BitNet b1.58: normalization happens *inside* each sublayer before quantization, stabilizing the input distribution.

---

## 3. BitConv1d — Quantization Math

This is the heart of the 1.58-bit optimization. Both weights and activations are quantized at every forward pass.

```mermaid
flowchart TD
    subgraph WQ ["Weight Quantization (Eq. 4-6)"]
        W["W ∈ ℝ^{cout×cin×K}"] --> BETA["β = mean(|W|)\n(per-tensor absmax scale)"]
        W --> DIV["W / β"]
        DIV --> CLIP["Clip(·, -1, 1)"]
        CLIP --> RND["Round(·)\n→ W' ∈ {-1, 0, 1}"]
        RND --> STE_W["STE backward:\ngrad flows to W, not W'"]
    end

    subgraph AQ ["Activation Quantization (Eq. 7-8)"]
        X["x (after LayerNorm)"] --> GAMMA["γ = ||x||∞\n(global abs-max)"]
        X --> SCALE["x × (127 / γ)"]
        SCALE --> CLIPX["Clip(·, -127, 127)"]
        CLIPX --> RNDX["Round(·)\n→ x' ∈ [-127, 127]"]
        RNDX --> STE_X["STE backward:\ngrad flows to x, not x'"]
    end

    RND --> CONV["Integer-like Conv1d\nf(W', x')"]
    RNDX --> CONV

    CONV --> RESCALE["Rescale output\ny = y_raw × (β × γ / 127)\n(Eq. 10: restore original magnitude)"]
    RESCALE --> OUT2["y (full-scale output)"]
```

### Why this works

| Property | Effect |
|---|---|
| W' ∈ {-1, 0, 1} | Only add/subtract needed, no multiply at inference |
| γ = global `\|\|x\|\|∞` | Single scale per tensor — consistent with paper Eq. 8 |
| β = mean(\|W\|) | Per-tensor scale captures weight distribution |
| Rescale by βγ/Qp | Exactly restores the magnitude of a full-precision conv output |

---

## 4. Straight-Through Estimator (STE)

The core training trick that makes quantization-aware training possible. `Round()` and `Clip()` have zero gradient almost everywhere — STE bypasses this.

```mermaid
flowchart LR
    subgraph FORWARD ["Forward Pass"]
        direction LR
        W_real["W (real-valued)"] -->|"÷ β, clip, round"| W_tern["W' ternary\n{-1, 0, 1}"]
        W_tern --> CONV2["conv(W', x')"]
    end

    subgraph BACKWARD ["Backward Pass (STE)"]
        direction RL
        GRAD_OUT["∂L/∂y"] -->|"backprop through conv"| GRAD_W_TERN["∂L/∂W'"]
        GRAD_W_TERN -->|"STE: treat round as identity\n∂W'/∂W = 1"| GRAD_W_REAL["∂L/∂W\n(updates real weights)"]
    end

    W_real -.->|"stored, updated by optimizer"| GRAD_W_REAL
```

**Key insight:** Real-valued weights `W` are always stored and updated by AdamW. `W'` (ternary) is only computed at each forward pass — it's not stored. The STE tells the autograd engine: "pretend `round()` is the identity function during backprop." This lets gradients flow to the real weights, which slowly converge to values that, when rounded, produce good ternary weights.

---

## 5. Model Architecture — Dimensions

```mermaid
block-beta
  columns 3

  block:encoder["BitEncoder"]:1
    EMB["Embedding\n32 → 256"]
    E1["BitConvBlock\n256→256, k=5"]
    E2["BitConvBlock\n256→256, k=5"]
    E3["BitConvBlock\n256→256, k=5"]
    E4["BitConvBlock\n256→256, k=5"]
    LN_E["LayerNorm 256"]
  end

  block:adaptor["VarianceAdaptor"]:1
    DP1["BitConvBlock\n256→256, k=3"]
    DP2["BitConvBlock\n256→256, k=3"]
    LINEAR["Linear 256→1\n(log duration)"]
    LR["Length Regulator\nrepeat_interleave"]
  end

  block:decoder["BitDecoder"]:1
    PROJ["Linear 256→192"]
    D1["BitConvBlock\n192→192, k=5"]
    D2["BitConvBlock\n192→192, k=5"]
    D3["BitConvBlock\n192→192, k=5"]
    D4["BitConvBlock\n192→192, k=5"]
    LN_D["LayerNorm 192"]
    OUT_PROJ["Linear 192→80"]
  end
```

---

## 6. VarianceAdaptor — Length Regulation

The adaptor bridges the text domain (short sequences) and the acoustic domain (longer mel frames).

```mermaid
flowchart TD
    ENC["Encoder Output\n[B, text_len, 256]"]

    subgraph PRED ["Duration Prediction"]
        ENC --> DUR_IN["2× BitConvBlock\n(k=3, context window)"]
        DUR_IN --> DUR_LIN["Linear → log_duration\n[B, text_len]"]
        DUR_LIN --> EXP["exp(·) - 1\n→ raw duration"]
    end

    subgraph TRAIN ["Training"]
        GT["Ground Truth\ndurations (from data)"]
    end

    subgraph INFER ["Inference"]
        EXP --> CTRL["× duration_control\n(speed adjustment)"]
        CTRL --> CLAMP["clamp(min=1)\n+ mask padding tokens"]
    end

    GT -->|"teacher forcing"| LR
    CLAMP --> LR

    ENC --> LR["Length Regulator\nrepeat_interleave(x, dur)"]
    LR --> EXP_OUT["Expanded Output\n[B, mel_len, 256]"]
```

**Padding mask:** During inference, padding tokens (index 0) are masked out before `clamp(min=1)` — otherwise padding would get `duration=1` and expand into the output mel.

---

## 7. Weight Indexing — Paper Algorithm 1

At inference/deployment, ternary weights are compressed to ~1/5 the size of uint8 storage using base-3 block encoding.

```mermaid
flowchart LR
    W_TERN["W' flattened\n[n elements]\nvalues: {-1, 0, 1}"]
    W_TERN --> ENCODE["Encode:\n0→0, 1→1, -1→2\n(base-3 digits)"]
    ENCODE --> BLOCK["Split into\nblocks of L*=5"]
    BLOCK --> INDEX["Each block → base-3 integer\nn = d₀×81 + d₁×27 + d₂×9 + d₃×3 + d₄\n∈ [0, 242] → fits in uint8"]
    INDEX --> STORE["Store as uint8\n5 values → 1 byte"]

    STORE --> LOAD["At load time:\ndecode each uint8\nback to 5 ternary values"]
    LOAD --> W_TERN2["W' reconstructed\n× β → float weights"]
```

**Size math for Conv1d(256, 256, k=5):**
- FP32: 256×256×5×4 bytes = **1,280 KB**
- uint8 (naive 1 val/byte): 256×256×5 = **320 KB**
- Algorithm 1 (5 vals/byte): 256×256×5÷5 = **64 KB** ✓ matches paper footnote 6

---

## 8. Training Optimizations

```mermaid
flowchart TD
    subgraph STABILITY ["Training Stability"]
        FILTER["Loss filter:\nskip batch if loss > 15\nor NaN/Inf"]
        ACCUM["Gradient accumulation\naccum_count tracks\nactual backward calls\n(not batch index)"]
        CLIP["Gradient clipping\nmax_norm = 1.0"]
        FILTER --> ACCUM --> CLIP
    end

    subgraph LR_SCHED ["LR Schedule (paper)"]
        WARMUP["Linear warmup\n1000 steps"]
        DECAY["Exponential decay\n× 0.9973 per step"]
        WARMUP --> DECAY
    end

    subgraph OPTIMIZER ["Optimizer (BitNet paper)"]
        ADAM["AdamW\nβ₁=0.8, β₂=0.99\nweight_decay=0"]
    end

    subgraph RESUME ["Reproducibility"]
        SEED["Fixed seed\n(torch + random + numpy)"]
        CKPT["Checkpoint saves:\n• weights\n• optimizer state (Adam moments)\n• scheduler state\n• global step"]
        SEED --- CKPT
    end
```

### Why these choices matter

**`accum_count` instead of `batch_idx % ACCUM_STEPS`:** If a batch is skipped due to loss explosion, `batch_idx` still increments. Using a separate `accum_count` ensures the optimizer always sees exactly `ACCUM_STEPS` worth of accumulated gradients.

**Saving optimizer state (Adam moments):** AdamW maintains running estimates of gradient mean (m) and variance (v) per parameter. Resuming without restoring these causes a "warm-up spike" as Adam re-adapts — especially bad after 200+ epochs of training.

**β₁=0.8 (not the default 0.9):** Lower β₁ means less momentum — the gradient estimate tracks recent gradients more closely. This is important for quantized networks where the STE introduces noise into gradients.

**weight_decay=0:** BitNet paper explicitly uses zero weight decay. Regular weight decay would penalize large weights — but for ternary quantization, β (the scale) carries the magnitude information and should not be decayed.

---

## 9. File Structure

```
bitts/
├── main.py                  # CLI dispatcher (train/infer/benchmark/sample)
├── src/
│   ├── hparams.py           # All hyperparameters + auto device detection
│   ├── layers.py            # BitConv1d, BitConvBlock, weight_quant, activation_quant
│   ├── models.py            # BitJETS, BitEncoder, BitDecoder, VarianceAdaptor
│   ├── dataset.py           # LJSpeechDataset, collate_fn
│   ├── train.py             # Training loop (step-based, infinite dataloader)
│   ├── inference.py         # Single-sentence inference + mel visualization
│   ├── sample_gen.py        # Batch audio sample generation
│   ├── benchmark.py         # Latency, RTF, size benchmarks
│   ├── vocoder.py           # HiFi-GAN wrapper
│   ├── models_gan.py        # HiFi-GAN generator architecture
│   ├── packing.py           # Weight indexing Algorithm 1 (base-3, L*=5)
│   ├── checkpoint.py        # Save/load/find checkpoint utilities
│   └── utils.py             # text_to_sequence, load_audio_wav
├── tests/
│   ├── conftest.py          # sys.path setup for pytest
│   ├── test_layers.py       # Unit tests: quantization math, shapes
│   ├── test_model.py        # Unit tests: model forward/backward, padding mask
│   ├── test_packing.py      # Unit tests: pack/unpack roundtrip, size reduction
│   └── test_smoke.py        # Integration: full training loop, checkpoint, resume
└── checkpoints/
    ├── latest.pth           # Most recent checkpoint (overwritten every step)
    ├── bitjets_ckpt_N.pth   # Named checkpoint every 10K steps
    ├── bitjets_packed_N.pth # Packed (Algorithm 1) version of named checkpoints
    └── UNIVERSAL_V1/        # Pre-trained HiFi-GAN vocoder
```
