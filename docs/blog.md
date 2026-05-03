# BitJETS — 1.58-bit Text-to-Speech: dari 12 MB jadi 2.5 MB

> **TL;DR:** Gue training TTS model yang semua convolutional weights-nya cuma 3 nilai: `{-1, 0, 1}`. Hasilnya? Model 5× lebih kecil (12 MB → 2.5 MB) tapi suaranya masih intelligible. Dibangun dari [BitNet b1.58](https://arxiv.org/abs/2402.17764) + [JETS](https://arxiv.org/abs/2203.16852), training di RTX 4060.

---

## Kenapa Ini Menarik?

Bayangin TTS model kayak Tortoise, VITS, atau NaturalSpeech. Mereka bagus, tapi gede — ratusan megabytes, bahkan gigabytes. Gimana kalau kita bikin model TTS yang **ukurannya cuma beberapa megabyte** tapi masih bisa ngomong jelas?

Ini bukan cuma soal "biar kecil aja". Model kecil berarti:

- **Edge deployment** — bisa jalan di HP, smart speaker, embedded device
- **Real-time inference** — latency rendah karena gak ada operasi floating-point multiply
- **Storage murah** — muat di embedded flash memory
- **Energy efficient** — cocok buat always-on devices

Nah, paper [BitTTS (Kawamura et al., 2024)](https://arxiv.org/abs/2409.10577) jawab ini dengan pendekatan radikal: **quantize semua convolutional layer ke 1.58-bit ternary weights** `{-1, 0, 1}`.

---

## 1.58-bit? Maksudnya Gimana?

Bayangin neural network biasa: setiap weight adalah angka float 32-bit. Di BitNet, weight di-quantize jadi cuma **tiga nilai**:

```
{-1, 0, 1}
```

Kenapa disebut 1.58-bit? Karena `log₂(3) ≈ 1.585`. Secara informasi teoritis, setiap ternary "trit" menyimpan ~1.58 bit informasi.

Di inference, perkalian weight × activation jadi operasi **tambah/kurang/ignore** — gak ada floating-point multiply sama sekali. Hardware bisa ngejalanin ini super cepat.

### Cara Kerja Quantization (STE)

```
Forward pass:
  W_real (float32) → W_ternary = round(clip(W_real / β, -1, 1))
  Conv pakai W_ternary

Backward pass (Straight-Through Estimator):
  Gradient mengalir ke W_real, BUKAN ke W_ternary
  ∇W_real = ∇W_ternary  (anggap round() = identity pas backward)
```

Real weights (`W_real`) tetap disimpan sebagai float32 dan di-update optimizer. Quantization hanya terjadi di forward pass. Ini disebut **Quantization-Aware Training (QAT)** — model belajar menghasilkan representasi yang robust terhadap quantization noise.

---

## Arsitektur

```
Text → Embedding → BitEncoder (4× BitConvBlock) → VarianceAdaptor → BitDecoder (4× BitConvBlock) → Mel → HiFi-GAN → Audio
       1.58-bit                                         dur predictor         1.58-bit              FP32
```

### BitConvBlock — Unit Quantized

Setiap convolution layer dibungkus dengan:

1. **LayerNorm** (Sub-LN) — normalisasi sebelum quantization, stabilkan distribusi input
2. **BitConv1d** — weight quantization `{-1, 0, 1}` + activation quantization 8-bit
3. **Residual connection** — skip connection bantu gradient flow

```python
class BitConvBlock(nn.Module):
    def forward(self, x):
        x_norm = self.layer_norm(x)        # Sub-LN normalization
        x_t = x_norm.transpose(1, 2)       # B,T,C → B,C,T
        out = self.bit_conv(x_t)           # Quantized conv
        return out.transpose(1, 2)         # Back to B,T,C
```

### Weight Packing — Algorithm 1

Di deployment, ternary weights dikompres pake base-3 block encoding:

```
5 ternary values → 1 byte (uint8)
  3^5 = 243 ≤ 256 ✓

Conv1d(256, 256, k=5):
  FP32:   256×256×5×4 bytes = 1,280 KB
  uint8:  256×256×5×1 bytes =   320 KB
  Packed: 256×256×5÷5 bytes =    64 KB  ← 20× lebih kecil!
```

---

## Hasil

### Model Size

| Format | Size |
|--------|------|
| FP32 (baseline) | 12.10 MB |
| Packed (actual) | **2.52 MB** |
| Compression | **79.2% reduction** |

### Inference Speed (RTX 4060)

| Input Length | Latency | RTF |
|-------------|---------|-----|
| 10 chars | 3.2 ms | 0.0047× |
| 30 chars | 5.8 ms | 0.0046× |
| 60 chars | 10.2 ms | 0.0046× |

RTF < 1.0 artinya **lebih cepat dari real-time**. Di RTX 4060, inference 200× lebih cepat dari durasi audio. Di CPU pun masih real-time karena gak ada operasi multiply.

### Parameter Distribution

- **92.9%** parameters di BitConv1d (1.58-bit)
- **7.1%** parameters di Embedding, LayerNorm, Linear (FP32)

Ini efisiensi maksimum — hampir semua parameters bisa di-quantize, sisanya komponen yang emang perlu presisi penuh.

---

## Training

### Setup

```bash
git clone <repo> && cd bitts
chmod +x scripts/*.sh
./scripts/bootstrap.sh    # one-command: uv + CUDA + LJSpeech + HiFi-GAN
./scripts/train.sh --resume   # mulai training
```

### Konfigurasi

| Hyperparameter | Value |
|---|---|
| Dataset | LJSpeech (~24 jam, single speaker) |
| Batch size | 32 |
| Learning rate | 2e-4 |
| LR schedule | Exponential decay (γ=0.9973/step) |
| Optimizer | AdamW (β₁=0.8, β₂=0.99) |
| Total steps | 500,000 |
| Hardware | RTX 4060 8GB |

### Training Loop Highlights

```python
# Loss filter: skip batch kalau loss meledak
if loss.item() > 15.0 or torch.isnan(loss):
    continue

# Gradient accumulation yang robust — track accum_count sendiri
(loss / ACCUM_STEPS).backward()
accum_count += 1

# Gradient clipping
clip_grad_norm_(model.parameters(), 1.0)
```

### Checkpoint System

- `latest.pth` — full state, di-overwrite tiap step (buat resume)
- `bitjets_ckpt_N.pth` — snapshot tiap 10K steps
- `bitjets_packed_N.pth` — compressed version (Algorithm 1)

---

## Kenapa Pakai HiFi-GAN (FP32) untuk Vocoder?

Paper BitTTS eksplisit nyebutin: **quantizing vocoder waveform generation layer causes severe audio degradation** (Section 3.2). Vocoder butuh presisi penuh buat reconstruct waveform dari mel spectrogram. Tapi karena vocoder cuma ~53 MB dan preprocessing step, ini gak ngaruh banyak ke total pipeline.

---

## Integrasi WandB

Training terintegrasi dengan [Weights & Biases](https://wandb.ai) buat monitoring:

- Loss curves (mel, duration, total)
- Gradient norm
- Learning rate
- Parameter histograms

Optional — bisa dimatiin dengan `--no-wandb` atau kosongin `WANDB_API_KEY` di `.env`.

---

## Cara Pakai

| Command | Purpose |
|---------|---------|
| `./scripts/bootstrap.sh` | Setup environment (sekali aja) |
| `./scripts/train.sh --resume` | Resume training |
| `./scripts/train.sh --resume --no-wandb` | Training offline |
| `./scripts/train.sh --preset high` | Training bs=64 (lebih cepat) |
| `./scripts/infer.sh --text "halo dunia"` | Generate audio |
| `./scripts/demo.sh` | Generate showcase artifacts |

---

## Demo

```bash
# Generate 6 audio samples + benchmark + spectrogram
./scripts/demo.sh
```

Output di `demo_output/`:
- 6 sample WAV files (beragam teks: pendek, panjang, teknis)
- `benchmark.txt` — latency, RTF, model size
- `mel_spectrogram.png` — visualisasi mel spectrogram
- `summary.txt` — ringkasan semua artifact

---

## Project Structure

```
bitts/
├── main.py            # CLI: train / infer / benchmark / sample
├── scripts/
│   ├── bootstrap.sh   # One-command setup
│   ├── train.sh       # Training launcher
│   ├── infer.sh       # Inference launcher
│   └── demo.sh        # Showcase generator
├── src/
│   ├── layers.py      # BitConv1d, BitConvBlock, quant functions
│   ├── models.py      # BitJETS, BitEncoder, BitDecoder, VarianceAdaptor
│   ├── train.py       # Training loop (step-based, infinite dataloader)
│   ├── packing.py     # Algorithm 1 weight indexing
│   ├── checkpoint.py  # Save/load/resume utilities
│   └── ...
├── tests/             # 19 tests: quantization math, model, packing, smoke
└── checkpoints/       # Model weights + HiFi-GAN vocoder
```

---

## Next Steps / TODO

- [ ] Multi-speaker training (VCTK, LibriTTS)
- [ ] ONNX/TFLite export untuk mobile deployment
- [ ] Quantize variance adaptor (saat ini masih FP32)
- [ ] Streaming inference (frame-by-frame)
- [ ] Bahasa Indonesia support

---

## Referensi

- [BitTTS: Quantized Text-to-Speech (Kawamura et al., 2024)](https://arxiv.org/abs/2409.10577)
- [BitNet b1.58: 1.58-bit Large Language Models (Ma et al., 2024)](https://arxiv.org/abs/2402.17764)
- [JETS: Jointly Training FastSpeech2 and HiFi-GAN (Lim et al., 2022)](https://arxiv.org/abs/2203.16852)
- [HiFi-GAN: Generative Adversarial Networks for Speech Synthesis (Kong et al., 2020)](https://arxiv.org/abs/2010.05646)

---

*Ditraining di RTX 4060 sambil ngopi. Model TTS 5× lebih kecil, masih bisa ngomong jelas. Quantization is the future.*
