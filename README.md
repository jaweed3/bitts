# BitJETS — 1.58-bit Quantized Text-to-Speech

Implementation of **BitTTS** ([Kawamura et al., 2024](https://arxiv.org/abs/2409.10577)) — a Text-to-Speech system where all convolutional layers use 1.58-bit ternary weights `{-1, 0, 1}`, achieving ~5× model size reduction vs FP32 with Quantization-Aware Training (QAT).

Built on [JETS](https://arxiv.org/abs/2203.16852) architecture with [BitNet b1.58](https://arxiv.org/abs/2402.17764) quantization. Trained on LJSpeech (single speaker, 22kHz).

> **📝 Blog post:** [docs/blog.md](docs/blog.md) — full writeup (Indonesian/English mixed)  
> **🎬 Demo script:** `./scripts/demo.sh` — generate showcase artifacts

---

## Key Results

| Metric | Value |
|---|---|
| Acoustic model size (FP32) | ~10 MB |
| Acoustic model size (packed, Algorithm 1) | ~2 MB |
| BitConv1d weight precision | 1.58-bit ternary `{-1, 0, 1}` |
| Activation precision | 8-bit int (via absmax scaling) |
| Vocoder | HiFi-GAN (full precision, intentional) |
| Training hardware | RTX 4060 / Apple M-series MPS |
| Dataset | LJSpeech (13,100 utterances, ~24h) |

> See [ARCHITECTURE.md](ARCHITECTURE.md) for full technical breakdown with diagrams.

---

## Quick Start (Recommended)

```bash
git clone <repo>
cd bitts
chmod +x scripts/*.sh

# One-command setup: uv + Python + CUDA deps + LJSpeech + HiFi-GAN
./scripts/bootstrap.sh

# Start training (auto-detects GPU, supports resume)
./scripts/train.sh --resume

# Inference
./scripts/infer.sh --text "hello world"
```

### Manual Setup (Alternative)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install deps (auto-selects PyTorch CUDA 12.1 on Linux)
uv sync --no-dev

# 3. LJSpeech dataset → place at:
#    data/speech/metadata.csv
#    data/speech/wavs/*.wav
#    Download: https://keithito.com/LJ-Speech-Dataset/
```

---

## Training

```bash
# Fresh training (auto-detects CUDA → MPS → CPU)
python main.py train

# Resume from latest checkpoint
python main.py train --auto-resume

# Resume from specific checkpoint
python main.py train --resume checkpoints/bitjets_ckpt_50000.pth

# Offline (no WandB)
python main.py train --auto-resume --no-wandb

# Custom config
python main.py train --device cuda --batch-size 64 --num-steps 1000000
```

### Hardware presets

| Hardware | `BATCH_SIZE` | `ACCUM_STEPS` | Effective BS | Notes |
|---|---|---|---|---|
| RTX 4060 (8GB) | 32 | 1 | 32 | Default |
| RTX 4060 (8GB) | 64 | 1 | 64 | Try this first |
| Apple M-series | 8 | 4 | 32 | MPS constraints |
| CPU (debug) | 2 | 1 | 2 | Slow, for testing |

Override via CLI: `--batch-size 64 --accum-steps 1`

### Checkpoints

Training saves:
- `checkpoints/latest.pth` — full state (weights + optimizer + scheduler + step), overwritten every step
- `checkpoints/bitjets_ckpt_N.pth` — named snapshot every 10,000 steps
- `checkpoints/bitjets_packed_N.pth` — compressed version (Algorithm 1) of named snapshots

Legacy checkpoints (epoch-based, e.g. `bitjets_ckpt_180.pth`) are supported — training will resume from step 0 with loaded weights.

---

## Inference

```bash
# Generate audio from text
python main.py infer \
  --model-path checkpoints/latest.pth \
  --text "the quick brown fox jumps over the lazy dog" \
  --output output.wav

# Adjust speaking speed
python main.py infer --model-path checkpoints/latest.pth \
  --text "hello world" --speed 0.8 --output slow.wav
```

---

## Generate Audio Samples

```bash
# Generate 8 benchmark sentences → samples/
python main.py sample \
  --model-path checkpoints/latest.pth \
  --output samples/

# Output:
# samples/01_hifigan_hello_world.wav
# samples/02_hifigan_the_quick_brown_fox...wav
# ...
# samples/mel_grid.png   (visual mel comparison)
```

---

## Benchmark

```bash
python main.py benchmark --model-path checkpoints/latest.pth
```

Example output:
```
============================================================
  BitJETS Inference Benchmark
============================================================

--- Parameters ---
  Total:                3,245,904
  BitConv1d:            3,014,656  (92.9%)
  Other (FP32):           231,248   (7.1%)

--- Model Size ---
  FP32 (baseline):    12,396.5 KB  (12.10 MB)
  Packed (actual):     2,580.2 KB   (2.52 MB)
  Compression:         79.2% reduction vs FP32

--- Inference Latency (device: cuda) ---
  Text length            Latency   Mel frames        RTF
  --------------------------------------------------------
  short  (10 chars)        3.21ms          87     0.0047x
  medium (30 chars)        5.84ms         261     0.0046x
  long   (60 chars)       10.22ms         522     0.0046x

  RTF < 1.0 means faster than real-time.
============================================================
```

---

## Run Tests

```bash
# All tests (19 total, no LJSpeech needed)
python -m pytest tests/ -v

# Smoke tests only (training loop integration)
python -m pytest tests/test_smoke.py -v

# Fast unit tests only
python -m pytest tests/test_layers.py tests/test_model.py -v
```

Test coverage:
- `test_layers.py` — quantization math, weight/activation quant ranges, shapes, gradients
- `test_model.py` — full model forward/backward, padding mask correctness, inference mode
- `test_packing.py` — Algorithm 1 pack/unpack roundtrip, actual size reduction
- `test_smoke.py` — training loop (50-100 steps), loss convergence, checkpoint save/resume

---

## Architecture Overview

```
Text → [Embedding] → [BitEncoder × 4] → [VarianceAdaptor] → [BitDecoder × 4] → [Mel] → [HiFi-GAN] → Audio
                      1.58-bit QAT        duration align       1.58-bit QAT              FP32
```

The 1.58-bit quantization uses **Straight-Through Estimator (STE)** for gradient flow through the non-differentiable `round()` operation. Real-valued weights are maintained throughout training and rounded to `{-1, 0, 1}` only during the forward pass.

See [ARCHITECTURE.md](ARCHITECTURE.md) for:
- Mermaid diagrams of every component
- Quantization math (Eq. 4-10 from paper)
- STE gradient flow explanation
- Weight indexing Algorithm 1
- Training optimization decisions

---

## Project Structure

```
src/
├── layers.py       # BitConv1d, BitConvBlock — core quantization
├── models.py       # BitJETS, BitEncoder, BitDecoder, VarianceAdaptor
├── train.py        # Step-based training loop
├── packing.py      # Weight indexing Algorithm 1 (base-3, L*=5)
├── checkpoint.py   # Save/load/find checkpoint utilities
├── dataset.py      # LJSpeechDataset
├── inference.py    # Single inference + mel plot
├── sample_gen.py   # Batch sample generation
├── benchmark.py    # Latency + size benchmarks
├── vocoder.py      # HiFi-GAN wrapper
└── hparams.py      # All hyperparameters
```

---

## References

- [BitTTS: Quantized Text-to-Speech](https://arxiv.org/abs/2409.10577) — Kawamura et al., 2024
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — Ma et al., 2024
- [JETS](https://arxiv.org/abs/2203.16852) — Lim et al., 2022
- [HiFi-GAN](https://arxiv.org/abs/2010.05646) — Kong et al., 2020
- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
