# Limitations & Quality Standards

This document is intentionally honest. BitJETS is a research implementation built on a single RTX 4060 with a public dataset. Understanding exactly where the ceiling is — and why — is more useful than vague claims.

---

## 1. What "Good Enough" Means Here

TTS quality is measured on a spectrum. The standard benchmark is **MOS (Mean Opinion Score)**: human listeners rate naturalness from 1–5. Here's where different systems land:

| System | MOS | Context |
|---|---|---|
| Human speech (LJSpeech) | ~4.5–4.7 | Ground truth ceiling |
| Tacotron 2 + WaveNet | ~4.5 | 2018 Google, large GPU cluster |
| FastSpeech 2 + HiFi-GAN | ~4.3 | 2020 Microsoft, full-precision |
| BitTTS (paper, FP32 baseline) | ~4.0 | A100, 2000K steps, MFA durations |
| BitTTS (paper, 1.58-bit) | ~3.8 | A100, 2000K steps, MFA durations |
| **BitJETS (this project)** | **~2.5–3.2 est.** | RTX 4060, dummy durations |
| Barely intelligible | ~2.0 | Words recognizable, very robotic |
| Unintelligible | < 1.5 | Can't understand words |

**Realistic target for this project: MOS 2.8–3.2** — clearly synthetic, robotic timing, but every word is understandable. This is enough to demonstrate the quantization works, the architecture is correctly implemented, and the model trained.

---

## 2. Hard Limitations (Cannot Be Fixed Without Fundamental Changes)

### 2.1 Dummy Duration — The Biggest Quality Ceiling

**What it is:** Every character is assigned `mel_len // text_len` frames — uniform duration regardless of phoneme.

**Impact:** 'a' and 'the' get the same duration. Stressed vs unstressed syllables sound identical. The decoder learns to generate different mels for the same duration label, which is a contradictory signal — it partially works but timing always sounds robotic.

**Magnitude:** This alone likely costs ~0.5–1.0 MOS points. The difference between MOS 2.5 and MOS 3.5 is mostly duration quality.

**What would fix it:** Montreal Forced Aligner (MFA) on LJSpeech — extracts per-character/phoneme durations from audio+transcript. This is the single highest-ROI improvement available.

**Why not done yet:** MFA requires installing additional tools and a separate alignment pipeline. It's planned but not implemented.

---

### 2.2 Character-Level Vocabulary, No Phonemes

**What it is:** Input is raw characters (`a`, `b`, ..., `z`, spaces, punctuation). There's no grapheme-to-phoneme (G2P) conversion.

**Impact:**
- Homographs like "read" (present vs past tense) sound identical
- No handling of abbreviations, numbers, or proper names
- The model must learn pronunciation directly from spelling, which is difficult for English (notorious for irregular pronunciation)

**Example failures:** "colonel", "worcestershire", "lead" (metal vs to lead) — all ambiguous at character level.

**What would fix it:** Add a G2P module (e.g., `phonemizer` library) to convert text to IPA or ARPAbet phonemes before encoding. This is a standard preprocessing step in production TTS.

---

### 2.3 Single Speaker, No Style Control

**What it is:** Trained only on LJSpeech (one female speaker reading audiobooks). No speaker embedding, no pitch/energy control, no speaking style variation.

**Impact:** One voice, one style. Cannot generalize to other speakers or emotional styles.

**Note:** This is a deliberate scope choice, not a bug. For a research demo of quantization, single-speaker is sufficient and actually easier to train.

---

### 2.4 HiFi-GAN Not Quantized

**What it is:** The vocoder runs at full FP32 precision. The system is not end-to-end 1.58-bit.

**Impact on size claim:** 
- Full system (acoustic + vocoder): HiFi-GAN UNIVERSAL_V1 ≈ 55 MB FP32
- BitJETS acoustic model (packed): ~2 MB
- If you count only the acoustic model: 5× compression ✓
- If you count the full system: compression ratio is much less impressive

**Why not quantized:** The paper itself excludes the output layer of HiFi-GAN. Quantizing the full vocoder causes "severe audio degradation" (robotic buzzing) even in the paper's experiments. This is an open research problem.

---

## 3. Hardware-Imposed Limitations

### 3.1 RTX 4060 vs A100 — What Actually Matters

| Factor | RTX 4060 | A100 | Impact on this project |
|---|---|---|---|
| VRAM | 8 GB | 80 GB | **Not a bottleneck** — model is <60 MB |
| FP32 TFLOPS | 15.1 | 77.6 | ~5× slower per step |
| Memory bandwidth | 272 GB/s | 2,000 GB/s | Matters for large batch; irrelevant here |
| Training duration | longer | shorter | Affects total steps achievable |

The RTX 4060 is not a meaningful hardware limitation for this model size. A model this small (3.2M parameters) trains primarily on compute throughput for the forward/backward pass, not memory bandwidth or VRAM capacity.

**Real hardware constraint:** Time. 500K steps on RTX 4060 ≈ 3–6 hours. 2000K steps (paper) ≈ 12–24 hours. Both are achievable in a single session.

### 3.2 Dataset Size vs Paper

| Dataset | Hours | Speakers | Sample rate |
|---|---|---|---|
| LibriTTS-R (paper) | 585h | 2,456 | 24 kHz |
| LJSpeech (this project) | 24h | 1 | 22 kHz |

LJSpeech is 24× smaller than LibriTTS-R. However, for single-speaker TTS, 24 hours is actually more than enough — most TTS systems can produce good quality with 2–10 hours of single-speaker data. The dataset is not the bottleneck.

---

## 4. Training Step Budget Reality

The paper trains for 2,000K steps. Given our step budget:

| Steps | Est. time (RTX 4060) | Expected quality |
|---|---|---|
| 50K | ~20 min | Loss dropping, incoherent audio |
| 100K | ~40 min | Words vaguely recognizable |
| 200K | ~80 min | Consistently intelligible |
| 500K | ~3–4 hrs | Good for demo, timing still rough |
| 1000K | ~6–8 hrs | Near plateau without MFA durations |
| 2000K | ~12–16 hrs | Paper-equivalent steps, still limited by dummy durations |

**Key insight:** Without MFA, training beyond ~500K steps gives diminishing returns. The model hits a quality ceiling imposed by dummy durations, not by insufficient training.

---

## 5. What This Project Can Legitimately Claim

### Can claim ✓

- Correct implementation of BitNet b1.58 quantization (weight + activation quantization per paper Eq. 4–10)
- Correct STE backpropagation through ternary weights
- Weight indexing Algorithm 1 (base-3, L*=5) matching paper footnote 6 size targets
- ~5× acoustic model compression vs FP32 (from 10 MB to ~2 MB)
- Working end-to-end TTS pipeline: text → intelligible speech
- Reproducible training with proper checkpoint/resume on commodity hardware
- Faster-than-real-time inference (RTF << 1.0) on both CPU and GPU

### Cannot claim ✗

- Production-quality naturalness (requires MFA, phonemes, prosody control)
- Paper-equivalent MOS scores (requires MFA durations + 2000K steps on large data)
- End-to-end 1.58-bit system (vocoder is FP32)
- Multi-speaker or cross-lingual capability
- Mobile/edge deployment (no ONNX/CoreML export yet)

### Gray area ⚠

- "1.58-bit TTS" — technically the acoustic model is 1.58-bit quantized, but the full pipeline is not
- Comparison to paper results — architecture matches but training conditions differ significantly

---

## 6. Path to Better Quality (Prioritized)

If you want to push quality further, in order of impact per effort:

### Tier 1 — High impact, medium effort

**MFA Duration Alignment**
Install Montreal Forced Aligner, run on LJSpeech, store per-sample duration files. Expected quality gain: +0.5–1.0 MOS. This is the single most impactful change possible.

```bash
# One-time setup
pip install montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa align data/speech/wavs data/speech/metadata.csv \
    english_us_arpa english_us_arpa data/alignments/
```

**Phoneme input (G2P)**
Replace character-level input with phoneme sequences via `phonemizer`. Expected gain: +0.2–0.4 MOS (especially on irregular English words).

### Tier 2 — Medium impact, medium effort

**Increase training steps to 1M+**
With MFA durations, the quality ceiling rises. Training to 1M+ steps will be meaningful.

**Scale up batch size**
RTX 4060 can handle batch_size=64 for this model. Larger effective batch → more stable gradient estimates → potentially faster convergence.

### Tier 3 — Low-hanging fruit for numbers

**Run actual MCD benchmark**
Mel Cepstral Distortion is a standard objective metric:

$$\text{MCD} = \frac{10\sqrt{2}}{\ln 10} \sqrt{\sum_{k=1}^{K} (c_k - \hat{c}_k)^2}$$

Lower is better (typical good TTS: MCD < 6 dB). This gives a concrete number to put in the README without needing human raters.

**WER via Whisper**
Run generated audio through `openai/whisper` and compute Word Error Rate vs the original text. A good system should have WER < 5% on LJSpeech test set. This is fully automatable.

---

## 7. Honest Summary

This project is a **correct, reproducible research implementation** of an interesting quantization technique applied to TTS. The architecture, quantization math, and training pipeline are sound.

The output audio will be robotic but intelligible — clearly synthetic, clearly the right words, clearly demonstrates that 1.58-bit quantization doesn't destroy the model entirely.

What it is not: a competitive TTS system. The dummy duration model and character-level input are known limitations that prevent reaching state-of-the-art quality regardless of hardware.

For a portfolio, the story is: "I implemented BitNet 1.58-bit quantization for TTS from scratch, correctly matching the paper's quantization math, and trained it on commodity hardware." That story is true and genuinely interesting. The audio quality doesn't need to match Google's TTS for the implementation to be impressive.
