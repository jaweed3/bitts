# Mathematical Theory & Optimization — BitJETS

This document covers the mathematical foundations behind every design decision in BitJETS: from the quantization formulas to gradient flow, optimizer tuning, and why training a 1.58-bit network converges at all.

---

## 1. The Quantization Problem

Standard neural networks store weights as 32-bit floats. At inference, a Conv1d layer computes:

$$y = W * x + b$$

where `*` is convolution, W ∈ ℝ^{c_out × c_in × K}, x is the input, and each multiply-accumulate (MAC) requires a full floating-point multiplication.

**The goal:** Replace W with W' ∈ {-1, 0, 1} so that the convolution reduces to additions and subtractions only — no multiplications. This is the defining property of 1-bit and 1.58-bit networks.

The challenge: how do you train a network whose weights must live in a discrete set? Discrete functions have zero gradient almost everywhere, which breaks standard backpropagation.

---

## 2. Weight Quantization

### 2.1 Mean-Absmax Scaling

Given a weight tensor W ∈ ℝ^{c_out × c_in × K}, we first compute a scale factor:

$$\beta = \frac{1}{c_{out} \cdot c_{in} \cdot K} \sum_{i,j,k} |W_{i,j,k}|$$

β is the **mean absolute value** of all weights — the L1-norm divided by the number of elements. This is different from the standard "absmax" used in 8-bit quantization (which uses the max). Using the mean is more robust to outliers.

### 2.2 Ternary Projection

The quantized weights are:

$$W' = \text{Round}\left(\text{Clip}\left(\frac{W}{\beta + \epsilon}, -1, 1\right)\right)$$

where:
- `Clip(u, -1, 1) = max(-1, min(1, u))` keeps values in [-1, 1]
- `Round(·)` maps to the nearest integer, so values in (-0.5, 0.5) → 0, values in [0.5, 1.5] → 1, values in [-1.5, -0.5] → -1

The result W' ∈ {-1, 0, 1}.

**Why this produces approximately 1.58 bits:**
- 3 possible values → log₂(3) ≈ 1.585 bits per weight
- The 0 value emerges naturally when weights are small relative to β — it's a form of automatic sparsity

### 2.3 Implementation in code (`layers.py`)

```python
def weight_quant(w):
    beta = w.abs().mean().clamp(min=1e-5)  # ε prevents div/0
    w_scaled = w / beta
    w_rounded = w_scaled.clamp(-1, 1).round()
    # STE: see Section 4
    w_quant = (w_rounded - w_scaled).detach() + w_scaled
    return w_quant, beta
```

---

## 3. Activation Quantization

### 3.1 Infinity Norm Scaling

For input tensor x (after LayerNorm), compute:

$$\gamma = \|x\|_\infty = \max_{i} |x_i|$$

This is the **global maximum absolute value** across the entire tensor. Then scale and clip to 8-bit integer range:

$$x' = \text{Clip}\left(\frac{x \cdot Q_p}{\gamma + \epsilon},\ -Q_p,\ Q_p\right)$$

where Q_p = 2^(p-1) = 127 for p=8 (8-bit precision).

### 3.2 Why global γ, not per-channel?

Per-channel normalization would give each channel its own scale, potentially better preserving fine-grained information. However, the paper uses global γ because:

1. **Consistency with rescaling:** The output rescale (Section 3.3) uses a single γ, which must match the quantization scale
2. **Simplicity:** One scale factor per tensor is easy to implement and store for inference
3. **LayerNorm already normalizes:** Since LayerNorm is applied before quantization, the activation distribution is already well-conditioned — per-channel scaling is less critical

### 3.3 Output Rescaling

After the quantized convolution:

$$y_{raw} = f(W', x')$$

where f is the convolution operation. The output y_raw is in a compressed integer-like range. To restore the original magnitude:

$$y = y_{raw} \cdot \frac{\beta \cdot \gamma}{Q_p}$$

**Derivation:** A true full-precision convolution would compute:

$$y_{fp} \approx f(\beta \cdot W', \frac{\gamma}{Q_p} \cdot x') = \beta \cdot \frac{\gamma}{Q_p} \cdot f(W', x') = \frac{\beta \gamma}{Q_p} \cdot y_{raw}$$

So multiplying y_raw by βγ/Q_p exactly cancels the quantization scaling applied to both weights and activations.

In code:
```python
rescale_factor = beta / scale_x   # = beta / (Q_p / gamma) = beta * gamma / Q_p
y = y_raw * rescale_factor
```

---

## 4. Straight-Through Estimator (STE)

### 4.1 The Problem

`Round(·)` has derivative zero almost everywhere and undefined at half-integers:

$$\frac{d}{du}\text{Round}(u) = 0 \quad \text{a.e.}$$

If we naively backprop through the quantization, every gradient is zero and nothing trains.

### 4.2 The STE Trick

Bengio et al. (2013) proposed treating the rounding as an identity during backpropagation. The STE approximates:

$$\frac{\partial \text{Round}(u)}{\partial u} \approx 1$$

**Implementation via the "stop gradient" trick:**

```python
w_quant = (w_rounded - w_scaled).detach() + w_scaled
```

Breaking this down:
- In the **forward pass**: `(w_rounded - w_scaled).detach() + w_scaled = w_rounded` ✓ (the detach cancels with w_scaled, leaving w_rounded)
- In the **backward pass**: `.detach()` blocks gradients through the first term. Gradients only flow through the final `+ w_scaled` term, which has gradient 1 with respect to w_scaled.

So: `∂w_quant/∂w_scaled = 1`, which gives `∂w_quant/∂w = 1/β` (from the w_scaled = w/β step).

### 4.3 What STE Actually Learns

The real weights W are updated by Adam using the gradient signal that flowed through the quantized forward pass. Over time, W converges to values such that W/β, when clipped and rounded, produces good ternary weights for the task.

Intuitively: weights that are far from ±β (i.e., small relative to the mean) get pulled toward 0 or ±1 by the task loss. Weights near ±β stay near ±1. Weights near 0 stay near 0 (sparsity emerges).

### 4.4 Why It Works Despite Being An Approximation

The STE is theoretically unjustified — it's an approximation that happens to work well in practice. The reasons:

1. **Soft quantization during early training:** At initialization, W is random and small. β is also small. W/β is often in (-1, 1) but not near 0.5, so rounding doesn't change values much. The STE is nearly exact early on.

2. **Progressive hardening:** As training progresses, weights polarize toward ±1 and 0. The "effective" quantization error decreases as weights become more committed.

3. **LayerNorm stabilizes:** By normalizing inputs before quantization, LayerNorm keeps activations in a predictable range, reducing the quantization error variance.

---

## 5. SubLN: Why LayerNorm Goes Inside the Sublayer

Standard transformer/conv blocks apply LayerNorm *before* the sublayer (Pre-LN):

```
x → LayerNorm → Sublayer → + x
```

BitNet b1.58 uses **SubLN** — LayerNorm is inside the sublayer, right before quantization:

```
x → [LayerNorm → Quantize → Conv → Rescale] → + x
     └────────── Sublayer ──────────────────┘
```

**Why this matters for quantization:**

The activation quantization uses γ = ‖x‖∞. If x has a very large magnitude (e.g., after a residual addition), γ becomes large, the scale Q_p/γ becomes small, and quantization loses precision — many activation values map to the same integer.

LayerNorm normalizes x to have approximately unit variance, bounding ‖x‖∞ to a predictable range. This maximizes the effective resolution of the 8-bit activation quantization.

---

## 6. Loss Functions

### 6.1 Mel Spectrogram Loss

$$\mathcal{L}_{mel} = \frac{1}{T \cdot M} \sum_{t=1}^{T} \sum_{m=1}^{M} (\hat{S}_{t,m} - S_{t,m})^2$$

MSE between predicted mel S_hat and ground truth mel S. T is the mel time dimension, M=80 mel bins.

**Why MSE and not L1?** L1 (MAE) tends to produce blurrier spectrograms — it over-penalizes outliers less, leading to median-seeking behavior. MSE is mean-seeking, which better preserves sharp spectral features. For mel spectrograms (which are already in log scale), MSE works well.

### 6.2 Duration Loss

The duration predictor outputs log-durations. The loss is MSE in log space:

$$\mathcal{L}_{dur} = \frac{1}{L} \sum_{i=1}^{L} (\hat{d}_i - \log(d_i + \epsilon))^2$$

where d_i is the ground truth duration (integer, number of mel frames per character), d_hat_i is the predicted log-duration, and ε=1e-4 prevents log(0).

**Why log space?** Durations are always positive and can vary over a large range (1 to 30+ frames). Log space:
1. Makes the distribution more Gaussian (easier to model)
2. Treats relative duration errors equally (10% error is the same whether duration is 2 or 20 frames)
3. Ensures predicted durations are always positive via `exp(·)`

### 6.3 Total Loss

$$\mathcal{L} = \mathcal{L}_{mel} + \mathcal{L}_{dur}$$

Unweighted sum. Both terms are MSE losses in comparable scales (mel is in log-mel scale, duration is in log-frame space), so no weighting coefficient is needed.

---

## 7. Optimizer: AdamW with BitNet Settings

### 7.1 Standard AdamW

AdamW maintains two moment estimates per parameter:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(first moment, mean)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(second moment, variance)}$$

The parameter update (with bias correction):

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta_{t-1}$$

where λ is weight decay.

### 7.2 Why β₁ = 0.8 (not the default 0.9)?

With β₁=0.9, the first moment m_t is a heavy exponential moving average — it takes ~10 steps to incorporate new gradient information (effective window ≈ 1/(1-β₁) = 10 steps).

With β₁=0.8, the effective window is 5 steps. This means:
- Gradient estimates react **faster** to new information
- The optimizer is **less smooth** but **more responsive**

For quantized networks, the STE introduces noise into gradients (the true gradient of the quantized function is different from the STE gradient). A lower β₁ prevents the optimizer from over-committing to a noisy gradient direction — it "forgets" stale estimates faster.

### 7.3 Why weight_decay = 0?

Standard weight decay adds a penalty that pulls weights toward 0:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \frac{\lambda}{2} \|W\|^2$$

This is designed to prevent large weights. But for ternary quantization, the **magnitude** of weights is captured by β, not by individual weight values. Applying weight decay would penalize large β, which would artificially constrain the scale of quantized operations. The paper explicitly uses zero weight decay for this reason.

### 7.4 Why β₂ = 0.99 (not the default 0.999)?

β₂ controls the second moment (variance) estimate. With β₂=0.999, the effective window is 1000 steps — the adaptive learning rate is very stable but slow to adapt.

With β₂=0.99 (window ≈ 100 steps), the per-parameter learning rate adapts more quickly to gradient magnitude changes. This is important in QAT because:

1. As weights polarize toward ternary values, the gradient landscape changes qualitatively
2. The variance of STE gradients can shift significantly during training
3. Faster adaptation means the effective learning rate self-corrects as the network transitions from "almost continuous" to "nearly ternary"

---

## 8. Learning Rate Schedule

### 8.1 Exponential Decay

The paper uses exponential LR decay:

$$\alpha_t = \alpha_0 \cdot \gamma^t$$

with γ=0.9973. Starting from α₀=2×10⁻⁴:

| Step | LR |
|---|---|
| 0 | 2.00 × 10⁻⁴ |
| 1,000 | 1.95 × 10⁻⁴ |
| 10,000 | 1.54 × 10⁻⁴ |
| 50,000 | 2.94 × 10⁻⁵ |
| 100,000 | 4.32 × 10⁻⁶ |
| 500,000 | ~10⁻¹⁰ (effectively 0) |

### 8.2 Why Exponential Decay for QAT?

The learning rate schedule interacts with the quantization process. Early in training:
- Weights are far from ternary values
- Large LR helps explore quickly
- STE gradient noise is relatively small compared to the task gradient

Late in training:
- Weights are nearly polarized to {-1, 0, 1}
- Small LR prevents oscillating across the rounding boundary
- The ternary assignment of most weights is stable; only a few weights are "on the boundary" and still updating

Exponential decay provides a smooth transition. Cosine annealing (alternately) can cause weights near the rounding boundary to oscillate as the LR periodically increases.

### 8.3 Linear Warmup (our addition)

We add a 1000-step linear warmup before the exponential decay begins:

$$\alpha_t = \alpha_0 \cdot \frac{t}{t_{warmup}}, \quad t < t_{warmup}$$

This is not in the original paper (which trains from scratch on large data). We add it because:
1. When **resuming** from a checkpoint with a fresh optimizer (lost Adam moments), the initial gradient estimates are unreliable — a low LR prevents a large disruptive update
2. On LJSpeech (smaller dataset), early instability is more damaging

---

## 9. Gradient Accumulation — Correctness Proof

### 9.1 Why Accumulation Works

For a batch of size N, the true gradient is:

$$g_N = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \mathcal{L}(x_i)$$

With accumulation over K micro-batches of size N/K each:

$$g_k = \frac{1}{N/K} \sum_{i \in \text{batch}_k} \nabla_\theta \mathcal{L}(x_i)$$

If we sum K micro-batch gradients and divide by K:

$$\frac{1}{K}\sum_{k=1}^{K} g_k = \frac{1}{K} \cdot K \cdot \frac{1}{N/K} \cdot \frac{1}{K} \sum_{i=1}^{N} \nabla_\theta \mathcal{L}(x_i) = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \mathcal{L}(x_i) = g_N$$

So: **accumulate gradients, then divide by K** is mathematically equivalent to a single large batch of size N. In code, this means the loss must be divided by K before calling `.backward()`:

```python
(loss / ACCUM_STEPS).backward()   # accumulate K times, then step
```

### 9.2 The Skip-Batch Bug (Fixed)

A subtle bug: if a batch is skipped (loss explosion), using `batch_idx % ACCUM_STEPS` as the trigger condition counts skipped batches toward the window:

```
batch 0: backward (1 accumulated)
batch 1: SKIPPED — loss > 15
batch 2: backward (2 accumulated)   ← batch_idx+1=3, 3%4≠0
batch 3: backward (3 accumulated)   ← batch_idx+1=4, 4%4=0 → STEP!
```

Optimizer steps with 3 accumulated batches, but loss was divided by 4. Each gradient is scaled by 1/4 instead of 1/3 — the effective learning rate is 25% lower than intended.

**Fix:** use `accum_count` that only increments on actual backward calls:

```python
accum_count = 0
...
(loss / ACCUM_STEPS).backward()
accum_count += 1
if accum_count == ACCUM_STEPS:
    optimizer.step()
    accum_count = 0
```

Now the optimizer always steps on exactly K accumulated gradients.

---

## 10. Weight Indexing — Information Theory Perspective

### 10.1 Theoretical Minimum

Each ternary weight value carries log₂(3) ≈ 1.585 bits of information. For a tensor of N values:

$$\text{Ideal storage} = N \cdot \log_2 3 \approx 1.585 N \text{ bits}$$

Standard hardware operates in multiples of 8 bits. Storing one ternary value per byte uses 8 bits for 1.585 bits of information — **80% waste**.

### 10.2 Base-3 Block Encoding

The insight: encode L* ternary values together as a single base-3 integer. Choose L* such that 3^L* ≤ 256 (fits in one uint8).

$$3^5 = 243 \leq 256 < 729 = 3^6$$

So L*=5 is optimal — 5 ternary values per uint8. The encoding:

For a block (v₁, v₂, v₃, v₄, v₅) where vᵢ ∈ {-1, 0, 1}:

1. Map to digits: 0→0, 1→1, -1→2
2. Compute: n = d₁×3⁴ + d₂×3³ + d₃×3² + d₄×3 + d₅

This n ∈ [0, 242], stored as uint8. The 13 unused values (243–255) are reserved.

**Storage efficiency:**
- Stores 5 ternary values in 1 byte
- vs. 1 value per byte (naive): 5× improvement
- vs. FP32 (4 bytes per value): 20× improvement
- Effective bit-per-weight: 8/5 = 1.6 bits (vs. theoretical 1.585 bits — nearly optimal)

### 10.3 Size Calculation for Conv1d(256, 256, k=5)

| Format | Formula | Size |
|---|---|---|
| FP32 | 256×256×5×4 bytes | 1,310,720 B = 1,280 KB |
| uint8 naive | 256×256×5×1 byte | 327,680 B = 320 KB |
| Algorithm 1 | ⌈256×256×5 / 5⌉ bytes | 65,536 B = 64 KB |
| Theoretical min | 256×256×5 × log₂3 / 8 | 64,881 B = 63.4 KB |

Algorithm 1 achieves 64 KB vs the theoretical minimum of 63.4 KB — only 0.9% overhead. ✓

---

## 11. Duration Modeling — Length Regulation

### 11.1 The Alignment Problem

TTS needs to map text (L tokens) to mel frames (T frames), where T >> L. A sentence of 30 characters might produce 300 mel frames.

The duration predictor learns d_i (frames per token i) such that Σd_i ≈ T. The length regulator then repeats each encoder hidden state d_i times:

$$h_{expanded} = [\underbrace{h_1, ..., h_1}_{d_1},\ \underbrace{h_2, ..., h_2}_{d_2},\ ...,\ \underbrace{h_L, ..., h_L}_{d_L}]$$

This is implemented via `torch.repeat_interleave`.

### 11.2 Log-Scale Prediction

The predictor outputs log(d), not d directly:

$$\hat{d}_i = \text{predictor}(h_i), \quad d_i = \exp(\hat{d}_i) - 1$$

The -1 shift comes from `exp(log(d+ε)) - 1 ≈ d` for small ε. This is because the loss uses:

$$\mathcal{L}_{dur} = \text{MSE}(\hat{d},\ \log(d_{GT} + \epsilon))$$

At inference: `durations = clamp(round(exp(pred) - 1), min=1)`. The `clamp(min=1)` ensures every **real** token gets at least 1 frame.

### 11.3 Dummy Duration Limitation

Currently, durations are computed as:

```python
avg_dur = max(1, mel_len // text_len)
durations = [avg_dur] * text_len  # uniform
durations[-1] += mel_len - sum(durations)  # fix remainder
```

This gives every character the same duration — factually wrong. 'a' and 'th' don't have the same acoustic duration. The model learns to predict equal durations and the decoder must compensate by generating different mel patterns for the same duration — a contradictory learning signal.

**Impact:** The duration loss converges quickly (predicting a constant is easy) but the mel loss suffers because the decoder receives incorrectly-timed expanded representations.

**Fix (planned):** Montreal Forced Aligner extracts per-phoneme durations from audio+transcript. This gives ground truth d_i per phoneme, resolving the contradictory signal.

---

## 12. Padding Mask in Inference

### 12.1 The Bug

In a batched text of shape [B, T_max], shorter sequences are padded with 0s. The duration predictor runs on all T_max positions. At inference:

```python
durations = clamp(round(exp(pred) - 1), min=1)  # all ≥ 1
```

Padding tokens (index 0) get `duration ≥ 1` → they expand into the output mel → output is too long.

### 12.2 The Fix

Generate a boolean mask before passing to VarianceAdaptor:

```python
src_mask = (text != 0)  # True = real token, False = padding
```

Then at inference, zero out predicted durations for padding positions before `repeat_interleave`:

```python
durations_to_use = durations * src_mask.float()
```

`repeat_interleave` with count=0 skips that position entirely. This ensures padding tokens contribute 0 frames to the output.

During training, padding already has `duration=0` (from `collate_fn`'s `padding_value=0`), so no masking needed — it's inference-only.

---

## 13. Numerical Stability

### 13.1 ε in β and γ

Both `weight_quant` and `activation_quant` clamp the scale to a minimum of 1e-5:

```python
beta  = w.abs().mean().clamp(min=1e-5)
gamma = x.abs().max().clamp(min=1e-5)
```

Without this: if W is all-zero (e.g., after initialization with some methods), β=0 → division by zero → NaN propagates through the entire network.

### 13.2 Loss Filter

The training loop skips batches where:

```python
loss > 15.0 or torch.isnan(loss) or torch.isinf(loss)
```

This is a hard guard against catastrophic loss spikes. In QAT, occasional spikes occur when the ternary assignment of a weight changes suddenly (the gradient landscape has discontinuities at the rounding boundary). Skipping these batches prevents a single bad batch from corrupting the accumulated gradients.

The threshold 15.0 is empirical — typical converged loss is ~0.5-2.0, so 15.0 catches genuine explosions while passing through normal early-training high losses.

### 13.3 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Clips the global gradient L2 norm to 1.0. This prevents any single large gradient (from a bad STE approximation or a hard mel target) from taking an excessively large step. Combined with the loss filter, this provides two layers of stability.

---

## 14. Why The Vocoder Is Full Precision

HiFi-GAN generates waveforms from mel spectrograms. The final convolution layer maps from a hidden representation to audio samples — it's the most perceptually sensitive layer in the entire system.

For 1.58-bit quantization of that layer: a weight error of ±β (the quantization error magnitude) in the output layer maps directly to an audio error. At 22050 Hz with 16-bit samples, the dynamic range requirement is very high. Ternary weights introduce systematic distortion patterns that manifest as audible buzzing artifacts.

The paper (Section 3.2) explicitly excludes the "convolutional layer closest to the waveform output" from quantization. We take the conservative choice of keeping the entire HiFi-GAN in FP32. The acoustic model (BitJETS) is quantized end-to-end; the vocoder is not.

This is a valid engineering tradeoff: the acoustic model accounts for the majority of parameters (the quantized ~92.9% of the system), so the size reduction is still substantial.
