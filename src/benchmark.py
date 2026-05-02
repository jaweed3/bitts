"""
BitJETS Inference Benchmark
Measures: parameter count, model size (FP32 vs packed), latency, RTF.
"""

import os
import time
import tempfile
import torch

from hparams import HParams
from models import BitJETS
from packing import pack_state_dict, save_packed


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_bitconv_params(model) -> int:
    total = 0
    for name, p in model.named_parameters():
        if "bit_conv.weight" in name:
            total += p.numel()
    return total


def _model_size_fp32_bytes(model) -> int:
    return sum(p.numel() * 4 for p in model.parameters())


def _packed_size_bytes(model) -> int:
    """Estimate packed size: BitConv1d weights as uint8 indices (L*=5), rest FP32."""
    packed_bytes = 0
    fp32_bytes   = 0
    for name, p in model.named_parameters():
        if "bit_conv.weight" in name:
            # ceil(numel / 5) uint8 indices
            packed_bytes += (p.numel() + 4) // 5
        else:
            fp32_bytes += p.numel() * 4
    return packed_bytes + fp32_bytes


def _actual_packed_size_bytes(model) -> int:
    """Save packed checkpoint to temp file and measure actual disk size."""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
    try:
        save_packed(model, tmp_path)
        return os.path.getsize(tmp_path)
    finally:
        os.remove(tmp_path)


def _measure_latency(model, text_len: int, n_runs: int = 50, device: str = "cpu") -> float:
    """Return mean inference latency in ms over n_runs."""
    model.eval()
    # Single-sentence inference (no target_durations → inference mode)
    text = torch.randint(1, HParams.VOCAB_SIZE, (1, text_len)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(text, duration_control=1.0)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            mel, _ = model(text, duration_control=1.0)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    return elapsed_ms / n_runs, mel.shape[1]


def main(args=None):
    device = HParams.DEVICE
    ckpt_path = getattr(args, "model_path", None) if args else None

    print("=" * 60)
    print("  BitJETS Inference Benchmark")
    print("=" * 60)

    model = BitJETS(
        vocab_size=HParams.VOCAB_SIZE,
        embed_dim=HParams.EMBED_DIM,
        hidden_dim=HParams.ENCODER_DIM,
        decoder_dim=HParams.DECODER_DIM,
        out_mel_dim=HParams.N_MELS,
    ).to(device)

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        print(f"Loaded: {ckpt_path}")
    else:
        print("No checkpoint provided — using random weights (structure benchmark only).")

    # --- Parameter count ---
    total_params    = _count_params(model)
    bitconv_params  = _count_bitconv_params(model)
    other_params    = total_params - bitconv_params

    print(f"\n--- Parameters ---")
    print(f"  Total:          {total_params:>12,}")
    print(f"  BitConv1d:      {bitconv_params:>12,}  ({bitconv_params/total_params*100:.1f}%)")
    print(f"  Other (FP32):   {other_params:>12,}  ({other_params/total_params*100:.1f}%)")

    # --- Model size ---
    fp32_bytes   = _model_size_fp32_bytes(model)
    packed_est   = _packed_size_bytes(model)
    packed_actual = _actual_packed_size_bytes(model)
    reduction    = (1 - packed_actual / fp32_bytes) * 100

    print(f"\n--- Model Size ---")
    print(f"  FP32 (baseline):   {fp32_bytes/1024:.1f} KB  ({fp32_bytes/1024/1024:.2f} MB)")
    print(f"  Packed (estimate): {packed_est/1024:.1f} KB")
    print(f"  Packed (actual):   {packed_actual/1024:.1f} KB  ({packed_actual/1024/1024:.2f} MB)")
    print(f"  Compression:       {reduction:.1f}% reduction vs FP32")

    # --- Latency ---
    test_cases = [
        ("short  (10 chars)", 10),
        ("medium (30 chars)", 30),
        ("long   (60 chars)", 60),
    ]

    print(f"\n--- Inference Latency (device: {device}) ---")
    print(f"  {'Text length':<22} {'Latency':>10} {'Mel frames':>12} {'RTF':>10}")
    print(f"  {'-'*56}")

    for label, text_len in test_cases:
        lat_ms, mel_frames = _measure_latency(model, text_len, n_runs=50, device=device)
        # Audio duration = mel_frames * hop_length / sample_rate
        audio_dur_s = mel_frames * HParams.HOP_LENGTH / HParams.SAMPLE_RATE
        # RTF = processing_time / audio_duration (lower is better)
        rtf = (lat_ms / 1000) / max(audio_dur_s, 1e-6)
        print(f"  {label:<22} {lat_ms:>8.2f}ms {mel_frames:>12} {rtf:>9.4f}x")

    print(f"\n  RTF < 1.0 means faster than real-time.")
    print("=" * 60)
