"""Test 2-bit weight packing round-trip."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from models import BitJETS
from packing import pack_state_dict, unpack_state_dict


def _make_model():
    return BitJETS(vocab_size=32, embed_dim=64, hidden_dim=64, decoder_dim=48, out_mel_dim=80)


def test_pack_unpack_roundtrip():
    """Unpacked weights should be ternary and close to original quantized weights."""
    model = _make_model()
    original_sd = model.state_dict()

    packed = pack_state_dict(original_sd)
    restored = unpack_state_dict(packed)

    assert set(restored.keys()) == set(original_sd.keys()), "Key mismatch after round-trip"

    # BitConv1d weights after unpack = beta * {-1, 0, 1}, so exactly 3 distinct values
    for key in packed.get("__packed_keys__", []):
        w = restored[key]
        n_unique = w.unique().numel()
        assert n_unique <= 3, f"{key}: expected ≤3 unique values (ternary * beta), got {n_unique}"


def test_packed_model_loads():
    """Model should load from unpacked state_dict without errors."""
    model = _make_model()
    packed = pack_state_dict(model.state_dict())
    restored_sd = unpack_state_dict(packed)
    model2 = _make_model()
    model2.load_state_dict(restored_sd)  # should not raise


def test_packed_size_smaller():
    """Packed indices (uint8, 5 values/byte) should be ~5x smaller than FP32."""
    model = _make_model()
    sd = model.state_dict()
    packed = pack_state_dict(sd)

    for key in packed.get("__packed_keys__", []):
        original_bytes = sd[key].numel() * 4  # float32 = 4 bytes per value
        # Each uint8 index encodes BLOCK_SIZE=5 ternary values
        index_bytes = packed[key]["indices"].numel()  # uint8 indices
        assert index_bytes < original_bytes, \
            f"{key}: indices ({index_bytes}B) not smaller than FP32 ({original_bytes}B)"
        # Should be roughly numel/5 bytes (5 values per uint8)
        expected_approx = sd[key].numel() / 5
        assert index_bytes <= expected_approx * 1.05, \
            f"{key}: index count {index_bytes} too large, expected ~{expected_approx:.0f}"
