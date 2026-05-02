"""
Weight indexing per BitTTS paper Algorithm 1.

Encoding: ternary {-1, 0, 1} → base-3 digits {2, 0, 1}
    0  →  0
    1  →  1
   -1  →  2

Block size L* = 5: each block of 5 ternary values is mapped to a base-3
integer n = d0*3^4 + d1*3^3 + d2*3^2 + d3*3 + d4, where d_i ∈ {0,1,2}.
Since 3^5 = 243 ≤ 256, each block fits in one uint8.

Size comparison for Conv1d(256, 256, kernel=5):
  FP32:   256 × 256 × 5 × 4 bytes  = 1,310,720 bytes = 1280 KB
  2-bit:  256 × 256 × 5 / 4 bytes  =   81,920 bytes  =   80 KB  (naive)
  Paper:  256 × 256 × 5 / 5 bytes  =   65,536 bytes  =   64 KB  ✓ matches paper footnote 6

Usage:
    packed = pack_state_dict(model.state_dict())
    torch.save(packed, "model_packed.pth")

    state_dict = unpack_state_dict(torch.load("model_packed.pth"))
    model.load_state_dict(state_dict)
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List

BLOCK_SIZE = 5          # L* from paper
_BIT_CONV_WEIGHT_SUFFIX = "bit_conv.weight"


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _ternary_to_code(v: torch.Tensor) -> torch.Tensor:
    """Map ternary {-1, 0, 1} → base-3 digit {2, 0, 1} (uint8)."""
    # 0→0, 1→1, -1→2
    code = torch.zeros_like(v, dtype=torch.uint8)
    code[v == 1]  = 1
    code[v == -1] = 2
    return code


def _code_to_ternary(code: torch.Tensor) -> torch.Tensor:
    """Map base-3 digit {0, 1, 2} → ternary {0, 1, -1} (float32)."""
    out = torch.zeros_like(code, dtype=torch.float32)
    out[code == 1] =  1.0
    out[code == 2] = -1.0
    return out


def _encode_blocks(codes: torch.Tensor) -> torch.Tensor:
    """
    Pack a 1-D uint8 tensor of base-3 digits into uint8 indices.
    Input length is padded to a multiple of BLOCK_SIZE.
    Algorithm 1 from paper: n = n*3 + digit, iterating MSB→LSB within block.
    Returns uint8 tensor of length ceil(numel / BLOCK_SIZE).
    """
    pad = (BLOCK_SIZE - len(codes) % BLOCK_SIZE) % BLOCK_SIZE
    if pad:
        codes = F.pad(codes.float(), (0, pad), value=0).to(torch.uint8)

    blocks = codes.reshape(-1, BLOCK_SIZE)  # [num_blocks, 5]

    # Compute base-3 index: n = d0*81 + d1*27 + d2*9 + d3*3 + d4
    multipliers = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32)
    indices = (blocks.to(torch.int32) * multipliers).sum(dim=1).to(torch.uint8)
    return indices


def _decode_blocks(indices: torch.Tensor, numel: int) -> torch.Tensor:
    """
    Unpack uint8 indices back to base-3 digits (uint8), then trim to numel.
    """
    # Decompose each index into 5 base-3 digits (MSB first)
    digits = torch.zeros(len(indices), BLOCK_SIZE, dtype=torch.uint8)
    idx = indices.to(torch.int32).clone()
    for pos in range(BLOCK_SIZE - 1, -1, -1):
        digits[:, pos] = (idx % 3).to(torch.uint8)
        idx = idx // 3

    return digits.flatten()[:numel]


# ---------------------------------------------------------------------------
# Tensor-level pack / unpack
# ---------------------------------------------------------------------------

def _pack_tensor(w: torch.Tensor) -> Dict[str, Any]:
    """
    Quantize weight tensor to ternary, then pack using Algorithm 1.
    Returns a dict with packed indices + metadata.
    """
    original_shape = w.shape
    beta = w.abs().mean().clamp(min=1e-5)

    # Quantize to ternary {-1, 0, 1}
    w_ternary = (w / beta).clamp(-1, 1).round()

    codes = _ternary_to_code(w_ternary.flatten())
    indices = _encode_blocks(codes)

    return {
        "indices": indices,          # uint8, one per block of 5
        "beta": beta,
        "shape": original_shape,
        "numel": w_ternary.numel(),
    }


def _unpack_tensor(d: Dict[str, Any]) -> torch.Tensor:
    """Reconstruct float32 ternary-scaled weight from packed dict."""
    codes = _decode_blocks(d["indices"], d["numel"])
    w_ternary = _code_to_ternary(codes)
    return (w_ternary * d["beta"]).reshape(d["shape"])


# ---------------------------------------------------------------------------
# State-dict level API
# ---------------------------------------------------------------------------

def pack_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Pack all BitConv1d weights in a state_dict using Algorithm 1.
    Non-BitConv weights (embeddings, LayerNorm, Linear) are kept as float32.
    """
    packed_sd: Dict[str, Any] = {}
    packed_keys: List[str] = []

    for key, tensor in state_dict.items():
        if key.endswith(_BIT_CONV_WEIGHT_SUFFIX):
            packed_sd[key] = _pack_tensor(tensor)
            packed_keys.append(key)
        else:
            packed_sd[key] = tensor

    packed_sd["__packed_keys__"] = packed_keys
    return packed_sd


def unpack_state_dict(packed_sd: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Reconstruct a regular state_dict from a packed one."""
    packed_keys = packed_sd.get("__packed_keys__", [])
    state_dict: Dict[str, torch.Tensor] = {}

    for key, value in packed_sd.items():
        if key == "__packed_keys__":
            continue
        if key in packed_keys:
            state_dict[key] = _unpack_tensor(value)
        else:
            state_dict[key] = value

    return state_dict


def save_packed(model, path: str) -> None:
    """Save model weights in packed format (Algorithm 1)."""
    packed = pack_state_dict(model.state_dict())
    torch.save(packed, path)
    # Compute size reduction stats
    n_packed = len(packed.get("__packed_keys__", []))
    print(f"Packed checkpoint saved: {path} ({n_packed} BitConv1d layers indexed)")


def load_packed(model, path: str, device: str = "cpu") -> None:
    """Load packed checkpoint into model (in-place)."""
    packed_sd = torch.load(path, map_location=device)
    state_dict = unpack_state_dict(packed_sd)
    model.load_state_dict(state_dict)
    print(f"Packed checkpoint loaded: {path}")
