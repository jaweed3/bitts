"""Test BitJETS model: shapes, forward pass, backward pass."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from models import BitJETS, VarianceAdaptor


def _make_model():
    return BitJETS(vocab_size=32, embed_dim=64, hidden_dim=64, decoder_dim=48, out_mel_dim=80)


def test_forward_with_target_durations():
    model = _make_model()
    text = torch.randint(0, 32, (2, 10))
    durations = torch.randint(1, 4, (2, 10))
    mel, log_dur = model(text, target_durations=durations)

    expected_mel_len = durations.sum(dim=1).max().item()
    assert mel.shape[0] == 2
    assert mel.shape[1] == expected_mel_len
    assert mel.shape[2] == 80
    assert log_dur.shape == (2, 10)


def test_forward_inference_no_duration():
    """Inference mode: no target_durations, model predicts own durations."""
    model = _make_model()
    model.eval()
    text = torch.randint(1, 32, (1, 8))  # no padding (all non-zero)
    with torch.no_grad():
        mel, log_dur = model(text, duration_control=1.0)
    assert mel.shape[0] == 1
    assert mel.shape[2] == 80


def test_padding_mask_inference():
    """Padding tokens (index 0) should not be expanded during inference."""
    model = _make_model()
    model.eval()

    # text with padding: real tokens 1..5, then padding 0s
    text_padded = torch.tensor([[1, 2, 3, 0, 0]])  # last 2 are padding
    text_full = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        mel_padded, _ = model(text_padded, duration_control=1.0)
        mel_full, _ = model(text_full, duration_control=1.0)

    # padded version should produce same mel length as unpadded
    assert mel_padded.shape[1] == mel_full.shape[1], (
        f"Padding leaked into output: padded={mel_padded.shape[1]}, full={mel_full.shape[1]}"
    )


def test_backward():
    model = _make_model()
    text = torch.randint(0, 32, (2, 6))
    durations = torch.randint(1, 3, (2, 6))
    mel, log_dur = model(text, target_durations=durations)
    loss = mel.mean() + log_dur.mean()
    loss.backward()
    # Check at least one parameter has gradient
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients flowed back through model"


def test_variance_adaptor_length_regulation():
    adaptor = VarianceAdaptor(hidden_dim=32)
    x = torch.randn(2, 4, 32)
    durations = torch.tensor([[2, 3, 1, 2], [1, 1, 4, 1]])
    out, log_dur = adaptor(x, target_durations=durations.float())
    expected_len = durations.sum(dim=1).max().item()
    assert out.shape == (2, expected_len, 32), f"Got {out.shape}"
