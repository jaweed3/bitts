"""
Smoke test: full training loop with a synthetic dataset.
No LJSpeech required — uses random mel/text tensors.
Verifies: forward pass, backward pass, optimizer step, checkpoint save, auto-resume.
"""
import sys
import os
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import BitJETS
from hparams import HParams


# ---------------------------------------------------------------------------
# Tiny synthetic dataset — no disk I/O, no LJSpeech dependency
# ---------------------------------------------------------------------------

class FakeDataset(Dataset):
    """Generates random (text, mel, duration) triples for testing."""
    def __init__(self, n=64, text_len=12, mel_len=48, n_mels=80):
        self.n       = n
        self.text_len = text_len
        self.mel_len  = mel_len
        self.n_mels   = n_mels

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # text: random vocab indices (1..VOCAB_SIZE-1, avoid 0=padding)
        text = torch.randint(1, HParams.VOCAB_SIZE, (self.text_len,))
        mel  = torch.randn(self.mel_len, self.n_mels)
        # durations that sum to mel_len
        dur  = torch.ones(self.text_len, dtype=torch.long) * (self.mel_len // self.text_len)
        dur[-1] += self.mel_len - dur.sum()
        return text, mel, dur.float()


def fake_collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    texts, mels, durs = zip(*batch)
    return (
        pad_sequence(texts, batch_first=True, padding_value=0),
        pad_sequence(mels,  batch_first=True, padding_value=-11.5),
        pad_sequence(durs,  batch_first=True, padding_value=0),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(device="cpu"):
    return BitJETS(
        vocab_size=HParams.VOCAB_SIZE,
        embed_dim=64,
        hidden_dim=64,
        decoder_dim=48,
        out_mel_dim=80,
    ).to(device)


def _run_steps(model, loader, n_steps, ckpt_dir, device="cpu"):
    """Run n_steps optimizer steps. Returns list of loss values."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.8, 0.99), weight_decay=0.0)
    criterion = nn.MSELoss()
    losses = []

    from checkpoint import save_checkpoint

    model.train()
    optimizer.zero_grad()
    data_iter = iter(loader)
    step = 0

    while step < n_steps:
        try:
            text, mel_target, dur_target = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            text, mel_target, dur_target = next(data_iter)

        text       = text.to(device).long()
        mel_target = mel_target.to(device).float()
        dur_target = dur_target.to(device).float()

        mel_pred, log_dur_pred = model(text, target_durations=dur_target)
        min_len    = min(mel_pred.shape[1], mel_target.shape[1])
        mel_pred   = mel_pred[:, :min_len, :]
        mel_target = mel_target[:, :min_len, :]

        loss_mel = criterion(mel_pred, mel_target)
        loss_dur = criterion(log_dur_pred, torch.log(dur_target + 1e-4))
        loss     = loss_mel + loss_dur

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        step += 1

        # Save checkpoint every 10 steps
        if step % 10 == 0:
            save_checkpoint(model, optimizer, None, step, loss.item(), ckpt_dir,
                            tag=str(step) if step == n_steps else None)

    return losses


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_backward_smoke():
    """Single forward + backward pass should not crash."""
    model = _make_model()
    text  = torch.randint(1, HParams.VOCAB_SIZE, (2, 10))
    dur   = torch.randint(1, 4, (2, 10)).float()
    mel   = torch.randn(2, dur.sum(dim=1).max().int().item(), 80)

    mel_pred, log_dur = model(text, target_durations=dur)
    min_len = min(mel_pred.shape[1], mel.shape[1])
    loss = nn.MSELoss()(mel_pred[:, :min_len], mel[:, :min_len]) + log_dur.mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed"


def test_training_loop_runs():
    """50 optimizer steps should complete without exception."""
    ds     = FakeDataset(n=32, text_len=10, mel_len=40)
    loader = DataLoader(ds, batch_size=4, collate_fn=fake_collate, shuffle=True)
    model  = _make_model()

    with tempfile.TemporaryDirectory() as ckpt_dir:
        losses = _run_steps(model, loader, n_steps=50, ckpt_dir=ckpt_dir)

    assert len(losses) == 50
    assert all(not (v != v) for v in losses), "NaN loss encountered"   # NaN check
    assert all(v < 1e6 for v in losses),      "Loss exploded"


def test_loss_decreases():
    """Loss should trend downward over 100 steps (overfit on tiny dataset)."""
    torch.manual_seed(0)
    ds     = FakeDataset(n=8, text_len=10, mel_len=40)   # tiny — model should overfit
    loader = DataLoader(ds, batch_size=8, collate_fn=fake_collate, shuffle=False)
    model  = _make_model()

    with tempfile.TemporaryDirectory() as ckpt_dir:
        losses = _run_steps(model, loader, n_steps=100, ckpt_dir=ckpt_dir)

    first_10 = sum(losses[:10]) / 10
    last_10  = sum(losses[-10:]) / 10
    assert last_10 < first_10, (
        f"Loss did not decrease: first_10_avg={first_10:.4f}, last_10_avg={last_10:.4f}"
    )


def test_checkpoint_save_and_resume():
    """Save checkpoint, create new model, load checkpoint, verify weights match."""
    from checkpoint import save_checkpoint, load_checkpoint

    ds     = FakeDataset(n=16, text_len=10, mel_len=40)
    loader = DataLoader(ds, batch_size=4, collate_fn=fake_collate)
    model  = _make_model()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as ckpt_dir:
        # Run 20 steps and save
        _run_steps(model, loader, n_steps=20, ckpt_dir=ckpt_dir)

        latest = os.path.join(ckpt_dir, "latest.pth")
        assert os.path.exists(latest), "latest.pth not created"

        # Load into fresh model
        model2    = _make_model()
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        step = load_checkpoint(latest, model2, optimizer2, None, device="cpu")

        assert step > 0, "Step count not restored from checkpoint"

        # Weights should match
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert torch.allclose(v1, v2), f"Weight mismatch after resume: {k1}"


def test_auto_resume_finds_latest():
    """find_latest_checkpoint should return latest.pth if present."""
    from checkpoint import find_latest_checkpoint

    with tempfile.TemporaryDirectory() as ckpt_dir:
        # No checkpoint yet
        assert find_latest_checkpoint(ckpt_dir) is None

        # Create latest.pth
        path = os.path.join(ckpt_dir, "latest.pth")
        torch.save({"global_step": 1000}, path)
        assert find_latest_checkpoint(ckpt_dir) == path


def test_no_nan_in_quantized_forward():
    """BitConv1d should not produce NaN even with extreme inputs."""
    from layers import BitConvBlock
    block = BitConvBlock(64, 64, kernel_size=3, padding=1)

    # Test with normal, large, and near-zero inputs
    for scale in [1.0, 100.0, 1e-6]:
        x = torch.randn(2, 20, 64) * scale
        y = block(x)
        assert not torch.isnan(y).any(), f"NaN output with scale={scale}"
        assert not torch.isinf(y).any(), f"Inf output with scale={scale}"
