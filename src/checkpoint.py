"""
Checkpoint utilities — isolated from dataset/torchaudio imports
so they can be used in tests without triggering heavy dependencies.
"""

import os
import glob
import torch

from packing import save_packed


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    """Return path to the latest checkpoint in ckpt_dir, or None.

    Tries latest.pth first. If it's corrupt, falls back to named checkpoints.
    """
    latest = os.path.join(ckpt_dir, "latest.pth")
    if os.path.exists(latest):
        # Verify it's a valid checkpoint
        try:
            torch.load(latest, map_location="cpu", weights_only=False)
            return latest
        except Exception:
            print(f"[WARN] latest.pth is corrupt — falling back to named checkpoint.")

    pattern = os.path.join(ckpt_dir, "bitjets_ckpt_*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _step_num(path):
        try:
            return int(os.path.basename(path).replace("bitjets_ckpt_", "").replace(".pth", ""))
        except ValueError:
            return -1

    return max(candidates, key=_step_num)


def _atomic_save(payload: dict, path: str) -> None:
    """Write to temp file then atomic rename — prevents corruption on Ctrl+C."""
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX, best-effort on Windows


def save_checkpoint(model, optimizer, scheduler, global_step: int, loss: float,
                    ckpt_dir: str, tag: str = None, extra_state: dict = None):
    """Save full training state (weights + optimizer + scheduler + step)."""
    payload = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }
    if extra_state:
        payload["extra_state"] = extra_state

    _atomic_save(payload, os.path.join(ckpt_dir, "latest.pth"))

    if tag:
        path = os.path.join(ckpt_dir, f"bitjets_ckpt_{tag}.pth")
        _atomic_save(payload, path)
        print(f"Checkpoint saved: {path}")
        packed_path = os.path.join(ckpt_dir, f"bitjets_packed_{tag}.pth")
        save_packed(model, packed_path)


def load_checkpoint(path: str, model, optimizer, scheduler, device: str) -> int:
    """Load full training state. Returns (global_step, extra_state dict or None)."""
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Optimizer state restored.")
        except Exception as e:
            print(f"Could not restore optimizer state: {e}. Fresh optimizer.")

    if scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print("Scheduler state restored.")
        except Exception as e:
            print(f"Could not restore scheduler state: {e}.")

    step = ckpt.get("global_step", 0)
    if step == 0:
        epoch = ckpt.get("epoch", 0)
        print(f"Legacy checkpoint (epoch {epoch}). Step counter reset to 0.")
    else:
        print(f"Resuming from step {step:,}.")

    extra_state = ckpt.get("extra_state", None)
    if extra_state:
        print("Extra state restored (alignment projector, etc.).")

    return step, extra_state
