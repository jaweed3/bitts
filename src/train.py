import os
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hparams import HParams
from dataset import LJSpeechDataset, collate_fn
from models import BitJETS
from checkpoint import find_latest_checkpoint, save_checkpoint, load_checkpoint

# ---------------------------------------------------------------------------
# WandB — optional, degrades gracefully if offline or not installed
# ---------------------------------------------------------------------------
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _wandb_log(data: dict, use_wandb: bool):
    if use_wandb and _WANDB_AVAILABLE:
        wandb.log(data)


def hparams_to_dict(cls):
    return {k: v for k, v in cls.__dict__.items() if k.isupper()}


# ---------------------------------------------------------------------------
# LR Scheduler: cosine annealing with warmup
#
# Exponential decay (gamma=0.9973) decays to ~0 within 10K steps for 500K
# training — way too aggressive. Cosine annealing sustains meaningful LR
# throughout training and is standard in modern TTS implementations.
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """
    Linear warmup → cosine annealing to min_lr_ratio × initial LR.

    Example (lr=2e-4, warmup=1000, total=500K, ratio=0.1):
      step 0:      0
      step 500:    1e-4   (warmup)
      step 1000:   2e-4   (peak)
      step 250K:   ~1.1e-4 (midpoint)
      step 500K:   2e-5   (floor, never below)
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Infinite dataloader — cycles through dataset continuously
# ---------------------------------------------------------------------------

def _infinite_loader(loader):
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main(args=None):
    device = HParams.DEVICE
    is_cuda = device.startswith("cuda")

    # --- Seed ---
    seed = getattr(args, "seed", None) or HParams.SEED
    random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed_all(seed)

    # --- WandB ---
    use_wandb = _WANDB_AVAILABLE and not getattr(args, "no_wandb", False)
    if use_wandb:
        try:
            wandb.init(
                project="BitTTS-M4-Erasmus",
                config=hparams_to_dict(HParams),
                name=f"run-bitnet-{device}",
                resume="allow",
            )
        except Exception as e:
            print(f"WandB init failed ({e}). Continuing without WandB.")
            use_wandb = False

    print(f"Training on: {device} | seed: {seed}")
    os.makedirs(HParams.CHECKPOINT_DIR, exist_ok=True)

    # --- Dataset ---
    print("Loading dataset...")
    train_ds = LJSpeechDataset(HParams.DATA_PATH, limit=None)
    train_loader = DataLoader(
        train_ds,
        batch_size=HParams.BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda,
    )
    data_iter = _infinite_loader(train_loader)

    # --- Model ---
    print("Initializing BitJETS (1.58-bit)...")
    model = BitJETS(
        vocab_size=HParams.VOCAB_SIZE,
        embed_dim=HParams.EMBED_DIM,
        hidden_dim=HParams.ENCODER_DIM,
        decoder_dim=HParams.DECODER_DIM,
        out_mel_dim=HParams.N_MELS,
    ).to(device)

    # --- Optimizer ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=HParams.LEARNING_RATE,
        betas=(0.8, 0.99),
        weight_decay=0.0,
    )

    # --- Scheduler ---
    ACCUM_STEPS = getattr(args, "accum_steps", None) or HParams.ACCUM_STEPS
    NUM_STEPS   = getattr(args, "num_steps", None)   or HParams.NUM_STEPS
    warmup_steps = HParams.WARMUP_STEPS
    scheduler = build_scheduler(optimizer, warmup_steps, NUM_STEPS, HParams.MIN_LR_RATIO)

    # --- Resume ---
    global_step = 0
    resume_path = None

    if args and getattr(args, "auto_resume", False):
        resume_path = find_latest_checkpoint(HParams.CHECKPOINT_DIR)
        if resume_path:
            print(f"Auto-resume: {resume_path}")
        else:
            print("Auto-resume: no checkpoint found, starting fresh.")
    elif args and getattr(args, "resume", None):
        resume_path = args.resume

    reset_lr = args and getattr(args, "reset_lr", False)

    if resume_path:
        global_step = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        if reset_lr:
            # Rebuild scheduler from scratch — useful when LR decayed to ~0
            # BUG FIX: Reset optimizer LR AND clear initial_lr so scheduler picks up the new base
            for param_group in optimizer.param_groups:
                param_group['lr'] = HParams.LEARNING_RATE
                if 'initial_lr' in param_group:
                    del param_group['initial_lr']

            scheduler = build_scheduler(optimizer, warmup_steps, NUM_STEPS, HParams.MIN_LR_RATIO)
            print(f"[reset-lr] Scheduler rebuilt. Warmup {warmup_steps}, "
                  f"peak lr={HParams.LEARNING_RATE:.1e}, floor={HParams.LEARNING_RATE * HParams.MIN_LR_RATIO:.1e}")
            # Force an initial step to update the optimizer LR immediately
            # otherwise it might stay at peak until the first optimizer.step()
            scheduler.step() 
            print(f"[reset-lr] Initial LR: {optimizer.param_groups[0]['lr']:.2e}")

    if use_wandb:
        wandb.watch(model, log="all", log_freq=200)

    # --- Loss ---
    criterion_mel = nn.MSELoss()
    criterion_dur = nn.MSELoss()

    eff_batch = HParams.BATCH_SIZE * ACCUM_STEPS
    print(f"Training | steps: {global_step:,} → {NUM_STEPS:,} | "
          f"batch: {HParams.BATCH_SIZE} × accum: {ACCUM_STEPS} = eff_bs: {eff_batch}")

    model.train()
    optimizer.zero_grad()
    accum_count = 0
    last_loss_val = float("nan")

    # --- Training health tracking ---
    loss_ema = None        # exponential moving average
    loss_ema_alpha = 0.05  # smoothing factor (lower = smoother)
    loss_best = float("inf")
    steps_since_best = 0
    plateau_warned = False

    # Adaptive loss gate: relaxed threshold for the first N steps after resume
    # so legacy checkpoints with moderate initial loss can break through.
    skip_threshold = HParams.LOSS_SKIP_THRESHOLD
    grace_threshold = HParams.LOSS_SKIP_THRESHOLD * 2.0
    steps_since_resume = 0
    in_grace = global_step == 0  # only apply grace when starting from step 0

    consecutive_skips = 0
    skip_diagnostics_printed = False

    while global_step < NUM_STEPS:
        text, mel_target, dur_target = next(data_iter)

        text       = text.to(device, non_blocking=is_cuda).long()
        mel_target = mel_target.to(device, non_blocking=is_cuda).float()
        dur_target = dur_target.to(device, non_blocking=is_cuda).float()

        if text.max() >= HParams.VOCAB_SIZE:
            continue

        mel_pred, log_dur_pred = model(text, target_durations=dur_target)

        min_len    = min(mel_pred.shape[1], mel_target.shape[1])
        mel_pred   = mel_pred[:, :min_len, :]
        mel_target = mel_target[:, :min_len, :]

        loss_mel = criterion_mel(mel_pred, mel_target)
        loss_dur = criterion_dur(log_dur_pred, torch.log(dur_target + 1e-4))
        loss     = loss_mel + loss_dur

        effective_threshold = grace_threshold if in_grace else skip_threshold

        if loss.item() > effective_threshold or torch.isnan(loss) or torch.isinf(loss):
            if not skip_diagnostics_printed:
                print(f"[DIAG] mel_pred  shape={list(mel_pred.shape)}  "
                      f"mean={mel_pred.mean().item():.3f}  std={mel_pred.std().item():.3f}  "
                      f"min={mel_pred.min().item():.3f}  max={mel_pred.max().item():.3f}")
                print(f"[DIAG] mel_target shape={list(mel_target.shape)} "
                      f"mean={mel_target.mean().item():.3f}  std={mel_target.std().item():.3f}  "
                      f"min={mel_target.min().item():.3f}  max={mel_target.max().item():.3f}")
                print(f"[DIAG] loss_mel={loss_mel.item():.3f}  loss_dur={loss_dur.item():.3f}  "
                      f"threshold={effective_threshold:.1f}" +
                      (" (grace period)" if in_grace else ""))
                skip_diagnostics_printed = True

            print(f"[SKIP] Loss {loss.item():.2f} > {effective_threshold:.0f} "
                  f"@ step {global_step}  (consecutive skips: {consecutive_skips + 1})")
            consecutive_skips += 1

            if consecutive_skips >= HParams.MAX_CONSECUTIVE_SKIPS:
                print(f"\n[FATAL] {consecutive_skips} consecutive batches skipped.")
                print(f"  Loss stuck at ~{loss.item():.1f} (threshold: {effective_threshold:.0f}).")
                if global_step == 0 and in_grace:
                    print(f"  Grace period ({grace_threshold:.0f}) also failing.")
                print(f"  Suggestions:")
                print(f"    1. Start fresh:  ./scripts/train.sh")
                print(f"    2. Raise threshold: edit src/hparams.py LOSS_SKIP_THRESHOLD=25")
                print(f"    3. Verify dataset:  ls data/speech/wavs/ | wc -l")
                print(f"  Exiting.")
                if use_wandb:
                    wandb.finish()
                return

            continue

        # Batch accepted — reset skip counter, exit grace after enough steps
        consecutive_skips = 0
        skip_diagnostics_printed = False

        if in_grace:
            steps_since_resume += 1
            if steps_since_resume >= HParams.SKIP_GRACE_STEPS:
                in_grace = False
                print(f"[INFO] Grace period ended. Threshold now {skip_threshold:.0f}.")

        (loss / ACCUM_STEPS).backward()
        accum_count += 1

        if accum_count < ACCUM_STEPS:
            continue

        # --- Optimizer step ---
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        accum_count = 0

        last_loss_val = loss.item()
        global_step  += 1

        # --- Health tracking ---
        if loss_ema is None:
            loss_ema = last_loss_val
        else:
            loss_ema = loss_ema_alpha * last_loss_val + (1 - loss_ema_alpha) * loss_ema

        if last_loss_val < loss_best:
            loss_best = last_loss_val
            steps_since_best = 0
        else:
            steps_since_best += 1

        # --- Logging ---
        if global_step % HParams.LOG_INTERVAL == 0:
            lr_now = optimizer.param_groups[0]["lr"]

            # Trend indicator
            plateau_watch = HParams.LOG_INTERVAL * HParams.PLATEAU_PATIENCE
            if steps_since_best >= plateau_watch * 3:
                trend = "🔴 STUCK"
            elif steps_since_best >= plateau_watch:
                trend = "🟡 flat"
            elif loss_ema < loss_best * 1.05:
                trend = "🟢 ok"
            else:
                trend = "🟠 slow"

            print(f"Step {global_step:>7,}/{NUM_STEPS:,} | "
                  f"loss: {last_loss_val:.4f} (ema={loss_ema:.4f} best={loss_best:.4f}) | "
                  f"mel={loss_mel.item():.4f} dur={loss_dur.item():.4f} | "
                  f"grad: {total_norm.item():.2f} | lr: {lr_now:.2e} | {trend}")

            # Plateau warning
            if steps_since_best >= plateau_watch and not plateau_warned:
                plateau_warned = True
                print(f"  ⚠ Plateau detected: no improvement for {steps_since_best} steps.")
                print(f"     Best loss: {loss_best:.4f} | Current EMA: {loss_ema:.4f}")
                if lr_now < 1e-7:
                    print(f"     LR is very low ({lr_now:.1e}). Consider --reset-lr.")

            if steps_since_best < plateau_watch:
                plateau_warned = False

            _wandb_log({
                "train_loss": last_loss_val,
                "loss_ema":   loss_ema,
                "loss_best":  loss_best,
                "loss_mel":   loss_mel.item(),
                "loss_dur":   loss_dur.item(),
                "grad_norm":  total_norm.item(),
                "lr":         lr_now,
                "global_step": global_step,
            }, use_wandb)

        # --- Checkpoint: latest every step (overwrite), named every CKPT_INTERVAL ---
        save_checkpoint(model, optimizer, scheduler, global_step, last_loss_val,
                        HParams.CHECKPOINT_DIR)
        if global_step % HParams.CKPT_INTERVAL == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, last_loss_val,
                            HParams.CHECKPOINT_DIR, tag=str(global_step))

    print(f"Training complete at step {global_step:,}.")
    if use_wandb:
        wandb.finish()
