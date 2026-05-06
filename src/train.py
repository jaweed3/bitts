import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from hparams import HParams
from dataset import LJSpeechDataset, collate_fn
from models import BitJETS
from checkpoint import find_latest_checkpoint, save_checkpoint, load_checkpoint
from mas import batch_extract_durations

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

    # --- Alignment head: projects encoder output → mel space for MAS ---
    # This small Linear layer learns what the encoder output "sounds like"
    # so MAS can find the right text↔mel alignment.
    align_proj = nn.Linear(HParams.ENCODER_DIM, HParams.N_MELS).to(device)

    # --- Optimizer (includes alignment head) ---
    optimizer = optim.AdamW(
        list(model.parameters()) + list(align_proj.parameters()),
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
        global_step, extra_state = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        if extra_state and "align_proj" in extra_state:
            try:
                align_proj.load_state_dict(extra_state["align_proj"])
                print("Alignment projector restored.")
            except Exception as e:
                print(f"Could not restore align_proj: {e}. Fresh projector.")
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
    criterion_mel = nn.L1Loss()   # L1 is more robust to alignment shifts than MSE
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

        # --- Encoder (shared) ---
        src_mask = text != 0
        encoder_out = model.encoder(text)  # [B, T_text, H]

        # --- MAS alignment: extract real durations from encoder↔mel match ---
        # Only after warmup — encoder needs time to learn basic phoneme representations.
        do_align = (global_step >= HParams.MAS_START_STEP and
                    global_step % HParams.ALIGN_INTERVAL == 0 and
                    global_step < NUM_STEPS - 100)
        do_align_loss = do_align and (global_step % HParams.ALIGN_LOSS_INTERVAL == 0)

        if do_align:
            with torch.no_grad():
                enc_proj = align_proj(encoder_out.detach())
                dur_target = batch_extract_durations(
                    enc_proj, mel_target, src_mask
                ).to(device)

        # --- Variance adaptor + decoder ---
        expanded_out, log_dur_pred = model.variance_adaptor(
            encoder_out, target_durations=dur_target, src_mask=src_mask
        )
        mel_pred = model.decoder(expanded_out)

        # --- Trim to matching length ---
        min_len    = min(mel_pred.shape[1], mel_target.shape[1])
        mel_pred   = mel_pred[:, :min_len, :]
        mel_target_trimmed = mel_target[:, :min_len, :]

        # --- Losses: L1 for mel (robust to alignment noise) ---
        loss_mel = criterion_mel(mel_pred, mel_target_trimmed)
        loss_dur = criterion_dur(log_dur_pred, torch.log(dur_target.float() + 1e-4))
        loss     = loss_mel + loss_dur

        # --- Alignment projection loss (trains the encoder→mel mapping) ---
        if do_align_loss:
            enc_proj_train = align_proj(encoder_out)  # with gradients
            # Length-regulate the projected encoder using the MAS durations
            dur_long = dur_target.long().clamp(min=1)
            enc_aligned_list = []
            for b in range(encoder_out.shape[0]):
                ea = torch.repeat_interleave(
                    enc_proj_train[b], dur_long[b], dim=0
                )  # [T_mel_est, 80]
                enc_aligned_list.append(ea)
            # Pad to same length
            enc_aligned = torch.nn.utils.rnn.pad_sequence(
                enc_aligned_list, batch_first=True
            )  # [B, max_T, 80]
            min_align = min(enc_aligned.shape[1], mel_target_trimmed.shape[1])
            align_loss = F.l1_loss(
                enc_aligned[:, :min_align, :],
                mel_target_trimmed[:, :min_align, :]
            )
            loss = loss + 0.3 * align_loss
        else:
            align_loss = torch.tensor(0.0)

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

            align_tag = "[MAS]" if do_align else "     "
            print(f"{align_tag} Step {global_step:>7,}/{NUM_STEPS:,} | "
                  f"loss: {last_loss_val:.4f} (ema={loss_ema:.4f} best={loss_best:.4f}) | "
                  f"mel={loss_mel.item():.4f} dur={loss_dur.item():.4f}" +
                  (f" align={align_loss.item():.4f}" if do_align_loss else "") +
                  f" | grad: {total_norm.item():.2f} | lr: {lr_now:.2e} | {trend}")

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
                        HParams.CHECKPOINT_DIR, extra_state={"align_proj": align_proj.state_dict()})
        if global_step % HParams.CKPT_INTERVAL == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, last_loss_val,
                            HParams.CHECKPOINT_DIR, tag=str(global_step),
                            extra_state={"align_proj": align_proj.state_dict()})

    print(f"Training complete at step {global_step:,}.")
    if use_wandb:
        wandb.finish()
