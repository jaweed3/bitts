import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import wandb

from hparams import HParams
from dataset import LJSpeechDataset, collate_fn
from models import BitJETS

def hparams_to_dict(cls):
    return {k: v for k, v in cls.__dict__.items() if k.isupper()}

def main(args=None):
    # Setup WandB
    wandb.init(
        project="BitTTS-M4-Erasmus",
        config=hparams_to_dict(HParams),
        name=f"run-bitnet-{HParams.DEVICE}",
        resume="allow"
    )

    print(f"🚀 Training on device => {HParams.DEVICE}")
    os.makedirs(HParams.CHECKPOINT_DIR, exist_ok=True)

    print("Loading dataset...")
    # Limit hapus kalo production
    train_ds = LJSpeechDataset(HParams.DATA_PATH, limit=None) 
    train_loader = DataLoader(
        train_ds,
        batch_size=HParams.BATCH_SIZE, # Fisik: 8
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0,      # WAJIB 0 BUAT MPS
        pin_memory=False    # WAJIB False BUAT MPS
    )

    print("Initializing BitJETS (1.58 bit)...")
    model = BitJETS(
        vocab_size=HParams.VOCAB_SIZE,
        embed_dim=HParams.EMBED_DIM,
        hidden_dim=HParams.ENCODER_DIM,
        decoder_dim=HParams.DECODER_DIM,
        out_mel_dim=HParams.N_MELS
    ).to(HParams.DEVICE)

    # --- [FIX 1] RESUME LOGIC (WEIGHTS ONLY) ---
    if args and args.resume:
        print(f"♻️  Resuming weights from: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=HParams.DEVICE)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Checkpoint weights loaded successfully!")
        except Exception as e:
            print(f"❌ Gagal load checkpoint: {e}")
            return

    wandb.watch(model, log="all", log_freq=50)

    # --- [FIX 2] OPTIMIZER SESUAI PAPER ---
    print("🔧 Initializing Fresh Optimizer (Betas=[0.8, 0.99])")
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=HParams.LEARNING_RATE,
        betas=(0.8, 0.99), # CRUCIAL
        weight_decay=0.0   # CRUCIAL
    )
    
    criterion_mel = nn.MSELoss()
    criterion_dur = nn.MSELoss()

    # --- [FIX 3] GRADIENT ACCUMULATION CONFIG ---
    # Target Effective Batch Size = 32
    # Jika Physical Batch Size (HParams) = 8, maka Accum Steps = 4
    ACCUM_STEPS = 4 
    print(f"🔥 Start training (ACCUMULATION MODE: {ACCUM_STEPS} steps)...")
    print(f"ℹ️  Effective Batch Size: {HParams.BATCH_SIZE * ACCUM_STEPS}")

    model.train()

    start_epoch = args.start_epoch if args else 0
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, HParams.NUM_EPOCHS):
        total_loss_epoch = 0
        
        # Zero grad di awal epoch
        optimizer.zero_grad()

        for batch_idx, (text, mel_target, dur_target) in enumerate(train_loader):
            
            # MPS Specifics
            text = text.to(HParams.DEVICE).long().contiguous()
            mel_target = mel_target.to(HParams.DEVICE).float().contiguous()
            dur_target = dur_target.to(HParams.DEVICE).float().contiguous()

            # Safety check index vocab
            if text.max() >= HParams.VOCAB_SIZE:
                print(f"[SKIP] Index out of bound: {text.max()}")
                continue

            # Forward
            mel_pred, log_dur_pred = model(text, target_durations=dur_target)

            # Slicing
            min_len = min(mel_pred.shape[1], mel_target.shape[1])
            mel_pred = mel_pred[:, :min_len, :]
            mel_target = mel_target[:, :min_len, :]

            # Loss Calc
            loss_mel = criterion_mel(mel_pred, mel_target)
            log_dur_target = torch.log(dur_target.float() + 1e-4) 
            loss_dur = criterion_dur(log_dur_pred, log_dur_target)

            loss = loss_mel + loss_dur

            # --- [FIX 4] SAFETY TRAP (THE FILTER) ---
            # Tangkap loss sampah SEBELUM masuk akumulasi
            if loss.item() > 15.0 or torch.isnan(loss) or torch.isinf(loss):
                print(f"\n🚨 [ALARM] Loss Explosion: {loss.item()} | Batch {batch_idx}")
                # Skip batch ini, jangan kotori gradien yang sedang terkumpul
                continue 

            # --- [FIX 5] NORMALIZE LOSS ---
            # Bagi loss dengan ACCUM_STEPS agar rata-rata gradien benar
            loss = loss / ACCUM_STEPS

            # Backward (Akumulasi Gradien terjadi di sini)
            loss.backward()

            # --- [FIX 6] STEP OPTIMIZER HANYA SETIAP 4 BATCH ---
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                
                # Clip Gradient (Atomic)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update & Bersihkan
                optimizer.step()
                optimizer.zero_grad()

                # Logging (Kembalikan nilai loss ke skala asli untuk display)
                real_loss_val = loss.item() * ACCUM_STEPS
                
                if global_step % HParams.LOG_INTERVAL == 0:
                    wandb.log({
                        "train_loss": real_loss_val,
                        "grad_norm": total_norm.item(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "lr": optimizer.param_groups[0]['lr']
                    })
                    print(f"Ep {epoch} | Step {global_step} | Loss: {real_loss_val:.4f} | Grad: {total_norm.item():.2f}")
                
                global_step += 1
                total_loss_epoch += real_loss_val

        # End of Epoch Log
        # Hitung rata-rata loss epoch (agak kasar karena akumulasi, tapi cukup buat indikator)
        avg_loss = total_loss_epoch / (len(train_loader) / ACCUM_STEPS)
        print(f"✅ Epoch {epoch} Completed | Est. Avg Loss: {avg_loss:.4f}")

        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = f"{HParams.CHECKPOINT_DIR}/bitjets_ckpt_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    wandb.finish()
