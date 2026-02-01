import argparse
import sys
import os
import torch
from dotenv import load_dotenv

# Load .env paling awal
load_dotenv()

# Setup Path: Pastikan folder src terbaca
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import HParams dan Pipelines
# Asumsi struktur foldermu:
# root/
#   main.py
#   src/
#     hparams.py
#     train.py
#     inference.py
#     models.py
from src.hparams import HParams
from src.train import main as train_pipeline
from src.inference import main as inference_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="BitTTS: BitNet 1.58-bit TTS framework.")

    parser.add_argument('mode', choices=['train', 'infer'], help="Mode: 'train' or 'infer'")
    
    # Overrides
    parser.add_argument('--device', type=str, default=None, help="Overide device (mps/cuda/cpu)")
    parser.add_argument('--epochs', type=int, default=None, help="Override epochs")
    parser.add_argument('--batch_size', type=int, default=None, help="Override batch size")
    parser.add_argument('--checkpoint_dir', type=str, default=None, help="Override checkpoint dir")
    
    # Inference specific
    parser.add_argument("--text" ,type=str, default="Hello world, this is BitNet TTS", help="Text to speak")
    parser.add_argument("--model_path", type=str, help="Path to .pth model")
    parser.add_argument("--output", type=str, default='output.wav', help="Output filename")

    # Training specific
    parser.add_argument('--resume', type=str, default=None, help='Path ke file checkpoint (.pth) untuk dilanjutkan')
    parser.add_argument('--start_epoch', type=int, default=0, help="Epoch awal (berguna jika resume)")

    return parser.parse_args()

def update_hparams(args):
    """
    Update HParams global state berdasarkan argumen CLI
    """
    # 1. Update Device
    if args.device:
        if args.device == 'auto':
            if torch.backends.mps.is_available():
                HParams.DEVICE = 'mps'
            elif torch.cuda.is_available():
                HParams.DEVICE = 'cuda'
            else:
                HParams.DEVICE = 'cpu'
        else:
            HParams.DEVICE = args.device
    
    # 2. Update Hyperparams lain
    if args.epochs: HParams.NUM_EPOCHS = args.epochs
    if args.batch_size: HParams.BATCH_SIZE = args.batch_size
    if args.checkpoint_dir: HParams.CHECKPOINT_DIR = args.checkpoint_dir
    
    # Print Konfirmasi Config
    print(f"⚙️  Active Config: Device={HParams.DEVICE} | BS={HParams.BATCH_SIZE} | Epochs={HParams.NUM_EPOCHS}")

def main():
    args = parse_args()
    
    # Update HParams sebelum modul lain jalan
    update_hparams(args)

    if args.mode == "train":
        print("🚀 Starting Training Pipeline...")
        train_pipeline(args) 

    elif args.mode == 'infer':
        if not args.model_path:
            print("❌ Error: Mode infer butuh --model_path")
            sys.exit(1)
            
        print(f"🗣️  Inference Text: {args.text}")
        inference_pipeline(args)

if __name__ == "__main__":
    main()
