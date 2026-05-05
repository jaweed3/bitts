import argparse
import sys
import os
import torch
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.hparams import HParams
from src.train import main as train_pipeline
from src.inference import main as inference_pipeline
from src.benchmark import main as benchmark_pipeline
from src.sample_gen import main as sample_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="BitJETS: BitNet 1.58-bit TTS")

    parser.add_argument("mode", choices=["train", "infer", "benchmark", "sample"],
                        help="train | infer | benchmark | sample")

    # Device / general
    parser.add_argument("--device", type=str, default=None,
                        help="Device override: cuda / mps / cpu / auto")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: HParams.SEED=42)")

    # Training
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Total optimizer steps (default: HParams.NUM_STEPS=500000)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=None,
                        help="Gradient accumulation steps (default: HParams.ACCUM_STEPS=1)")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging (useful when offline)")
    parser.add_argument("--loss-threshold", type=float, default=None,
                        help="Max loss before skipping batch (default: 15. Lower = stricter)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pth to resume from")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Auto-find and resume from latest checkpoint in checkpoint-dir")

    # Inference / sample
    parser.add_argument("--text", type=str, default="Hello world, this is BitNet TTS")
    parser.add_argument("--model-path", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output wav (infer) or output dir (sample)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speaking speed multiplier (1.0=normal)")

    return parser.parse_args()


def update_hparams(args):
    if args.device:
        if args.device == "auto":
            if torch.backends.mps.is_available():
                HParams.DEVICE = "mps"
            elif torch.cuda.is_available():
                HParams.DEVICE = "cuda"
            else:
                HParams.DEVICE = "cpu"
        else:
            HParams.DEVICE = args.device

    if args.num_steps:
        HParams.NUM_STEPS = args.num_steps
    if args.batch_size:
        HParams.BATCH_SIZE = args.batch_size
    if args.accum_steps:
        HParams.ACCUM_STEPS = args.accum_steps
    if args.loss_threshold:
        HParams.LOSS_SKIP_THRESHOLD = args.loss_threshold
    if args.checkpoint_dir:
        HParams.CHECKPOINT_DIR = args.checkpoint_dir

    print(f"Config: device={HParams.DEVICE} | bs={HParams.BATCH_SIZE} | "
          f"accum={HParams.ACCUM_STEPS} | effective_bs={HParams.BATCH_SIZE * HParams.ACCUM_STEPS} | "
          f"target_steps={HParams.NUM_STEPS:,}")


def main():
    args = parse_args()
    update_hparams(args)

    if args.mode == "train":
        train_pipeline(args)

    elif args.mode == "infer":
        if not args.model_path:
            print("Error: --model-path required for infer mode")
            sys.exit(1)
        inference_pipeline(args)

    elif args.mode == "benchmark":
        benchmark_pipeline(args)

    elif args.mode == "sample":
        if not args.model_path:
            print("Error: --model-path required for sample mode")
            sys.exit(1)
        sample_pipeline(args)


if __name__ == "__main__":
    main()
