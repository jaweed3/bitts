import torch

def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _auto_batch_config():
    """Returns (batch_size, accum_steps) tuned per device.

    CUDA (RTX 4060 8GB) → bs=64, accum=1   (80% VRAM headroom)
    MPS (Apple Silicon)  → bs=8,  accum=4   (shared memory, conservative)
    CPU                   → bs=2,  accum=1   (debug only)
    """
    if torch.cuda.is_available():
        # RTX 4060 / Ada Lovelace: 8 GB VRAM, bs=64 fits comfortably.
        # For smaller GPUs (<6 GB) fall back to bs=32.
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            if vram_gb >= 10:
                return 96, 1
            elif vram_gb >= 7:
                return 64, 1   # RTX 4060 / 4070 sweet spot
            else:
                return 32, 1   # GTX 1060 / laptop GPU
        except Exception:
            return 32, 1

    if torch.backends.mps.is_available():
        return 8, 4            # MPS: smaller batch, accumulate to match effective bs

    return 2, 1                # CPU: tiny, for smoke tests only

class HParams:
    # >>> audio config <<<
    SAMPLE_RATE = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    N_MELS = 80

    # >>> text config <<<
    VOCAB = "_abcdefghijklmnopqrstuvwxyz!?,. "
    VOCAB_SIZE = len(VOCAB)

    # >>> model architecture <<<
    EMBED_DIM = 256
    ENCODER_DIM = 256
    DECODER_DIM = 192
    ENCODER_LAYERS = 4
    DECODER_LAYERS = 4
    KERNEL_SIZE = 5
    DROPOUT = 0.1

    # >>> training config <<<
    # Auto-tuned per device (see _auto_batch_config above).
    # Override via CLI: --batch-size N --accum-steps N
    BATCH_SIZE, ACCUM_STEPS = _auto_batch_config()
    LEARNING_RATE = 2e-4
    NUM_STEPS = 500_000      # total optimizer steps (paper: 2000K, we target 500K)
    WARMUP_STEPS = 1000      # linear warmup before cosine decay
    MIN_LR_RATIO = 0.1       # cosine floor: lr decays to 10% of peak (never 0)
    CHECKPOINT_DIR = "./checkpoints"
    LOG_INTERVAL = 100       # log every N optimizer steps
    CKPT_INTERVAL = 10_000   # save named checkpoint every N optimizer steps
    SEED = 42

    # Loss gate: skip batches above this threshold to avoid gradient explosion.
    # Set higher (25-30) when resuming legacy checkpoints with high initial loss.
    LOSS_SKIP_THRESHOLD = 15.0
    MAX_CONSECUTIVE_SKIPS = 200   # exit if this many batches skipped in a row
    SKIP_GRACE_STEPS = 50         # first N optimizer steps after resume: relaxed threshold
    PLATEAU_PATIENCE = 10         # warn if no improvement for N × LOG_INTERVAL steps
    ALIGN_INTERVAL = 2000          # run MAS duration extraction every N steps

    # >>> Dataset PAth <<<
    DATA_PATH = "./data/speech"

    # >>> VOCODER CONFIG <<<
    VOCODER_CKPT = "checkpoints/UNIVERSAL_V1/g_02500000"
    VOCODER_CONFIG = "checkpoints/UNIVERSAL_V1/config.json"

    # >>> Device (auto-detect: cuda > mps > cpu) <<<
    DEVICE = _auto_device()
