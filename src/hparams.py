import torch

def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

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
    # RTX 4060 (CUDA): BATCH_SIZE=32, ACCUM_STEPS=1
    # Apple MPS:       BATCH_SIZE=8,  ACCUM_STEPS=4
    BATCH_SIZE = 32
    ACCUM_STEPS = 1          # effective batch = BATCH_SIZE * ACCUM_STEPS
    LEARNING_RATE = 2e-4
    NUM_STEPS = 500_000      # total optimizer steps (paper: 2000K, we target 500K)
    CHECKPOINT_DIR = "./checkpoints"
    LOG_INTERVAL = 100       # log every N optimizer steps
    CKPT_INTERVAL = 10_000   # save named checkpoint every N optimizer steps
    SEED = 42

    # >>> Dataset PAth <<<
    DATA_PATH = "./data/speech"

    # >>> VOCODER CONFIG <<<
    VOCODER_CKPT = "checkpoints/UNIVERSAL_V1/g_02500000"
    VOCODER_CONFIG = "checkpoints/UNIVERSAL_V1/config.json"

    # >>> Device (auto-detect: cuda > mps > cpu) <<<
    DEVICE = _auto_device()
