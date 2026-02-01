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
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 300
    CHECKPOINT_DIR = "./checkpoints"
    LOG_INTERVAL = 10

    # >>> Dataset PAth <<<
    DATA_PATH = "./data/speech"

    # >>> VOCODER CONFIG <<<
    VOCODER_CKPT = "checkpoints/UNIVERSAL_V1/g_02500000"
    VOCODER_CONFIG = "checkpoints/UNIVERSAL_V1/config.json"

    # >>> Inference Device <<<
    DEVICE = "mps"
