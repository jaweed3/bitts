import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import LJSpeechDataset, collate_fn, vocab_size

LJ_PATH = "./speech_data/speech"

if __name__ == "__main__":
    print(">>> Testing data pipeline <<<")
    ds = LJSpeechDataset(LJ_PATH, limit=10)

    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    text_batch, mel_batch, dur_batch = next(iter(loader))

    print(f"Vocab Size: {vocab_size}")
    print(f"Text batch shape: {text_batch.shape} -> [Batch, Text_len]")
    print(f"Mel batch shape: {mel_batch.shape} -> [Batch, Mel_len, 80]")
    print(f"Duration batch shape: {dur_batch.shape} -> [batch, text_len]")

    plt.imshow(mel_batch[0].T, origin='lower', aspect='auto')
    plt.show()

    print(">>> dataset ready to serve <<<")
