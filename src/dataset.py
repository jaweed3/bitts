import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from hparams import HParams
from utils import text_to_sequence, load_audio_wav

class LJSpeechDataset(Dataset):
    def __init__(self, root_dir=HParams.DATA_PATH, limit=None):
        self.root_dir = root_dir
        self.wav_dir = os.path.join(root_dir, "wavs")
        self.meta_path = os.path.join(root_dir, "metadata.csv")
        
        # Load Metadata
        self.meta = pd.read_csv(self.meta_path, sep='|', header=None, quoting=3)
        if limit:
            self.meta = self.meta.iloc[:limit]
            
        # Transform Setup
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=HParams.SAMPLE_RATE,
            n_fft=HParams.N_FFT,
            hop_length=HParams.HOP_LENGTH,
            n_mels=HParams.N_MELS,
            power=1.0,
            normalized=True
        )
        
        # Resampler (disiapin lazy load nanti)
        self.resampler = None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_name = row[0]
        text_raw = str(row[2])
        
        wav_path = os.path.join(self.wav_dir, f"{file_name}.wav")
        
        # 1. Load Audio (Pake Utils bersih)
        waveform, sr = load_audio_wav(wav_path)
        if waveform is None: # Fallback kalau error
            return self.__getitem__(0)

        # 2. Resample on demand
        if sr != HParams.SAMPLE_RATE:
            if self.resampler is None:
                 self.resampler = torchaudio.transforms.Resample(sr, HParams.SAMPLE_RATE)
            waveform = self.resampler(waveform)
            
        # 3. Mel Spec
        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.squeeze(0).transpose(0, 1) # [Time, Mels]
        
        # 4. Text
        text_seq = torch.tensor(text_to_sequence(text_raw), dtype=torch.long)
        
        # 5. Dummy Duration (Hack sementara)
        mel_len = mel.shape[0]
        text_len = text_seq.shape[0]
        avg_dur = max(1, mel_len // text_len)
        durations = torch.ones(text_len, dtype=torch.long) * avg_dur
        # Fix sisa duration
        diff = mel_len - durations.sum()
        if diff != 0: durations[-1] += diff
            
        return text_seq, mel, durations

def collate_fn(batch):
    # Sort by text len desc
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    text_seqs, mels, durations = zip(*batch)
    
    # Pad
    text_padded = pad_sequence(text_seqs, batch_first=True, padding_value=0)
    mel_padded = pad_sequence(mels, batch_first=True, padding_value=-11.5)
    dur_padded = pad_sequence(durations, batch_first=True, padding_value=0)
    
    return text_padded, mel_padded, dur_padded
