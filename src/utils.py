import torch
import soundfile as sf
from hparams import HParams

def text_to_sequence(text):
    """Change string into list of integer based on vocab in HParams"""
    char_to_id = {char: i for i, char in enumerate(HParams.VOCAB)}
    text = text.lower()
    return [char_to_id[c] for c in text if c in char_to_id]

def load_audio_wav(path):
    """Load sound audio, bypassing buggy torchaudio in Mac. using soundfile"""
    try:
        wav_numpy, sr = sf.read(path)
    except Exception as e:
        print(f"Error reading audio {path}: {e}")
        return None, None

    waveform = torch.from_numpy(wav_numpy).float()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)

    return waveform, sr
