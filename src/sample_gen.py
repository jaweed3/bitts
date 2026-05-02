"""
BitJETS Audio Sample Generator
Generates a set of benchmark TTS sentences and saves to samples/.
Sentences cover: short/long, punctuation, numbers spelled out, prosody variety.
"""

import os
import torch
import soundfile as sf
import matplotlib.pyplot as plt

from hparams import HParams
from models import BitJETS
from vocoder import Vocoder
from utils import text_to_sequence
from packing import unpack_state_dict

# Standard TTS benchmark sentences — chosen for phoneme diversity and length variety
BENCHMARK_SENTENCES = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck",
    "to be or not to be that is the question",
    "speech synthesis is the artificial production of human speech",
    "a journey of a thousand miles begins with a single step",
    "the birch canoe slid on the smooth planks",
]


def _load_model(ckpt_path: str, device: str) -> BitJETS:
    model = BitJETS(
        vocab_size=HParams.VOCAB_SIZE,
        embed_dim=HParams.EMBED_DIM,
        hidden_dim=HParams.ENCODER_DIM,
        decoder_dim=HParams.DECODER_DIM,
        out_mel_dim=HParams.N_MELS,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "__packed_keys__" in ckpt:
        sd = unpack_state_dict(ckpt)
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    model.load_state_dict(sd)
    model.eval()
    return model


def _load_vocoder(device: str):
    try:
        return Vocoder(HParams.VOCODER_CKPT, HParams.VOCODER_CONFIG, device=device)
    except Exception as e:
        print(f"HiFi-GAN unavailable ({e}). Will use Griffin-Lim fallback.")
        return None


def _griffin_lim(mel_pred: torch.Tensor, device: str) -> torch.Tensor:
    import torchaudio
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=HParams.N_FFT // 2 + 1,
        n_mels=HParams.N_MELS,
        sample_rate=HParams.SAMPLE_RATE,
    ).to(device)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=HParams.N_FFT, hop_length=HParams.HOP_LENGTH
    ).to(device)
    mel_linear = torch.exp(mel_pred.squeeze(0).transpose(0, 1))
    return griffin_lim(inverse_mel(mel_linear))


def main(args):
    device    = HParams.DEVICE
    ckpt_path = args.model_path
    out_dir   = getattr(args, "output", "samples")
    speed     = getattr(args, "speed", 1.0)

    if not ckpt_path:
        print("Error: --model-path required for sample mode.")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating {len(BENCHMARK_SENTENCES)} samples → {out_dir}/")

    model   = _load_model(ckpt_path, device)
    vocoder = _load_vocoder(device)

    results = []
    for i, sentence in enumerate(BENCHMARK_SENTENCES, 1):
        seq = text_to_sequence(sentence)
        if not seq:
            print(f"  [{i:02d}] SKIP (no encodable chars): '{sentence}'")
            continue

        text_tensor = torch.tensor(seq).unsqueeze(0).to(device)

        with torch.no_grad():
            mel_pred, _ = model(text_tensor, duration_control=speed)

        # Vocoding
        if vocoder:
            try:
                waveform = vocoder.infer(mel_pred)
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.squeeze().cpu().numpy()
                method = "hifigan"
            except Exception as e:
                print(f"  HiFi-GAN failed ({e}), falling back to Griffin-Lim.")
                waveform = _griffin_lim(mel_pred, device).cpu().numpy()
                method = "griffinlim"
        else:
            waveform = _griffin_lim(mel_pred, device).cpu().numpy()
            method = "griffinlim"

        fname = f"{i:02d}_{method}_{sentence[:30].replace(' ', '_')}.wav"
        fpath = os.path.join(out_dir, fname)
        sf.write(fpath, waveform, HParams.SAMPLE_RATE)

        dur_s = len(waveform) / HParams.SAMPLE_RATE
        results.append((sentence, fname, dur_s))
        print(f"  [{i:02d}] {dur_s:.2f}s → {fname}")

    # Save mel grid for visual inspection
    _save_mel_grid(model, device, out_dir)

    print(f"\nDone. {len(results)}/{len(BENCHMARK_SENTENCES)} samples saved to {out_dir}/")


def _save_mel_grid(model, device, out_dir):
    """Save a 2x4 grid of mel spectrograms for the benchmark sentences."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()

    for i, sentence in enumerate(BENCHMARK_SENTENCES):
        seq = text_to_sequence(sentence)
        if not seq:
            continue
        text_tensor = torch.tensor(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            mel, _ = model(text_tensor, duration_control=1.0)
        mel_np = mel.squeeze(0).transpose(0, 1).cpu().numpy()
        axes[i].imshow(mel_np, origin="lower", aspect="auto", cmap="viridis")
        axes[i].set_title(sentence[:28], fontsize=7)
        axes[i].axis("off")

    plt.tight_layout()
    grid_path = os.path.join(out_dir, "mel_grid.png")
    plt.savefig(grid_path, dpi=120)
    plt.close()
    print(f"  Mel grid saved → {grid_path}")
