import torch
import matplotlib.pyplot as plt
import soundfile as sf

from hparams import HParams
from models import BitJETS
from vocoder import Vocoder
from dataset import text_to_sequence
from packing import unpack_state_dict

def main(args):
    checkpoint_path = args.model_path
    output_wav = args.output
    text_str = args.text
    
    print(f"⏳ Loading Model from {checkpoint_path}...")
    
    # Init Model
    model = BitJETS(
        vocab_size=HParams.VOCAB_SIZE,
        embed_dim=HParams.EMBED_DIM, 
        hidden_dim=HParams.ENCODER_DIM,
        decoder_dim=HParams.DECODER_DIM,
        out_mel_dim=80
    ).to(HParams.DEVICE)

    # Load Weights
    print(f"🔍 Inspecting checkpoint keys...")
    checkpoint = torch.load(checkpoint_path, map_location=HParams.DEVICE)

    # Handle packed checkpoint (has __packed_keys__), training checkpoint, or raw weights
    if "__packed_keys__" in checkpoint:
        print("📦 Packed checkpoint detected. Unpacking...")
        state_dict = unpack_state_dict(checkpoint)
    elif 'model_state_dict' in checkpoint:
        print("✅ Found 'model_state_dict' key. Extracting weights...")
        state_dict = checkpoint['model_state_dict']
    else:
        print("⚠️ No 'model_state_dict' key found. Assuming raw weights.")
        state_dict = checkpoint

    # Load ke model
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Weights loaded successfully (Strict Mode)!")
    except Exception as e:
        print(f"⚠️ Strict loading failed. Retrying with strict=False... Error: {e}")
        # Gunakan strict=False jika ada mismatch minor (misal buffer yang gak kepakai)
        model.load_state_dict(state_dict, strict=False) 
        
    model.eval()

    # Load Vocoder (HiFi-GAN)
    print("⏳ Loading HiFi-GAN Vocoder...")
    try:
        vocoder = Vocoder(HParams.VOCODER_CKPT, HParams.VOCODER_CONFIG, device=HParams.DEVICE)
        use_vocoder = True
    except Exception as e:
        print(f"⚠️ Warning: Gagal load HiFi-GAN ({e}). Fallback ke Griffin-Lim.")
        use_vocoder = False

    # Prepare Text
    print(f"Testing Text: '{text_str}'")
    sequence = text_to_sequence(text_str)
    text_tensor = torch.tensor(sequence).unsqueeze(0).to(HParams.DEVICE)

    # Inference Mel
    print("Generating Mel-Spectrogram...")
    with torch.no_grad():
        # Duration control 1.0 = normal speed
        mel_pred, log_dur_pred = model(text_tensor, duration_control=1.0)

    # Simpan Gambar Mel (Buat debugging visual)
    plt.figure(figsize=(10, 4))
    # Squeeze batch dim -> [Time, 80] -> Transpose buat plot [80, Time]
    mel_plot = mel_pred.squeeze(0).transpose(0, 1).cpu().numpy()
    plt.imshow(mel_plot, origin='lower', aspect='auto')
    plt.title(f"Spectrogram: '{text_str}'")
    plt.savefig('output_mel.png')
    print("📸 Spectrogram image saved to output_mel.png")

    # --- VOCODING PROCESS (CRITICAL FIX) ---
    print("Synthesizing Audio...")
    waveform = None
    
    # Opsi 1: HiFi-GAN (Prioritas Utama)
    if use_vocoder:
        try:
            # Mel Pred masuk ke Vocoder
            waveform = vocoder.infer(mel_pred)
            # Hasil dari vocoder biasanya Tensor [Time] atau [1, Time]
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.squeeze().cpu().numpy()
            print("✨ Audio generated using HiFi-GAN!")
        except Exception as e:
            print(f"❌ HiFi-GAN Inference Error: {e}")
            waveform = None

    # Opsi 2: Griffin-Lim (Fallback kalau HiFi-GAN gagal/gak ada)
    if waveform is None:
        print("🐢 Using Griffin-Lim (Fallback Mode)...")
        # Implementasi Griffin-Lim sederhana pake Torchaudio
        import torchaudio
        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=HParams.N_FFT // 2 + 1, n_mels=HParams.N_MELS, sample_rate=HParams.SAMPLE_RATE
        ).to(HParams.DEVICE)
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=HParams.N_FFT, hop_length=HParams.HOP_LENGTH
        ).to(HParams.DEVICE)
        
        # Log-Mel -> Linear -> Audio
        mel_linear = torch.exp(mel_pred.squeeze(0).transpose(0, 1)) # [80, Time]
        spec = inverse_mel(mel_linear)
        waveform = griffin_lim(spec).cpu().numpy()

    # Save
    sf.write(output_wav, waveform, HParams.SAMPLE_RATE)
    print(f"✅ DONE! Audio saved to: {output_wav}")
