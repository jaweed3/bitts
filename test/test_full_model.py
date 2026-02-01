import torch
from models import BitJETS

def test_full_flow():
    print(">>> TESTING FULL BIT-JETS ARCHITECTURE <<<")

    vocab_size = 50
    batch_size = 2
    text_len = 10

    text_input = torch.randint(0, vocab_size, (batch_size, text_len))

    # simulation for target duration
    target_durations = torch.randint(2, 4, (batch_size, text_len))

    expected_len_b0 = target_durations[0].sum().item()
    expected_len_b1 = target_durations[1].sum().item()
    max_len = max(expected_len_b0, expected_len_b1)

    print(f"Input Text shape: {text_input.shape}")
    print(f"target durations sum (batch 0): {expected_len_b0}")
    print(f"target durations sum (batch 1): {expected_len_b1}")

    model = BitJETS(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=256,
        decoder_dim=192,
        out_mel_dim=80
    )

    print("\nRunning fast forward!.....")
    mel_out, log_dur_pred = model(text_input, target_durations=target_durations)

    print(f"mel output shape: {mel_out.shape}")
    print(f"log duration prediction shape: {log_dur_pred.shape}")

    if mel_out.shape[1] == max_len and mel_out.shape[2] == 80:
        print("Success: arsitektur utuh berfungsi!")
        print("MOdel berhasil mengubah text -> latent -> expanded -> mel-spectrogram")
    else:
        print(f"fail: dimensi salah, harusnya [2, {max_len}, 80], dapetnya {mel_out.shape}")

    # dummy loss
    loss = mel_out.mean()
    loss.backward()
    print("Gradient check passed (Backward aman)")

if __name__ == "__main__":
    test_full_flow()
