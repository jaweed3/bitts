import torch
from models import VarianceAdaptor

if __name__ == "__main__":
    print("---testing variance adaptor---")

    batch_size = 2
    text_len = 3
    hidden_dim = 4

    encoder_output = torch.randn(batch_size, text_len, hidden_dim)

    target_durations = torch.tensor([
        [2, 2, 1],
        [1, 3, 2]
    ])

    adaptor = VarianceAdaptor(hidden_dim=hidden_dim)

    output, log_pred = adaptor(encoder_output, target_durations=target_durations)

    print(f"Encoder output shape: {encoder_output.shape} -> [B, 3, 4]")
    print(f"Adapter Output shape: {output.shape}")

    expected_shape = (2, 6, 4)
    if output.shape == expected_shape:
        print(f"Success: output shape sesuai prediksi (expanded & padded)")
    else:
        print(f"fail: shape salah, harusnya {expected_shape}, bukan => {output.shape}")

    print(f"Log duration pred shape: {log_pred.shape} -> harusnya [2, 3]")
    print(f"sample prediction (still random coz havent trained.)", log_pred[0])
