import torch
import torch.nn as nn
from layers import BitConvBlock

class VarianceAdaptor(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()

        # Duration predictor
        self.duration_predictor = nn.Sequential(
            BitConvBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # context underxtanding
            BitConvBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Do not quantize
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, target_durations=None, duration_control=0.1):
        """
        x => encoder output [batch, text_len, hidden_dim
        target_duration: Real duration (ground truth) [batch, text_len] -> used when training
        duration_control: for managing speaking speed when inference (1.0=normal, 0.8=fast)
        duration prediction => log scale (because duration is always positive so we use log)
        """

        log_duration_prediction = self.duration_predictor(x).squeeze(-1)

        if target_durations is not None:
            # training mode, use real duration(raw)
            durations_to_use = target_durations
        else:
            duration_prediction = torch.exp(log_duration_prediction) - 1

            duration_prediction = duration_prediction * duration_control

            durations_to_use = torch.clamp(torch.round(duration_prediction), min=1)

        output = self.length_regulator(x, durations_to_use)

        return output, log_duration_prediction

    def length_regulator(self, x, durations):
        output = []
        batch_size = x.shape[0]

        for i in range(batch_size):
            xi = x[i]
            di = durations[i]
            
            # TODO: MASKING FILTER FOR PADDING

            # repeat_interleave function
            expanded = torch.repeat_interleave(xi, di.long(), dim=0)

            output.append(expanded)

        # padding logic (handling batch)
        output_tensor = torch.nn.utils.rnn.pad_sequence(output, batch_first=True)

        return output_tensor



class BitEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.layer = nn.ModuleList([
            BitConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=5,
                padding=2
            ) for _ in range(num_layers)
            ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layer:
            residual = x
            out = layer(x)
            out = torch.relu(out)
            x = out + residual
        x = self.final_norm(x)
        return x

class BitDecoder(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=80, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            BitConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=5,
                padding=2
            ) for _ in range(num_layers)
        ])

        self.final_form = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.input_proj(x)

        for layer in self.layers:
            residual = x
            out = layer(x)
            out = torch.relu(out)
            x = out + residual

        x = self.final_form(x)

        mel_out = self.output_proj(x)

        return mel_out

class BitJETS(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, decoder_dim=192, out_mel_dim=80):
        super().__init__()

        self.encoder = BitEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )

        self.variance_adaptor = VarianceAdaptor(hidden_dim=hidden_dim)

        self.decoder = BitDecoder(
            in_dim=hidden_dim,
            hidden_dim=decoder_dim,
            out_dim=out_mel_dim
        )

    def forward(self, text, target_durations=None, duration_control=1.0):
        encoder_out = self.encoder(text)

        expanded_out, log_duration_preds = self.variance_adaptor(
            encoder_out,
            target_durations=target_durations,
            duration_control=duration_control
        )

        mel_output = self.decoder(expanded_out)

        return mel_output, log_duration_preds
