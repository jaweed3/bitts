import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding

# layer norm untuk menerima input x
def activation_quant(x):
    """
    Activation path
    """
    # gamma => absmax in each channel
    gamma = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)

    # Scaling and clipping => Q_p = 127 (8-bit int)
    Q_p = 127.0
    scale = Q_p / gamma

    # forward: rounding (diskrit)
    # backward: identity ()
    x_scaled = x * scale
    x_rounded = x_scaled.round().clamp(-Q_p, Q_p)

    x_quant = (x_rounded - x_scaled).detach() + x_scaled

    # x_quant untuk conv dan scale untuk gamma
    return x_quant, scale

def weight_quant(w):
    """
    Weight Path
    """
    beta = w.abs().mean().clamp(min=1e-5)

    # scale weights
    w_scaled = w / beta

    # quantize -> 'beta' box {-1, 0, 1}
    # clip to [-1, 1], then round
    w_rounded = w_scaled.clamp(-1, 1).round()

    # STE:  make the gradient flows to real weights
    w_quant = (w_rounded - w_scaled).detach() + w_scaled

    return w_quant, beta

class BitConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BitConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )

        self.bias = None

    def forward(self, x):
        # 1. Preprocessing
        # TODO: Make LayerNorm class!

        # 2. Quantization activation
        x_quant, scale_x = activation_quant(x)

        # weight quantization, w_quant ==> ternary weight {-1, 0, 1}
        w_quant, beta = weight_quant(self.weight)

        y_raw = F.conv1d(
            x_quant,
            w_quant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

        rescale_factor = beta / scale_x

        y = y_raw * rescale_factor

        return y

class BitConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BitConvBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(in_channels)
        self.bit_conv = BitConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        
        # transpose
        x_t = x_norm.transpose(1,2)


        out_conv = self.bit_conv(x_t)

        # return it again transposed
        out = out_conv.transpose(1, 2)

        return out

if __name__ == "__main__":
    batch_size = 1
    channels = 256
    length = 100

    x = torch.randn(batch_size, channels, length)

    model = BitConvBlock(in_channels=channels, out_channels=channels, kernel_size=5, padding=2)

    y = model(x)

    print(f"input shape: {x.shape}")
    print(f"output shape: {y.shape}")
    print(f"output mean: {y.mean().item()}")
    print(f"output std: {y.mean().item()}")
    print(f"internal weight shape: {model.bit_conv.weight.shape}")
