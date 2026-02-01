import torch
import torch.nn as nn
import torch.nn.functional as F

# Simulasi fungsi aktivasi BitNet 1.58
def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y + x - x.detach()

def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u + w - w.detach()

class BitLinear(nn.Linear):
    def forward(self, x):
        w_quant = weight_quant(self.weight)
        x_quant = activation_quant(x)
        return F.linear(x_quant, w_quant, self.bias)

def test_mps():
    if not torch.backends.mps.is_available():
        print("❌ MPS Gak ada cuy.")
        return

    device = torch.device("mps")
    print(f"🍏 Testing BitLinear on {device}...")

    try:
        # 1. Init Layer
        layer = BitLinear(256, 256).to(device)
        
        # 2. Dummy Input
        x = torch.randn(4, 50, 256).to(device) # [Batch, Seq, Dim]
        
        # 3. Forward Pass
        print("👉 Running Forward Pass...")
        y = layer(x)
        print("✅ Forward Sukses! Shape:", y.shape)
        
        # 4. Backward Pass (Biasanya crash disini)
        print("👉 Running Backward Pass...")
        loss = y.sum()
        loss.backward()
        print("✅ Backward Sukses!")
        
        print("🎉 KESIMPULAN: MPS Aman buat BitNet. Masalahnya ada di Dataloader/Layer lain.")

    except Exception as e:
        print(f"💥 CRASH TERDETEKSI: {e}")
    except:
        print("💀 SEGMENTATION FAULT TERJADI DI SINI.")

if __name__ == "__main__":
    test_mps()
