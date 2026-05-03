"""Test quantization layers (BitConv1d, BitConvBlock)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from layers import BitConv1d, BitConvBlock, weight_quant, activation_quant


def test_weight_quant_ternary():
    w = torch.randn(64, 32, 3)
    w_q, beta = weight_quant(w)
    unique = w_q.unique()
    assert set(unique.tolist()).issubset({-1.0, 0.0, 1.0}), f"Non-ternary values: {unique}"


def test_activation_quant_range():
    x = torch.randn(2, 256, 50)
    x_q, scale = activation_quant(x)
    assert x_q.abs().max() <= 127.0 + 1e-4


def test_bitconv1d_shape():
    layer = BitConv1d(256, 256, kernel_size=5, padding=2)
    x = torch.randn(2, 256, 100)
    y = layer(x)
    assert y.shape == (2, 256, 100), f"Expected (2,256,100), got {y.shape}"


def test_bitconvblock_shape():
    block = BitConvBlock(256, 256, kernel_size=5, padding=2)
    x = torch.randn(2, 50, 256)  # [batch, time, channels]
    y = block(x)
    assert y.shape == (2, 50, 256), f"Expected (2,50,256), got {y.shape}"


def test_bitconvblock_backward():
    block = BitConvBlock(64, 64, kernel_size=3, padding=1)
    x = torch.randn(2, 20, 64, requires_grad=True)
    y = block(x)
    y.mean().backward()
    assert x.grad is not None
