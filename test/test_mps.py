import torch

input=torch.rand(32, 32, device='mps')
assert input.device.type == "mps"
weight=torch.randint(-100, 100, (32, 32), dtype=torch.int8, device='mps')
assert weight.device.type == "mps"

print(torch.nn.functional.linear(input, weight.to(dtype=input.dtype)))
