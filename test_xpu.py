import torch

print("PyTorch:", torch.__version__)
print("XPU available:", torch.xpu.is_available())

if torch.xpu.is_available():

    print("XPU device:", torch.xpu.get_device_name(0))

    x = torch.rand(2000, 2000, device="xpu")
    y = torch.rand(2000, 2000, device="xpu")

    z = x @ y

    torch.xpu.synchronize()

    print("GPU calculation successful!")

else:
    print("Intel GPU not available")