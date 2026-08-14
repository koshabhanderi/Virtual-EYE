import sys
import os
import cv2
import torch
import numpy as np
import time

sys.path.append("training")

from networks.xception import Xception


# ==========================================
# 1. DEVICE
# ==========================================

if torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Device:", device)


# ==========================================
# 2. CREATE MODEL
# ==========================================

config = {
    "mode": "original",
    "num_classes": 2,
    "inc": 3,
    "dropout": False
}

print("Creating Xception model...")

model = Xception(config)


# ==========================================
# 3. LOAD CHECKPOINT
# ==========================================

checkpoint_path = "training/weights/xception_best.pth"

print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)


# ==========================================
# 4. FIX CHECKPOINT KEYS
# ==========================================

new_checkpoint = {}

for key, value in checkpoint.items():

    if key.startswith("backbone."):
        key = key[len("backbone."):]

    new_checkpoint[key] = value


# ==========================================
# 5. LOAD WEIGHTS
# ==========================================

model.load_state_dict(new_checkpoint)

model = model.to(device)

model.eval()

print("Model ready.")


# ==========================================
# 6. CREATE TEST IMAGE
# ==========================================

dummy_image = np.random.rand(
    1,
    3,
    256,
    256
).astype(np.float32)

dummy_tensor = torch.from_numpy(
    dummy_image
).to(device)


# ==========================================
# 7. WARM-UP
# ==========================================

print("Warming up XPU...")

for i in range(10):

    with torch.no_grad():

        output, features = model(
            dummy_tensor
        )


# ==========================================
# 8. SYNCHRONIZE XPU
# ==========================================

if device.type == "xpu":

    torch.xpu.synchronize()


# ==========================================
# 9. MEASURE INFERENCE
# ==========================================

print("Measuring inference speed...")

iterations = 50

start_time = time.perf_counter()


for i in range(iterations):

    with torch.no_grad():

        output, features = model(
            dummy_tensor
        )


if device.type == "xpu":

    torch.xpu.synchronize()


end_time = time.perf_counter()


# ==========================================
# 10. CALCULATE RESULTS
# ==========================================

total_time = (
    end_time -
    start_time
)

average_time = (
    total_time /
    iterations
)

inference_fps = (
    1.0 /
    average_time
)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print()
print("===================================")
print("       PERFORMANCE RESULTS")
print("===================================")

print(
    f"Iterations: {iterations}"
)

print(
    f"Total time: {total_time:.2f} seconds"
)

print(
    f"Average inference time: "
    f"{average_time * 1000:.2f} ms"
)

print(
    f"Xception inference FPS: "
    f"{inference_fps:.2f}"
)

print("===================================")