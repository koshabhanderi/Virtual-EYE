import sys
import torch

sys.path.append("training")

from networks.xception import Xception


# -----------------------------
# 1. Select device
# -----------------------------

if torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Device:", device)


# -----------------------------
# 2. Create Xception model
# -----------------------------

config = {
    "mode": "original",
    "num_classes": 2,
    "inc": 3,
    "dropout": False
}

print("Creating Xception model...")

model = Xception(config)

print("Model created successfully.")


# -----------------------------
# 3. Load checkpoint
# -----------------------------

checkpoint_path = "training/weights/xception_best.pth"

print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

print("Checkpoint loaded successfully.")


# -----------------------------
# 4. Fix checkpoint key names
# -----------------------------

print("Preparing checkpoint weights...")

new_checkpoint = {}

for key, value in checkpoint.items():

    if key.startswith("backbone."):
        new_key = key[len("backbone."):]
    else:
        new_key = key

    new_checkpoint[new_key] = value


# -----------------------------
# 5. Load weights into model
# -----------------------------

print("Loading weights into model...")

model.load_state_dict(new_checkpoint)

print("Weights loaded successfully.")


# -----------------------------
# 6. Move model to device
# -----------------------------

model = model.to(device)

model.eval()

print("Model moved to:", device)

print("Xception model is ready!")