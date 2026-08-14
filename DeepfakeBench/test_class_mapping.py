import sys
import torch

sys.path.append("training")

from networks.xception import Xception


# ==========================================
# DEVICE
# ==========================================

if torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Device:", device)


# ==========================================
# CREATE MODEL
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
# LOAD CHECKPOINT
# ==========================================

checkpoint_path = "training/weights/xception_best.pth"

print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)


# ==========================================
# PREPARE CHECKPOINT
# ==========================================

new_checkpoint = {}

for key, value in checkpoint.items():

    if key.startswith("backbone."):

        key = key[len("backbone."):]

    new_checkpoint[key] = value


# ==========================================
# LOAD WEIGHTS
# ==========================================

model.load_state_dict(new_checkpoint)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ==========================================
# SHOW MODEL MODULES
# ==========================================

print()
print("======================================")
print("MODEL MODULES")
print("======================================")

for name, module in model.named_modules():

    print(name, "->", module.__class__.__name__)


# ==========================================
# SHOW PARAMETER NAMES
# ==========================================

print()
print("======================================")
print("FINAL PARAMETERS")
print("======================================")

parameter_names = list(
    model.state_dict().keys()
)

for name in parameter_names[-20:]:

    print(name)


# ==========================================
# CREATE DUMMY INPUT
# ==========================================

print()
print("======================================")
print("TESTING OUTPUT")
print("======================================")


dummy = torch.randn(
    1,
    3,
    256,
    256
).to(device)


with torch.no_grad():

    output, features = model(
        dummy
    )


if device.type == "xpu":

    torch.xpu.synchronize()


print()
print("Output:")
print(output)

print()
print("Output shape:")
print(output.shape)

print()
print("Features shape:")
print(features.shape)


# ==========================================
# SOFTMAX
# ==========================================

probabilities = torch.softmax(
    output,
    dim=1
)

print()
print("Probabilities:")
print(probabilities)


print()
print("Class 0 probability:")
print(
    probabilities[0][0].item()
)

print()
print("Class 1 probability:")
print(
    probabilities[0][1].item()
)


print()
print("======================================")
print("IMPORTANT")
print("======================================")

print(
    "Class 0 and Class 1 are the two model outputs."
)

print(
    "We need the DeepfakeBench detector configuration"
)

print(
    "to determine which one corresponds to REAL/FAKE."
)

print("======================================")