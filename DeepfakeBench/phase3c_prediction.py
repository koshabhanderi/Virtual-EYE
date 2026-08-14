import sys
import cv2
import torch
import numpy as np

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
# 4. FIX CHECKPOINT NAMES
# ==========================================

new_checkpoint = {}

for key, value in checkpoint.items():

    if key.startswith("backbone."):
        new_key = key[len("backbone."):]
    else:
        new_key = key

    new_checkpoint[new_key] = value


# ==========================================
# 5. LOAD WEIGHTS
# ==========================================

print("Loading weights...")

model.load_state_dict(new_checkpoint)

print("Weights loaded successfully.")


# ==========================================
# 6. MOVE MODEL TO XPU
# ==========================================

model = model.to(device)
model.eval()

print("Model ready.")


# ==========================================
# 7. LOAD IMAGE
# ==========================================

image_path = "test_face.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Could not load image:", image_path)
    sys.exit()


# ==========================================
# 8. RESIZE
# ==========================================

image = cv2.resize(image, (299, 299))


# ==========================================
# 9. BGR → RGB
# ==========================================

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==========================================
# 10. CONVERT TO FLOAT
# ==========================================

image = image.astype(np.float32) / 255.0


# ==========================================
# 11. HWC → CHW
# ==========================================

image = np.transpose(image, (2, 0, 1))


# ==========================================
# 12. ADD BATCH DIMENSION
# ==========================================

image = np.expand_dims(image, axis=0)


# ==========================================
# 13. NUMPY → PYTORCH
# ==========================================

tensor = torch.from_numpy(image)

tensor = tensor.to(device)


# ==========================================
# 14. PREDICTION
# ==========================================

# ==========================================
# 14. PREDICTION
# ==========================================


print("Running prediction...")

with torch.no_grad():
    output, features = model(tensor)

print("Classification output:")
print(output)

probabilities = torch.softmax(output, dim=1)

real_probability = probabilities[0][0].item()
fake_probability = probabilities[0][1].item()

print("Real probability:", real_probability)
print("Fake probability:", fake_probability)

print("Real:", round(real_probability * 100, 2), "%")
print("Fake:", round(fake_probability * 100, 2), "%")

print("Feature shape:")
print(features.shape)