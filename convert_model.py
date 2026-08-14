import sys
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

TRAINING_DIR = BASE_DIR / "DeepfakeBench" / "training"
CHECKPOINT_PATH = BASE_DIR / "ai_face_weights" / "xception_ai_face.pth"
OUTPUT_PATH = BASE_DIR / "ai_face_weights" / "xception_ai_face.onnx"

sys.path.insert(0, str(TRAINING_DIR))

from networks.xception import Xception


MODEL_CONFIG = {
    "mode": "original",
    "num_classes": 1,
    "inc": 3,
    "dropout": False,
}


print("Loading Xception model...")

model = Xception(MODEL_CONFIG)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]


new_checkpoint = {}

for key, value in checkpoint.items():

    if key.startswith("backbone."):
        key = key[len("backbone."):]

    new_checkpoint[key] = value


model.load_state_dict(
    new_checkpoint,
    strict=True
)

model.eval()

print("Model loaded successfully.")

dummy_input = torch.randn(
    1,
    3,
    299,
    299
)

print("Converting to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    str(OUTPUT_PATH),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {
            0: "batch"
        },
        "output": {
            0: "batch"
        }
    }
)

print()
print("===================================")
print("ONNX conversion successful!")
print("Saved to:")
print(OUTPUT_PATH)
print("===================================")