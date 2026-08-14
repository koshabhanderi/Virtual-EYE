import sys
import os
import json
import cv2
import torch
import numpy as np

# ============================================================
# PATHS
# ============================================================

sys.path.append("training")

from networks.xception import Xception


# ============================================================
# 1. DEVICE
# ============================================================

if torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Device:", device)


# ============================================================
# 2. CREATE MODEL
# ============================================================

config = {
    "mode": "original",
    "num_classes": 2,
    "inc": 3,
    "dropout": False
}

print("Creating Xception model...")

model = Xception(config)

print("Model created successfully.")


# ============================================================
# 3. LOAD CHECKPOINT
# ============================================================

checkpoint_path = "training/weights/xception_best.pth"

print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

print("Checkpoint loaded successfully.")


# ============================================================
# 4. FIX CHECKPOINT NAMES
# ============================================================

new_checkpoint = {}

for key, value in checkpoint.items():

    if key.startswith("backbone."):
        new_key = key[len("backbone."):]
    else:
        new_key = key

    new_checkpoint[new_key] = value


# ============================================================
# 5. LOAD WEIGHTS
# ============================================================

print("Loading weights...")

model.load_state_dict(new_checkpoint)

print("Weights loaded successfully.")


# ============================================================
# 6. MOVE MODEL TO XPU
# ============================================================

model = model.to(device)
model.eval()

print("Model ready.")


# ============================================================
# 7. BENCHMARK PATH
# ============================================================

benchmark_dir = (
    "phase3h_test/benchmark_raw/"
    "faceforensics_benchmark_images"
)

results_dir = "phase3h_test/results"

os.makedirs(results_dir, exist_ok=True)

submission_path = (
    results_dir +
    "/phase3h_submission.json"
)


# ============================================================
# 8. GET ALL PNG IMAGES
# ============================================================

image_files = [
    f
    for f in os.listdir(benchmark_dir)
    if f.lower().endswith(".png")
]

image_files.sort()

print()
print("==========================================")
print("       PHASE 3H BENCHMARK")
print("==========================================")

print("Benchmark directory:")
print(benchmark_dir)

print()
print("Images found:", len(image_files))


if len(image_files) != 1000:

    print()
    print("WARNING!")
    print("Expected 1000 benchmark images.")
    print("Found:", len(image_files))
    print()


# ============================================================
# 9. PREDICTION DICTIONARY
# ============================================================

predictions = {}

processed = 0
failed = 0


# ============================================================
# 10. PROCESS IMAGES
# ============================================================

for index, filename in enumerate(image_files):

    image_path = os.path.join(
        benchmark_dir,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:

        print(
            "Could not read:",
            filename
        )

        failed += 1

        continue


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (299, 299)
    )


    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Convert to float
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    ) / 255.0


    # --------------------------------------------------------
    # Normalize
    #
    # Your working Phase 3C script used:
    # image values from 0 to 1.
    #
    # We keep the same preprocessing here.
    # --------------------------------------------------------


    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    image = np.transpose(
        image,
        (2, 0, 1)
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )


    # --------------------------------------------------------
    # NumPy -> PyTorch
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        image
    )

    tensor = tensor.to(device)


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        output, features = model(
            tensor
        )


    # --------------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output,
        dim=1
    )


    class_0_probability = (
        probabilities[0][0].item()
    )

    class_1_probability = (
        probabilities[0][1].item()
    )


    # ========================================================
    # CLASS MAPPING
    #
    # Based on our existing Phase 3C/3G setup:
    #
    # Class 0 = REAL
    # Class 1 = FAKE
    #
    # We use the same mapping here.
    # ========================================================

    if class_1_probability >= class_0_probability:

        prediction = "fake"

    else:

        prediction = "real"


    predictions[filename] = prediction

    processed += 1


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        processed % 25 == 0
        or processed == len(image_files)
    ):

        print(
            f"Processed {processed}/"
            f"{len(image_files)}"
        )


# ============================================================
# 11. SAVE JSON
# ============================================================

print()
print("Saving submission JSON...")

with open(
    submission_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        predictions,
        f,
        indent=4
    )


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print()
print("==========================================")
print("       PHASE 3H COMPLETE")
print("==========================================")

print(
    "Total images:",
    len(image_files)
)

print(
    "Processed:",
    processed
)

print(
    "Failed:",
    failed
)

print()
print(
    "REAL predictions:",
    sum(
        1
        for x in predictions.values()
        if x == "real"
    )
)

print(
    "FAKE predictions:",
    sum(
        1
        for x in predictions.values()
        if x == "fake"
    )
)

print()
print("Submission saved to:")
print(submission_path)

print()
print("==========================================")