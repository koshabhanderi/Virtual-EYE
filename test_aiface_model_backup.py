import os
import sys
import cv2
import torch
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

# Add DeepfakeBench training folder
sys.path.insert(0, r".\DeepfakeBench\training")

from networks.xception import Xception


# ============================================================
# DEVICE
# ============================================================

if torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")


print("=" * 60)
print("              AI-FACE XCEPTION TEST")
print("=" * 60)

print("Device:", device)


# ============================================================
# CREATE MODEL
# ============================================================

# IMPORTANT:
# AI-Face Xception checkpoint has ONE output class/logit.
#
# Therefore:
# num_classes = 1
#
# We will use SIGMOID during prediction.
# We will NOT use softmax.

config = {
    "mode": "original",
    "num_classes": 1,
    "inc": 3,
    "dropout": False
}


print()
print("Creating Xception model...")

model = Xception(config)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint_path = r".\ai_face_weights\xception_ai_face.pth"

print("Loading AI-Face checkpoint...")
print("Checkpoint:", checkpoint_path)


checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

print("Checkpoint loaded.")


# ============================================================
# FIX CHECKPOINT KEY NAMES
# ============================================================

new_checkpoint = {}


for key, value in checkpoint.items():

    if key.startswith("backbone."):

        new_key = key[len("backbone."):]

    else:

        new_key = key

    new_checkpoint[new_key] = value


# ============================================================
# LOAD WEIGHTS
# ============================================================

print("Loading weights...")


model.load_state_dict(
    new_checkpoint,
    strict=True
)


print("Weights loaded successfully!")


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(device)

model.eval()


print("Model ready.")


# ============================================================
# IMAGE PREDICTION FUNCTION
# ============================================================

def predict(image_path):

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(image_path)


    if image is None:

        print("Could not load:", image_path)

        return None


    # --------------------------------------------------------
    # RESIZE
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
    # UINT8 -> FLOAT
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    ) / 255.0


    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    image = np.transpose(
        image,
        (2, 0, 1)
    )


    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )


    # --------------------------------------------------------
    # NUMPY -> PYTORCH
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        image
    )


    tensor = tensor.to(device)


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        result = model(tensor)


    # --------------------------------------------------------
    # MODEL RETURNS:
    #
    #     output, features
    #
    # We only need output for classification.
    # --------------------------------------------------------

    if isinstance(result, tuple):

        output = result[0]

    else:

        output = result


    # --------------------------------------------------------
    # CONVERT OUTPUT TO SINGLE VALUE
    # --------------------------------------------------------

    output = output.squeeze()


    # --------------------------------------------------------
    # ONE-OUTPUT MODEL
    #
    # The checkpoint has ONE output.
    #
    # Therefore use SIGMOID.
    #
    # sigmoid output:
    #
    # close to 0 -> REAL
    # close to 1 -> FAKE
    # --------------------------------------------------------

    fake_probability = torch.sigmoid(
        output
    ).item()


    # --------------------------------------------------------
    # REAL PROBABILITY
    # --------------------------------------------------------

    real_probability = 1.0 - fake_probability


    # --------------------------------------------------------
    # CLASSIFICATION
    #
    # 0.5 is the normal decision threshold.
    # --------------------------------------------------------

    if fake_probability >= 0.5:

        prediction = "FAKE"

    else:

        prediction = "REAL"


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return (
        prediction,
        real_probability,
        fake_probability
    )


# ============================================================
# TEST A FOLDER
# ============================================================

def test_folder(folder_path, expected_label):


    print()
    print("=" * 60)

    print(
        "Testing folder:",
        folder_path
    )

    print(
        "Expected label:",
        expected_label
    )

    print("=" * 60)


    # --------------------------------------------------------
    # CHECK FOLDER
    # --------------------------------------------------------

    if not os.path.exists(folder_path):

        print()
        print("ERROR: Folder does not exist!")

        return 0, 0


    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    )


    image_files = []


    for filename in os.listdir(folder_path):

        if filename.lower().endswith(
            valid_extensions
        ):

            image_files.append(
                filename
            )


    image_files.sort()


    print(
        "Images found:",
        len(image_files)
    )


    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    correct = 0

    total = len(image_files)


    # --------------------------------------------------------
    # PREDICT EACH IMAGE
    # --------------------------------------------------------

    for index, filename in enumerate(
        image_files
    ):


        image_path = os.path.join(
            folder_path,
            filename
        )


        result = predict(
            image_path
        )


        if result is None:

            continue


        prediction, real_probability, fake_probability = result


        # ----------------------------------------------------
        # CHECK CORRECTNESS
        # ----------------------------------------------------

        if prediction == expected_label:

            correct += 1

            status = "CORRECT"

        else:

            status = "WRONG"


        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            f"{index + 1:03d}. "
            f"{filename:20s} "
            f"Predicted: {prediction:4s} "
            f"REAL: {real_probability * 100:6.2f}% "
            f"FAKE: {fake_probability * 100:6.2f}% "
            f"[{status}]"
        )


    # --------------------------------------------------------
    # FOLDER ACCURACY
    # --------------------------------------------------------

    if total > 0:

        accuracy = (
            correct / total
        ) * 100

    else:

        accuracy = 0


    print()

    print("-" * 60)

    print(
        "Expected:",
        expected_label
    )

    print(
        "Correct:",
        correct
    )

    print(
        "Total:",
        total
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print("-" * 60)


    return correct, total


# ============================================================
# MAIN TEST
# ============================================================

print()

print("=" * 60)
print("                  STARTING TEST")
print("=" * 60)


# ============================================================
# REAL IMAGES
# ============================================================

real_folder = r".\ai_face_training\val\real"

real_correct, real_total = test_folder(
    real_folder,
    "REAL"
)


# ============================================================
# FAKE IMAGES
# ============================================================

fake_folder = r".\ai_face_training\val\fake"

fake_correct, fake_total = test_folder(
    fake_folder,
    "FAKE"
)


# ============================================================
# FINAL RESULTS
# ============================================================

total_correct = (
    real_correct +
    fake_correct
)


total_images = (
    real_total +
    fake_total
)


if total_images > 0:

    overall_accuracy = (
        total_correct /
        total_images
    ) * 100

else:

    overall_accuracy = 0


# ============================================================
# FINAL REPORT
# ============================================================

print()

print("=" * 60)
print("                    FINAL RESULTS")
print("=" * 60)


print()

print(
    "REAL images:",
    real_total
)

print(
    "REAL correctly detected:",
    real_correct
)


if real_total > 0:

    print(
        "REAL accuracy:",
        f"{real_correct / real_total * 100:.2f}%"
    )


print()

print(
    "FAKE images:",
    fake_total
)

print(
    "FAKE correctly detected:",
    fake_correct
)


if fake_total > 0:

    print(
        "FAKE accuracy:",
        f"{fake_correct / fake_total * 100:.2f}%"
    )


print()

print(
    "Total correct:",
    total_correct
)

print(
    "Total images:",
    total_images
)

print(
    "Overall accuracy:",
    f"{overall_accuracy:.2f}%"
)


print()

print("=" * 60)
print("                         DONE")
print("=" * 60)
