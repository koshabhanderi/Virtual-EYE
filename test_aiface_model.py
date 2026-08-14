import os
import sys
import cv2
import torch
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

sys.path.insert(
    0,
    r".\DeepfakeBench\training"
)

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

print()
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

print()
print("Loading model weights...")


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
# IMAGE PREDICTION
# ============================================================

def predict_probability(image_path):

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )


    if image is None:

        print(
            "Could not load:",
            image_path
        )

        return None


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Keep the preprocessing that gave us:
    #
    # REAL = 100%
    #
    # We are NOT doing face cropping here.
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
    #
    # Keep /255 only.
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
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        result = model(
            tensor
        )


    # --------------------------------------------------------
    # GET MODEL OUTPUT
    # --------------------------------------------------------

    if isinstance(
        result,
        tuple
    ):

        output = result[0]

    else:

        output = result


    # --------------------------------------------------------
    # SINGLE VALUE
    # --------------------------------------------------------

    output = output.squeeze()


    # --------------------------------------------------------
    # SIGMOID
    #
    # close to 0 -> REAL
    # close to 1 -> FAKE
    # --------------------------------------------------------

    fake_probability = torch.sigmoid(
        output
    ).item()


    return fake_probability


# ============================================================
# COLLECT SCORES FROM A FOLDER
# ============================================================

def collect_scores(
    folder_path,
    expected_label
):

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    )


    # --------------------------------------------------------
    # CHECK FOLDER
    # --------------------------------------------------------

    if not os.path.exists(
        folder_path
    ):

        print()
        print(
            "ERROR: Folder does not exist:"
        )

        print(
            folder_path
        )

        return []


    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    image_files = []


    for filename in os.listdir(
        folder_path
    ):

        if filename.lower().endswith(
            valid_extensions
        ):

            image_files.append(
                filename
            )


    image_files.sort()


    print()
    print("=" * 60)

    print(
        "Collecting:",
        expected_label
    )

    print(
        "Folder:",
        folder_path
    )

    print(
        "Images:",
        len(image_files)
    )

    print("=" * 60)


    results = []


    # --------------------------------------------------------
    # PREDICT EVERY IMAGE
    # --------------------------------------------------------

    for index, filename in enumerate(
        image_files
    ):

        image_path = os.path.join(
            folder_path,
            filename
        )


        fake_probability = predict_probability(
            image_path
        )


        if fake_probability is None:

            continue


        real_probability = (
            1.0 - fake_probability
        )


        results.append(
            {
                "filename": filename,
                "expected": expected_label,
                "fake_probability": fake_probability,
                "real_probability": real_probability
            }
        )


        print(
            f"{index + 1:03d}. "
            f"{filename:20s} "
            f"REAL: {real_probability * 100:6.2f}% "
            f"FAKE: {fake_probability * 100:6.2f}%"
        )


    return results


# ============================================================
# CALCULATE ACCURACY FOR A THRESHOLD
# ============================================================

def calculate_accuracy(
    real_results,
    fake_results,
    threshold
):

    real_correct = 0

    fake_correct = 0


    # --------------------------------------------------------
    # REAL IMAGES
    #
    # fake_probability < threshold -> REAL
    # --------------------------------------------------------

    for result in real_results:

        fake_probability = (
            result["fake_probability"]
        )


        if fake_probability < threshold:

            real_correct += 1


    # --------------------------------------------------------
    # FAKE IMAGES
    #
    # fake_probability >= threshold -> FAKE
    # --------------------------------------------------------

    for result in fake_results:

        fake_probability = (
            result["fake_probability"]
        )


        if fake_probability >= threshold:

            fake_correct += 1


    # --------------------------------------------------------
    # TOTALS
    # --------------------------------------------------------

    real_total = len(
        real_results
    )

    fake_total = len(
        fake_results
    )


    total_correct = (
        real_correct +
        fake_correct
    )


    total_images = (
        real_total +
        fake_total
    )


    if real_total > 0:

        real_accuracy = (
            real_correct /
            real_total
        ) * 100

    else:

        real_accuracy = 0


    if fake_total > 0:

        fake_accuracy = (
            fake_correct /
            fake_total
        ) * 100

    else:

        fake_accuracy = 0


    if total_images > 0:

        overall_accuracy = (
            total_correct /
            total_images
        ) * 100

    else:

        overall_accuracy = 0


    return (
        real_correct,
        fake_correct,
        real_accuracy,
        fake_accuracy,
        overall_accuracy
    )


# ============================================================
# FIND BEST THRESHOLD
#
# REQUIREMENT:
#
# REAL accuracy MUST remain 100%.
#
# Among all thresholds that keep REAL at 100%,
# choose the threshold with the highest FAKE accuracy.
# ============================================================

def find_best_threshold(
    real_results,
    fake_results
):

    print()
    print("=" * 60)
    print("           SEARCHING BEST THRESHOLD")
    print("=" * 60)

    best_threshold = None

    best_fake_accuracy = -1

    best_overall_accuracy = -1

    best_real_accuracy = 0


    # --------------------------------------------------------
    # Test thresholds from 0.01 to 0.99
    # --------------------------------------------------------

    for threshold_number in range(
        1,
        100
    ):

        threshold = (
            threshold_number /
            100.0
        )


        (
            real_correct,
            fake_correct,
            real_accuracy,
            fake_accuracy,
            overall_accuracy
        ) = calculate_accuracy(
            real_results,
            fake_results,
            threshold
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We ONLY accept thresholds where REAL = 100%.
        # ----------------------------------------------------

        if real_accuracy >= 100.0:

            # ------------------------------------------------
            # Prefer highest FAKE accuracy.
            #
            # If FAKE accuracy is equal, prefer higher
            # overall accuracy.
            # ------------------------------------------------

            if (
                fake_accuracy > best_fake_accuracy
                or
                (
                    fake_accuracy == best_fake_accuracy
                    and
                    overall_accuracy > best_overall_accuracy
                )
            ):

                best_threshold = threshold

                best_fake_accuracy = (
                    fake_accuracy
                )

                best_overall_accuracy = (
                    overall_accuracy
                )

                best_real_accuracy = (
                    real_accuracy
                )


    # --------------------------------------------------------
    # IF NO THRESHOLD FOUND
    # --------------------------------------------------------

    if best_threshold is None:

        print()
        print(
            "No threshold can give 100% REAL accuracy."
        )

        print(
            "Keeping threshold = 0.50"
        )

        return 0.50


    # --------------------------------------------------------
    # DISPLAY BEST THRESHOLD
    # --------------------------------------------------------

    print()
    print(
        "Best threshold found:",
        f"{best_threshold:.2f}"
    )

    print(
        "REAL accuracy:",
        f"{best_real_accuracy:.2f}%"
    )

    print(
        "FAKE accuracy:",
        f"{best_fake_accuracy:.2f}%"
    )

    print(
        "Overall accuracy:",
        f"{best_overall_accuracy:.2f}%"
    )

    print("=" * 60)


    return best_threshold


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

def print_final_results(
    real_results,
    fake_results,
    threshold
):

    (
        real_correct,
        fake_correct,
        real_accuracy,
        fake_accuracy,
        overall_accuracy
    ) = calculate_accuracy(
        real_results,
        fake_results,
        threshold
    )


    real_total = len(
        real_results
    )

    fake_total = len(
        fake_results
    )


    total_correct = (
        real_correct +
        fake_correct
    )


    total_images = (
        real_total +
        fake_total
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("                    FINAL RESULTS")
    print("=" * 60)


    print()

    print(
        "DECISION THRESHOLD:",
        f"{threshold:.2f}"
    )


    print()

    print(
        "REAL images:",
        real_total
    )

    print(
        "REAL correctly detected:",
        real_correct
    )

    print(
        "REAL accuracy:",
        f"{real_accuracy:.2f}%"
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

    print(
        "FAKE accuracy:",
        f"{fake_accuracy:.2f}%"
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


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 60)
print("               STARTING VALIDATION")
print("=" * 60)


# ============================================================
# DATASET PATHS
# ============================================================

real_folder = (
    r".\ai_face_training\val\real"
)

fake_folder = (
    r".\ai_face_training\val\fake"
)


# ============================================================
# COLLECT REAL SCORES
# ============================================================

real_results = collect_scores(
    real_folder,
    "REAL"
)


# ============================================================
# COLLECT FAKE SCORES
# ============================================================

fake_results = collect_scores(
    fake_folder,
    "FAKE"
)


# ============================================================
# CHECK DATASET
# ============================================================

if (
    len(real_results) == 0
    or
    len(fake_results) == 0
):

    print()
    print(
        "ERROR: Could not collect enough images."
    )

    sys.exit(1)


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

best_threshold = find_best_threshold(
    real_results,
    fake_results
)


# ============================================================
# FINAL RESULTS USING BEST THRESHOLD
# ============================================================

print_final_results(
    real_results,
    fake_results,
    best_threshold
)