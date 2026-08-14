import sys
import os
import cv2
import torch
import numpy as np
import csv
import time


# ==========================================
# 1. DEEPFAKEBENCH PATH
# ==========================================

sys.path.append("training")

from networks.xception import Xception


# ==========================================
# 2. PATHS
# ==========================================

REAL_DIR = "phase3g_test/real"
FAKE_DIR = "phase3g_test/fake"
RESULT_DIR = "phase3g_test/results"

os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "phase3g_results.csv"
)


# ==========================================
# 3. DEVICE
# ==========================================

if torch.xpu.is_available():

    device = torch.device("xpu")

else:

    device = torch.device("cpu")


print("Device:", device)


# ==========================================
# 4. CREATE XCEPTION MODEL
# ==========================================

config = {
    "mode": "original",
    "num_classes": 2,
    "inc": 3,
    "dropout": False
}

print("Creating Xception model...")

model = Xception(config)

print("Model created successfully.")


# ==========================================
# 5. LOAD CHECKPOINT
# ==========================================

checkpoint_path = "training/weights/xception_best.pth"

print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

print("Checkpoint loaded successfully.")


# ==========================================
# 6. PREPARE CHECKPOINT
# ==========================================

new_checkpoint = {}

for key, value in checkpoint.items():

    if key.startswith("backbone."):

        key = key[len("backbone."):]

    new_checkpoint[key] = value


# ==========================================
# 7. LOAD MODEL WEIGHTS
# ==========================================

print("Loading weights...")

model.load_state_dict(new_checkpoint)

model = model.to(device)

model.eval()

print("Weights loaded successfully.")
print("Model ready.")


# ==========================================
# 8. LOAD YUNET
# ==========================================

yunet_path = os.path.join(
    "..",
    "phase1",
    "face_detection_yunet_2026may.onnx"
)

print("Loading face detector...")

detector = cv2.FaceDetectorYN.create(
    yunet_path,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

print("Face detector ready.")


# ==========================================
# 9. IMAGE EXTENSIONS
# ==========================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ==========================================
# 10. GET IMAGE FILES
# ==========================================

def get_images(folder):

    files = []

    if not os.path.exists(folder):

        return files

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            VALID_EXTENSIONS
        ):

            files.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    files.sort()

    return files


# ==========================================
# 11. PREPROCESS FACE
# ==========================================

def preprocess_face(face_crop):

    image = cv2.resize(
        face_crop,
        (256, 256),
        interpolation=cv2.INTER_CUBIC
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = (
        image - 0.5
    ) / 0.5

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    tensor = torch.from_numpy(
        image
    )

    return tensor


# ==========================================
# 12. PREDICT ONE IMAGE
# ==========================================

def predict_image(image_path):

    frame = cv2.imread(image_path)

    if frame is None:

        return None


    height, width = frame.shape[:2]


    # --------------------------------------
    # Detect faces
    # --------------------------------------

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)


    if faces is None or len(faces) == 0:

        return {
            "faces": 0,
            "prediction": "NO_FACE",
            "real": 0.0,
            "fake": 0.0
        }


    tensors = []


    # ======================================
    # PREPARE ALL DETECTED FACES
    # ======================================

    for face in faces:

        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])


        x1 = max(
            0,
            x
        )

        y1 = max(
            0,
            y
        )

        x2 = min(
            width,
            x + w
        )

        y2 = min(
            height,
            y + h
        )


        face_crop = frame[
            y1:y2,
            x1:x2
        ]


        if face_crop.size == 0:

            continue


        tensor = preprocess_face(
            face_crop
        )

        tensors.append(
            tensor.squeeze(0)
        )


    if len(tensors) == 0:

        return {
            "faces": 0,
            "prediction": "NO_FACE",
            "real": 0.0,
            "fake": 0.0
        }


    # ======================================
    # BATCH ALL FACES
    # ======================================

    batch = torch.stack(
        tensors
    ).to(device)


    # ======================================
    # MODEL INFERENCE
    # ======================================

    with torch.no_grad():

        output, features = model(
            batch
        )


    if device.type == "xpu":

        torch.xpu.synchronize()


    # ======================================
    # PROBABILITIES
    # ======================================

    probabilities = torch.softmax(
        output,
        dim=1
    )


    # ======================================
    # COMBINE MULTIPLE FACE RESULTS
    # ======================================

    real_probabilities = (
        probabilities[:, 0]
        .detach()
        .cpu()
        .numpy()
    )

    fake_probabilities = (
        probabilities[:, 1]
        .detach()
        .cpu()
        .numpy()
    )


    # Average prediction across faces

    real_probability = float(
        np.mean(real_probabilities)
    )

    fake_probability = float(
        np.mean(fake_probabilities)
    )


    real_percent = (
        real_probability * 100
    )

    fake_percent = (
        fake_probability * 100
    )


    # ======================================
    # FINAL CLASSIFICATION
    # ======================================

    if fake_probability >= 0.5:

        prediction = "FAKE"

    else:

        prediction = "REAL"


    return {
        "faces": len(tensors),
        "prediction": prediction,
        "real": real_percent,
        "fake": fake_percent
    }


# ==========================================
# 13. COLLECT TEST IMAGES
# ==========================================

real_images = get_images(
    REAL_DIR
)

fake_images = get_images(
    FAKE_DIR
)


print()
print("======================================")
print("       PHASE 3G ACCURACY TEST")
print("======================================")

print(
    "Real images:",
    len(real_images)
)

print(
    "Fake images:",
    len(fake_images)
)

print()


# ==========================================
# 14. RESULTS STORAGE
# ==========================================

results = []


# ==========================================
# 15. TEST REAL IMAGES
# ==========================================

print("Testing REAL images...")

for index, image_path in enumerate(
    real_images,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    print(
        f"[REAL {index}/{len(real_images)}] "
        f"{filename}"
    )


    start = time.perf_counter()

    result = predict_image(
        image_path
    )

    elapsed = (
        time.perf_counter()
        - start
    )


    if result is None:

        print("  Could not read image.")

        continue


    actual = "REAL"

    predicted = result["prediction"]


    results.append({
        "filename": filename,
        "actual": actual,
        "predicted": predicted,
        "faces": result["faces"],
        "real_percent": result["real"],
        "fake_percent": result["fake"],
        "time_seconds": elapsed
    })


    print(
        f"  Prediction: {predicted}"
    )

    print(
        f"  Real: {result['real']:.2f}%"
    )

    print(
        f"  Fake: {result['fake']:.2f}%"
    )

    print()


# ==========================================
# 16. TEST FAKE IMAGES
# ==========================================

print("Testing FAKE images...")

for index, image_path in enumerate(
    fake_images,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    print(
        f"[FAKE {index}/{len(fake_images)}] "
        f"{filename}"
    )


    start = time.perf_counter()

    result = predict_image(
        image_path
    )

    elapsed = (
        time.perf_counter()
        - start
    )


    if result is None:

        print("  Could not read image.")

        continue


    actual = "FAKE"

    predicted = result["prediction"]


    results.append({
        "filename": filename,
        "actual": actual,
        "predicted": predicted,
        "faces": result["faces"],
        "real_percent": result["real"],
        "fake_percent": result["fake"],
        "time_seconds": elapsed
    })


    print(
        f"  Prediction: {predicted}"
    )

    print(
        f"  Real: {result['real']:.2f}%"
    )

    print(
        f"  Fake: {result['fake']:.2f}%"
    )

    print()


# ==========================================
# 17. CALCULATE METRICS
# ==========================================

total = len(results)

correct = 0

false_positives = 0

false_negatives = 0

real_total = 0

fake_total = 0


for result in results:

    actual = result["actual"]

    predicted = result["predicted"]


    if actual == predicted:

        correct += 1


    # REAL incorrectly classified as FAKE

    if actual == "REAL":

        real_total += 1

        if predicted == "FAKE":

            false_positives += 1


    # FAKE incorrectly classified as REAL

    if actual == "FAKE":

        fake_total += 1

        if predicted == "REAL":

            false_negatives += 1


# ==========================================
# 18. METRIC CALCULATIONS
# ==========================================

if total > 0:

    accuracy = (
        correct / total
    ) * 100

else:

    accuracy = 0


if real_total > 0:

    false_positive_rate = (
        false_positives /
        real_total
    ) * 100

else:

    false_positive_rate = 0


if fake_total > 0:

    false_negative_rate = (
        false_negatives /
        fake_total
    ) * 100

else:

    false_negative_rate = 0


# ==========================================
# 19. SAVE CSV
# ==========================================

with open(
    RESULT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "actual",
            "predicted",
            "faces",
            "real_percent",
            "fake_percent",
            "time_seconds"
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ==========================================
# 20. FINAL REPORT
# ==========================================

print()

print("======================================")
print("       PHASE 3G RESULTS")
print("======================================")

print(
    f"Total images: {total}"
)

print(
    f"Real images: {real_total}"
)

print(
    f"Fake images: {fake_total}"
)

print(
    f"Correct predictions: {correct}"
)

print(
    f"Incorrect predictions: "
    f"{total - correct}"
)

print()

print(
    f"Accuracy: "
    f"{accuracy:.2f}%"
)

print(
    f"False Positive Rate: "
    f"{false_positive_rate:.2f}%"
)

print(
    f"False Negative Rate: "
    f"{false_negative_rate:.2f}%"
)

print()

print(
    "Detailed results saved to:"
)

print(
    RESULT_FILE
)

print("======================================")