import sys
import os
import cv2
import torch
import numpy as np
import csv
import time
import dlib

from skimage import transform as trans


# ============================================================
# 1. DEEPFAKEBENCH PATH
# ============================================================

sys.path.append("training")

from networks.xception import Xception


# ============================================================
# 2. PATHS
# ============================================================

REAL_DIR = "phase3g_test/real"
FAKE_DIR = "phase3g_test/fake"

RESULT_DIR = "phase3g_test/results"

os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "phase3g_aligned_results.csv"
)


# ============================================================
# 3. DEVICE
# ============================================================

if torch.xpu.is_available():

    device = torch.device("xpu")

else:

    device = torch.device("cpu")


print("Device:", device)


# ============================================================
# 4. CREATE XCEPTION MODEL
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
# 5. LOAD CHECKPOINT
# ============================================================

checkpoint_path = "training/weights/xception_best.pth"

print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

print("Checkpoint loaded successfully.")


# ============================================================
# 6. PREPARE CHECKPOINT
# ============================================================

new_checkpoint = {}

for key, value in checkpoint.items():

    if key.startswith("backbone."):

        key = key[len("backbone."):]

    new_checkpoint[key] = value


# ============================================================
# 7. LOAD WEIGHTS
# ============================================================

print("Loading weights...")

model.load_state_dict(new_checkpoint)

model = model.to(device)

model.eval()

print("Weights loaded successfully.")
print("Model ready.")


# ============================================================
# 8. LOAD DLIB FACE DETECTOR
# ============================================================

print("Loading dlib face detector...")

face_detector = dlib.get_frontal_face_detector()


# ============================================================
# 9. LOAD 81-POINT LANDMARK MODEL
# ============================================================

predictor_path = (
    "preprocessing/dlib_tools/"
    "shape_predictor_81_face_landmarks.dat"
)

print("Loading landmark predictor...")

if not os.path.exists(predictor_path):

    print()
    print("ERROR:")
    print("Landmark predictor was not found:")
    print(predictor_path)
    print()

    sys.exit(1)


predictor = dlib.shape_predictor(
    predictor_path
)

print("Landmark predictor loaded.")


# ============================================================
# 10. IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# 11. GET IMAGE FILES
# ============================================================

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


# ============================================================
# 12. GET DEEPFAKEBENCH 5 LANDMARKS
# ============================================================

def get_keypts(image, face):

    # DeepfakeBench uses the 81-point predictor
    shape = predictor(
        image,
        face
    )

    # --------------------------------------------------------
    # Left eye
    # landmark 37
    # --------------------------------------------------------

    leye = np.array(
        [
            shape.part(37).x,
            shape.part(37).y
        ]
    ).reshape(-1, 2)


    # --------------------------------------------------------
    # Right eye
    # landmark 44
    # --------------------------------------------------------

    reye = np.array(
        [
            shape.part(44).x,
            shape.part(44).y
        ]
    ).reshape(-1, 2)


    # --------------------------------------------------------
    # Nose
    # landmark 30
    # --------------------------------------------------------

    nose = np.array(
        [
            shape.part(30).x,
            shape.part(30).y
        ]
    ).reshape(-1, 2)


    # --------------------------------------------------------
    # Left mouth
    # landmark 49
    # --------------------------------------------------------

    lmouth = np.array(
        [
            shape.part(49).x,
            shape.part(49).y
        ]
    ).reshape(-1, 2)


    # --------------------------------------------------------
    # Right mouth
    # landmark 55
    # --------------------------------------------------------

    rmouth = np.array(
        [
            shape.part(55).x,
            shape.part(55).y
        ]
    ).reshape(-1, 2)


    pts = np.concatenate(
        [
            leye,
            reye,
            nose,
            lmouth,
            rmouth
        ],
        axis=0
    )

    return pts


# ============================================================
# 13. DEEPFAKEBENCH ALIGNMENT
# ============================================================

def img_align_crop(
    img,
    landmark,
    outsize=(256, 256),
    scale=1.3
):

    # --------------------------------------------------------
    # DeepfakeBench target landmarks
    # --------------------------------------------------------

    target_size = [
        112,
        112
    ]


    dst = np.array(
        [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Original DeepfakeBench adjustment
    # --------------------------------------------------------

    if target_size[1] == 112:

        dst[:, 0] += 8.0


    # --------------------------------------------------------
    # Scale destination points
    # --------------------------------------------------------

    dst[:, 0] = (
        dst[:, 0]
        * outsize[0]
        / target_size[0]
    )

    dst[:, 1] = (
        dst[:, 1]
        * outsize[1]
        / target_size[1]
    )


    target_size = outsize


    # --------------------------------------------------------
    # Add margin
    # scale = 1.3
    # --------------------------------------------------------

    margin_rate = scale - 1

    x_margin = (
        target_size[0]
        * margin_rate
        / 2
    )

    y_margin = (
        target_size[1]
        * margin_rate
        / 2
    )


    # --------------------------------------------------------
    # Move
    # --------------------------------------------------------

    dst[:, 0] += x_margin

    dst[:, 1] += y_margin


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    dst[:, 0] *= (
        target_size[0]
        /
        (
            target_size[0]
            + 2 * x_margin
        )
    )

    dst[:, 1] *= (
        target_size[1]
        /
        (
            target_size[1]
            + 2 * y_margin
        )
    )


    # --------------------------------------------------------
    # Source landmarks
    # --------------------------------------------------------

    src = landmark.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Similarity transformation
    # --------------------------------------------------------

    tform = trans.SimilarityTransform()

    tform.estimate(
        src,
        dst
    )

    M = tform.params[
        0:2,
        :
    ]


    # --------------------------------------------------------
    # Warp image
    # --------------------------------------------------------

    aligned = cv2.warpAffine(
        img,
        M,
        (
            target_size[1],
            target_size[0]
        )
    )


    # --------------------------------------------------------
    # Resize if necessary
    # --------------------------------------------------------

    if outsize is not None:

        aligned = cv2.resize(
            aligned,
            (
                outsize[1],
                outsize[0]
            )
        )


    return aligned


# ============================================================
# 14. ALIGN ONE IMAGE
# ============================================================

def align_face(image):

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    frame = cv2.imread(
        image
    )

    if frame is None:

        return None


    # --------------------------------------------------------
    # Convert BGR → RGB
    # DeepfakeBench does this before dlib
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = face_detector(
        rgb,
        1
    )


    if len(faces) == 0:

        return None


    # --------------------------------------------------------
    # DeepfakeBench chooses biggest face
    # --------------------------------------------------------

    face = max(
        faces,
        key=lambda rect:
        rect.width() * rect.height()
    )


    # --------------------------------------------------------
    # Get five key landmarks
    # --------------------------------------------------------

    landmarks = get_keypts(
        rgb,
        face
    )


    # --------------------------------------------------------
    # Align and crop
    # --------------------------------------------------------

    cropped_face = img_align_crop(
        rgb,
        landmarks,
        outsize=(256, 256),
        scale=1.3
    )


    return cropped_face


# ============================================================
# 15. PREPROCESS FOR XCEPTION
# ============================================================

def preprocess_face(face):

    # face is RGB

    image = face.astype(
        np.float32
    ) / 255.0


    # --------------------------------------------------------
    # DeepfakeBench normalization
    # mean = [0.5, 0.5, 0.5]
    # std  = [0.5, 0.5, 0.5]
    # --------------------------------------------------------

    image = (
        image - 0.5
    ) / 0.5


    # --------------------------------------------------------
    # HWC → CHW
    # --------------------------------------------------------

    image = np.transpose(
        image,
        (2, 0, 1)
    )


    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        image
    ).float()


    return tensor


# ============================================================
# 16. PREDICT IMAGE
# ============================================================

def predict_image(image_path):

    aligned_face = align_face(
        image_path
    )


    if aligned_face is None:

        return {
            "faces": 0,
            "prediction": "NO_FACE",
            "real": 0.0,
            "fake": 0.0
        }


    tensor = preprocess_face(
        aligned_face
    )


    tensor = tensor.unsqueeze(
        0
    )


    tensor = tensor.to(
        device
    )


    # --------------------------------------------------------
    # Xception inference
    # --------------------------------------------------------

    with torch.no_grad():

        output, features = model(
            tensor
        )


    if device.type == "xpu":

        torch.xpu.synchronize()


    # --------------------------------------------------------
    # Softmax
    # Class 0 = REAL
    # Class 1 = FAKE
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output,
        dim=1
    )


    real_probability = (
        probabilities[0, 0]
        .detach()
        .cpu()
        .item()
    )


    fake_probability = (
        probabilities[0, 1]
        .detach()
        .cpu()
        .item()
    )


    real_percent = (
        real_probability * 100
    )


    fake_percent = (
        fake_probability * 100
    )


    if fake_probability >= 0.5:

        prediction = "FAKE"

    else:

        prediction = "REAL"


    return {
        "faces": 1,
        "prediction": prediction,
        "real": real_percent,
        "fake": fake_percent
    }


# ============================================================
# 17. GET TEST IMAGES
# ============================================================

real_images = get_images(
    REAL_DIR
)

fake_images = get_images(
    FAKE_DIR
)


print()
print("==========================================")
print("     PHASE 3G ALIGNED ACCURACY TEST")
print("==========================================")

print(
    "Real images:",
    len(real_images)
)

print(
    "Fake images:",
    len(fake_images)
)

print()


# ============================================================
# 18. RESULTS
# ============================================================

results = []


# ============================================================
# 19. TEST REAL IMAGES
# ============================================================

print("Testing REAL images...")
print()


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


    try:

        result = predict_image(
            image_path
        )

    except Exception as e:

        print(
            "  ERROR:",
            str(e)
        )

        continue


    elapsed = (
        time.perf_counter()
        - start
    )


    if result["faces"] == 0:

        print(
            "  No face detected."
        )

        predicted = "NO_FACE"

    else:

        predicted = result[
            "prediction"
        ]


    results.append({

        "filename": filename,

        "actual": "REAL",

        "predicted": predicted,

        "faces": result["faces"],

        "real_percent":
            result["real"],

        "fake_percent":
            result["fake"],

        "time_seconds":
            elapsed
    })


    print(
        f"  Prediction: {predicted}"
    )

    print(
        f"  Real: "
        f"{result['real']:.2f}%"
    )

    print(
        f"  Fake: "
        f"{result['fake']:.2f}%"
    )

    print()


# ============================================================
# 20. TEST FAKE IMAGES
# ============================================================

print("Testing FAKE images...")
print()


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


    try:

        result = predict_image(
            image_path
        )

    except Exception as e:

        print(
            "  ERROR:",
            str(e)
        )

        continue


    elapsed = (
        time.perf_counter()
        - start
    )


    if result["faces"] == 0:

        print(
            "  No face detected."
        )

        predicted = "NO_FACE"

    else:

        predicted = result[
            "prediction"
        ]


    results.append({

        "filename": filename,

        "actual": "FAKE",

        "predicted": predicted,

        "faces": result["faces"],

        "real_percent":
            result["real"],

        "fake_percent":
            result["fake"],

        "time_seconds":
            elapsed
    })


    print(
        f"  Prediction: {predicted}"
    )

    print(
        f"  Real: "
        f"{result['real']:.2f}%"
    )

    print(
        f"  Fake: "
        f"{result['fake']:.2f}%"
    )

    print()


# ============================================================
# 21. CALCULATE RESULTS
# ============================================================

total = len(results)

correct = 0

false_positives = 0

false_negatives = 0

real_total = 0

fake_total = 0


for result in results:

    actual = result["actual"]

    predicted = result["predicted"]


    # Correct

    if actual == predicted:

        correct += 1


    # REAL → FAKE

    if actual == "REAL":

        real_total += 1

        if predicted == "FAKE":

            false_positives += 1


    # FAKE → REAL

    if actual == "FAKE":

        fake_total += 1

        if predicted == "REAL":

            false_negatives += 1


# ============================================================
# 22. METRICS
# ============================================================

if total > 0:

    accuracy = (
        correct / total
    ) * 100

else:

    accuracy = 0


if real_total > 0:

    false_positive_rate = (
        false_positives
        / real_total
    ) * 100

else:

    false_positive_rate = 0


if fake_total > 0:

    false_negative_rate = (
        false_negatives
        / fake_total
    ) * 100

else:

    false_negative_rate = 0


# ============================================================
# 23. SAVE CSV
# ============================================================

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


# ============================================================
# 24. FINAL REPORT
# ============================================================

print()

print("==========================================")
print("          PHASE 3G RESULTS")
print("==========================================")

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

print("==========================================")