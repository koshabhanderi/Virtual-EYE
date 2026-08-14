import sys
import os
import cv2
import torch
import numpy as np
from collections import deque


# ==========================================
# 1. DEEPFAKEBENCH PATH
# ==========================================

sys.path.append("training")

from networks.xception import Xception


# ==========================================
# 2. DEVICE
# ==========================================

if torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Device:", device)


# ==========================================
# 3. CREATE XCEPTION MODEL
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
# 4. LOAD CHECKPOINT
# ==========================================

checkpoint_path = "training/weights/xception_best.pth"

print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

print("Checkpoint loaded successfully.")


# ==========================================
# 5. PREPARE CHECKPOINT
# ==========================================

new_checkpoint = {}

for key, value in checkpoint.items():

    if key.startswith("backbone."):
        new_key = key[len("backbone."):]
    else:
        new_key = key

    new_checkpoint[new_key] = value


# ==========================================
# 6. LOAD WEIGHTS
# ==========================================

print("Loading weights...")

model.load_state_dict(new_checkpoint)

print("Weights loaded successfully.")


# ==========================================
# 7. MOVE MODEL TO DEVICE
# ==========================================

model = model.to(device)

model.eval()

print("Model moved to:", device)
print("Xception model is ready!")


# ==========================================
# 8. LOAD YUNET FACE DETECTOR
# ==========================================

model_path = os.path.join(
    "..",
    "phase1",
    "face_detection_yunet_2026may.onnx"
)

print("Loading face detector...")

detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

print("Face detector ready.")


# ==========================================
# 9. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open webcam.")

    sys.exit()


print("Webcam started.")
print("Press Q to quit.")


# ==========================================
# 10. STABILITY SETTINGS
# ==========================================

HISTORY_SIZE = 10

# Separate prediction history
# for every detected face.

prediction_history = {}


# ==========================================
# 11. MAIN LOOP
# ==========================================

while True:

    # --------------------------------------
    # Read frame
    # --------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("Failed to read webcam.")

        break


    # --------------------------------------
    # Frame dimensions
    # --------------------------------------

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )


    # --------------------------------------
    # Detect faces
    # --------------------------------------

    _, faces = detector.detect(frame)


    # ======================================
    # NO FACES
    # ======================================

    if faces is None:

        cv2.putText(
            frame,
            "Faces: 0",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "FakeShield - Phase 3D",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        continue


    # ======================================
    # SORT FACES LEFT TO RIGHT
    # ======================================

    faces = sorted(
        faces,
        key=lambda face: face[0]
    )


    # ======================================
    # DISPLAY FACE COUNT
    # ======================================

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ======================================
    # REMOVE OLD FACE HISTORY
    # ======================================

    active_faces = set()

    for face_number in range(
        1,
        len(faces) + 1
    ):

        active_faces.add(face_number)

        if face_number not in prediction_history:

            prediction_history[face_number] = deque(
                maxlen=HISTORY_SIZE
            )


    old_faces = list(
        prediction_history.keys()
    )

    for face_number in old_faces:

        if face_number not in active_faces:

            del prediction_history[
                face_number
            ]


    # ======================================
    # ANALYZE EVERY FACE
    # ======================================

    for face_number, face in enumerate(
        faces,
        start=1
    ):

        # ----------------------------------
        # Bounding box
        # ----------------------------------

        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])


        # ----------------------------------
        # Keep inside image
        # ----------------------------------

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


        # ----------------------------------
        # Crop face
        # ----------------------------------

        face_crop = frame[
            y1:y2,
            x1:x2
        ]


        if face_crop.size == 0:
            continue


        # ==================================
        # PREPROCESS
        # ==================================

        image = cv2.resize(
            face_crop,
            (256, 256),
            interpolation=cv2.INTER_CUBIC
        )


        # BGR -> RGB

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # uint8 -> float32

        image = image.astype(
            np.float32
        ) / 255.0


        # Normalize

        image = (
            image - 0.5
        ) / 0.5


        # HWC -> CHW

        image = np.transpose(
            image,
            (2, 0, 1)
        )


        # Add batch dimension

        image = np.expand_dims(
            image,
            axis=0
        )


        # NumPy -> Tensor

        tensor = torch.from_numpy(
            image
        ).to(device)


        # ==================================
        # MODEL PREDICTION
        # ==================================

        with torch.no_grad():

            output, features = model(
                tensor
            )


        # ==================================
        # SOFTMAX
        # ==================================

        probabilities = torch.softmax(
            output,
            dim=1
        )


        # Class 0 = REAL
        # Class 1 = FAKE

        real_probability = (
            probabilities[0][0].item()
        )

        fake_probability = (
            probabilities[0][1].item()
        )


        # ==================================
        # ADD CURRENT RESULT TO HISTORY
        # ==================================

        prediction_history[
            face_number
        ].append(
            fake_probability
        )


        # ==================================
        # CALCULATE AVERAGE
        # ==================================

        history = prediction_history[
            face_number
        ]

        average_fake = (
            sum(history) /
            len(history)
        )

        average_real = (
            1.0 - average_fake
        )


        real_percent = (
            average_real * 100
        )

        fake_percent = (
            average_fake * 100
        )


        # ==================================
        # CLASSIFICATION
        # ==================================

        if fake_percent >= 70:

            result = "FAKE"

        elif real_percent >= 70:

            result = "REAL"

        else:

            result = "UNCERTAIN"


        # ==================================
        # DRAW FACE BOX
        # ==================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ==================================
        # LABEL 1
        # ==================================

        label1 = (
            f"Face {face_number}: {result}"
        )


        # ==================================
        # LABEL 2
        # ==================================

        label2 = (
            f"Real {real_percent:.1f}% | "
            f"Fake {fake_percent:.1f}%"
        )


        # ==================================
        # LABEL 3
        # ==================================

        label3 = (
            f"Samples: {len(history)}/"
            f"{HISTORY_SIZE}"
        )


        # ==================================
        # DISPLAY LABEL 1
        # ==================================

        cv2.putText(
            frame,
            label1,
            (
                x1,
                max(25, y1 - 55)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        # ==================================
        # DISPLAY LABEL 2
        # ==================================

        cv2.putText(
            frame,
            label2,
            (
                x1,
                max(50, y1 - 30)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2
        )


        # ==================================
        # DISPLAY LABEL 3
        # ==================================

        cv2.putText(
            frame,
            label3,
            (
                x1,
                max(75, y1 - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )


    # ======================================
    # SHOW FRAME
    # ======================================

    cv2.imshow(
        "FakeShield - Phase 3D",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Phase 3D stopped.")