import sys
import os
import cv2
import torch
import numpy as np


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


# ==========================================
# 4. LOAD CHECKPOINT
# ==========================================

checkpoint_path = "training/weights/xception_best.pth"

print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)


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
# 7. MOVE MODEL TO XPU
# ==========================================

model = model.to(device)

model.eval()

print("Model ready.")


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
# 10. MAIN WEBCAM LOOP
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
            "FakeShield - Phase 3C",
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
    # ANALYZE EVERY FACE
    # ======================================

    for face_number, face in enumerate(
        faces,
        start=1
    ):

        # ----------------------------------
        # Face bounding box
        # ----------------------------------

        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])


        # ----------------------------------
        # Keep coordinates inside frame
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
        # PREPROCESS FACE
        # ==================================

        # DeepfakeBench Xception input
        # size: 256 x 256

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


        # Convert uint8 -> float32

        image = image.astype(
            np.float32
        ) / 255.0


        # DeepfakeBench-style normalization
        #
        # mean = 0.5
        # std  = 0.5

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


        # NumPy -> PyTorch

        tensor = torch.from_numpy(
            image
        ).to(device)


        # ==================================
        # XCEPTION PREDICTION
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


        real_percent = (
            real_probability * 100
        )

        fake_percent = (
            fake_probability * 100
        )


        # ==================================
        # RESULT
        # ==================================

        if fake_percent >= 70:

            result = "FAKE"
            confidence = fake_percent

        elif real_percent >= 70:

            result = "REAL"
            confidence = real_percent

        else:

            result = "UNCERTAIN"
            confidence = max(
                real_percent,
                fake_percent
            )


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
        # DISPLAY LABEL 1
        # ==================================

        cv2.putText(
            frame,
            label1,
            (
                x1,
                max(25, y1 - 35)
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
                max(50, y1 - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2
        )


    # ======================================
    # SHOW FRAME
    # ======================================

    cv2.imshow(
        "FakeShield - Phase 3C",
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

print("Phase 3C stopped.")