import sys
import os
import cv2
import torch
import numpy as np
import time


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
        key = key[len("backbone."):]

    new_checkpoint[key] = value


# ==========================================
# 6. LOAD WEIGHTS
# ==========================================

print("Loading weights...")

model.load_state_dict(new_checkpoint)

model = model.to(device)

model.eval()

print("Weights loaded successfully.")
print("Model ready.")


# ==========================================
# 7. LOAD YUNET
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
# 8. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open webcam.")
    sys.exit()


print("Webcam started.")
print("Batch inference enabled.")
print("Test with 1 face and 2 faces.")
print("Press Q to quit.")


# ==========================================
# 9. FPS VARIABLES
# ==========================================

frame_count = 0

start_time = time.perf_counter()

fps = 0.0

total_inference_time = 0.0

inference_count = 0


# ==========================================
# 10. MAIN LOOP
# ==========================================

while True:

    frame_start = time.perf_counter()

    # --------------------------------------
    # Read frame
    # --------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("Failed to read webcam.")
        break


    height, width = frame.shape[:2]


    # ======================================
    # FACE DETECTION
    # ======================================

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)


    if faces is None:

        faces = []


    face_count = len(faces)


    # ======================================
    # FACE TENSORS
    # ======================================

    tensors = []

    face_boxes = []


    # ======================================
    # PREPARE ALL FACES
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


        # ----------------------------------
        # Resize
        # ----------------------------------

        image = cv2.resize(
            face_crop,
            (256, 256),
            interpolation=cv2.INTER_CUBIC
        )


        # ----------------------------------
        # BGR -> RGB
        # ----------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------
        # Normalize
        # ----------------------------------

        image = image.astype(
            np.float32
        ) / 255.0


        image = (
            image - 0.5
        ) / 0.5


        # ----------------------------------
        # HWC -> CHW
        # ----------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )


        # ----------------------------------
        # NumPy -> Tensor
        # ----------------------------------

        tensor = torch.from_numpy(
            image
        )

        tensors.append(tensor)

        face_boxes.append(
            (x1, y1, x2, y2)
        )


    # ======================================
    # BATCH INFERENCE
    # ======================================

    if len(tensors) > 0:

        batch = torch.stack(
            tensors
        ).to(device)


        inference_start = time.perf_counter()


        with torch.no_grad():

            output, features = model(
                batch
            )


        # ----------------------------------
        # XPU synchronization
        # ----------------------------------

        if device.type == "xpu":

            torch.xpu.synchronize()


        inference_end = time.perf_counter()


        inference_time = (
            inference_end -
            inference_start
        )


        total_inference_time += (
            inference_time
        )

        inference_count += 1


        # ==================================
        # SOFTMAX FOR ALL FACES
        # ==================================

        probabilities = torch.softmax(
            output,
            dim=1
        )


        # ==================================
        # PROCESS RESULTS
        # ==================================

        for i, box in enumerate(
            face_boxes
        ):

            x1, y1, x2, y2 = box


            # ----------------------------------
            # Probabilities
            # ----------------------------------

            real_probability = (
                probabilities[i][0].item()
            )

            fake_probability = (
                probabilities[i][1].item()
            )


            real_percent = (
                real_probability * 100
            )

            fake_percent = (
                fake_probability * 100
            )


            # ----------------------------------
            # Classification
            # ----------------------------------

            if fake_percent >= 70:

                result = "FAKE"

            elif real_percent >= 70:

                result = "REAL"

            else:

                result = "UNCERTAIN"


            # ==================================
            # DRAW BOX
            # ==================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ==================================
            # LABEL
            # ==================================

            label = (
                f"Face {i + 1}: {result} | "
                f"R:{real_percent:.1f}% "
                f"F:{fake_percent:.1f}%"
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                2
            )


    # ======================================
    # CALCULATE FPS
    # ======================================

    frame_count += 1


    elapsed_time = (
        time.perf_counter()
        - start_time
    )


    if elapsed_time > 0:

        fps = (
            frame_count /
            elapsed_time
        )


    # ======================================
    # AVERAGE BATCH INFERENCE TIME
    # ======================================

    if inference_count > 0:

        average_inference_ms = (
            total_inference_time /
            inference_count
        ) * 1000

    else:

        average_inference_ms = 0


    # ======================================
    # DISPLAY FACE COUNT
    # ======================================

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ======================================
    # DISPLAY FPS
    # ======================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    # ======================================
    # DISPLAY INFERENCE TIME
    # ======================================

    cv2.putText(
        frame,
        f"Batch inference: "
        f"{average_inference_ms:.1f} ms",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


    # ======================================
    # TARGET
    # ======================================

    if fps >= 15:

        status = "TARGET: PASSED"

    else:

        status = "TARGET: BELOW 15 FPS"


    cv2.putText(
        frame,
        status,
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


    # ======================================
    # SHOW WINDOW
    # ======================================

    cv2.imshow(
        "FakeShield - Phase 3F Batch",
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


# ==========================================
# FINAL RESULTS
# ==========================================

print()

print("======================================")
print("       PHASE 3F BATCH RESULTS")
print("======================================")

print(
    f"Average webcam FPS: "
    f"{fps:.2f}"
)

print(
    f"Average batch inference time: "
    f"{average_inference_ms:.2f} ms"
)

print(
    f"Total batch inferences: "
    f"{inference_count}"
)

print(
    "Target: 15 FPS"
)


if fps >= 15:

    print(
        "RESULT: PASSED"
    )

else:

    print(
        "RESULT: NEEDS OPTIMIZATION"
    )


print("======================================")