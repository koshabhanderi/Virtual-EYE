import os
import sys
import cv2
import torch
import numpy as np
from collections import deque


# ============================================================
# FAKESHIELD - GOOGLE MEET LARGEST FACE DETECTOR
# ============================================================

print("=" * 65)
print("       FAKESHIELD - GOOGLE MEET LARGEST FACE")
print("=" * 65)


# ============================================================
# DEEPFAKEBENCH PATH
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

print("Device:", device)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

config = {
    "mode": "original",
    "num_classes": 1,
    "inc": 3,
    "dropout": False
}


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("Creating Xception model...")

model = Xception(config)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint_path = (
    r".\ai_face_weights\xception_ai_face.pth"
)

if not os.path.exists(checkpoint_path):

    print()
    print("ERROR: Checkpoint not found:")
    print(checkpoint_path)

    sys.exit(1)


print("Loading AI-Face checkpoint...")

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
# LOAD MODEL WEIGHTS
# ============================================================

print("Loading model weights...")

model.load_state_dict(
    new_checkpoint,
    strict=True
)

model = model.to(device)

model.eval()

print("Weights loaded successfully!")
print("Model ready.")


# ============================================================
# YUNET FACE DETECTOR
# ============================================================

yunet_path = (
    r".\phase1\face_detection_yunet_2026may.onnx"
)

if not os.path.exists(yunet_path):

    print()
    print("ERROR: YuNet model not found:")
    print(yunet_path)

    sys.exit(1)


print()
print("Loading YuNet face detector...")

detector = cv2.FaceDetectorYN.create(
    yunet_path,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

print("YuNet ready.")


# ============================================================
# FIND OBS VIRTUAL CAMERA
# ============================================================

print()
print("Searching for OBS Virtual Camera...")

try:

    from pygrabber.dshow_graph import FilterGraph

    graph = FilterGraph()

    camera_names = graph.get_input_devices()

except Exception as error:

    print()
    print("ERROR: Could not enumerate cameras.")
    print(error)
    print()
    print("Install pygrabber using:")
    print("pip install pygrabber")

    sys.exit(1)


print()
print("Available cameras:")

for index, name in enumerate(camera_names):

    print(
        f"{index}: {name}"
    )


# ============================================================
# FIND OBS CAMERA
# ============================================================

obs_index = None

for index, name in enumerate(camera_names):

    name_lower = name.lower()

    if (
        "obs virtual camera" in name_lower
        or
        (
            "obs" in name_lower
            and
            "camera" in name_lower
        )
    ):

        obs_index = index
        break


# ============================================================
# OBS CAMERA NOT FOUND
# ============================================================

if obs_index is None:

    print()
    print("=" * 65)
    print("ERROR: OBS VIRTUAL CAMERA NOT FOUND")
    print("=" * 65)
    print()

    print("Make sure OBS Virtual Camera is running.")
    print()

    print("Available cameras:")

    for index, name in enumerate(camera_names):

        print(
            f"{index}: {name}"
        )

    sys.exit(1)


print()
print(
    "OBS Virtual Camera found at index:",
    obs_index
)


# ============================================================
# OPEN OBS VIRTUAL CAMERA
# ============================================================

camera = cv2.VideoCapture(
    obs_index,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    print()
    print("ERROR: Could not open OBS Virtual Camera.")

    sys.exit(1)


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

print(
    "OBS Virtual Camera opened successfully."
)


# ============================================================
# FACE PREDICTION FUNCTION
# ============================================================

def predict_face(face):

    # Resize
    face = cv2.resize(
        face,
        (299, 299),
        interpolation=cv2.INTER_AREA
    )

    # BGR -> RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # Normalize
    face = (
        face.astype(np.float32)
        / 255.0
    )

    # HWC -> CHW
    face = np.transpose(
        face,
        (2, 0, 1)
    )

    # Add batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )

    # NumPy -> PyTorch
    tensor = torch.from_numpy(
        face
    ).to(device)

    # Prediction
    with torch.no_grad():

        result = model(tensor)

    # Get output
    if isinstance(result, tuple):

        output = result[0]

    else:

        output = result

    output = output.squeeze()

    # Sigmoid
    fake_probability = torch.sigmoid(
        output
    ).item()

    return fake_probability


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

prediction_history = deque(
    maxlen=20
)


# ============================================================
# CREATE ONLY ONE WINDOW
# ============================================================

WINDOW_NAME = (
    "FakeShield - Remote Detection"
)

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1000,
    700
)


# ============================================================
# START
# ============================================================

print()
print("=" * 65)
print("FAKESHIELD READY")
print("=" * 65)
print()
print("Input: OBS Virtual Camera")
print()
print("The largest visible face will be analyzed.")
print()
print("Press Q to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # READ OBS FRAME
        # ----------------------------------------------------

        ret, frame = camera.read()

        if not ret or frame is None:

            print(
                "Could not read OBS Virtual Camera."
            )

            break


        height, width = frame.shape[:2]


        # ----------------------------------------------------
        # DETECT FACES
        # ----------------------------------------------------

        detector.setInputSize(
            (
                width,
                height
            )
        )

        _, faces = detector.detect(
            frame
        )


        # ====================================================
        # NO FACE
        # ====================================================

        if faces is None or len(faces) == 0:

            prediction_history.clear()

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (
                    20,
                    50
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (
                    0,
                    255,
                    255
                ),
                2
            )


        else:

            # =================================================
            # FIND LARGEST FACE
            # =================================================

            largest_face = None
            largest_area = 0

            for face in faces:

                x, y, w, h = (
                    face[:4].astype(int)
                )

                area = w * h

                if area > largest_area:

                    largest_area = area
                    largest_face = face


            # =================================================
            # PROCESS LARGEST FACE
            # =================================================

            if largest_face is not None:

                x, y, w, h = (
                    largest_face[:4].astype(int)
                )


                # ------------------------------------------------
                # KEEP COORDINATES INSIDE FRAME
                # ------------------------------------------------

                x = max(
                    0,
                    x
                )

                y = max(
                    0,
                    y
                )

                w = min(
                    w,
                    width - x
                )

                h = min(
                    h,
                    height - y
                )


                # ------------------------------------------------
                # IGNORE VERY SMALL FACE
                # ------------------------------------------------

                if w >= 70 and h >= 70:

                    face_crop = frame[
                        y:y + h,
                        x:x + w
                    ]


                    if face_crop.size > 0:

                        # ========================================
                        # PREDICT
                        # ========================================

                        fake_probability = (
                            predict_face(
                                face_crop
                            )
                        )


                        # ========================================
                        # STORE PREDICTION
                        # ========================================

                        prediction_history.append(
                            fake_probability
                        )


                        # ========================================
                        # SMOOTH PREDICTION
                        # ========================================

                        smoothed_fake = float(
                            np.mean(
                                prediction_history
                            )
                        )


                        real_probability = (
                            1.0
                            - smoothed_fake
                        )


                        # ========================================
                        # CLASSIFICATION
                        #
                        # 0 - 55%  = REAL
                        # 55 - 80% = UNCERTAIN
                        # 80 - 100% = FAKE
                        # ========================================

                        if smoothed_fake >= 0.80:

                            label = "FAKE"

                            color = (
                                0,
                                0,
                                255
                            )

                            confidence = (
                                smoothed_fake
                                * 100
                            )


                        elif smoothed_fake <= 0.55:

                            label = "REAL"

                            color = (
                                0,
                                255,
                                0
                            )

                            confidence = (
                                real_probability
                                * 100
                            )


                        else:

                            label = "UNCERTAIN"

                            color = (
                                0,
                                255,
                                255
                            )

                            confidence = 50.0


                        # ========================================
                        # FACE RECTANGLE
                        # ========================================

                        cv2.rectangle(
                            frame,
                            (
                                x,
                                y
                            ),
                            (
                                x + w,
                                y + h
                            ),
                            color,
                            3
                        )


                        # ========================================
                        # LABEL POSITION
                        # ========================================

                        label_y = max(
                            30,
                            y - 45
                        )

                        result_y = max(
                            60,
                            y - 15
                        )


                        # ========================================
                        # LARGEST FACE LABEL
                        # ========================================

                        cv2.putText(
                            frame,
                            "LARGEST FACE",
                            (
                                x,
                                label_y
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2
                        )


                        # ========================================
                        # RESULT
                        # ========================================

                        cv2.putText(
                            frame,
                            f"{label} "
                            f"{confidence:.1f}%",
                            (
                                x,
                                result_y
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            color,
                            2
                        )


                        # ========================================
                        # REAL / FAKE PROBABILITIES
                        # ========================================

                        probability_text = (
                            f"REAL: "
                            f"{real_probability * 100:.1f}%  "
                            f"FAKE: "
                            f"{smoothed_fake * 100:.1f}%"
                        )


                        probability_y = (
                            y + h + 30
                        )


                        if probability_y >= height:

                            probability_y = (
                                height - 15
                            )


                        cv2.putText(
                            frame,
                            probability_text,
                            (
                                x,
                                probability_y
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2
                        )


            # =================================================
            # NUMBER OF FACES
            # =================================================

            cv2.putText(
                frame,
                f"FACES DETECTED: {len(faces)}",
                (
                    20,
                    45
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (
                    0,
                    255,
                    0
                ),
                2
            )


        # ====================================================
        # WATERMARK
        # ====================================================

        cv2.putText(
            frame,
            "FAKE SHIELD - GOOGLE MEET ANALYSIS",
            (
                20,
                height - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                255,
                255
            ),
            2
        )


        # ====================================================
        # DISPLAY ONLY ONE WINDOW
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )


        # ====================================================
        # QUIT WITH Q
        # ====================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break


        # ====================================================
        # CHECK WINDOW
        # ====================================================

        try:

            visible = cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE
            )

            if visible < 1:

                break

        except Exception:

            break


finally:

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("FakeShield stopped.")