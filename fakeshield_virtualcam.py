import os
import sys
import cv2
import torch
import numpy as np
import pyvirtualcam
from collections import deque


# ============================================================
# FAKE SHIELD - STABLE VIRTUAL CAMERA
# PHASE 3K
# ============================================================


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


print("=" * 70)
print("              FAKE SHIELD")
print("       STABLE VIRTUAL CAMERA")
print("=" * 70)

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

print()
print("Loading checkpoint...")
print("Checkpoint:", checkpoint_path)


if not os.path.exists(checkpoint_path):

    print("ERROR: Checkpoint not found.")

    sys.exit(1)


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

        new_key = key[
            len("backbone.") :
        ]

    else:

        new_key = key

    new_checkpoint[
        new_key
    ] = value


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

model = model.to(
    device
)

model.eval()

print("Model ready.")


# ============================================================
# LOAD YUNET
# ============================================================

print()
print("Loading YuNet face detector...")


yunet_path = (
    r".\phase1\face_detection_yunet_2026may.onnx"
)


if not os.path.exists(yunet_path):

    print()
    print("ERROR:")
    print("YuNet model not found:")
    print(yunet_path)

    sys.exit(1)


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
# STABILITY SETTINGS
# ============================================================

# Number of predictions stored for each face.
HISTORY_LENGTH = 15


# ------------------------------------------------------------
# Deadband
#
# If the smoothed probability is near the decision boundary,
# don't rapidly switch labels.
# ------------------------------------------------------------

LOW_THRESHOLD = 0.45
HIGH_THRESHOLD = 0.55


# ------------------------------------------------------------
# Normal classification threshold
# ------------------------------------------------------------

DEFAULT_THRESHOLD = 0.50


# ============================================================
# FACE HISTORY
# ============================================================

face_histories = {}


# ============================================================
# FACE ID
# ============================================================

next_face_id = 1


# ============================================================
# FACE CENTER DISTANCE
# ============================================================

MAX_FACE_DISTANCE = 100


# ============================================================
# PREVIOUS FACE POSITIONS
# ============================================================

previous_faces = {}


# ============================================================
# PREVIOUS LABELS
# ============================================================

previous_labels = {}


# ============================================================
# PREDICT ONE FACE
# ============================================================

def predict_face(face):

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    face = cv2.resize(
        face,
        (299, 299),
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # FLOAT
    # --------------------------------------------------------

    face = (
        face.astype(
            np.float32
        )
        / 255.0
    )


    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    face = np.transpose(
        face,
        (2, 0, 1)
    )


    # --------------------------------------------------------
    # ADD BATCH
    # --------------------------------------------------------

    face = np.expand_dims(
        face,
        axis=0
    )


    # --------------------------------------------------------
    # NUMPY -> PYTORCH
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        face
    ).to(device)


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    with torch.no_grad():

        result = model(
            tensor
        )


    # --------------------------------------------------------
    # GET OUTPUT
    # --------------------------------------------------------

    if isinstance(
        result,
        tuple
    ):

        output = result[0]

    else:

        output = result


    output = output.squeeze()


    # --------------------------------------------------------
    # SIGMOID
    # --------------------------------------------------------

    fake_probability = torch.sigmoid(
        output
    ).item()


    return fake_probability


# ============================================================
# FACE CENTER
# ============================================================

def get_face_center(
    x,
    y,
    w,
    h
):

    center_x = (
        x + w // 2
    )

    center_y = (
        y + h // 2
    )

    return (
        center_x,
        center_y
    )


# ============================================================
# FIND EXISTING FACE ID
# ============================================================

def find_face_id(
    center,
    used_ids
):

    best_id = None

    best_distance = MAX_FACE_DISTANCE


    for face_id, old_center in previous_faces.items():

        if face_id in used_ids:

            continue


        distance = np.sqrt(
            (
                center[0]
                -
                old_center[0]
            ) ** 2
            +
            (
                center[1]
                -
                old_center[1]
            ) ** 2
        )


        if distance < best_distance:

            best_distance = distance

            best_id = face_id


    return best_id


# ============================================================
# OPEN WEBCAM
# ============================================================

print()
print("Opening webcam...")


cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print()
    print("ERROR: Could not open webcam.")

    sys.exit(1)


# ============================================================
# CAMERA SETTINGS
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# ============================================================
# OBS VIRTUAL CAMERA
# ============================================================

print()
print("Connecting to OBS Virtual Camera...")


try:

    virtual_camera = pyvirtualcam.Camera(
        width=640,
        height=480,
        fps=30,
        fmt=pyvirtualcam.PixelFormat.BGR
    )

except Exception as error:

    cap.release()

    print()
    print("ERROR: Could not start virtual camera.")
    print()
    print(error)

    print()
    print("Make sure OBS Virtual Camera is NOT")
    print("already running manually in OBS.")

    sys.exit(1)


print()
print(
    "Virtual camera:",
    virtual_camera.device
)


print()
print("=" * 70)
print("             FAKE SHIELD IS LIVE")
print("=" * 70)

print()
print("Press Q to quit.")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print(
            "Failed to read webcam frame."
        )

        break


    height, width = (
        frame.shape[:2]
    )


    # --------------------------------------------------------
    # YUNET INPUT SIZE
    # --------------------------------------------------------

    detector.setInputSize(
        (width, height)
    )


    # --------------------------------------------------------
    # DETECT FACES
    # --------------------------------------------------------

    _, faces = detector.detect(
        frame
    )


    face_count = 0

    used_ids = set()

    current_faces = {}


    # ========================================================
    # PROCESS ALL FACES
    # ========================================================

    if faces is not None:

        for face_index, face in enumerate(
            faces
        ):

            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            x, y, w, h = (
                face[:4]
                .astype(int)
            )


            # ------------------------------------------------
            # KEEP INSIDE FRAME
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


            if w <= 10 or h <= 10:

                continue


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center = get_face_center(
                x,
                y,
                w,
                h
            )


            # ------------------------------------------------
            # FIND FACE ID
            # ------------------------------------------------

            face_id = find_face_id(
                center,
                used_ids
            )


            # ------------------------------------------------
            # NEW FACE
            # ------------------------------------------------

            if face_id is None:

                face_id = next_face_id

                next_face_id += 1


            used_ids.add(
                face_id
            )


            current_faces[
                face_id
            ] = center


            previous_faces[
                face_id
            ] = center


            # ------------------------------------------------
            # CREATE HISTORY
            # ------------------------------------------------

            if face_id not in face_histories:

                face_histories[
                    face_id
                ] = deque(
                    maxlen=HISTORY_LENGTH
                )


            # ------------------------------------------------
            # CROP FACE
            # ------------------------------------------------

            face_crop = frame[
                y:y + h,
                x:x + w
            ]


            if face_crop.size == 0:

                continue


            face_count += 1


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            fake_probability = predict_face(
                face_crop
            )


            # =================================================
            # ADD TO HISTORY
            # =================================================

            face_histories[
                face_id
            ].append(
                fake_probability
            )


            # =================================================
            # SMOOTHED PROBABILITY
            # =================================================

            history = face_histories[
                face_id
            ]


            smoothed_probability = float(
                np.mean(
                    list(history)
                )
            )


            real_probability = (
                1.0
                -
                smoothed_probability
            )


            # =================================================
            # STABLE CLASSIFICATION
            # =================================================

            previous_label = previous_labels.get(
                face_id,
                None
            )


            # -------------------------------------------------
            # If clearly REAL
            # -------------------------------------------------

            if smoothed_probability <= LOW_THRESHOLD:

                label = "REAL"


            # -------------------------------------------------
            # If clearly FAKE
            # -------------------------------------------------

            elif smoothed_probability >= HIGH_THRESHOLD:

                label = "FAKE"


            # -------------------------------------------------
            # Borderline zone
            # -------------------------------------------------

            else:

                # Keep previous result instead of flipping
                # every frame.

                if previous_label is not None:

                    label = previous_label

                else:

                    if smoothed_probability >= DEFAULT_THRESHOLD:

                        label = "FAKE"

                    else:

                        label = "REAL"


            previous_labels[
                face_id
            ] = label


            # =================================================
            # CONFIDENCE
            # =================================================

            if label == "FAKE":

                confidence = (
                    smoothed_probability
                    * 100
                )

                box_color = (
                    0,
                    0,
                    255
                )

            else:

                confidence = (
                    real_probability
                    * 100
                )

                box_color = (
                    0,
                    255,
                    0
                )


            # =================================================
            # DRAW FACE BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                3
            )


            # =================================================
            # FACE ID
            # =================================================

            cv2.putText(
                frame,
                f"Face {face_id}",
                (
                    x,
                    max(
                        y - 42,
                        25
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2,
                cv2.LINE_AA
            )


            # =================================================
            # MAIN LABEL
            # =================================================

            cv2.putText(
                frame,
                f"{label} {confidence:.1f}%",
                (
                    x,
                    max(
                        y - 12,
                        50
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                box_color,
                2,
                cv2.LINE_AA
            )


            # =================================================
            # PROBABILITY
            # =================================================

            probability_text = (
                f"REAL: "
                f"{real_probability * 100:.1f}% "
                f"FAKE: "
                f"{smoothed_probability * 100:.1f}%"
            )


            text_y = (
                y + h + 25
            )


            if text_y >= height:

                text_y = (
                    height - 10
                )


            cv2.putText(
                frame,
                probability_text,
                (
                    x,
                    text_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                box_color,
                2,
                cv2.LINE_AA
            )


    # ========================================================
    # REMOVE OLD FACE HISTORIES
    # ========================================================

    active_ids = set(
        current_faces.keys()
    )


    old_ids = list(
        face_histories.keys()
    )


    for face_id in old_ids:

        if face_id not in active_ids:

            del face_histories[
                face_id
            ]


            if face_id in previous_labels:

                del previous_labels[
                    face_id
                ]


            if face_id in previous_faces:

                del previous_faces[
                    face_id
                ]


    # ========================================================
    # FACE COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"Faces detected: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # WATERMARK
    # ========================================================

    cv2.putText(
        frame,
        "FAKE SHIELD - LIVE DETECTION",
        (
            20,
            height - 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # SEND TO OBS
    # ========================================================

    virtual_camera.send(
        frame
    )


    # ========================================================
    # MAINTAIN 30 FPS
    # ========================================================

    virtual_camera.sleep_until_next_frame()


    # ========================================================
    # LOCAL PREVIEW
    # ========================================================

    cv2.imshow(
        "FakeShield - Stable Virtual Camera",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

print()
print("Stopping FakeShield...")


cap.release()

virtual_camera.close()

cv2.destroyAllWindows()


print("FakeShield stopped.")