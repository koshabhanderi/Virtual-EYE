import os
import sys
import cv2
import torch
import numpy as np


# ============================================================
# FAKE SHIELD - PHASE 3K
# STABLE REAL-TIME DEEPFAKE DETECTION
# MULTI-FACE TRACKING
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
print("             FAKE SHIELD - PHASE 3K")
print("          STABLE REAL-TIME DETECTION")
print("=" * 70)

print("Device:", device)


# ============================================================
# XCEPTION CONFIGURATION
# ============================================================

config = {
    "mode": "original",
    "num_classes": 1,
    "inc": 3,
    "dropout": False
}


# ============================================================
# CREATE XCEPTION MODEL
# ============================================================

print()
print("Creating Xception model...")

model = Xception(config)


# ============================================================
# LOAD AI-FACE CHECKPOINT
# ============================================================

checkpoint_path = (
    r".\ai_face_weights\xception_ai_face.pth"
)

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
# LOAD YUNET
# ============================================================

yunet_path = (
    r".\phase1\face_detection_yunet_2026may.onnx"
)

print()
print("Loading YuNet face detector...")
print("YuNet:", yunet_path)


face_detector = cv2.FaceDetectorYN.create(
    yunet_path,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

print("YuNet ready.")


# ============================================================
# LIVE DETECTION SETTINGS
# ============================================================

# ------------------------------------------------------------
# Face-box smoothing
# ------------------------------------------------------------

BOX_ALPHA = 0.20


# ------------------------------------------------------------
# Extra padding around face
# ------------------------------------------------------------

PADDING = 0.05


# ------------------------------------------------------------
# Prediction smoothing
# ------------------------------------------------------------

EMA_ALPHA = 0.10


# ------------------------------------------------------------
# Classification thresholds
#
# Fake probability:
#
# <= 35%  -> REAL
# 35-65%  -> UNCERTAIN
# >= 65%  -> FAKE
# ------------------------------------------------------------

REAL_THRESHOLD = 0.35

FAKE_THRESHOLD = 0.65


# ------------------------------------------------------------
# Minimum frames before classification
# ------------------------------------------------------------

MIN_FRAMES = 5


# ------------------------------------------------------------
# Maximum distance for face tracking
#
# Increase to 150 if faces move very quickly.
# ------------------------------------------------------------

MAX_FACE_DISTANCE = 120


# ------------------------------------------------------------
# Number of frames a face can disappear before
# its tracking information is deleted.
# ------------------------------------------------------------

MAX_MISSED_FRAMES = 10


# ============================================================
# FACE HISTORY
# ============================================================

face_history = {}


# ============================================================
# FACE TRACKING
# ============================================================

tracked_faces = {}

next_face_id = 0


# ============================================================
# GET FACE CENTER
# ============================================================

def get_center(box):

    x, y, w, h = box

    center_x = x + (w / 2)

    center_y = y + (h / 2)

    return (
        center_x,
        center_y
    )


# ============================================================
# CALCULATE DISTANCE BETWEEN TWO CENTERS
# ============================================================

def center_distance(center1, center2):

    dx = center1[0] - center2[0]

    dy = center1[1] - center2[1]

    return np.sqrt(
        dx * dx + dy * dy
    )


# ============================================================
# MATCH CURRENT DETECTIONS TO OLD TRACKS
# ============================================================

def match_faces(detections, tracked_faces):

    matches = {}

    used_track_ids = set()

    # --------------------------------------------------------
    # Calculate every possible detection -> track distance
    # --------------------------------------------------------

    possible_matches = []

    for detection_index, detection_box in enumerate(
        detections
    ):

        detection_center = get_center(
            detection_box
        )

        for face_id, track_data in tracked_faces.items():

            old_center = track_data["center"]

            distance = center_distance(
                detection_center,
                old_center
            )

            if distance <= MAX_FACE_DISTANCE:

                possible_matches.append(
                    (
                        distance,
                        detection_index,
                        face_id
                    )
                )

    # --------------------------------------------------------
    # Closest matches first
    # --------------------------------------------------------

    possible_matches.sort(
        key=lambda item: item[0]
    )

    # --------------------------------------------------------
    # Assign each detection to only ONE track
    # --------------------------------------------------------

    for (
        distance,
        detection_index,
        face_id
    ) in possible_matches:

        if detection_index in matches:

            continue

        if face_id in used_track_ids:

            continue

        matches[detection_index] = face_id

        used_track_ids.add(
            face_id
        )

    return matches


# ============================================================
# FACE PREDICTION FUNCTION
# ============================================================

def predict_face(face_crop):

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    image = cv2.resize(
        face_crop,
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
    # FLOAT
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

    if isinstance(result, tuple):

        output = result[0]

    else:

        output = result


    # --------------------------------------------------------
    # SINGLE VALUE
    # --------------------------------------------------------

    output = output.squeeze()


    # --------------------------------------------------------
    # SIGMOID
    # --------------------------------------------------------

    fake_probability = torch.sigmoid(
        output
    ).item()


    return fake_probability


# ============================================================
# START WEBCAM
# ============================================================

print()
print("=" * 70)
print("Starting webcam...")
print("Press Q to quit.")
print("=" * 70)


cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Cannot open webcam.")

    sys.exit(1)


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print("Failed to read webcam.")

        break


    height, width = frame.shape[:2]


    # --------------------------------------------------------
    # SET YUNET INPUT SIZE
    # --------------------------------------------------------

    face_detector.setInputSize(
        (width, height)
    )


    # --------------------------------------------------------
    # DETECT FACES
    # --------------------------------------------------------

    _, faces = face_detector.detect(
        frame
    )


    # --------------------------------------------------------
    # CURRENT FACE IDs
    # --------------------------------------------------------

    current_face_ids = set()


    # ========================================================
    # NO FACES
    # ========================================================

    if faces is None:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    # ========================================================
    # FACES DETECTED
    # ========================================================

    else:

        # ----------------------------------------------------
        # Create list of raw detection boxes
        # ----------------------------------------------------

        detection_boxes = []


        for face in faces:

            raw_x = float(face[0])

            raw_y = float(face[1])

            raw_w = float(face[2])

            raw_h = float(face[3])


            detection_boxes.append(
                (
                    raw_x,
                    raw_y,
                    raw_w,
                    raw_h
                )
            )


        # ----------------------------------------------------
        # Match detections to existing tracked faces
        # ----------------------------------------------------

        matches = match_faces(
            detection_boxes,
            tracked_faces
        )


        # ----------------------------------------------------
        # Process every detected face
        # ----------------------------------------------------

        for detection_index, raw_box in enumerate(
            detection_boxes
        ):

            raw_x = raw_box[0]

            raw_y = raw_box[1]

            raw_w = raw_box[2]

            raw_h = raw_box[3]


            # =================================================
            # GET EXISTING FACE ID
            # =================================================

            if detection_index in matches:

                face_id = matches[
                    detection_index
                ]

            else:

                face_id = (
                    f"face_{next_face_id}"
                )

                next_face_id += 1


            # =================================================
            # UPDATE TRACK
            # =================================================

            tracked_faces[face_id] = {

                "center": get_center(
                    raw_box
                ),

                "box": raw_box,

                "missed": 0
            }


            current_face_ids.add(
                face_id
            )


            # =================================================
            # GET PREVIOUS BOX
            # =================================================

            if face_id in face_history:

                old_box = face_history[
                    face_id
                ]["box"]

                old_x = old_box[0]

                old_y = old_box[1]

                old_w = old_box[2]

                old_h = old_box[3]


                # ------------------------------------------------
                # SMOOTH X
                # ------------------------------------------------

                smooth_x = (

                    BOX_ALPHA * raw_x

                    +

                    (1 - BOX_ALPHA)
                    * old_x
                )


                # ------------------------------------------------
                # SMOOTH Y
                # ------------------------------------------------

                smooth_y = (

                    BOX_ALPHA * raw_y

                    +

                    (1 - BOX_ALPHA)
                    * old_y
                )


                # ------------------------------------------------
                # SMOOTH WIDTH
                # ------------------------------------------------

                smooth_w = (

                    BOX_ALPHA * raw_w

                    +

                    (1 - BOX_ALPHA)
                    * old_w
                )


                # ------------------------------------------------
                # SMOOTH HEIGHT
                # ------------------------------------------------

                smooth_h = (

                    BOX_ALPHA * raw_h

                    +

                    (1 - BOX_ALPHA)
                    * old_h
                )


            else:

                smooth_x = raw_x

                smooth_y = raw_y

                smooth_w = raw_w

                smooth_h = raw_h


            # =================================================
            # CONVERT BOX TO INTEGER
            # =================================================

            x = int(smooth_x)

            y = int(smooth_y)

            w = int(smooth_w)

            h = int(smooth_h)


            # =================================================
            # KEEP BOX INSIDE FRAME
            # =================================================

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


            if w <= 0 or h <= 0:

                continue


            # =================================================
            # EXPAND FACE CROP
            # =================================================

            padding_x = int(
                w * PADDING
            )

            padding_y = int(
                h * PADDING
            )


            crop_x1 = max(
                0,
                x - padding_x
            )

            crop_y1 = max(
                0,
                y - padding_y
            )

            crop_x2 = min(
                width,
                x + w + padding_x
            )

            crop_y2 = min(
                height,
                y + h + padding_y
            )


            # =================================================
            # EXTRACT FACE
            # =================================================

            face_crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]


            if face_crop.size == 0:

                continue


            # =================================================
            # XCEPTION PREDICTION
            # =================================================

            fake_probability = predict_face(
                face_crop
            )


            # =================================================
            # CREATE FACE HISTORY
            # =================================================

            if face_id not in face_history:

                face_history[face_id] = {

                    "box": (
                        smooth_x,
                        smooth_y,
                        smooth_w,
                        smooth_h
                    ),

                    "probability":
                        fake_probability,

                    "frames": 1
                }


            # =================================================
            # UPDATE FACE HISTORY
            # =================================================

            else:

                old_probability = (
                    face_history[face_id]
                    ["probability"]
                )


                # ------------------------------------------------
                # EMA SMOOTHING
                # ------------------------------------------------

                smoothed_probability = (

                    EMA_ALPHA
                    * fake_probability

                    +

                    (1 - EMA_ALPHA)
                    * old_probability
                )


                face_history[face_id][
                    "probability"
                ] = smoothed_probability


                face_history[face_id][
                    "box"
                ] = (

                    smooth_x,
                    smooth_y,
                    smooth_w,
                    smooth_h
                )


                face_history[face_id][
                    "frames"
                ] += 1


            # =================================================
            # GET SMOOTHED PROBABILITY
            # =================================================

            smoothed_fake = (
                face_history[face_id]
                ["probability"]
            )


            smoothed_real = (
                1.0 - smoothed_fake
            )


            frames_seen = (
                face_history[face_id]
                ["frames"]
            )


            # =================================================
            # CLASSIFICATION
            # =================================================

            if frames_seen < MIN_FRAMES:

                label = "ANALYZING"

                confidence = 50.0

                box_color = (
                    0,
                    255,
                    255
                )


            elif smoothed_fake >= FAKE_THRESHOLD:

                label = "FAKE"

                confidence = (
                    smoothed_fake * 100
                )

                box_color = (
                    0,
                    0,
                    255
                )


            elif smoothed_fake <= REAL_THRESHOLD:

                label = "REAL"

                confidence = (
                    smoothed_real * 100
                )

                box_color = (
                    0,
                    255,
                    0
                )


            else:

                label = "UNCERTAIN"

                confidence = 50.0

                box_color = (
                    0,
                    255,
                    255
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
            # DISPLAY FACE ID
            # =================================================

            cv2.putText(
                frame,
                face_id,
                (
                    x,
                    max(
                        20,
                        y - 35
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )


            # =================================================
            # DISPLAY RESULT
            # =================================================

            cv2.putText(
                frame,
                f"{label} {confidence:.1f}%",
                (
                    x,
                    max(
                        45,
                        y - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )


            # =================================================
            # DISPLAY PROBABILITIES
            # =================================================

            probability_text = (
                f"REAL: "
                f"{smoothed_real * 100:.1f}%  "
                f"FAKE: "
                f"{smoothed_fake * 100:.1f}%"
            )


            probability_y = min(
                height - 20,
                y + h + 25
            )


            cv2.putText(
                frame,
                probability_text,
                (
                    x,
                    probability_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2
            )


    # ========================================================
    # UPDATE MISSED TRACKS
    # ========================================================

    tracked_face_ids = list(
        tracked_faces.keys()
    )


    for face_id in tracked_face_ids:

        if face_id not in current_face_ids:

            tracked_faces[
                face_id
            ]["missed"] += 1


            if (
                tracked_faces[
                    face_id
                ]["missed"]
                > MAX_MISSED_FRAMES
            ):

                tracked_faces.pop(
                    face_id,
                    None
                )

                face_history.pop(
                    face_id,
                    None
                )


    # ========================================================
    # DISPLAY FACE COUNT
    # ========================================================

    face_count = (
        len(faces)
        if faces is not None
        else 0
    )


    cv2.putText(
        frame,
        f"Faces detected: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ========================================================
    # PROJECT TITLE
    # ========================================================

    cv2.putText(
        frame,
        "FAKE SHIELD - PHASE 3K",
        (
            20,
            height - 45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "Q = Quit",
        (
            20,
            height - 15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # ========================================================
    # SHOW WEBCAM
    # ========================================================

    cv2.imshow(
        "FakeShield - Phase 3K",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print("=" * 70)
print("PHASE 3K STOPPED")
print("=" * 70)