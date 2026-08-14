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
# 8. LOAD FACE DETECTOR
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
# 10. TRACKING SETTINGS
# ==========================================

HISTORY_SIZE = 10

IOU_THRESHOLD = 0.30

MAX_MISSED_FRAMES = 15


# ==========================================
# 11. TRACK STORAGE
# ==========================================

tracks = {}

next_track_id = 1


# ==========================================
# 12. IOU FUNCTION
# ==========================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b


    # Intersection

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    intersection_width = max(
        0,
        ix2 - ix1
    )

    intersection_height = max(
        0,
        iy2 - iy1
    )


    intersection_area = (
        intersection_width *
        intersection_height
    )


    # Area of A

    area_a = max(
        0,
        ax2 - ax1
    ) * max(
        0,
        ay2 - ay1
    )


    # Area of B

    area_b = max(
        0,
        bx2 - bx1
    ) * max(
        0,
        by2 - by1
    )


    union_area = (
        area_a +
        area_b -
        intersection_area
    )


    if union_area <= 0:

        return 0.0


    return (
        intersection_area /
        union_area
    )


# ==========================================
# 13. MAIN LOOP
# ==========================================

while True:

    # --------------------------------------
    # Read webcam frame
    # --------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("Failed to read webcam.")
        break


    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )


    # ======================================
    # DETECT FACES
    # ======================================

    _, faces = detector.detect(frame)


    # ======================================
    # NO FACES
    # ======================================

    if faces is None:

        for track_id in tracks:

            tracks[
                track_id
            ]["missed"] += 1


        # Remove old tracks

        remove_ids = []

        for track_id in tracks:

            if tracks[
                track_id
            ]["missed"] > MAX_MISSED_FRAMES:

                remove_ids.append(
                    track_id
                )


        for track_id in remove_ids:

            del tracks[
                track_id
            ]


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
            "FakeShield - Phase 3E",
            frame
        )


        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        continue


    # ======================================
    # CONVERT DETECTIONS TO BOXES
    # ======================================

    detected_boxes = []

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

        detected_boxes.append(
            (x1, y1, x2, y2)
        )


    # ======================================
    # MATCH DETECTIONS TO EXISTING TRACKS
    # ======================================

    matched_tracks = set()

    matched_detections = set()

    matches = []


    for detection_index, detection_box in enumerate(
        detected_boxes
    ):

        best_track_id = None

        best_iou = 0.0


        for track_id, track in tracks.items():

            if track_id in matched_tracks:
                continue


            old_box = track["box"]


            iou = calculate_iou(
                detection_box,
                old_box
            )


            if iou > best_iou:

                best_iou = iou

                best_track_id = track_id


        if (
            best_track_id is not None
            and best_iou >= IOU_THRESHOLD
        ):

            matches.append(
                (
                    detection_index,
                    best_track_id
                )
            )

            matched_detections.add(
                detection_index
            )

            matched_tracks.add(
                best_track_id
            )


    # ======================================
    # UPDATE MATCHED TRACKS
    # ======================================

    for detection_index, track_id in matches:

        box = detected_boxes[
            detection_index
        ]


        tracks[
            track_id
        ]["box"] = box


        tracks[
            track_id
        ]["missed"] = 0


    # ======================================
    # CREATE NEW TRACKS
    # ======================================

    for detection_index, box in enumerate(
        detected_boxes
    ):

        if detection_index in matched_detections:
            continue


        track_id = next_track_id

        next_track_id += 1


        tracks[track_id] = {

            "box": box,

            "missed": 0,

            "history": deque(
                maxlen=HISTORY_SIZE
            )
        }


    # ======================================
    # UPDATE MISSED TRACKS
    # ======================================

    for track_id in list(
        tracks.keys()
    ):

        if track_id not in matched_tracks:

            # New tracks are not considered
            # missed immediately.

            if tracks[
                track_id
            ]["missed"] > 0:

                tracks[
                    track_id
                ]["missed"] += 1


    # ======================================
    # REMOVE DEAD TRACKS
    # ======================================

    remove_ids = []

    for track_id in tracks:

        if tracks[
            track_id
        ]["missed"] > MAX_MISSED_FRAMES:

            remove_ids.append(
                track_id
            )


    for track_id in remove_ids:

        del tracks[
            track_id
        ]


    # ======================================
    # DISPLAY FACE COUNT
    # ======================================

    cv2.putText(
        frame,
        f"Faces: {len(detected_boxes)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ======================================
    # ANALYZE EACH DETECTION
    # ======================================

    for detection_index, box in enumerate(
        detected_boxes
    ):

        x1, y1, x2, y2 = box


        # ----------------------------------
        # Find corresponding track
        # ----------------------------------

        track_id = None


        for current_detection, current_track in matches:

            if current_detection == detection_index:

                track_id = current_track

                break


        # ----------------------------------
        # New detection
        # ----------------------------------

        if track_id is None:

            for possible_id, track in tracks.items():

                if (
                    track["box"] == box
                    and possible_id not in matched_tracks
                ):

                    track_id = possible_id

                    break


        if track_id is None:

            continue


        # ==================================
        # CROP FACE
        # ==================================

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
        ).to(device)


        # ==================================
        # MODEL
        # ==================================

        with torch.no_grad():

            output, features = model(
                tensor
            )


        # ==================================
        # PROBABILITY
        # ==================================

        probabilities = torch.softmax(
            output,
            dim=1
        )


        real_probability = (
            probabilities[0][0].item()
        )

        fake_probability = (
            probabilities[0][1].item()
        )


        # ==================================
        # ADD TO TRACK HISTORY
        # ==================================

        tracks[
            track_id
        ]["history"].append(
            fake_probability
        )


        history = tracks[
            track_id
        ]["history"]


        # ==================================
        # STABLE AVERAGE
        # ==================================

        average_fake = (
            sum(history) /
            len(history)
        )

        average_real = (
            1.0 -
            average_fake
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

        label1 = (
            f"Person {track_id}: {result}"
        )


        label2 = (
            f"Real {real_percent:.1f}% | "
            f"Fake {fake_percent:.1f}%"
        )


        label3 = (
            f"Track {track_id} | "
            f"Samples {len(history)}/{HISTORY_SIZE}"
        )


        # ==================================
        # DISPLAY LABEL 1
        # ==================================

        cv2.putText(
            frame,
            label1,
            (
                x1,
                max(
                    25,
                    y1 - 55
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
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
                max(
                    50,
                    y1 - 30
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
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
                max(
                    75,
                    y1 - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 0),
            1
        )


    # ======================================
    # SHOW FRAME
    # ======================================

    cv2.imshow(
        "FakeShield - Phase 3E",
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

print("Phase 3E stopped.")