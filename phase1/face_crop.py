import cv2
import numpy as np


# ============================================================
# YUNET MODEL
# ============================================================

model_path = r".\phase1\face_detection_yunet_2026may.onnx"

detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (320, 320),

    # Face detection confidence
    0.9,

    # NMS threshold
    0.3,

    # Maximum number of faces
    5000
)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("Failed to access webcam")
    exit()


# ============================================================
# FACE TRACKING / SMOOTHING
# ============================================================

# Stores the previous bounding box for each face.
previous_boxes = {}


# How much the new detection affects the box.
# Smaller value = more stable.
BOX_ALPHA = 0.20


# Extra space around the detected face.
# This prevents the crop from becoming too tight
# when the mouth opens or expression changes.
PADDING = 0.15


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print("Failed to read webcam frame")
        break


    height, width = frame.shape[:2]


    # ========================================================
    # YUNET INPUT SIZE
    # ========================================================

    detector.setInputSize(
        (width, height)
    )


    # ========================================================
    # DETECT FACES
    # ========================================================

    _, faces = detector.detect(
        frame
    )


    # List of all face crops
    face_crops = []


    # Keep track of currently detected face IDs
    current_face_ids = set()


    # ========================================================
    # PROCESS DETECTED FACES
    # ========================================================

    if faces is not None:

        for i, face in enumerate(faces):


            # ------------------------------------------------
            # ORIGINAL YUNET BOX
            # ------------------------------------------------

            raw_x = float(face[0])
            raw_y = float(face[1])
            raw_w = float(face[2])
            raw_h = float(face[3])


            # ------------------------------------------------
            # FACE CENTER
            # ------------------------------------------------

            center_x = (
                raw_x + raw_w / 2
            )

            center_y = (
                raw_y + raw_h / 2
            )


            # ------------------------------------------------
            # CREATE A STABLE FACE ID
            #
            # For this detector test we use the approximate
            # position of the face.
            # ------------------------------------------------

            face_id = (

                round(center_x / 100),

                round(center_y / 100)

            )


            current_face_ids.add(
                face_id
            )


            # =================================================
            # SMOOTH BOUNDING BOX
            # =================================================

            if face_id in previous_boxes:

                old_x, old_y, old_w, old_h = (
                    previous_boxes[face_id]
                )


                smooth_x = (
                    BOX_ALPHA * raw_x
                    +
                    (1 - BOX_ALPHA) * old_x
                )


                smooth_y = (
                    BOX_ALPHA * raw_y
                    +
                    (1 - BOX_ALPHA) * old_y
                )


                smooth_w = (
                    BOX_ALPHA * raw_w
                    +
                    (1 - BOX_ALPHA) * old_w
                )


                smooth_h = (
                    BOX_ALPHA * raw_h
                    +
                    (1 - BOX_ALPHA) * old_h
                )


            else:

                smooth_x = raw_x
                smooth_y = raw_y
                smooth_w = raw_w
                smooth_h = raw_h


            # Save smoothed box
            previous_boxes[face_id] = (

                smooth_x,
                smooth_y,
                smooth_w,
                smooth_h

            )


            # =================================================
            # CONVERT TO INTEGER
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
            # EXPANDED FACE CROP
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
            # EXTRACT STABLE FACE CROP
            # =================================================

            face_crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]


            if face_crop.size > 0:

                face_crops.append(
                    face_crop
                )


            # =================================================
            # DRAW SMOOTHED FACE BOX
            # =================================================

            cv2.rectangle(

                frame,

                (x, y),

                (x + w, y + h),

                (0, 255, 0),

                2

            )


            # =================================================
            # DRAW FACE NUMBER
            # =================================================

            cv2.putText(

                frame,

                f"Face {i + 1}",

                (
                    x,
                    max(25, y - 8)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2

            )


            # =================================================
            # DRAW CROP AREA
            # =================================================

            cv2.rectangle(

                frame,

                (crop_x1, crop_y1),

                (crop_x2, crop_y2),

                (255, 255, 0),

                1

            )


    # ========================================================
    # REMOVE OLD FACE HISTORY
    # ========================================================

    old_ids = list(
        previous_boxes.keys()
    )


    for old_id in old_ids:

        if old_id not in current_face_ids:

            del previous_boxes[
                old_id
            ]


    # ========================================================
    # DISPLAY ALL FACE CROPS
    # ========================================================

    if len(face_crops) > 0:

        resized_faces = []


        for i, face_crop in enumerate(
            face_crops
        ):


            # -----------------------------------------------
            # RESIZE
            # -----------------------------------------------

            resized_face = cv2.resize(

                face_crop,

                (250, 250)

            )


            # -----------------------------------------------
            # FACE NUMBER
            # -----------------------------------------------

            cv2.putText(

                resized_face,

                f"Face {i + 1}",

                (10, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2

            )


            resized_faces.append(
                resized_face
            )


        # ====================================================
        # COMBINE FACES
        # ====================================================

        face_display = np.hstack(
            resized_faces
        )


        cv2.imshow(

            "Stable Face Crops",

            face_display

        )


    else:

        # ====================================================
        # NO FACE
        # ====================================================

        blank = np.zeros(

            (250, 250, 3),

            dtype=np.uint8

        )


        cv2.putText(

            blank,

            "No Face",

            (70, 130),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 255),

            2

        )


        cv2.imshow(

            "Stable Face Crops",

            blank

        )


    # ========================================================
    # FACE COUNT
    # ========================================================

    face_count = (

        len(faces)

        if faces is not None

        else 0

    )


    cv2.putText(

        frame,

        f"Faces: {face_count}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 255, 0),

        2

    )


    # ========================================================
    # PROJECT TITLE
    # ========================================================

    cv2.putText(

        frame,

        "FAKE SHIELD - PHASE 3A",

        (20, height - 25),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2

    )


    # ========================================================
    # SHOW ORIGINAL FRAME
    # ========================================================

    cv2.imshow(

        "FakeShield - Phase 3A",

        frame

    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    # ========================================================
    # CLOSE WINDOW
    # ========================================================

    if cv2.getWindowProperty(

        "FakeShield - Phase 3A",

        cv2.WND_PROP_VISIBLE

    ) < 1:

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("=" * 60)
print("PHASE 3A STOPPED")
print("=" * 60)