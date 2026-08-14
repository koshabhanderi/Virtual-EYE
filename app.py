import os
import sys
import base64
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

XCEPTION_ONNX_PATH = (
    BASE_DIR
    / "ai_face_weights"
    / "xception_ai_face.onnx"
)

YUNET_PATH = (
    BASE_DIR
    / "phase1"
    / "face_detection_yunet_2026may.onnx"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

# Maximum request size: 8 MB
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# ============================================================
# XCEPTION ONNX MODEL
# ============================================================

session = None
model_error = None

try:

    if not XCEPTION_ONNX_PATH.exists():

        raise FileNotFoundError(
            f"Xception ONNX model not found: "
            f"{XCEPTION_ONNX_PATH}"
        )

    print("Loading FakeShield Xception ONNX model...")

    # Use CPU only.
    session = ort.InferenceSession(
        str(XCEPTION_ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    print(
        "FakeShield Xception ONNX model "
        "loaded successfully."
    )

    print(
        "ONNX providers:",
        session.get_providers()
    )

    print(
        "ONNX inputs:",
        [
            inp.name
            for inp in session.get_inputs()
        ]
    )

    print(
        "ONNX outputs:",
        [
            out.name
            for out in session.get_outputs()
        ]
    )


except Exception as exc:

    session = None
    model_error = str(exc)

    print(
        "MODEL LOAD ERROR:",
        exc
    )


# ============================================================
# YuNet FACE DETECTOR
# ============================================================

face_detector = None
yunet_error = None

try:

    if not YUNET_PATH.exists():

        raise FileNotFoundError(
            f"YuNet model not found: "
            f"{YUNET_PATH}"
        )

    print("Loading YuNet...")

    face_detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH),
        "",
        (320, 320),
        0.8,
        0.3,
        5000
    )

    print(
        "YuNet loaded successfully."
    )


except Exception as exc:

    face_detector = None
    yunet_error = str(exc)

    print(
        "YUNET LOAD ERROR:",
        exc
    )


# ============================================================
# PREDICT FACE USING ONNX
# ============================================================

def predict_face(face_bgr):

    if session is None:

        raise RuntimeError(
            model_error
            or "FakeShield ONNX model "
               "is not loaded."
        )

    # --------------------------------------------------------
    # Resize face
    # --------------------------------------------------------

    face = cv2.resize(
        face_bgr,
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
    # Convert to float
    # --------------------------------------------------------

    face = face.astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    face = np.transpose(
        face,
        (2, 0, 1)
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    face = np.expand_dims(
        face,
        axis=0
    )

    # Make sure array is contiguous.
    face = np.ascontiguousarray(
        face,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Get input name
    # --------------------------------------------------------

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    # --------------------------------------------------------
    # ONNX inference
    # --------------------------------------------------------

    outputs = session.run(
        None,
        {
            input_name: face
        }
    )

    if not outputs:

        raise RuntimeError(
            "ONNX model returned no output."
        )

    output = outputs[0]

    # --------------------------------------------------------
    # Convert output to scalar
    # --------------------------------------------------------

    output = np.asarray(
        output
    )

    output = output.squeeze()

    if output.size == 0:

        raise RuntimeError(
            "ONNX model returned an empty output."
        )

    # Take first value if necessary.
    raw_value = float(
        output.reshape(-1)[0]
    )

    # --------------------------------------------------------
    # Apply sigmoid
    #
    # The original PyTorch code used:
    #
    # torch.sigmoid(output)
    #
    # so we reproduce that here.
    # --------------------------------------------------------

    raw_value = np.clip(
        raw_value,
        -50.0,
        50.0
    )

    fake_probability = (
        1.0 /
        (
            1.0 +
            np.exp(-raw_value)
        )
    )

    return float(
        fake_probability
    )


# ============================================================
# DECODE BROWSER IMAGE
# ============================================================

def decode_image(data_url):

    if not data_url:

        raise ValueError(
            "No image was supplied."
        )

    # --------------------------------------------------------
    # Handle data URL
    # --------------------------------------------------------

    if "," in data_url:

        _, encoded = data_url.split(
            ",",
            1
        )

    else:

        encoded = data_url

    try:

        raw = base64.b64decode(
            encoded
        )

    except Exception as exc:

        raise ValueError(
            "Invalid base64 image data."
        ) from exc

    # --------------------------------------------------------
    # Convert bytes -> NumPy array
    # --------------------------------------------------------

    array = np.frombuffer(
        raw,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Decode JPEG/PNG
    # --------------------------------------------------------

    frame = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise ValueError(
            "Could not decode the image."
        )

    return frame


# ============================================================
# DETECT LARGEST FACE
# ============================================================

def detect_largest_face(frame):

    if face_detector is None:

        raise RuntimeError(
            yunet_error
            or "YuNet is not loaded."
        )

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Tell YuNet the current image size
    # --------------------------------------------------------

    face_detector.setInputSize(
        (
            int(width),
            int(height)
        )
    )

    _, faces = face_detector.detect(
        frame
    )

    # --------------------------------------------------------
    # No faces
    # --------------------------------------------------------

    if faces is None or len(faces) == 0:

        return None, 0

    largest = None
    largest_area = 0

    # --------------------------------------------------------
    # Find largest face
    # --------------------------------------------------------

    for face in faces:

        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])

        # Keep coordinates inside image.

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
            int(width - x)
        )

        h = min(
            h,
            int(height - y)
        )

        if w <= 0 or h <= 0:

            continue

        area = int(
            w * h
        )

        if area > largest_area:

            largest_area = area

            largest = (
                int(x),
                int(y),
                int(w),
                int(h)
            )

    return (
        largest,
        int(len(faces))
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(fake_probability):

    fake_probability = float(
        fake_probability
    )

    # Keep probability between 0 and 1.

    fake_probability = max(
        0.0,
        min(
            1.0,
            fake_probability
        )
    )

    real_probability = (
        1.0 -
        fake_probability
    )

    # --------------------------------------------------------
    # Decision thresholds
    # --------------------------------------------------------

    if fake_probability >= 0.80:

        label = "FAKE"

    elif fake_probability <= 0.55:

        label = "REAL"

    else:

        label = "UNCERTAIN"

    return {

        "label": str(
            label
        ),

        "fake_probability": float(
            round(
                fake_probability * 100,
                1
            )
        ),

        "real_probability": float(
            round(
                real_probability * 100,
                1
            )
        )
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return jsonify({

        "status": "ok",

        "model_loaded": bool(
            session is not None
        ),

        "yunet_loaded": bool(
            face_detector is not None
        ),

        "model_error": (
            str(model_error)
            if model_error
            else None
        ),

        "yunet_error": (
            str(yunet_error)
            if yunet_error
            else None
        )
    })


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict():

    try:

        # ----------------------------------------------------
        # Read JSON safely
        # ----------------------------------------------------

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        image_data = data.get(
            "image"
        )

        if not image_data:

            return jsonify({

                "ok": False,

                "error":
                    "No image supplied."

            }), 400

        # ----------------------------------------------------
        # Decode image
        # ----------------------------------------------------

        frame = decode_image(
            image_data
        )

        # ----------------------------------------------------
        # Detect face
        # ----------------------------------------------------

        face_box, face_count = (
            detect_largest_face(
                frame
            )
        )

        # ----------------------------------------------------
        # No face
        # ----------------------------------------------------

        if face_box is None:

            return jsonify({

                "ok": True,

                "face_detected": False,

                "face_count":
                    int(face_count),

                "message":
                    "No face detected."

            }), 200

        # ----------------------------------------------------
        # Get largest face
        # ----------------------------------------------------

        x, y, w, h = face_box

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        face_count = int(
            face_count
        )

        # ----------------------------------------------------
        # Ignore very small faces
        # ----------------------------------------------------

        if w < 70 or h < 70:

            return jsonify({

                "ok": True,

                "face_detected": False,

                "face_count":
                    int(face_count),

                "message":
                    "Face is too small "
                    "for reliable analysis."

            }), 200

        # ----------------------------------------------------
        # Crop face
        # ----------------------------------------------------

        face_crop = frame[
            y:y + h,
            x:x + w
        ]

        if face_crop.size == 0:

            raise ValueError(
                "Invalid face crop."
            )

        # ----------------------------------------------------
        # AI prediction
        # ----------------------------------------------------

        fake_probability = (
            predict_face(
                face_crop
            )
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        result = classify(
            fake_probability
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        response = {

            "ok": True,

            "face_detected": True,

            "face_count":
                int(face_count),

            "box": {

                "x": int(x),

                "y": int(y),

                "w": int(w),

                "h": int(h)
            },

            "label":
                str(
                    result["label"]
                ),

            "fake_probability":
                float(
                    result[
                        "fake_probability"
                    ]
                ),

            "real_probability":
                float(
                    result[
                        "real_probability"
                    ]
                )
        }

        print(
            "Prediction:",
            response["label"],
            "| Fake:",
            response["fake_probability"],
            "%",
            "| Real:",
            response["real_probability"],
            "%"
        )

        return jsonify(
            response
        ), 200

    except Exception as exc:

        print(
            "PREDICTION ERROR:",
            repr(exc)
        )

        return jsonify({

            "ok": False,

            "error":
                str(exc)

        }), 500


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def too_large(_error):

    return jsonify({

        "ok": False,

        "error":
            "Image is too large. "
            "Please use a smaller image."

    }), 413


# ============================================================
# GENERAL SERVER ERROR
# ============================================================

@app.errorhandler(500)
def internal_error(error):

    print(
        "SERVER ERROR:",
        repr(error)
    )

    return jsonify({

        "ok": False,

        "error":
            "Internal server error."

    }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )