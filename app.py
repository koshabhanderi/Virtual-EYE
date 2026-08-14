import os

# Reduce CPU/thread memory usage on Render Free
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import sys
import base64
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEEPFAKEBENCH_TRAINING = (
    BASE_DIR / "DeepfakeBench" / "training"
)

CHECKPOINT_PATH = (
    BASE_DIR / "ai_face_weights" / "xception_ai_face.pth"
)

sys.path.insert(
    0,
    str(DEEPFAKEBENCH_TRAINING)
)

from networks.xception import Xception


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# ============================================================
# JSON RESPONSE HELPER
# ============================================================

def json_response(data, status=200):

    response = app.response_class(
        response=json.dumps(
            data,
            separators=(",", ":")
        ),
        status=status,
        mimetype="application/json"
    )

    # Do NOT manually specify Content-Length.
    # Render/Gunicorn will handle it.
    response.headers.pop(
        "Content-Length",
        None
    )

    return response


# ============================================================
# Model
# ============================================================

DEVICE = torch.device("cpu")

# Keep PyTorch from creating large CPU thread pools
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

MODEL_CONFIG = {
    "mode": "original",
    "num_classes": 1,
    "inc": 3,
    "dropout": False,
}

model = None
model_error = None


def load_model():

    global model
    global model_error

    try:

        if not CHECKPOINT_PATH.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}"
            )

        print(
            "Loading FakeShield Xception model..."
        )

        model = Xception(
            MODEL_CONFIG
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu"
        )

        # Handle common checkpoint formats.
        if isinstance(
            checkpoint,
            dict
        ):

            if "state_dict" in checkpoint:

                checkpoint = (
                    checkpoint["state_dict"]
                )

            elif "model_state_dict" in checkpoint:

                checkpoint = (
                    checkpoint["model_state_dict"]
                )

        new_checkpoint = {}

        for key, value in checkpoint.items():

            if key.startswith("backbone."):

                new_key = key[len("backbone."):]

            else:

                new_key = key

            new_checkpoint[
                new_key
            ] = value

        model.load_state_dict(
            new_checkpoint,
            strict=True
        )

        model.to(
            DEVICE
        )

        model.eval()

        print(
            "FakeShield model loaded successfully."
        )

    except Exception as exc:

        model = None

        model_error = str(
            exc
        )

        print(
            "MODEL LOAD ERROR:",
            exc
        )


load_model()


# ============================================================
# YuNet Face Detector
# ============================================================

YUNET_PATH = (
    BASE_DIR
    / "phase1"
    / "face_detection_yunet_2026may.onnx"
)

face_detector = None
yunet_error = None


try:

    if not YUNET_PATH.exists():

        raise FileNotFoundError(
            f"YuNet model not found: {YUNET_PATH}"
        )

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

    yunet_error = str(
        exc
    )

    print(
        "YUNET LOAD ERROR:",
        exc
    )


# ============================================================
# Face Prediction
# ============================================================

def predict_face(face_bgr):

    if model is None:

        raise RuntimeError(
            model_error
            or "FakeShield model is not loaded."
        )

    face = cv2.resize(
        face_bgr,
        (299, 299),
        interpolation=cv2.INTER_AREA
    )

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    face = face.astype(
        np.float32
    ) / 255.0

    face = np.transpose(
        face,
        (2, 0, 1)
    )

    face = np.expand_dims(
        face,
        axis=0
    )

    tensor = torch.from_numpy(
        face
    ).to(DEVICE)

    with torch.inference_mode():

        result = model(
            tensor
        )

    output = (
        result[0]
        if isinstance(result, tuple)
        else result
    )

    output = output.squeeze()

    fake_probability = torch.sigmoid(
        output
    ).item()

    return float(
        fake_probability
    )


# ============================================================
# Decode Browser Image
# ============================================================

def decode_image(data_url):

    if not data_url:

        raise ValueError(
            "No image was supplied."
        )

    if "," in data_url:

        _, encoded = data_url.split(
            ",",
            1
        )

    else:

        encoded = data_url

    raw = base64.b64decode(
        encoded
    )

    array = np.frombuffer(
        raw,
        dtype=np.uint8
    )

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
# Detect Largest Face
# ============================================================

def detect_largest_face(frame):

    if face_detector is None:

        raise RuntimeError(
            yunet_error
            or "YuNet is not loaded."
        )

    height, width = (
        frame.shape[:2]
    )

    face_detector.setInputSize(
        (
            int(width),
            int(height)
        )
    )

    _, faces = face_detector.detect(
        frame
    )

    if faces is None or len(faces) == 0:

        return None, 0

    largest = None

    largest_area = 0

    for face in faces:

        x = int(
            face[0]
        )

        y = int(
            face[1]
        )

        w = int(
            face[2]
        )

        h = int(
            face[3]
        )

        # Keep box inside image.

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
# Classification
# ============================================================

def classify(fake_probability):

    fake_probability = float(
        fake_probability
    )

    real_probability = float(
        1.0 - fake_probability
    )

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
# Home Route
# ============================================================

@app.get("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():

    return jsonify({

        "status": "ok",

        "model_loaded": bool(
            model is not None
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
# Prediction
# ============================================================

@app.post("/predict")
def predict():

    try:

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

            return json_response({

                "ok": False,

                "error":
                    "No image supplied."

            }, 400)

        frame = decode_image(
            image_data
        )

        face_box, face_count = (
            detect_largest_face(
                frame
            )
        )

        # ----------------------------------------------------
        # No face detected
        # ----------------------------------------------------

        if face_box is None:

            return json_response({

                "ok": True,

                "face_detected": False,

                "face_count": int(
                    face_count
                ),

                "message":
                    "No face detected."

            })


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

            return json_response({

                "ok": True,

                "face_detected": False,

                "face_count": int(
                    face_count
                ),

                "message":
                    "Face is too small for reliable analysis."

            })


        # ----------------------------------------------------
        # Crop largest face
        # ----------------------------------------------------

        face_crop = frame[
            y:y + h,
            x:x + w
        ]


        # ----------------------------------------------------
        # AI classification
        # ----------------------------------------------------

        fake_probability = (
            predict_face(
                face_crop
            )
        )

        result = classify(
            fake_probability
        )


        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return json_response({

            "ok": True,

            "face_detected": True,

            "face_count": int(
                face_count
            ),

            "box": {

                "x": int(x),

                "y": int(y),

                "w": int(w),

                "h": int(h)
            },

            "label": str(
                result["label"]
            ),

            "fake_probability": float(
                result[
                    "fake_probability"
                ]
            ),

            "real_probability": float(
                result[
                    "real_probability"
                ]
            )

        })


    except Exception as exc:

        print(
            "PREDICTION ERROR:",
            exc
        )

        return json_response({

            "ok": False,

            "error": str(
                exc
            )

        }, 500)


# ============================================================
# File Too Large
# ============================================================

@app.errorhandler(413)
def too_large(_error):

    return json_response({

        "ok": False,

        "error":
            "Image is too large. Please use a smaller image."

    }, 413)


# ============================================================
# Run Application
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