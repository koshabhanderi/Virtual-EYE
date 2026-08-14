---
title: FakeShield
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
---

# FakeShield

AI-powered face authenticity detection demo.

The web deployment uses Flask, YuNet for face detection, and the FakeShield Xception model for classification.

## Local project structure

The deployment expects these existing project folders/files:

- `DeepfakeBench/training/networks/xception.py`
- `ai_face_weights/xception_ai_face.pth`
- `phase1/face_detection_yunet_2026may.onnx`

The deployed web app analyzes the largest visible face from the browser webcam frame.

## Decision thresholds

- Fake score <= 55%: REAL
- Fake score 55% to <80%: UNCERTAIN
- Fake score >= 80%: FAKE

These thresholds are presentation thresholds, not a guarantee of ground-truth authenticity.
