import cv2
import time

model_path = "face_detection_yunet_2026may.onnx"

detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to access webcam")
        break

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Draw face rectangles
    face_count = 0

    if faces is not None:
        face_count = len(faces)

        for face in faces:
            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display number of faces
    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("FakeShield - Phase 2", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if cv2.getWindowProperty(
        "FakeShield - Phase 2",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break

cap.release()
cv2.destroyAllWindows()