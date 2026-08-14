import cv2
import time

cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to access webcam")
        break

    current_time = time.time()

    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("FakeShield - Phase 1", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if cv2.getWindowProperty("FakeShield - Phase 1", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()