from raw.cam import RawCameraView
from raw.landmarks import LandmarkViewer
from drowsiness.main import DrowsinessDetector

import cv2

raw_view = RawCameraView()
landmark_view = LandmarkViewer()
drowsy = DrowsinessDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    raw_frame = raw_view.process(frame.copy())
    landmark_frame = landmark_view.process(frame.copy())
    sleep_frame = drowsy.process(frame.copy())

    cv2.imshow("Raw", raw_frame)
    cv2.imshow("Landmarks", landmark_frame)
    cv2.imshow("Drowsiness", sleep_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
