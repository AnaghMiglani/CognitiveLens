from raw.cam import RawCameraView
from raw.landmarks import LandmarkViewer
from drowsiness.main import DrowsinessDetector
from attention.main import AttentionDetector
from stress.main import StressDetector

import cv2
import numpy as np

raw_view = RawCameraView()
landmark_view = LandmarkViewer()
drowsy = DrowsinessDetector()
attention = AttentionDetector()
stress = StressDetector()

cap = cv2.VideoCapture(0)

#tile size
TILE_W = 420
TILE_H = 320

def resize_frame(frame):
    return cv2.resize(frame, (TILE_W, TILE_H))

#make window resizable
cv2.namedWindow("CognitiveLens Dashboard", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    raw_frame = raw_view.process(frame.copy())
    landmark_frame = landmark_view.process(frame.copy())
    sleep_frame = drowsy.process(frame.copy())
    attention_frame = attention.process(frame.copy())
    stress_frame = stress.process(frame.copy())
    placeholder_frame = raw_view.process(frame.copy())

    raw_frame = resize_frame(raw_frame)
    landmark_frame = resize_frame(landmark_frame)
    sleep_frame = resize_frame(sleep_frame)
    attention_frame = resize_frame(attention_frame)
    stress_frame = resize_frame(stress_frame)
    placeholder_frame = resize_frame(placeholder_frame)

    top_row = np.hstack((raw_frame, landmark_frame, sleep_frame))
    bottom_row = np.hstack((attention_frame, stress_frame, placeholder_frame))
    dashboard = np.vstack((top_row, bottom_row))

    #fit to screen size
    screen_w = 1920
    screen_h = 1080

    dash_h, dash_w = dashboard.shape[:2]
    scale_w = screen_w / dash_w
    scale_h = screen_h / dash_h
    scale = min(scale_w, scale_h)

    new_w = int(dash_w * scale)
    new_h = int(dash_h * scale)

    dashboard = cv2.resize(dashboard, (new_w, new_h))

    cv2.imshow("CognitiveLens Dashboard", dashboard)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
