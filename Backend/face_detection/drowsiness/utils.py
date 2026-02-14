import numpy as np
import cv2

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_eye_points(indices, landmarks, w, h):
    pts = []
    for i in indices:
        lm = landmarks[i]
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts

def compute_ear(points):
    vertical1 = euclidean(points[1], points[5])
    vertical2 = euclidean(points[2], points[4])
    horizontal = euclidean(points[0], points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def compute_mar(landmarks, w, h):
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[61]
    right = landmarks[291]

    top = (int(top.x * w), int(top.y * h))
    bottom = (int(bottom.x * w), int(bottom.y * h))
    left = (int(left.x * w), int(left.y * h))
    right = (int(right.x * w), int(right.y * h))

    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)

    mar = vertical / horizontal
    return mar, [top, bottom, left, right]

def extract_eye_features(landmarks, frame_shape):
    h, w = frame_shape[:2]

    left_eye = get_eye_points(LEFT_EYE_INDICES, landmarks, w, h)
    right_eye = get_eye_points(RIGHT_EYE_INDICES, landmarks, w, h)

    left_ear = compute_ear(left_eye)
    right_ear = compute_ear(right_eye)

    ear = (left_ear + right_ear) / 2.0

    return ear, left_eye, right_eye

def extract_mouth_features(landmarks, frame_shape):
    h, w = frame_shape[:2]
    return compute_mar(landmarks, w, h)

def is_sleepy(ear, mar):
    return ear <= 0.15 or (ear <= 0.20 and mar >= 0.35)

def draw_eye_points(frame, left_eye, right_eye):
    for p in left_eye + right_eye:
        cv2.circle(frame, p, 2, (255, 0, 0), -1)

def draw_mouth_points(frame, mouth_points):
    for p in mouth_points:
        cv2.circle(frame, p, 3, (0, 255, 255), -1)
