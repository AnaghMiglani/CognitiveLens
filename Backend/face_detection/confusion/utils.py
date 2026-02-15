import numpy as np
import math


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def to_pixel(landmark, shape):
    h, w = shape[:2]
    return (int(landmark.x * w), int(landmark.y * h))

def compute_brow_metrics(landmarks, shape):

    left_brow = to_pixel(landmarks[70], shape)
    right_brow = to_pixel(landmarks[300], shape)

    left_eye_top = to_pixel(landmarks[159], shape)
    right_eye_top = to_pixel(landmarks[386], shape)

    inner_left_brow = to_pixel(landmarks[105], shape)
    inner_right_brow = to_pixel(landmarks[334], shape)

    #vertical raise
    left_raise = euclidean(left_brow, left_eye_top)
    right_raise = euclidean(right_brow, right_eye_top)

    avg_raise = (left_raise + right_raise) / 2.0

    asymmetry = abs(left_raise - right_raise)

    inward_distance = euclidean(inner_left_brow, inner_right_brow)

    return avg_raise, asymmetry, inward_distance, [
        left_brow, right_brow,
        inner_left_brow, inner_right_brow
    ]

def compute_head_tilt(landmarks, shape):

    left_eye = to_pixel(landmarks[33], shape)
    right_eye = to_pixel(landmarks[263], shape)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = math.degrees(math.atan2(dy, dx))

    return abs(angle), [left_eye, right_eye]

def compute_ear(landmarks, shape):

    left_indices = [33, 160, 158, 133, 153, 144]
    right_indices = [362, 385, 387, 263, 373, 380]

    def ear_eye(indices):
        pts = [to_pixel(landmarks[i], shape) for i in indices]
        v1 = euclidean(pts[1], pts[5])
        v2 = euclidean(pts[2], pts[4])
        h = euclidean(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h), pts

    left_ear, left_pts = ear_eye(left_indices)
    right_ear, right_pts = ear_eye(right_indices)

    ear = (left_ear + right_ear) / 2.0

    return ear, left_pts + right_pts

def compute_yaw_ratio(landmarks, shape):

    nose = to_pixel(landmarks[1], shape)
    left_cheek = to_pixel(landmarks[234], shape)
    right_cheek = to_pixel(landmarks[454], shape)

    left_dist = euclidean(nose, left_cheek)
    right_dist = euclidean(nose, right_cheek)

    if right_dist == 0:
        return 0, [nose, left_cheek, right_cheek]

    ratio = left_dist / right_dist

    return ratio, [nose, left_cheek, right_cheek]
