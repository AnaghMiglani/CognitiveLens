import numpy as np


# -------------- BASIC --------------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def landmark_to_point(lm, shape):
    h, w = shape[:2]
    return (int(lm.x * w), int(lm.y * h))


# -------------- EYE LANDMARKS --------------
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


def compute_eye_opening(landmarks, shape):

    lt = landmark_to_point(landmarks[LEFT_EYE_TOP], shape)
    lb = landmark_to_point(landmarks[LEFT_EYE_BOTTOM], shape)

    rt = landmark_to_point(landmarks[RIGHT_EYE_TOP], shape)
    rb = landmark_to_point(landmarks[RIGHT_EYE_BOTTOM], shape)

    left_open = euclidean(lt, lb)
    right_open = euclidean(rt, rb)

    avg_open = (left_open + right_open) / 2

    return avg_open, lt, lb, rt, rb


# -------------- BROW --------------
def compute_brow_distance(landmarks, shape):

    lb = landmark_to_point(landmarks[105], shape)
    rb = landmark_to_point(landmarks[334], shape)

    le = landmark_to_point(landmarks[159], shape)
    re = landmark_to_point(landmarks[386], shape)

    left_dist = euclidean(lb, le)
    right_dist = euclidean(rb, re)

    avg = (left_dist + right_dist) / 2

    return avg, lb, rb, le, re


# -------------- LIP --------------
def compute_lip_ratio(landmarks, shape):

    top = landmark_to_point(landmarks[13], shape)
    bottom = landmark_to_point(landmarks[14], shape)
    left = landmark_to_point(landmarks[61], shape)
    right = landmark_to_point(landmarks[291], shape)

    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)

    ratio = vertical / horizontal if horizontal != 0 else 0

    return ratio, top, bottom, left, right

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def compute_ear(landmarks, frame_shape):

    h, w, _ = frame_shape

    def get_points(indices):
        pts = []
        for i in indices:
            lm = landmarks[i]
            pts.append((int(lm.x * w), int(lm.y * h)))
        return pts

    left = get_points(LEFT_EYE)
    right = get_points(RIGHT_EYE)

    def ear_calc(points):
        v1 = euclidean(points[1], points[5])
        v2 = euclidean(points[2], points[4])
        h = euclidean(points[0], points[3])
        return (v1 + v2) / (2.0 * h)

    left_ear = ear_calc(left)
    right_ear = ear_calc(right)

    ear = (left_ear + right_ear) / 2.0

    return ear, left + right
