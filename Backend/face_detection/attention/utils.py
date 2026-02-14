import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_head_yaw(landmarks, frame_shape):
    """
    Returns yaw ratio based on nose and cheek distances.
    """

    h, w = frame_shape[:2]

    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]

    nose = (int(nose.x * w), int(nose.y * h))
    left_cheek = (int(left_cheek.x * w), int(left_cheek.y * h))
    right_cheek = (int(right_cheek.x * w), int(right_cheek.y * h))

    dist_left = euclidean(nose, left_cheek)
    dist_right = euclidean(nose, right_cheek)

    if dist_right == 0:
        return 0

    yaw_ratio = dist_left / dist_right

    return yaw_ratio, nose, left_cheek, right_cheek


def get_yaw_direction(yaw_ratio, left_threshold=0.75, right_threshold=1.25):
    """
    Determines head direction from yaw ratio.
    """

    if yaw_ratio < left_threshold:
        return "RIGHT"
    elif yaw_ratio > right_threshold:
        return "LEFT"
    else:
        return "CENTER"
