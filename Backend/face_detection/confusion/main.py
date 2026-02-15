import cv2
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .utils import (
    compute_brow_metrics,
    compute_head_tilt,
    compute_ear,
    compute_yaw_ratio
)

class ConfusionDetector:

    def __init__(self, calibration_frames=60):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "face_landmarker.task")
        model_path = os.path.abspath(model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.frame_count = 0

        # calibration
        self.calibration_frames = calibration_frames
        self.calib_count = 0

        self.raise_sum = 0
        self.inward_sum = 0
        self.ear_sum = 0

        self.baseline_raise = None
        self.baseline_inward = None
        self.baseline_ear = None

        # side glance stability
        self.last_yaw = 1
        self.yaw_change_count = 0
        self.start_time = time.time()

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect_for_video(mp_image, self.frame_count)
        self.frame_count += 1

        if not result.face_landmarks:
            return frame

        landmarks = result.face_landmarks[0]

        raise_val, asymmetry, inward_dist, brow_pts = compute_brow_metrics(landmarks, frame.shape)
        tilt_angle, tilt_pts = compute_head_tilt(landmarks, frame.shape)
        ear, eye_pts = compute_ear(landmarks, frame.shape)
        yaw_ratio, yaw_pts = compute_yaw_ratio(landmarks, frame.shape)

        if self.baseline_raise is None:

            self.raise_sum += raise_val
            self.inward_sum += inward_dist
            self.ear_sum += ear
            self.calib_count += 1

            cv2.putText(frame, "Calibrating Confusion...", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if self.calib_count >= self.calibration_frames:
                self.baseline_raise = self.raise_sum / self.calibration_frames
                self.baseline_inward = self.inward_sum / self.calibration_frames
                self.baseline_ear = self.ear_sum / self.calibration_frames

            return frame

        raise_score = max(0, (raise_val - self.baseline_raise) / self.baseline_raise) * 100
        raise_score = min(100, raise_score)

        asymmetry_score = min(100, asymmetry * 3)

        inward_score = max(0, (self.baseline_inward - inward_dist) / self.baseline_inward) * 100
        inward_score = min(100, inward_score)

        tilt_score = min(100, abs(tilt_angle) * 2)

        if 0.75 * self.baseline_ear < ear < 0.95 * self.baseline_ear:
            squint_score = 60
        else:
            squint_score = 0

        if abs(yaw_ratio - self.last_yaw) > 0.08:
            self.yaw_change_count += 1

        self.last_yaw = yaw_ratio

        elapsed = time.time() - self.start_time
        yaw_rate = self.yaw_change_count / elapsed if elapsed > 0 else 0
        yaw_score = min(100, yaw_rate * 40)

        confusion_score = (
            0.25 * raise_score +
            0.15 * inward_score +
            0.15 * asymmetry_score +
            0.20 * tilt_score +
            0.15 * squint_score +
            0.10 * yaw_score
        )

        confusion_score = min(100, confusion_score)

        # ---------- DRAW ----------
        for p in brow_pts:
            cv2.circle(frame, p, 4, (255, 0, 0), -1)

        for p in eye_pts:
            cv2.circle(frame, p, 3, (0, 255, 0), -1)

        for p in yaw_pts:
            cv2.circle(frame, p, 4, (0, 255, 255), -1)

        for p in tilt_pts:
            cv2.circle(frame, p, 4, (255, 0, 255), -1)

        cv2.putText(frame, f"Confusion: {confusion_score:.0f}/100", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        return frame
