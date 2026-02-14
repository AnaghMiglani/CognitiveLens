import cv2
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .utils import compute_head_yaw


class AttentionDetector:
    def __init__(self, consec_frames=10, calibration_frames=40, deviation_threshold=0.30):

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
        self.turn_count = 0
        self.last_direction = "CENTER"
        self.yaw_stable_frames = 0
        self.MIN_STABLE_FRAMES = consec_frames
        self.start_time = time.time()

        self.calibration_frames = calibration_frames
        self.calibration_count = 0
        self.yaw_sum = 0
        self.baseline_yaw = None
        self.deviation_threshold = deviation_threshold

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect_for_video(mp_image, self.frame_count)
        self.frame_count += 1

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            yaw_ratio, nose, left_cheek, right_cheek = compute_head_yaw(
                landmarks, frame.shape
            )

            if self.baseline_yaw is None:
                self.yaw_sum += yaw_ratio
                self.calibration_count += 1

                cv2.putText(frame, "Calibrating...", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if self.calibration_count >= self.calibration_frames:
                    self.baseline_yaw = self.yaw_sum / self.calibration_frames

                return frame

            deviation = yaw_ratio - self.baseline_yaw

            if deviation > self.deviation_threshold:
                direction = "LEFT"
            elif deviation < -self.deviation_threshold:
                direction = "RIGHT"
            else:
                direction = "CENTER"

            cv2.circle(frame, nose, 4, (0, 255, 255), -1)
            cv2.circle(frame, left_cheek, 4, (255, 0, 0), -1)
            cv2.circle(frame, right_cheek, 4, (255, 0, 0), -1)

            cv2.putText(frame, f"Yaw: {yaw_ratio:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(frame, f"Baseline: {self.baseline_yaw:.2f}", (30, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.putText(frame, f"Direction: {direction}", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            if direction != self.last_direction and direction != "CENTER":
                self.yaw_stable_frames += 1

                if self.yaw_stable_frames >= self.MIN_STABLE_FRAMES:
                    self.turn_count += 1
                    self.last_direction = direction
                    self.yaw_stable_frames = 0
            else:
                self.yaw_stable_frames = 0

        elapsed_time = time.time() - self.start_time
        turns_per_minute = (self.turn_count * 60 / elapsed_time) if elapsed_time > 0 else 0

        cv2.putText(frame, f"Turn Count: {self.turn_count}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.putText(frame, f"Turns/Min: {turns_per_minute:.2f}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame
