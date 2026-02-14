import cv2
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .utils import (
    compute_brow_distance,
    compute_lip_ratio,
    compute_ear
)


class StressDetector:
    def __init__(self, calibration_frames=50):

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
        self.start_time = time.time()

        # Calibration for brow + lip only
        self.calibration_frames = calibration_frames
        self.calib_count = 0

        self.brow_sum = 0
        self.lip_sum = 0

        self.baseline_brow = None
        self.baseline_lip = None

        # Blink system (simple + reliable)
        self.blink_counter = 0
        self.eye_state = "OPEN"

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect_for_video(mp_image, self.frame_count)
        self.frame_count += 1

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            brow_dist, lb, rb, le, re = compute_brow_distance(
                landmarks, frame.shape
            )
            lip_ratio, top, bottom, left, right = compute_lip_ratio(
                landmarks, frame.shape
            )
            ear, eye_points = compute_ear(
                landmarks, frame.shape
            )

            # ---------- CALIBRATION ----------
            if self.baseline_brow is None:

                self.brow_sum += brow_dist
                self.lip_sum += lip_ratio
                self.calib_count += 1

                cv2.putText(frame, "Calibrating Stress...", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if self.calib_count >= self.calibration_frames:
                    self.baseline_brow = self.brow_sum / self.calibration_frames
                    self.baseline_lip = self.lip_sum / self.calibration_frames

                return frame

            # ---------- BROW SCORE ----------
            brow_deviation = max(0, self.baseline_brow - brow_dist)
            brow_score = min(100, (brow_deviation / self.baseline_brow) * 200)

            # ---------- LIP SCORE ----------
            lip_deviation = max(0, self.baseline_lip - lip_ratio)
            lip_score = min(100, (lip_deviation / self.baseline_lip) * 250)

            # ---------- BLINK DETECTION (EAR FIXED THRESHOLD) ----------

            EAR_THRESHOLD = 0.22  # adjust between 0.20–0.24
            REOPEN_THRESHOLD = 0.25

            if self.eye_state == "OPEN" and ear < EAR_THRESHOLD:
                self.eye_state = "CLOSED"

            elif self.eye_state == "CLOSED" and ear > REOPEN_THRESHOLD:
                self.blink_counter += 1
                self.eye_state = "OPEN"

            elapsed_time = time.time() - self.start_time
            blinks_per_min = (
                self.blink_counter * 60 / elapsed_time
                if elapsed_time > 0 else 0
            )

            blink_score = min(100, blinks_per_min * 3)

            # ---------- FINAL SCORE ----------
            stress_score = (
                0.40 * brow_score +
                0.30 * lip_score +
                0.30 * blink_score
            )

            stress_score = min(100, stress_score)

            # ---------- DRAW MARKERS ----------

            # Eyes
            for p in eye_points:
                cv2.circle(frame, p, 3, (0, 255, 0), -1)

            # Brows
            cv2.circle(frame, lb, 4, (255, 0, 0), -1)
            cv2.circle(frame, rb, 4, (255, 0, 0), -1)

            # Lips
            for p in [top, bottom, left, right]:
                cv2.circle(frame, p, 4, (0, 255, 255), -1)

            # ---------- TEXT ----------
            cv2.putText(frame, f"Stress: {stress_score:.0f}/100", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

            cv2.putText(frame, f"EAR: {ear:.3f}", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Blinks/min: {blinks_per_min:.1f}", (30, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(frame, f"Total Blinks: {self.blink_counter}", (30, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)

        return frame
