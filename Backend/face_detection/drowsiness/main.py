import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

from .utils import (
    extract_eye_features,
    extract_mouth_features,
    draw_eye_points,
    draw_mouth_points,
    is_sleepy
)


class DrowsinessDetector:
    def __init__(self, model_path="../face_landmarker.task", consec_frames=15):

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
        self.sleep_counter = 0
        self.CONSEC_FRAMES = consec_frames

    def process(self, frame):
        """
        Takes a single frame.
        Returns processed frame.
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect_for_video(mp_image, self.frame_count)
        self.frame_count += 1

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            ear, left_eye, right_eye = extract_eye_features(face_landmarks, frame.shape)
            mar, mouth_points = extract_mouth_features(face_landmarks, frame.shape)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            draw_eye_points(frame, left_eye, right_eye)
            draw_mouth_points(frame, mouth_points)

            if is_sleepy(ear, mar):
                self.sleep_counter += 1
            else:
                self.sleep_counter = 0

            if self.sleep_counter >= self.CONSEC_FRAMES:
                cv2.putText(frame, "SLEEPY", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return frame
