import cv2


class RawCameraView:
    def __init__(self):
        pass

    def process(self, frame):
        """
        Takes a frame and returns it (optionally with overlay text).
        """

        cv2.putText(
            frame,
            "RAW CAMERA",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        return frame
