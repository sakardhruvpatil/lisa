# video_processing.py

import cv2
from logger import log_bug, log_print
from config import VIDEO_SOURCE_LEFT, VIDEO_SOURCE_RIGHT  # Assuming you define separate sources for left and right cameras

class CameraManager:
    def __init__(self, side=None, source=None):
        self.cameras = {}

        if side and source:
            self.initialize_video_capture(side, source)

    def initialize_video_capture(self, side, source):
        if side in self.cameras:
            log_print(f"{side.capitalize()} camera already initialized.")
            return  # Skip if already initialized
        try:
            cap = cv2.VideoCapture(source)
            log_print(f"Attempting to initialize {side} camera with source {source}.")
            if not cap.isOpened():
                raise Exception(f"Could not open the video source: {source}.")
            
            # Check if the other camera is already initialized and open
            if side == 'right' and 'left' in self.cameras:
                left_cap = self.cameras['left']
                if left_cap is not None and not left_cap.isOpened():
                    raise Exception("Left camera is not opened, cannot proceed with right camera.")
            
            self.cameras[side] = cap
            log_print(f"Video capture initialized for {side} camera on source {source}.")
        except Exception as e:
            log_bug(f"{side.capitalize()} camera: Video capture initialization failed. Exception: {e}")
            raise

    def release_video_resources(self):
        for side, cap in self.cameras.items():
            try:
                if cap.isOpened():
                    cap.release()
                    log_print(f"{side.capitalize()} camera: Video capture released.")
            except Exception as e:
                log_bug(f"{side.capitalize()} camera: Failed to release video resources. Exception: {e}")
        cv2.destroyAllWindows()
        log_print("All OpenCV windows destroyed.")

        # Ensure no resources are held after release
        self.cameras = {}  # Reset camera dictionary after release


    def get_frame(self, side):
        if side in self.cameras:
            cap = self.cameras[side]
            ret, frame = cap.read()
            if not ret:
                log_bug(f"{side.capitalize()} camera: Failed to capture frame.")
                return None
            return frame
        else:
            log_bug(f"{side.capitalize()} camera: Video capture not initialized.")
            return None