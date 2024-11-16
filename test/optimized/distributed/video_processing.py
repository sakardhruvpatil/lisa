# video_processing.py

import cv2
from config import VIDEO_SOURCE
from logger import log_bug, log_print

def initialize_video_capture(source=VIDEO_SOURCE):
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise Exception("Could not open the video source.")
        log_print(f"Video capture initialized on source {source}.")
        return cap
    except Exception as e:
        log_bug(f"Video capture initialization failed. Exception: {e}")
        raise

def release_video_resources(cap):
    try:
        if cap.isOpened():
            cap.release()
            log_print("Video capture released.")
        cv2.destroyAllWindows()
        log_print("All OpenCV windows destroyed.")
    except Exception as e:
        log_bug(f"Failed to release video resources. Exception: {e}")
