# video_processing.py

import cv2
from logger import log_bug, log_print
import pypylon.pylon as pylon

def initialize_video_capture():
    try:
        # Initialize Basler camera
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()

        # Reset camera settings to default
        camera.UserSetSelector.SetValue("Default")
        camera.UserSetLoad.Execute()

        # Set pixel format
        camera.PixelFormat.SetValue("BayerRG8")

        # Configure auto white balance settings
        camera.AutoFunctionAOISelector.Value = "AOI2"
        camera.AutoFunctionAOIUsageWhiteBalance.Value = True
        camera.BalanceWhiteAuto.Value = "Continuous"

        # Start grabbing
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        return camera
    except Exception as e:
        log_bug(f"Video capture initialization failed. Exception: {e}")
        raise


def release_video_resources(camera):
    try:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        camera.Close()
        log_print("Video capture released.")
        cv2.destroyAllWindows()
        log_print("All OpenCV windows destroyed.")
    except Exception as e:
        log_bug(f"Failed to release video resources. Exception: {e}")
