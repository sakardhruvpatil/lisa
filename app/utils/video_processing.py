import os
import time
import cv2
from pypylon import pylon
from config.config import VIDEO_SOURCE_LEFT, VIDEO_SOURCE_RIGHT, LEFT_CAMERA_PFS, RIGHT_CAMERA_PFS
from utils.logger import log_bug, log_print


class CameraManager:
    def __init__(self, side=None, source=None):
        self.cameras = {}
        if side and source:
            self.initialize_video_capture(side, source)

    def initialize_video_capture(self, side, source):
        # If already initialized, skip
        if side in self.cameras:
            log_print(f"{side.capitalize()} camera already initialized.")
            return

        # Try connecting, retry on failure
        while True:
            try:
                log_print(f"Attempting to initialize {side} camera with source {source}.")
                device_info = pylon.CDeviceInfo()
                device_info.SetIpAddress(source)

                camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device_info))
                camera.Open()
                if not camera.IsOpen():
                    raise Exception(f"Could not open the Basler camera source: {source}")

                # Load settings from PFS file
                pfs_file = LEFT_CAMERA_PFS if side == 'left' else RIGHT_CAMERA_PFS
                if os.path.exists(pfs_file):
                    try:
                        pylon.FeaturePersistence.Load(pfs_file, camera.GetNodeMap(), True)
                        log_print(f"{side.capitalize()} camera settings loaded from {pfs_file}.")
                    except Exception as e:
                        error_code = 1006
                        log_bug(f"{side.capitalize()} camera: Failed to load PFS file {pfs_file}. "
                                f"Exception: {e}(Error Code: {error_code})")
                else:
                    error_code = 1007
                    log_bug(f"{side.capitalize()} camera: PFS file {pfs_file} not found."
                            f"(Error Code: {error_code})")

                # If right camera depends on left camera:
                if side == 'right' and 'left' in self.cameras:
                    left_cam_data = self.cameras['left']
                    left_cam = left_cam_data['camera']
                    if left_cam is None or not left_cam.IsOpen():
                        log_print("Left camera is not opened yet, continuing but the right camera will retry if needed.")

                # Start grabbing
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                converter = pylon.ImageFormatConverter()
                converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

                self.cameras[side] = {
                    'camera': camera,
                    'converter': converter,
                    'source': source  # Keep track of source for reconnect
                }
                log_print(f"Video capture initialized for {side} camera on source {source}.")
                break

            except Exception as e:
                error_code = 1008
                log_bug(f"{side.capitalize()} camera: Video capture initialization failed. Exception: {e}(Error Code: {error_code})")
                log_print(f"Retrying {side} camera connection in 5 seconds...")
                time.sleep(5)

    def reconnect_camera(self, side):
        """
        Attempt to close and reinitialize the camera if it is disconnected.
        """
        if side not in self.cameras:
            log_bug(f"{side.capitalize()} camera: No camera data found to reconnect.")
            return

        camera_data = self.cameras[side]
        camera = camera_data.get('camera', None)
        source = camera_data.get('source', None)

        # Safely close existing camera if it's open
        try:
            if camera and camera.IsOpen():
                camera.StopGrabbing()
                camera.Close()
        except Exception as e:
            error_code = 1014
            log_bug(f"{side.capitalize()} camera: Error closing camera during reconnect. "
                    f"Exception: {e}(Error Code: {error_code})")

        # Remove the old camera reference from the dict
        del self.cameras[side]

        # Attempt to re-initialize
        if source:
            log_print(f"Reconnecting {side} camera with source {source}...")
            self.initialize_video_capture(side, source)
        else:
            log_bug(f"{side.capitalize()} camera: Source not found, cannot reconnect.")

    def release_video_resources(self):
        for side, cam_data in self.cameras.items():
            camera = cam_data['camera']
            try:
                if camera and camera.IsOpen():
                    camera.StopGrabbing()
                    camera.Close()
                    log_print(f"{side.capitalize()} camera: Video capture released.")
            except Exception as e:
                error_code = 1009
                log_bug(f"{side.capitalize()} camera: Failed to release video resources. Exception: {e}(Error Code: {error_code})")

        cv2.destroyAllWindows()
        log_print("All OpenCV windows destroyed.")
        self.cameras = {}

    def get_frame(self, side):
        if side not in self.cameras:
            error_code = 1010
            log_bug(f"{side.capitalize()} camera: Video capture not initialized.(Error Code: {error_code})")
            return None

        cam_data = self.cameras[side]
        camera = cam_data['camera']
        converter = cam_data['converter']

        # Check if camera is open
        if not camera or not camera.IsOpen():
            log_print(f"{side.capitalize()} camera: Detected a disconnect, attempting to reconnect...")
            self.reconnect_camera(side)
            return None

        # Grab a frame
        if camera.IsGrabbing():
            try:
                grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    image = converter.Convert(grab_result)
                    frame = image.GetArray()
                    grab_result.Release()
                    return frame
                else:
                    grab_result.Release()
                    error_code = 1011
                    log_bug(f"{side.capitalize()} camera: Failed to capture frame.(Error code: {error_code})")
                    return None
            except pylon.TimeoutException:
                error_code = 1012
                log_bug(f"{side.capitalize()} camera: Frame retrieval timed out.(Error code: {error_code})")
                return None
        else:
            error_code = 1013
            log_bug(f"{side.capitalize()} camera: Not grabbing frames.(Error code: {error_code})")
            return None