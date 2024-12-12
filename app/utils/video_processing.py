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
        if side in self.cameras:
            log_print(f"{side.capitalize()} camera already initialized.")
            return

        while True:
            try:
                log_print(f"Attempting to initialize {side} camera with source {source}.")
                device_info = pylon.CDeviceInfo()
                device_info.SetIpAddress(source)

                camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device_info))
                camera.Open()
                if not camera.IsOpen():
                    raise Exception(f"Could not open the Basler camera source: {source}")

                # Apply the camera-specific configuration from PFS file
                pfs_file = LEFT_CAMERA_PFS if side == 'left' else RIGHT_CAMERA_PFS
                if os.path.exists(pfs_file):
                    try:
                        pylon.FeaturePersistence.Load(pfs_file, camera.GetNodeMap(), True)
                        log_print(f"{side.capitalize()} camera settings loaded from {pfs_file}.")
                    except Exception as e:
                        log_bug(f"{side.capitalize()} camera: Failed to load PFS file {pfs_file}. Exception: {e}")
                else:
                    log_bug(f"{side.capitalize()} camera: PFS file {pfs_file} not found.")

                # If right camera depends on left camera:
                if side == 'right' and 'left' in self.cameras:
                    left_cam_data = self.cameras['left']
                    left_cam = left_cam_data['camera']
                    if left_cam is None or not left_cam.IsOpen():
                        log_print("Left camera is not opened yet, will continue and keep right camera waiting.")

                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                converter = pylon.ImageFormatConverter()
                converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

                self.cameras[side] = {
                    'camera': camera,
                    'converter': converter
                }
                log_print(f"Video capture initialized for {side} camera on source {source}.")
                break

            except Exception as e:
                log_bug(f"{side.capitalize()} camera: Video capture initialization failed. Exception: {e}")
                log_print(f"Retrying {side} camera connection in 5 seconds...")
                time.sleep(5)

    def release_video_resources(self):
        for side, cam_data in self.cameras.items():
            camera = cam_data['camera']
            try:
                if camera.IsOpen():
                    camera.StopGrabbing()
                    camera.Close()
                    log_print(f"{side.capitalize()} camera: Video capture released.")
            except Exception as e:
                log_bug(f"{side.capitalize()} camera: Failed to release video resources. Exception: {e}")

        cv2.destroyAllWindows()
        log_print("All OpenCV windows destroyed.")
        self.cameras = {}

    def get_frame(self, side):
        if side not in self.cameras:
            log_bug(f"{side.capitalize()} camera: Video capture not initialized.")
            return None

        camera = self.cameras[side]['camera']
        converter = self.cameras[side]['converter']

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
                    log_bug(f"{side.capitalize()} camera: Failed to capture frame.")
                    return None
            except pylon.TimeoutException:
                log_bug(f"{side.capitalize()} camera: Frame retrieval timed out.")
                return None
        else:
            log_bug(f"{side.capitalize()} camera: Not grabbing frames.")
            return None
