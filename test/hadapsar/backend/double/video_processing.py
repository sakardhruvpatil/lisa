# video_processing.py

import cv2
import numpy as np
from logger import log_bug, log_print
import neoapi
import logging
from threading import Lock
import time
import sys


class CamBuffer(neoapi.BufferBase):
    def __init__(self, size):
        neoapi.BufferBase.__init__(self)
        self.cpu_mat = np.ndarray(size, np.uint8)
        self.RegisterMemory(self.cpu_mat, size)

    def FreeCamBuffers(self):
        while self._buffers:
            self._camera.RevokeUserBuffer(self._buffers.pop())

    def __del__(self):
        self.UnregisterMemory()


class CameraInitializationError(Exception):
    """Custom exception for camera initialization errors"""

    pass


# Camera Initialization

def initialize_camera(serial_number):
    try:
        camera_lock = Lock()
        camera = None
        print(f"Initializing camera with serial number: {serial_number}")
        if camera is None:
            while camera is None:
                try:
                    with camera_lock:
                        camera = neoapi.Cam()
                        camera.Connect(serial_number)
                        set_camera_params(camera)
                        print(f"Camera {serial_number} connected successfully.")
                except (neoapi.NeoException, Exception) as exc:
                    print(f"Failed to connect camera {serial_number}: {exc}. Retrying...")
                    camera = None
                    time.sleep(3)
            if camera is None:
                raise CameraInitializationError(
                    f"Failed to initialize camera {serial_number} after multiple attempts."
                )
    except Exception as e:
        log_bug(f"Initialization failed for camera {serial_number}. Exception: {e}")
        raise
    return camera



def set_camera_params(camera):
    try:
        pixel_format = (
            "BGR8"
            if camera.f.PixelFormat.GetEnumValueList().IsReadable("BGR8")
            else "Mono8"
        )
        camera.f.PixelFormat.SetString(pixel_format)
        camera.f.TriggerMode.value = neoapi.TriggerMode_Off
        camera.f.ExposureAuto.SetString("Off")
        camera.f.ExposureMode.SetString("Timed")
        camera.f.ExposureTime.Set(14000.0)
        camera.f.GainAuto.SetString("Off")
        camera.f.Gain.Set(0.0)
        camera.f.LUTEnable.Set(True)
        camera.f.Gamma.Set(0.80)
        camera.f.BalanceWhiteAuto.SetString("Continuous")
        camera.f.AcquisitionFrameRateEnable.value = True
        camera.f.AcquisitionFrameRate.value = 55.0
    except Exception as e:
        print(f"Error setting camera parameters: {e}")
        sys.exit(0)


# Frame Capture
def capture_frames(camera, frame_queue, stop_event, serial_number):
    if camera is None:
        logging.error(f"No camera available for frame capture (Serial: {serial_number}).")
        return
    while not stop_event.is_set():
        try:
            payloadsize = camera.f.PayloadSize.Get()
            buf = CamBuffer(payloadsize)
            camera.AddUserBuffer(buf)
            camera.SetUserBufferMode(True)
            img = camera.GetImage().GetNPArray()

            if img.size != 0:
                # Rotate the frame 90 degrees clockwise
                img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                if frame_queue.full():
                    frame_queue.get()
                frame_queue.put(img_rotated)

        except Exception as e:
            logging.error(
                f"Error capturing frame from camera {serial_number}: {e}. Retrying camera initialization..."
            )
            camera.Disconnect()
            camera = initialize_camera(serial_number)  # Retry initializing the camera
        except neoapi.NoImageBufferException as e:
            logging.error(
                f"NoImageBufferException on camera {serial_number}: {e}. Retrying camera initialization..."
            )
            camera.Disconnect()
            camera = initialize_camera(serial_number)  # Reinitialize the camera
        finally:
            camera.RevokeUserBuffer(buf)  # Always revoke the buffer

    # Clear all frames in the queue
    with frame_queue.mutex:
        frame_queue.queue.clear()


def release_video_resources(camera, buf):
    try:
        # Clean up
        if camera:
            camera.RevokeUserBuffer(buf)
            camera.Disconnect()
            cv2.destroyAllWindows()
            log_print("Video capture released.")
        log_print("All OpenCV windows destroyed.")
    except Exception as e:
        log_bug(f"Failed to release video resources. Exception: {e}")
