# video_processing.py

import cv2
import numpy as np
from logger import log_bug, log_print
import neoapi
import threading
import time
import sys
from config import SERIAL_NUMBER_LEFT, SERIAL_NUMBER_RIGHT
import queue

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


class CameraManager:
    def __init__(self):
        self.camera_serial_numbers = [SERIAL_NUMBER_LEFT, SERIAL_NUMBER_RIGHT]
        self.cameras = {}  # Map serial number to camera object
        self.camera_locks = {}  # Map serial number to lock
        self.camera_connected_events = {}  # Map serial number to threading.Event
        self.frame_queues = {}  # Map serial number to frame queue
        self.bufs = {}  # Map serial number to buf
        self.stop_event = threading.Event()

        for serial in self.camera_serial_numbers:
            self.cameras[serial] = None
            self.camera_locks[serial] = threading.Lock()
            self.camera_connected_events[serial] = threading.Event()
            self.frame_queues[serial] = queue.Queue(maxsize=200)
            self.bufs[serial] = None

    def start(self):
        # Start camera initialization threads for each camera
        for serial in self.camera_serial_numbers:
            threading.Thread(target=self.initialize_camera, args=(serial,), daemon=True).start()
            threading.Thread(target=self.capture_frames, args=(serial,), daemon=True).start()

    def stop(self):
        self.stop_event.set()
        # Wait a moment to ensure threads have stopped
        time.sleep(1)
        # Release video resources
        for serial in self.camera_serial_numbers:
            self.release_video_resources(serial)

    def initialize_camera(self, serial_number):
        while not self.stop_event.is_set():
            try:
                with self.camera_locks[serial_number]:
                    if self.cameras[serial_number] is None:
                        log_print(f"Initializing camera with serial number: {serial_number}")
                        camera = neoapi.Cam()
                        camera.Connect(serial_number)
                        self.set_camera_params(camera)
                        self.cameras[serial_number] = camera
                        log_print(f"Camera {serial_number} connected successfully.")
                        self.camera_connected_events[serial_number].set()
            except neoapi.NoAccessException as exc:
                log_print(f"No access to camera {serial_number}: {exc}. Waiting before retrying...")
                time.sleep(10)  # Wait longer before retrying
            except neoapi.NeoException as exc:
                log_print(f"Failed to connect camera {serial_number}: {exc}. Retrying...")
                self.cameras[serial_number] = None
                self.camera_connected_events[serial_number].clear()
                time.sleep(3)
            except Exception as exc:
                log_print(f"Unexpected error when connecting to camera {serial_number}: {exc}")
                self.cameras[serial_number] = None
                self.camera_connected_events[serial_number].clear()
                time.sleep(3)
            time.sleep(0.1)  # Slight delay to prevent tight loop

    def set_camera_params(self, camera):
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
            log_print(f"Error setting camera parameters: {e}")
            sys.exit(0)

    def capture_frames(self, serial_number):
        while not self.stop_event.is_set():
            if not self.camera_connected_events[serial_number].is_set():
                log_print(f"Camera {serial_number} not connected. Waiting to capture frames...")
                time.sleep(1)
                continue
            try:
                with self.camera_locks[serial_number]:
                    camera = self.cameras[serial_number]
                    if camera is None:
                        continue
                    
                    # Fetch payloadsize every time to ensure it's up-to-date
                    payloadsize = camera.f.PayloadSize.Get()
                    #log_print(f"Camera {serial_number} payload size: {payloadsize}")
                    buf = CamBuffer(payloadsize)
                    self.bufs[serial_number] = buf  # Store buf

                    camera.AddUserBuffer(buf)
                    camera.SetUserBufferMode(True)

                # Capture image outside of the lock to prevent blocking
                img = camera.GetImage().GetNPArray()

                if img.size != 0:
                    # Rotate the frame 90 degrees clockwise
                    #img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                    if self.frame_queues[serial_number].full():
                        self.frame_queues[serial_number].get()
                    self.frame_queues[serial_number].put(img)

                with self.camera_locks[serial_number]:
                    camera.RevokeUserBuffer(buf)
                    self.bufs[serial_number] = None  # Clear buf after revoking

            except neoapi.NeoException as exc:
                log_print(f"Camera {serial_number} error: {exc}. Attempting to reconnect...")
                with self.camera_locks[serial_number]:
                    if self.cameras[serial_number]:
                        # Revoke buf if it's still allocated
                        if self.bufs[serial_number]:
                            try:
                                self.cameras[serial_number].RevokeUserBuffer(self.bufs[serial_number])
                            except Exception as e:
                                log_print(f"Error revoking buffer for camera {serial_number}: {e}")
                            self.bufs[serial_number] = None
                        try:
                            self.cameras[serial_number].StopStreaming()
                        except Exception as e:
                            log_print(f"Error stopping streaming for camera {serial_number}: {e}")
                        try:
                            self.cameras[serial_number].Disconnect()
                        except Exception as e:
                            log_print(f"Error disconnecting camera {serial_number}: {e}")
                        del self.cameras[serial_number]  # Delete the camera object
                        self.cameras[serial_number] = None
                        self.camera_connected_events[serial_number].clear()
                time.sleep(3)
            except Exception as e:
                log_print(f"Unexpected error in camera {serial_number} capture: {e}")
                log_bug(f"Unexpected error in camera {serial_number} capture: {e}")
                time.sleep(3)
            time.sleep(0.01)


    def get_frame(self, serial_number):
        if serial_number not in self.frame_queues:
            return None
        if not self.frame_queues[serial_number].empty():
            return self.frame_queues[serial_number].get()
        else:
            return None

    def release_video_resources(self, serial_number):
        try:
            with self.camera_locks[serial_number]:
                if self.cameras[serial_number]:
                    # Revoke buf if it's still allocated
                    if self.bufs[serial_number]:
                        self.cameras[serial_number].RevokeUserBuffer(self.bufs[serial_number])
                        self.bufs[serial_number] = None
                    self.cameras[serial_number].Disconnect()
                    log_print(f"Camera {serial_number} disconnected.")
        except Exception as e:
            log_bug(f"Failed to release video resources for camera {serial_number}. Exception: {e}")
