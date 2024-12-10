# Unit Test for Basler 2 Cameras
# To run use 'python3 -m unittest camera_test.py'

import unittest
from pypylon import pylon

class TestActualCameras(unittest.TestCase):
    def test_cameras_working(self):
        # List of camera IPs
        camera_ips = ["192.168.1.10", "192.168.1.20"]

        # Initialize cameras
        cameras = []
        try:
            for ip in camera_ips:
                device_info = pylon.CDeviceInfo()
                device_info.SetIpAddress(ip)
                camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device_info))
                camera.Open()
                cameras.append(camera)

            # Check if cameras are grabbing successfully
            for i, camera in enumerate(cameras):
                with self.subTest(camera=i):
                    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                    self.assertTrue(camera.IsGrabbing(), f"Camera {i+1} is not grabbing.")
                    self.assertTrue(grab_result.GrabSucceeded(), f"Camera {i+1} failed to grab an image.")
                    grab_result.Release()

        finally:
            # Release resources
            for camera in cameras:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
                camera.Close()

if __name__ == '__main__':
    unittest.main()
