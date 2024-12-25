import os
import time
from pypylon import pylon
import cv2
import numpy as np

# Relative paths to PFS files (located in the "config" folder at the same level as "dataset")
LEFT_CAMERA_PFS = os.path.join("..", "config", "left_camera_config.pfs")
RIGHT_CAMERA_PFS = os.path.join("..", "config", "right_camera_config.pfs")

# Relative directory for saving captures
base_captures_dir = os.path.join("capture")
timestamp = int(time.time())  # Current timestamp in seconds

# Directory for the current run within horizontal_dataset
current_capture_dir = os.path.join(base_captures_dir, "horizontal_dataset", f"run_{timestamp}")
os.makedirs(current_capture_dir, exist_ok=True)

# List of camera IPs and their configurations
camera_configs = [
    {"ip": "192.168.1.20", "side": "left", "pfs": LEFT_CAMERA_PFS},
    {"ip": "192.168.1.10", "side": "right", "pfs": RIGHT_CAMERA_PFS}
]

# Dictionary to associate each InstantCamera instance with a side label
camera_side_map = {}

# Initialize cameras
cameras = []
for config in camera_configs:
    device_info = pylon.CDeviceInfo()
    device_info.SetIpAddress(config["ip"])
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device_info))
    camera.Open()

    # Apply the camera-specific configuration from the PFS file
    if os.path.exists(config["pfs"]):
        try:
            pylon.FeaturePersistence.Load(config["pfs"], camera.GetNodeMap(), True)
            print(f"{config['side'].capitalize()} camera settings loaded from {config['pfs']}.")
        except Exception as e:
            print(f"{config['side'].capitalize()} camera: Failed to load PFS file {config['pfs']}. Exception: {e}")
    else:
        print(f"{config['side'].capitalize()} camera: PFS file {config['pfs']} not found.")

    # Keep track of which side corresponds to this camera
    camera_side_map[camera] = config["side"]

    cameras.append(camera)

# Start grabbing on all cameras
for camera in cameras:
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Image converter for OpenCV
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

try:
    frame_count = 0
    last_capture_time = time.time()

    while True:
        frames = {}
        for camera in cameras:
            if camera.IsGrabbing():
                grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    # Convert image to an OpenCV-compatible format
                    image = converter.Convert(grab_result)
                    frame = image.GetArray()

                    # Use our dictionary to grab the correct side ("left" or "right")
                    side = camera_side_map[camera]
                    frames[side] = frame

                grab_result.Release()

        # If we got frames for both cameras, process them
        if len(frames) == 2:  # or == len(cameras)
            left_frame = frames["left"]
            right_frame = frames["right"]

            # Resize frames to have the same height for horizontal concatenation
            min_height = min(left_frame.shape[0], right_frame.shape[0])
            left_resized = cv2.resize(
                left_frame,
                (int(left_frame.shape[1] * min_height / left_frame.shape[0]), min_height)
            )
            right_resized = cv2.resize(
                right_frame,
                (int(right_frame.shape[1] * min_height / right_frame.shape[0]), min_height)
            )

            # Concatenate frames horizontally
            concatenated_frame = np.concatenate([left_resized, right_resized], axis=1)

            # Show concatenated feed
            cv2.imshow("Concatenated Feed", concatenated_frame)

            # Save concatenated frame every 1 second
            current_time = time.time()
            if current_time - last_capture_time >= 1:
                save_path = os.path.join(current_capture_dir, f"capture_{frame_count:04d}.jpg")
                cv2.imwrite(save_path, concatenated_frame)
                print(f"Saved: {save_path}")

                frame_count += 1
                last_capture_time = current_time  # Update the last capture time

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    for camera in cameras:
        camera.StopGrabbing()
        camera.Close()
    cv2.destroyAllWindows()
