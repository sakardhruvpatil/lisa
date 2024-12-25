import os
import time
from pypylon import pylon
import cv2
import numpy as np

# Relative paths to PFS files
LEFT_CAMERA_PFS = os.path.join("..", "config", "left_camera_config.pfs")
RIGHT_CAMERA_PFS = os.path.join("..", "config", "right_camera_config.pfs")

# Relative directory for saving captures
base_dataset_dir = os.path.join("capture")
timestamp = int(time.time())  # Current timestamp in seconds

# Directories for left and right dataset (with run timestamp)
left_dataset_timestamp_dir = os.path.join(base_dataset_dir, "left_dataset", f"run_{timestamp}")
right_dataset_timestamp_dir = os.path.join(base_dataset_dir, "right_dataset", f"run_{timestamp}")

# Create all required directories if they don't exist
os.makedirs(left_dataset_timestamp_dir, exist_ok=True)
os.makedirs(right_dataset_timestamp_dir, exist_ok=True)

# Camera configurations
camera_configs = [
    {"ip": "192.168.1.20", "side": "left", "pfs": LEFT_CAMERA_PFS},
    {"ip": "192.168.1.10", "side": "right", "pfs": RIGHT_CAMERA_PFS}
]

# For tracking which side corresponds to each camera
camera_side_map = {}

# Initialize cameras
cameras = []
for config in camera_configs:
    device_info = pylon.CDeviceInfo()
    device_info.SetIpAddress(config["ip"])
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device_info))
    camera.Open()

    # Apply the camera-specific configuration from PFS file
    pfs_file = config["pfs"]
    side = config["side"]
    if os.path.exists(pfs_file):
        try:
            pylon.FeaturePersistence.Load(pfs_file, camera.GetNodeMap(), True)
            print(f"{side.capitalize()} camera settings loaded from {pfs_file}.")
        except Exception as e:
            print(f"{side.capitalize()} camera: Failed to load PFS file {pfs_file}. Exception: {e}")
    else:
        print(f"{side.capitalize()} camera: PFS file {pfs_file} not found.")

    # Attach the device (optional in some scripts; do not store side as a node in the camera)
    camera.Attach(pylon.TlFactory.GetInstance().CreateDevice(device_info))

    # Store the camera object and its side in a Python dictionary
    camera_side_map[camera] = side

    cameras.append(camera)

# Start grabbing on all cameras
for camera in cameras:
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Image converter for OpenCV
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

try:
    frame_count = 0  # Counter for image filenames
    last_capture_time = time.time()  # Track the last capture time

    while True:
        frames = {}
        for camera in cameras:
            if camera.IsGrabbing():
                grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    # Convert image to OpenCV format
                    image = converter.Convert(grab_result)
                    frame = image.GetArray()

                    # Retrieve which 'side' this camera is (left or right) from our dictionary
                    side = camera_side_map[camera]

                    frames[side] = frame

                grab_result.Release()

        # If we got frames from both sides, display and/or save them
        if len(frames) == len(cameras):
            left_frame = frames["left"]
            right_frame = frames["right"]

            # Resize frames to have the same height for stitching display
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
            stitched_frame = np.concatenate([left_resized, right_resized], axis=1)

            # Display stitched feed
            cv2.imshow("Stitched Feed", stitched_frame)

            # Save frames every 1 second
            current_time = time.time()
            if current_time - last_capture_time >= 1:
                left_save_path = os.path.join(left_dataset_timestamp_dir, f"left_{frame_count:04d}.jpg")
                right_save_path = os.path.join(right_dataset_timestamp_dir, f"right_{frame_count:04d}.jpg")

                cv2.imwrite(left_save_path, left_frame)
                cv2.imwrite(right_save_path, right_frame)

                print(f"Saved left image: {left_save_path}")
                print(f"Saved right image: {right_save_path}")

                frame_count += 1
                last_capture_time = current_time

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    for camera in cameras:
        camera.StopGrabbing()
        camera.Close()
    cv2.destroyAllWindows()
