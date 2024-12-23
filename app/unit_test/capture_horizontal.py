import os
import time
from pypylon import pylon
import cv2
import numpy as np

# Absolute paths for PFS files
LEFT_CAMERA_PFS = "/home/dp/Documents/lisa/app/config/left_camera_config.pfs"
RIGHT_CAMERA_PFS = "/home/dp/Documents/lisa/app/config/right_camera_config.pfs"

# Directory for saving captures
base_captures_dir = "./captures"
timestamp = int(time.time())  # Current timestamp in seconds
current_capture_dir = os.path.join(base_captures_dir, f"run_{timestamp}")

# Create the directory for the current run
os.makedirs(current_capture_dir, exist_ok=True)

# List of camera IPs and their configurations
camera_configs = [
    {"ip": "192.168.1.20", "side": "left", "pfs": LEFT_CAMERA_PFS},
    {"ip": "192.168.1.10", "side": "right", "pfs": RIGHT_CAMERA_PFS}
]

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
        frames = []
        for camera in cameras:
            if camera.IsGrabbing():
                grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    # Convert image to OpenCV format
                    image = converter.Convert(grab_result)
                    frame = image.GetArray()
                    frames.append(frame)
                grab_result.Release()

        # If all frames are captured, concatenate and display
        if len(frames) == len(cameras):
            # Resize frames to have the same height
            min_height = min(frame.shape[0] for frame in frames)
            resized_frames = [
                cv2.resize(frame, (int(frame.shape[1] * min_height / frame.shape[0]), min_height))
                for frame in frames
            ]

            # Concatenate frames horizontally
            stitched_frame = np.concatenate(resized_frames, axis=1)

            # Display the stitched feed
            cv2.imshow("Stitched Feed", stitched_frame)

            # Save the stitched frame every 1 second
            current_time = time.time()
            if current_time - last_capture_time >= 1:  # Check if 1 second has passed
                save_path = os.path.join(current_capture_dir, f"capture_{frame_count:04d}.jpg")
                cv2.imwrite(save_path, stitched_frame)
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
