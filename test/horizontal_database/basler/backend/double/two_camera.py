from pypylon import pylon
import cv2

# List of camera IPs
camera_ips = ["192.168.1.10", "192.168.1.20"]

# Initialize cameras
cameras = []
for ip in camera_ips:
    device_info = pylon.CDeviceInfo()
    device_info.SetIpAddress(ip)
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device_info))
    camera.Open()
    cameras.append(camera)

# Start grabbing on all cameras
for camera in cameras:
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Image converter for OpenCV
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

try:
    while True:
        for i, camera in enumerate(cameras):
            if camera.IsGrabbing():
                grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    # Convert image to OpenCV format
                    image = converter.Convert(grab_result)
                    frame = image.GetArray()
                    # Display the image
                    cv2.imshow(f"Camera {i+1}", frame)
                grab_result.Release()
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    for camera in cameras:
        camera.StopGrabbing()
        camera.Close()
    cv2.destroyAllWindows()
