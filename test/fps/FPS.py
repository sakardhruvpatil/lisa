import cv2
import time

# Open the AVI video file
video_path = 'test/media/video012.avi'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video's FPS and calculate the wait time per frame
video_fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / video_fps)  # milliseconds

# Initialize variables
display_fps = 0
prev_time = 0

# Get the original dimensions of the video frames
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate half dimensions
half_width = original_width // 2
half_height = original_height // 2

# Loop to read frames from the video
while cap.isOpened():
    ret, frame = cap.read()
    
    # Break the loop if there are no frames left
    if not ret:
        print("End of video.")
        break
    
    # Calculate display FPS
    current_time = time.time()
    display_fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Display both Video FPS and Display FPS on the frame with yellow color
    cv2.putText(frame, f"Video FPS: {int(video_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Display FPS: {int(display_fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Resize the frame to half its original size
    frame_resized = cv2.resize(frame, (half_width, half_height))
    
    # Display the resulting frame
    cv2.imshow('Video with FPS (Resized)', frame_resized)
    
    # Press 'q' to exit the video
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()
