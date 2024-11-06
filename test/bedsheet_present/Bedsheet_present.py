import cv2
import time
from ultralytics import YOLO

# Load the trained YOLOv8 segmentation model (replace 'best.pt' with your model path)
model = YOLO('/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/bedsheet.pt')

# Open the AVI video file
video_path = '/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/media/6times012_rotated.mp4'
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

    # Resize the frame to half its original size before inference
    frame_resized = cv2.resize(frame, (half_width, half_height))

    # Run YOLOv8 detection on the resized frame
    results = model.predict(source=frame_resized)

    # Check if a bedsheet is detected (based on model classes or labels)
    bedsheet_present = False
    for result in results:
        if result.boxes and result.masks:  # Ensure boxes and masks are not None
            boxes = result.boxes.xyxy  # Bounding boxes
            classes = result.boxes.cls  # Class indices
            num_detections = len(result.boxes)
            for idx in range(num_detections):
                class_id = int(classes[idx])
                
                # Check if the detected class is 'bedsheet' (adjust class_id if necessary)
                if class_id == 0:  # Replace 0 with your bedsheet class index
                    bedsheet_present = True

                    # Draw bounding box on the resized frame
                    x1, y1, x2, y2 = map(int, boxes[idx])
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Print presence information
    if bedsheet_present:
        print("Bedsheet Present")
        cv2.putText(frame_resized, "Bedsheet Present", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print("Bedsheet Not Present")
        cv2.putText(frame_resized, "Bedsheet Not Present", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate display FPS
    current_time = time.time()
    display_fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display both Video FPS and Display FPS on the frame with yellow color
    cv2.putText(frame_resized, f"Video FPS: {int(video_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame_resized, f"Display FPS: {int(display_fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    annotated_frame_bedsheet = results[0].plot()

    # Display the resulting frame
    cv2.imshow('Video with FPS and Detection Status', annotated_frame_bedsheet)
    
    # Press 'q' to exit the video
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()
