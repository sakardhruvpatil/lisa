import cv2
import time
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/bedsheet.pt')

# Open the video file
video_path = '/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/media/6times012_rotated.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video FPS and calculate wait time per frame
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps == 0:
    video_fps = 25  # Default FPS if unable to get FPS
wait_time = int(1000 / video_fps)  # milliseconds

# Initialize variables
display_fps = 0
prev_time = time.time()

# Get original frame dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate half dimensions for resizing
half_width = original_width // 2
half_height = original_height // 2

# Edge detection variables
conf_threshold = 0.8

# State variables
starting_edge_active = False
ending_edge_active = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (half_width, half_height))

    # Run YOLOv8 detection on the resized frame
    results = model.predict(source=frame_resized, conf=conf_threshold)

    # Initialize variables for this frame
    bedsheet_present = False

    y1_positions = []
    y2_positions = []

    for result in results:
        if result.boxes:
            boxes = result.boxes.xyxy  # Bounding boxes
            classes = result.boxes.cls  # Class indices
            confidences = result.boxes.conf  # Confidence scores
            num_detections = len(result.boxes)

            for idx in range(num_detections):
                class_id = int(classes[idx])
                conf = confidences[idx]

                if class_id == 0 and conf > conf_threshold:
                    bedsheet_present = True

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = boxes[idx]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    y1_positions.append(y1)
                    y2_positions.append(y2)

    frame_height = frame_resized.shape[0]

    if bedsheet_present and y1_positions and y2_positions:
        y1_min = min(y1_positions)  # Top edge of the bedsheet
        y2_max = max(y2_positions)  # Bottom edge of the bedsheet

        # Starting Edge Logic
        if not starting_edge_active:
            # Activate when top edge enters from bottom 25% of the frame
            if y1_min > frame_height * 0.75:
                starting_edge_active = True

        if starting_edge_active:
            if y1_min > frame_height * 0.05:
                # Continue printing "Starting Edge"
                print("Starting Edge")
            else:
                # Deactivate when top edge reaches the top 5% of the frame
                starting_edge_active = False

        # Ending Edge Logic
        if not ending_edge_active:
            # Activate when bottom edge crosses above bottom 25% of the frame
            if y2_max < frame_height * 0.95:
                ending_edge_active = True

        if ending_edge_active:
            if y2_max > frame_height * 0.05:
                # Continue printing "Ending Edge"
                print("Ending Edge")
            else:
                # Deactivate when bottom edge reaches the top 5% of the frame
                ending_edge_active = False

    else:
        # Reset states if bedsheet is not present
        starting_edge_active = False
        ending_edge_active = False

    # Print bedsheet presence
    if bedsheet_present:
        print("Bedsheet Present")
    else:
        print("Bedsheet Not Present")

    # Calculate display FPS
    current_time = time.time()
    display_fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Draw annotations on frame_resized
    # Display FPS information
    cv2.putText(frame_resized, f"Video FPS: {int(video_fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame_resized, f"Display FPS: {int(display_fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display bedsheet presence
    if bedsheet_present:
        cv2.putText(frame_resized, "Bedsheet Present", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame_resized, "Bedsheet Not Present", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display edge information
    if starting_edge_active:
        cv2.putText(frame_resized, "Starting Edge", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    if ending_edge_active:
        cv2.putText(frame_resized, "Ending Edge", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    annotated_frame_bedsheet = results[0].plot()

    # Display the resulting frame
    cv2.imshow('Video with FPS and Detection Status', annotated_frame_bedsheet)

    # Press 'q' to exit
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
