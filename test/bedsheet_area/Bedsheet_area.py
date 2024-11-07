import cv2
import time
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/bedsheet.pt')

# Open the video file
video_path = '/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/media/6times001.mp4'
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
bedsheet_processing_active = False  # To manage processing state

# Area calculation variables
total_bedsheet_area = 0
prev_bbox = None  # Previous bounding box
area_accumulating = False

# Initialize bedsheet counter
bedsheet_count = 0

# Open the text file in append mode to log bedsheet areas
log_file_path = 'bedsheet_areas.txt'
try:
    log_file = open(log_file_path, 'a')  # 'a' mode appends to the file
except IOError:
    print(f"Error: Could not open or create {log_file_path}.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (half_width, half_height))

    # Run YOLOv8 detection on the resized frame
    results = model.predict(source=frame_resized, conf=conf_threshold, verbose=False)

    # Initialize variables for this frame
    bedsheet_present = False

    y1_positions = []
    y2_positions = []
    current_bbox = None

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

                    current_bbox = (x1, y1, x2, y2)

    frame_height = frame_resized.shape[0]

    if bedsheet_present and y1_positions and y2_positions:
        y1_min = min(y1_positions)  # Top edge of the bedsheet
        y2_max = max(y2_positions)  # Bottom edge of the bedsheet

        # Debug: Print y2_max and frame_height
        print(f"y2_max: {y2_max}, frame_height: {frame_height}")

        if not bedsheet_processing_active:
            # Starting Edge Logic
            if y1_min > frame_height * 0.75:
                starting_edge_active = True
                area_accumulating = True  # Start accumulating area
                bedsheet_processing_active = True  # Indicate that processing has started
                print("Starting Edge Detected")

        if starting_edge_active:
            if y1_min > frame_height * 0.05:
                # Continue printing "Starting Edge"
                print("Starting Edge")
            else:
                # Deactivate when top edge reaches the top 5% of the frame
                starting_edge_active = False

        if bedsheet_processing_active and not ending_edge_active:
            # Ending Edge Logic
            if y2_max < frame_height * 0.75:  # Trigger when bottom edge crosses above 75%
                ending_edge_active = True
                area_accumulating = False  # Stop accumulating area
                print("Ending Edge Detected")
                # Print total area when ending edge appears
                print(f"Total Bedsheet Area at Ending Edge Appearance: {total_bedsheet_area}")

        if ending_edge_active:
            if y2_max > frame_height * 0.25:
                # Continue printing "Ending Edge"
                print("Ending Edge")
            else:
                # Deactivate when bottom edge reaches the top 5% of the frame
                ending_edge_active = False
                # Increment bedsheet counter
                bedsheet_count += 1
                # Log the bedsheet area to the text file
                log_entry = f"Bedsheet {bedsheet_count}: Area = {total_bedsheet_area}\n"
                try:
                    log_file.write(log_entry)
                except IOError:
                    print(f"Error: Could not write to {log_file_path}.")
                # Print to terminal
                print(f"Bedsheet {bedsheet_count}: Area = {total_bedsheet_area}")
                # Reset the total area
                total_bedsheet_area = 0
                prev_bbox = None  # Reset previous bounding box
                bedsheet_processing_active = False  # Reset processing flag
                print("Ending Edge Reached Top")
                print("Total Bedsheet Area Reset")

        # Area Calculation
        if area_accumulating and current_bbox is not None:
            x1, y1, x2, y2 = current_bbox
            current_area = (x2 - x1) * (y2 - y1)

            if prev_bbox is not None:
                # Calculate overlap between current_bbox and prev_bbox
                x1_prev, y1_prev, x2_prev, y2_prev = prev_bbox

                # Find coordinates of the intersection rectangle
                x_left = max(x1, x1_prev)
                y_top = max(y1, y1_prev)
                x_right = min(x2, x2_prev)
                y_bottom = min(y2, y2_prev)

                if x_right < x_left or y_bottom < y_top:
                    # No overlap
                    overlap_area = 0
                else:
                    overlap_area = (x_right - x_left) * (y_bottom - y_top)

                # New area is current area minus overlap area
                new_area = current_area - overlap_area

                if new_area < 0:
                    new_area = 0  # Avoid negative area due to numerical errors
            else:
                # First frame, no previous bbox
                new_area = current_area

            total_bedsheet_area += new_area
            prev_bbox = current_bbox  # Update previous bounding box

            # Debug: Print accumulated area
            print(f"Accumulated Area: {total_bedsheet_area}")

    else:
        # Reset states if bedsheet is not present
        starting_edge_active = False
        ending_edge_active = False
        area_accumulating = False
        prev_bbox = None
        bedsheet_processing_active = False
        total_bedsheet_area = 0  # Ensure area is reset when bedsheet is not present

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

    # Display the accumulated area
    cv2.putText(frame_resized, f"Accumulated Area: {int(total_bedsheet_area)}", (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video with FPS and Detection Status', frame_resized)

    # Press 'q' to exit
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Close the log file
log_file.close()
