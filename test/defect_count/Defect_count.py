import cv2
import time
from ultralytics import YOLO
import logging
import datetime

# Set up logging to a new file with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"bedsheet_log_{timestamp}.txt"
defect_log_filename = f"defect_log_{timestamp}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

# Define a helper function to log and print simultaneously
def log_print(message):
    print(message)          # Print to console
    logging.info(message)   # Write to log file

# Load the trained YOLOv8 models
bedsheet_model = YOLO('/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/bedsheet.pt')
defect_model = YOLO('/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/defect.pt')

# Open the video file
video_path = '/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/media/6times002.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    log_print("Error: Could not open video.")
    exit()

# Get video properties
video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
wait_time = int(1000 / video_fps)
original_width, original_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
half_width, half_height = original_width // 2, original_height // 2

# Detection thresholds
conf_threshold = 0.8
defect_conf_threshold = 0.5

# State and area calculation variables
display_fps, prev_time = 0, time.time()
total_bedsheet_area, bedsheet_count, defect_count = 0, 0, 0
starting_edge_active, ending_edge_active, bedsheet_processing_active = False, False, False
area_accumulating, prev_bbox = False, None

# Open separate log files for bedsheet areas and defects
log_file_path = 'bedsheet_areas.txt'
log_file = open(log_file_path, 'a')
defect_log_file = open(defect_log_filename, 'a')

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        log_print("End of video.")
        break

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (half_width, half_height))

    # Run YOLOv8 detection for bedsheet
    bedsheet_results = bedsheet_model.predict(source=frame_resized, conf=conf_threshold, verbose=False)
    bedsheet_present, y1_positions, y2_positions, current_bbox = False, [], [], None
    unique_defect_ids = set()  # Reset defect IDs for each bedsheet cycle

    # Process bedsheet detection results
    for result in bedsheet_results:
        if result.boxes:
            boxes, classes, confidences = result.boxes.xyxy, result.boxes.cls, result.boxes.conf
            for idx, class_id in enumerate(classes):
                if int(class_id) == 0 and confidences[idx] > conf_threshold:
                    bedsheet_present = True
                    x1, y1, x2, y2 = map(int, boxes[idx])
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    y1_positions.append(y1)
                    y2_positions.append(y2)
                    current_bbox = (x1, y1, x2, y2)

    log_print("Bedsheet Present" if bedsheet_present else "Bedsheet Not Present")

    # Calculate display FPS
    current_time = time.time()
    display_fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Edge detection and area calculation
    if bedsheet_present and y1_positions and y2_positions:
        y1_min, y2_max = min(y1_positions), max(y2_positions)
        frame_height = frame_resized.shape[0]

        if not bedsheet_processing_active and y1_min > frame_height * 0.75:
            starting_edge_active, area_accumulating, bedsheet_processing_active = True, True, True
            log_print("Starting Edge Detected")

        if starting_edge_active and y1_min <= frame_height * 0.05:
            starting_edge_active = False

        # Detect the ending edge of the bedsheet as soon as it appears
        if bedsheet_processing_active and not ending_edge_active and y2_max < frame_height * 0.75:
            ending_edge_active, area_accumulating = True, False
            log_print(f"Total Bedsheet Area at Ending Edge Appearance: {total_bedsheet_area}")

            # Log and reset bedsheet count and defect details at the ending edge detection
            bedsheet_count += 1
            log_file.write(f"Bedsheet {bedsheet_count}: Area = {total_bedsheet_area}\n")
            log_file.flush()
            log_print(f"Bedsheet {bedsheet_count}: Area = {total_bedsheet_area}")
            defect_log_file.write(f"Bedsheet {bedsheet_count}: Total Defects = {defect_count} (IDs: {list(unique_defect_ids)})\n")
            defect_log_file.flush()
            log_print(f"Bedsheet {bedsheet_count}: Total Defects = {defect_count} (IDs: {list(unique_defect_ids)})")
            
            # Reset values for next bedsheet
            total_bedsheet_area, prev_bbox = 0, None
            ending_edge_active, bedsheet_processing_active = False, False
            defect_count = 0  # Reset defect count for the next bedsheet
            log_print("Ending Edge Detected and Counted")

        if area_accumulating and current_bbox:
            x1, y1, x2, y2 = current_bbox
            current_area = (x2 - x1) * (y2 - y1)
            if prev_bbox:
                x1_prev, y1_prev, x2_prev, y2_prev = prev_bbox
                x_left, y_top = max(x1, x1_prev), max(y1, y1_prev)
                x_right, y_bottom = min(x2, x2_prev), min(y2, y2_prev)
                overlap_area = (x_right - x_left) * (y_bottom - y_top) if x_right > x_left and y_bottom > y_top else 0
                new_area = max(current_area - overlap_area, 0)
            else:
                new_area = current_area
            total_bedsheet_area += new_area
            prev_bbox = current_bbox

    # Defect detection and tracking logic for the full frame
    if bedsheet_present:
        # Track defects in the full resized frame
        defect_results = defect_model.track(source=frame_resized, conf=defect_conf_threshold, verbose=False, persist=True, tracker="/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/botsort_defect.yaml")
        
        if defect_results:
            # Count defects only within the bedsheet region
            for defect_result in defect_results:
                if defect_result.boxes and defect_result.boxes.id is not None:
                    for box, track_id in zip(defect_result.boxes.xyxy, defect_result.boxes.id):
                        x1_d, y1_d, x2_d, y2_d = box.int().tolist()
                        # Check if defect is within the bedsheet bounding box
                        if current_bbox and x1 <= x1_d <= x2 and y1 <= y1_d <= y2:
                            unique_defect_ids.add(track_id.item())  # Track unique defect IDs
                            defect_count += 1  # Increment defect count for current bedsheet
                            cv2.rectangle(frame_resized, (x1_d, y1_d), (x2_d, y2_d), (0, 0, 255), 2)
                    log_print(f"Defects Present - Unique IDs: {list(unique_defect_ids)}")

    # Display annotations on frame
    cv2.putText(frame_resized, f"Video FPS: {int(video_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame_resized, f"Display FPS: {int(display_fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame_resized, "Bedsheet Present" if bedsheet_present else "Bedsheet Not Present", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if bedsheet_present else (0, 0, 255), 2)
    if starting_edge_active:
        cv2.putText(frame_resized, "Starting Edge", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        log_print("Starting Edge")

    if ending_edge_active:
        cv2.putText(frame_resized, "Ending Edge", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        log_print("Ending Edge")

    cv2.putText(frame_resized, f"Accumulated Area: {int(total_bedsheet_area)}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame even when no bedsheet is detected
    cv2.imshow('Video with FPS and Detection Status', frame_resized)

    # Exit if 'q' is pressed
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
log_file.close()
defect_log_file.close()
