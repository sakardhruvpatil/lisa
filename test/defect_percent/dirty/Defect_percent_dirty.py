import cv2
import time
from ultralytics import YOLO
import logging
import datetime
import numpy as np

# Set up logging with timestamps for unique filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"bedsheet_log_{timestamp}.txt"
defect_log_filename = f"defect_log_{timestamp}.txt"
defect_coordinates_log_filename = f"defect_coordinates_{timestamp}.txt"
bedsheet_areas_log_filename = f"bedsheet_areas_{timestamp}.txt"
defect_area_log_filename = f"defect_area_{timestamp}.txt"  # Unique defect area log filename
# Set up logging for total defect area
total_defect_area_log_filename = f"total_defect_area_{timestamp}.txt"
total_defect_area_log_file = open(total_defect_area_log_filename, 'a')  # File to log total defect area for each bedsheet
# Set up logging for defect percent
defect_percent_log_filename = f"defect_percent_{timestamp}.txt"
defect_percent_log_file = open(defect_percent_log_filename, 'a')  # File to log defect percentage for each bedsheet

# Configure the main log
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

# Define a helper function to log and print simultaneously
def log_print(message):
    print(message)          # Print to console
    logging.info(message)   # Write to log file

# Load the trained YOLOv8 models
bedsheet_model = YOLO('/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/bedsheet.pt')
defect_model = YOLO('/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/defect.pt')

# Open the video file
video_path = '/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/media/video001.avi'
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
total_bedsheet_area, bedsheet_count = 0, 0
starting_edge_active, ending_edge_active, bedsheet_processing_active = False, False, False
area_accumulating, prev_bbox = False, None
unique_defect_ids = set()  # Track unique defect IDs across the bedsheet

# Open separate log files for bedsheet areas, defects, defect coordinates, and defect areas
log_file = open(bedsheet_areas_log_filename, 'a')
defect_log_file = open(defect_log_filename, 'a')
defect_coordinates_log_file = open(defect_coordinates_log_filename, 'a')
defect_area_log_file = open(defect_area_log_filename, 'a')  # File to log defect areas

# Dictionary to store maximum area of each unique defect ID
defect_max_areas = {}

# Initialize variables to track the visible bedsheet area
total_bedsheet_area = 0
previous_y2 = None  # Tracks the lowest visible y2 in the current frame
tracking_active = False  # Flag to indicate if area tracking is active

# Initialize total defect area at the beginning
total_defect_area = 0

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

    # Process bedsheet detection results
    for result in bedsheet_results:
        if result.boxes:
            boxes, classes, confidences = result.boxes.xyxy, result.boxes.cls, result.boxes.conf
            for idx, class_id in enumerate(classes):
                if int(class_id) == 0 and confidences[idx] > conf_threshold:
                    bedsheet_present = True
                    x1, y1, x2, y2 = map(int, boxes[idx])
                    box_width = x2 - x1  # Width of the bounding box
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    y1_positions.append(y1)
                    y2_positions.append(y2)
                    current_bbox = (x1, y1, x2, y2)

                    # Start tracking when the starting edge is detected
                    if not tracking_active and y1 > frame_resized.shape[0] * 0.75:
                        tracking_active = True
                        previous_y2 = y2
                        log_print("Starting Edge Detected")
                        log_file.write(f"Bedsheet {bedsheet_count + 1}: Starting Edge Detected\n")
                        log_file.flush()

                    # If tracking is active, calculate new non-overlapping area
                    if tracking_active and y2 < previous_y2:
                        # Calculate the new segment height
                        new_segment_height = previous_y2 - y2
                        new_area = new_segment_height * box_width  # Use the bounding box width for area calculation
                        total_bedsheet_area += new_area
                        log_print(f"Added new area segment: Height = {new_segment_height}, Width = {box_width}, Area = {new_area}")
                        log_file.write(f"Bedsheet {bedsheet_count + 1}: Added new area segment: Height = {new_segment_height}, Width = {box_width}, Area = {new_area}\n")
                        log_file.flush()

                        # Update previous_y2
                        previous_y2 = y2

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
            log_file.write(f"Bedsheet {bedsheet_count + 1}: Starting Edge Detected\n")
            log_file.flush()

        if starting_edge_active and y1_min <= frame_height * 0.05:
            starting_edge_active = False

        # Detect the ending edge of the bedsheet as soon as it appears
        # Adjusted threshold to 0.90 for better detection
        if tracking_active and not ending_edge_active and y2_max < frame_height * 0.90:
            ending_edge_active, area_accumulating = True, False
            log_print(f"Ending Edge Detected. y2_max: {y2_max}, frame_height: {frame_height}")
            log_file.write(f"Ending Edge Detected. y2_max: {y2_max}, frame_height: {frame_height}\n")
            log_file.flush()

            log_print(f"Total Bedsheet Area at Ending Edge Appearance: {total_bedsheet_area}")

            # Calculate defect percent
            if total_bedsheet_area > 0:
                defect_percent = (total_defect_area / total_bedsheet_area) * 100
            else:
                defect_percent = 0.0

            # Log defect percent
            log_print(f"Bedsheet {bedsheet_count + 1}: Total Defect Area = {total_defect_area}")
            log_print(f"Bedsheet {bedsheet_count + 1}: Defect Percent = {defect_percent:.2f}%")
            total_defect_area_log_file.write(f"Bedsheet {bedsheet_count + 1}: Total Defect Area = {total_defect_area}\n")
            defect_percent_log_file.write(f"Bedsheet {bedsheet_count + 1}: Total Bedsheet Area = {total_bedsheet_area}, Total Defect Area = {total_defect_area}, Defect Percent = {defect_percent:.2f}%\n")
            defect_percent_log_file.flush()

            # Log bedsheet area and total defect count
            log_file.write(f"Bedsheet {bedsheet_count + 1}: Area = {total_bedsheet_area}\n")
            defect_log_file.write(f"Bedsheet {bedsheet_count + 1}: Total Defects = {len(unique_defect_ids)}\n")
            log_file.flush()
            defect_log_file.flush()

            log_print(f"Bedsheet {bedsheet_count + 1}: Area = {total_bedsheet_area}")
            log_print(f"Bedsheet {bedsheet_count + 1}: Total Defects = {len(unique_defect_ids)}")

            # Reset values for the next bedsheet
            bedsheet_count += 1
            total_bedsheet_area, prev_bbox = 0, None
            total_defect_area = 0  # Reset total defect area
            ending_edge_active, bedsheet_processing_active = False, False
            unique_defect_ids.clear()  # Clear tracked defects for the next bedsheet
            defect_max_areas.clear()   # Reset defect area tracking for the next bedsheet
            tracking_active = False    # Reset tracking
            log_print("Ending Edge Detected and Counted")
            log_file.write(f"Bedsheet {bedsheet_count}: Ending Edge Detected and Counted\n")
            log_file.flush()

    # Defect detection and tracking logic for the full frame
    if bedsheet_present and tracking_active:
        # Track defects in the full resized frame
        defect_results = defect_model.track(
            source=frame_resized,
            conf=defect_conf_threshold,
            verbose=False,
            persist=True,
            tracker="/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/botsort_defect.yaml"
        )

        if defect_results:
            # Count defects only within the bedsheet region
            for defect_result in defect_results:
                masks = defect_result.masks
                tracks = defect_result.boxes.id.cpu().numpy() if defect_result.boxes.id is not None else None

                if masks is not None and tracks is not None:
                    mask_array = masks.data
                    for j, mask in enumerate(mask_array):
                        defect_mask = mask.cpu().numpy()
                        defect_id = tracks[j]
                        defect_area = np.sum(defect_mask)  # Calculate defect area as the sum of mask pixels

                        # Track unique defect IDs for the current bedsheet
                        unique_defect_ids.add(defect_id)

                        # Update maximum area for each defect ID
                        if defect_id in defect_max_areas:
                            defect_max_areas[defect_id] = max(defect_max_areas[defect_id], defect_area)
                        else:
                            defect_max_areas[defect_id] = defect_area

                        # Update the running total defect area
                        total_defect_area = sum(defect_max_areas.values())

                        # Log the defect coordinates and area with bedsheet ID
                        x1_d, y1_d, x2_d, y2_d = defect_result.boxes.xyxy[j].int().tolist()
                        defect_area_log_file.write(
                            f"Bedsheet {bedsheet_count + 1} - Defect ID {defect_id}: Coordinates ({x1_d}, {y1_d}, {x2_d}, {y2_d}): Area = {defect_area} pixels\n"
                        )
                        defect_coordinates_log_file.write(
                            f"Bedsheet {bedsheet_count + 1} - Defect ID {defect_id}: Coordinates ({x1_d}, {y1_d}, {x2_d}, {y2_d})\n"
                        )

                        cv2.rectangle(frame_resized, (x1_d, y1_d), (x2_d, y2_d), (0, 0, 255), 2)
                        log_print(
                            f"Bedsheet {bedsheet_count + 1} - Defects Present - Unique ID: {defect_id}: Coordinates ({x1_d}, {y1_d}, {x2_d}, {y2_d}) Area: {defect_area}"
                        )

    # Display annotations on frame
    cv2.putText(frame_resized, f"Video FPS: {int(video_fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame_resized, f"Display FPS: {int(display_fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame_resized,
                "Bedsheet Present" if bedsheet_present else "Bedsheet Not Present",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if bedsheet_present else (0, 0, 255),
                2)
    cv2.putText(frame_resized, f"Total Defect Area: {total_defect_area}", (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Display total defect area

    # Display defect percentage if ending edge has been detected for the current bedsheet
    if ending_edge_active:
        # Calculate defect percent
        if total_bedsheet_area > 0:
            defect_percent = (total_defect_area / total_bedsheet_area) * 100
        else:
            defect_percent = 0.0
        cv2.putText(frame_resized, f"Defect Percent: {defect_percent:.2f}%", (10, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        log_print(f"Defect Percent: {defect_percent:.2f}%")

    if starting_edge_active:
        cv2.putText(frame_resized, "Starting Edge", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        log_print("Starting Edge")

    if ending_edge_active:
        cv2.putText(frame_resized, "Ending Edge", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        log_print("Ending Edge")

    cv2.putText(frame_resized, f"Accumulated Area: {int(total_bedsheet_area)}", (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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
defect_coordinates_log_file.close()
defect_area_log_file.close()
total_defect_area_log_file.close()
defect_percent_log_file.close()
