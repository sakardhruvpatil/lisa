import cv2
import time
from ultralytics import YOLO
import logging
import datetime
import numpy as np
from enum import Enum

# Define FSM States
class State(Enum):
    IDLE = 0
    TRACKING_SCANNING = 1
    TRACKING_DECIDED_NOT_CLEAN_PREMATURE = 2
    TRACKING_DECIDED_CLEAN = 3

# Set up logging with timestamps for unique filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"bedsheet_log_{timestamp}.txt"
defect_log_filename = f"defect_log_{timestamp}.txt"
defect_coordinates_log_filename = f"defect_coordinates_{timestamp}.txt"
bedsheet_areas_log_filename = f"bedsheet_areas_{timestamp}.txt"
defect_area_log_filename = f"defect_area_{timestamp}.txt"  # Unique defect area log filename
# Set up logging for total defect area
total_defect_area_log_filename = f"total_defect_area_{timestamp}.txt"
total_defect_area_log_file = open(
    total_defect_area_log_filename, "a"
)  # File to log total defect area for each bedsheet
# Set up logging for defect percent
defect_percent_log_filename = f"defect_percent_{timestamp}.txt"
defect_percent_log_file = open(
    defect_percent_log_filename, "a"
)  # File to log defect percentage for each bedsheet
# Set up logging for cleanliness analysis
input_clean_log_filename = f"realtime_clean_{timestamp}.txt"
input_clean_log_file = open(
    input_clean_log_filename, "a"
)  # File to log cleanliness analysis

# Configure the main log
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(message)s")

# Define a helper function to log and print simultaneously
def log_print(message):
    print(message)  # Print to console
    logging.info(message)  # Write to log file

# Define cleanliness threshold and default bedsheet area
CLEAN_THRESHOLD = 95.0  # Predefined cleanliness percentage
DEFAULT_BEDSHEET_AREA = 70000  # Predefined bedsheet area in pixels

# Load the trained YOLOv8 models
bedsheet_model = YOLO(
    "/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/bedsheet.pt"
)
defect_model = YOLO(
    "/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/defect.pt"
)

# Open the video file
video_path = (
    "/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/media/video001.avi"
)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    log_print("Error: Could not open video.")
    exit()

# Get video properties
video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
wait_time = int(1000 / video_fps)
original_width, original_height = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)
half_width, half_height = original_width // 2, original_height // 2

# Detection thresholds
conf_threshold = 0.8
defect_conf_threshold = 0.01

# Initialize state
state = State.IDLE

# State-related variables
display_fps, prev_time = 0, time.time()
bedsheet_count = 0  # Start counting from 0; increment after processing
unique_defect_ids = set()  # Track unique defect IDs across the bedsheet

# Open separate log files for bedsheet areas, defects, defect coordinates, and defect areas
log_file = open(bedsheet_areas_log_filename, "a")
defect_log_file = open(defect_log_filename, "a")
defect_coordinates_log_file = open(defect_coordinates_log_filename, "a")
# File to log defect areas
defect_area_log_file = open(defect_area_log_filename, "a")

# Dictionary to store maximum area of each unique defect ID
defect_max_areas = {}

# Initialize variables to track the visible bedsheet area
previous_y2 = None  # Tracks the lowest visible y2 in the current frame
total_bedsheet_area = 0
total_defect_area = 0  # Initialize total defect area

# Flags for state management
await_ending_edge = False  # Flag to await ending edge after premature decision
display_not_clean = False
ending_edge_detected = False  # Flag to prevent multiple ending edge detections
final_decision_made = False  # Flag to indicate if a final decision has been made

# Main processing loop with new FSM structure
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            log_print("End of video.")
            break

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (half_width, half_height))
        frame_height = frame_resized.shape[0]

        # Calculate display FPS
        current_time = time.time()
        display_fps = (
            1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        )
        prev_time = current_time

        # Initialize bedsheet presence flags for display
        bedsheet_present = False
        y1_positions = []
        y2_positions = []
        current_bbox = None

        # FSM Logic
        if state == State.IDLE:
            # Detect starting edge to transition from IDLE to TRACKING_SCANNING
            bedsheet_results = bedsheet_model.predict(source=frame_resized, conf=conf_threshold, verbose=False)

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

                            if y1 > frame_height * 0.75:  # Starting edge detected
                                state = State.TRACKING_SCANNING
                                previous_y2 = y2
                                total_bedsheet_area = 0
                                total_defect_area = 0
                                unique_defect_ids.clear()
                                defect_max_areas.clear()
                                await_ending_edge = False  # Reset await flag
                                display_not_clean = False
                                log_print("Transitioned to TRACKING_SCANNING: Starting edge detected.")
                                log_file.write(
                                    f"Bedsheet {bedsheet_count + 1}: Starting Edge Detected\n"
                                )
                                log_file.flush()
                                break  # Assuming one bedsheet per frame

            log_print("Bedsheet Present" if bedsheet_present else "Bedsheet Not Present")

        elif state == State.TRACKING_SCANNING:
            # Continue tracking and scanning
            # Detect defects and update cleanliness percentage
            defect_results = defect_model.track(
                source=frame_resized,
                conf=defect_conf_threshold,
                verbose=False,
                persist=True,
                tracker="/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/botsort_defect.yaml",
            )

            if defect_results:
                # Count defects only within the bedsheet region
                for defect_result in defect_results:
                    masks = defect_result.masks
                    tracks = (
                        defect_result.boxes.id.cpu().numpy()
                        if defect_result.boxes.id is not None
                        else None
                    )

                    if masks is not None and tracks is not None:
                        mask_array = masks.data
                        for j, mask in enumerate(mask_array):
                            defect_mask = mask.cpu().numpy()
                            defect_id = tracks[j]
                            defect_area = np.sum(defect_mask)  # Calculate defect area as the sum of mask pixels

                            # Track unique defect IDs for the current bedsheet
                            unique_defect_ids.add(defect_id)

                            # Check if this defect ID already exists in defect_max_areas
                            if defect_id in defect_max_areas:
                                # Only update if the new area is larger than the last maximum area
                                if defect_area > defect_max_areas[defect_id]:
                                    # Adjust total_defect_area to account for the increase
                                    total_defect_area += defect_area - defect_max_areas[defect_id]
                                    # Update the maximum area for this defect ID
                                    defect_max_areas[defect_id] = defect_area
                            else:
                                # New defect ID: add its area to total_defect_area and store it
                                defect_max_areas[defect_id] = defect_area
                                total_defect_area += defect_area

                            # Calculate real-time clean percent based on DEFAULT_BEDSHEET_AREA
                            defect_percent_real_time = (
                                total_defect_area / DEFAULT_BEDSHEET_AREA
                            ) * 100
                            clean_percent_real_time = 100 - defect_percent_real_time

                            # Check cleanliness threshold
                            if clean_percent_real_time < CLEAN_THRESHOLD:
                                state = State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                                await_ending_edge = True
                                display_not_clean = True

                                # Log cleanliness analysis
                                analysis_message = (
                                    f"Threshold: {CLEAN_THRESHOLD}%, "
                                    f"Bedsheet {bedsheet_count + 1}: Not Clean Prematurely. "
                                    f"Dirty Percent: {defect_percent_real_time:.2f}%, "
                                    f"Clean Percent: {clean_percent_real_time:.2f}%"
                                )
                                log_print(analysis_message)
                                input_clean_log_file.write(analysis_message + "\n")
                                input_clean_log_file.flush()

                                # Log defect percent and clean percent
                                log_print(
                                    f"Bedsheet {bedsheet_count + 1}: Total Defect Area = {total_defect_area}"
                                )
                                log_print(
                                    f"Bedsheet {bedsheet_count + 1}: Defect Percent = {defect_percent_real_time:.2f}%"
                                )
                                log_print(
                                    f"Bedsheet {bedsheet_count + 1}: Clean Percent = {clean_percent_real_time:.2f}%"
                                )
                                defect_percent_log_file.write(
                                    f"Bedsheet {bedsheet_count + 1}: Total Bedsheet Area = {DEFAULT_BEDSHEET_AREA}, "
                                    f"Total Defect Area = {total_defect_area}, Defect Percent = {defect_percent_real_time:.2f}%, "
                                    f"Clean Percent = {clean_percent_real_time:.2f}%\n"
                                )
                                defect_percent_log_file.flush()

                                # Log bedsheet area and total defect count
                                log_file.write(
                                    f"Bedsheet {bedsheet_count + 1}: Area = {total_bedsheet_area}\n"
                                )
                                defect_log_file.write(
                                    f"Bedsheet {bedsheet_count + 1}: Total Defects = {len(unique_defect_ids)}\n"
                                )
                                log_file.flush()
                                defect_log_file.flush()

                                log_print(f"Bedsheet {bedsheet_count + 1}: Area = {total_bedsheet_area}")
                                log_print(
                                    f"Bedsheet {bedsheet_count + 1}: Total Defects = {len(unique_defect_ids)}"
                                )

                                # Increment bedsheet count only because it's classified as "Not Clean"
                                bedsheet_count += 1

                                # Reset area calculations but continue tracking until ending edge
                                total_bedsheet_area = 0
                                total_defect_area = 0  # Reset total defect area
                                unique_defect_ids.clear()  # Clear tracked defects for the next bedsheet
                                defect_max_areas.clear()  # Reset defect area tracking for the next bedsheet

                                log_print("Ending Edge Detection Awaited for Not Clean Decision")
                                log_file.write(
                                    f"Bedsheet {bedsheet_count}: Ending Edge Detection Awaited for Not Clean Decision\n"
                                )
                                log_file.flush()

                                # **Important:** Break out of defect processing to avoid further detections in this frame
                                break

                            # **Draw Bounding Boxes Around Defects**
                            # Check if bounding box coordinates are available
                            if hasattr(defect_result.boxes, 'xyxy') and len(defect_result.boxes.xyxy) > j:
                                x1_d, y1_d, x2_d, y2_d = defect_result.boxes.xyxy[j].int().tolist()
                                # Draw rectangle around defect
                                cv2.rectangle(
                                    frame_resized,
                                    (x1_d, y1_d),
                                    (x2_d, y2_d),
                                    (0, 0, 255),  # Red color for defects
                                    2,
                                )
                                # Annotate defect ID
                                cv2.putText(
                                    frame_resized,
                                    f"ID: {defect_id}",
                                    (x1_d, y1_d - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    1,
                                )

                                # Log defect coordinates and area
                                defect_area_log_file.write(
                                    f"Bedsheet {bedsheet_count}: Defect ID {defect_id}: Coordinates ({x1_d}, {y1_d}, {x2_d}, {y2_d}) Area = {defect_area} pixels\n"
                                )
                                defect_coordinates_log_file.write(
                                    f"Bedsheet {bedsheet_count}: Defect ID {defect_id}: Coordinates ({x1_d}, {y1_d}, {x2_d}, {y2_d})\n"
                                )

                                log_print(
                                    f"Bedsheet {bedsheet_count}: Defect Present - Unique ID: {defect_id}: Coordinates ({x1_d}, {y1_d}, {x2_d}, {y2_d}) Area: {defect_area}"
                                )

            # Detect ending edge to transition to IDLE or other states
            bedsheet_results = bedsheet_model.predict(source=frame_resized, conf=conf_threshold, verbose=False)
            bedsheet_present = False
            y2_positions = []

            for result in bedsheet_results:
                if result.boxes:
                    boxes, classes, confidences = result.boxes.xyxy, result.boxes.cls, result.boxes.conf
                    for idx, class_id in enumerate(classes):
                        if int(class_id) == 0 and confidences[idx] > conf_threshold:
                            bedsheet_present = True
                            x1, y1, x2, y2 = map(int, boxes[idx])
                            y2_positions.append(y2)

            if y2_positions:
                y2_max = max(y2_positions)
                if y2_max < frame_height * 0.90:  # Ending edge detected
                    if state == State.TRACKING_SCANNING:
                        # Clean decision upon ending edge detection
                        defect_percent_real_time = (
                            total_defect_area / DEFAULT_BEDSHEET_AREA
                        ) * 100
                        clean_percent_real_time = 100 - defect_percent_real_time

                        if clean_percent_real_time >= CLEAN_THRESHOLD:
                            state = State.TRACKING_DECIDED_CLEAN
                            display_not_clean = False  # No need to display "Not Clean"

                            # Log cleanliness analysis
                            analysis_message = (
                                f"Threshold: {CLEAN_THRESHOLD}%, "
                                f"Bedsheet {bedsheet_count + 1}: Clean. "
                                f"Dirty Percent: {defect_percent_real_time:.2f}%, "
                                f"Clean Percent: {clean_percent_real_time:.2f}%"
                            )
                            log_print(analysis_message)
                            input_clean_log_file.write(analysis_message + "\n")
                            input_clean_log_file.flush()

                            # Log defect percent and clean percent
                            log_print(
                                f"Bedsheet {bedsheet_count + 1}: Total Defect Area = {total_defect_area}"
                            )
                            log_print(
                                f"Bedsheet {bedsheet_count + 1}: Defect Percent = {defect_percent_real_time:.2f}%"
                            )
                            log_print(
                                f"Bedsheet {bedsheet_count + 1}: Clean Percent = {clean_percent_real_time:.2f}%"
                            )
                            defect_percent_log_file.write(
                                f"Bedsheet {bedsheet_count + 1}: Total Bedsheet Area = {DEFAULT_BEDSHEET_AREA}, "
                                f"Total Defect Area = {total_defect_area}, Defect Percent = {defect_percent_real_time:.2f}%, "
                                f"Clean Percent = {clean_percent_real_time:.2f}%\n"
                            )
                            defect_percent_log_file.flush()

                            # Log bedsheet area and total defect count
                            log_file.write(
                                f"Bedsheet {bedsheet_count + 1}: Area = {total_bedsheet_area}\n"
                            )
                            defect_log_file.write(
                                f"Bedsheet {bedsheet_count + 1}: Total Defects = {len(unique_defect_ids)}\n"
                            )
                            log_file.flush()
                            defect_log_file.flush()

                            log_print(f"Bedsheet {bedsheet_count + 1}: Area = {total_bedsheet_area}")
                            log_print(
                                f"Bedsheet {bedsheet_count + 1}: Total Defects = {len(unique_defect_ids)}"
                            )

                            # Log cleanliness analysis to input_clean_log_file
                            input_clean_log_file.write(analysis_message + "\n")
                            input_clean_log_file.flush()

                            # Increment bedsheet count because it's classified as "Clean"
                            bedsheet_count += 1

                            # Reset area calculations but continue tracking until ending edge
                            total_bedsheet_area = 0
                            total_defect_area = 0  # Reset total defect area
                            unique_defect_ids.clear()  # Clear tracked defects for the next bedsheet
                            defect_max_areas.clear()  # Reset defect area tracking for the next bedsheet

                            log_print("Ending Edge Detected and Counted as Clean")
                            log_file.write(
                                f"Bedsheet {bedsheet_count}: Ending Edge Detected and Counted as Clean\n"
                            )
                            log_file.flush()

                        else:
                            # If clean percent is still below threshold upon ending edge
                            state = State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                            await_ending_edge = True
                            display_not_clean = True

                            # Log cleanliness analysis
                            analysis_message = (
                                f"Threshold: {CLEAN_THRESHOLD}%, "
                                f"Bedsheet {bedsheet_count + 1}: Not Clean at Ending Edge. "
                                f"Dirty Percent: {defect_percent_real_time:.2f}%, "
                                f"Clean Percent: {clean_percent_real_time:.2f}%"
                            )
                            log_print(analysis_message)
                            input_clean_log_file.write(analysis_message + "\n")
                            input_clean_log_file.flush()

                            # Log defect percent and clean percent
                            log_print(
                                f"Bedsheet {bedsheet_count + 1}: Total Defect Area = {total_defect_area}"
                            )
                            log_print(
                                f"Bedsheet {bedsheet_count + 1}: Defect Percent = {defect_percent_real_time:.2f}%"
                            )
                            log_print(
                                f"Bedsheet {bedsheet_count + 1}: Clean Percent = {clean_percent_real_time:.2f}%"
                            )
                            defect_percent_log_file.write(
                                f"Bedsheet {bedsheet_count + 1}: Total Bedsheet Area = {DEFAULT_BEDSHEET_AREA}, "
                                f"Total Defect Area = {total_defect_area}, Defect Percent = {defect_percent_real_time:.2f}%, "
                                f"Clean Percent = {clean_percent_real_time:.2f}%\n"
                            )
                            defect_percent_log_file.flush()

                            # Log bedsheet area and total defect count
                            log_file.write(
                                f"Bedsheet {bedsheet_count + 1}: Area = {total_bedsheet_area}\n"
                            )
                            defect_log_file.write(
                                f"Bedsheet {bedsheet_count + 1}: Total Defects = {len(unique_defect_ids)}\n"
                            )
                            log_file.flush()
                            defect_log_file.flush()

                            log_print(f"Bedsheet {bedsheet_count + 1}: Area = {total_bedsheet_area}")
                            log_print(
                                f"Bedsheet {bedsheet_count + 1}: Total Defects = {len(unique_defect_ids)}"
                            )

                            # Log cleanliness analysis to input_clean_log_file
                            input_clean_log_file.write(analysis_message + "\n")
                            input_clean_log_file.flush()

                            # Increment bedsheet count only because it's classified as "Not Clean"
                            bedsheet_count += 1

                            # Reset area calculations but continue tracking until ending edge
                            total_bedsheet_area = 0
                            total_defect_area = 0  # Reset total defect area
                            unique_defect_ids.clear()  # Clear tracked defects for the next bedsheet
                            defect_max_areas.clear()  # Reset defect area tracking for the next bedsheet

                            log_print("Ending Edge Detected and Counted as Not Clean")
                            log_file.write(
                                f"Bedsheet {bedsheet_count}: Ending Edge Detected and Counted as Not Clean\n"
                            )
                            log_file.flush()

            elif state == State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE:
                # Await ending edge detection
                bedsheet_results = bedsheet_model.predict(source=frame_resized, conf=conf_threshold, verbose=False)
                bedsheet_present = False
                y2_positions = []

                for result in bedsheet_results:
                    if result.boxes:
                        boxes, classes, confidences = result.boxes.xyxy, result.boxes.cls, result.boxes.conf
                        for idx, class_id in enumerate(classes):
                            if int(class_id) == 0 and confidences[idx] > conf_threshold:
                                bedsheet_present = True
                                x1, y1, x2, y2 = map(int, boxes[idx])
                                y2_positions.append(y2)

                if y2_positions:
                    y2_max = max(y2_positions)
                    if y2_max < frame_height * 0.90:  # Ending edge detected
                        state = State.IDLE
                        await_ending_edge = False
                        display_not_clean = False
                        ending_edge_detected = False  # Reset ending edge flag for next bedsheet
                        log_print("Transitioned to IDLE: Ending edge detected after Not Clean decision.")
                        log_file.write(
                            f"Bedsheet {bedsheet_count}: Ending Edge Detected after Not Clean Decision\n"
                        )
                        log_file.flush()

        # Display annotations on frame
        cv2.putText(
            frame_resized,
            f"Video FPS: {int(video_fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame_resized,
            f"Display FPS: {int(display_fps)}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame_resized,
            "Bedsheet Present" if bedsheet_present else "Bedsheet Not Present",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if bedsheet_present else (0, 0, 255),
            2,
        )
        cv2.putText(
            frame_resized,
            f"Total Defect Area: {total_defect_area}",
            (10, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )  # Display total defect area

        # Display defect percentage and clean percentage if active
        if state in [State.TRACKING_SCANNING, State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE]:
            # Calculate defect percent and clean percent
            if DEFAULT_BEDSHEET_AREA > 0:
                defect_percent = (total_defect_area / DEFAULT_BEDSHEET_AREA) * 100
                clean_percent = 100 - defect_percent
            else:
                defect_percent = 0.0
                clean_percent = 100.0

            # Display on frame
            cv2.putText(
                frame_resized,
                f"Defect Percent: {defect_percent:.2f}%",
                (10, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame_resized,
                f"Clean Percent: {clean_percent:.2f}%",
                (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Display cleanliness status if not already classified as Not Clean or Clean
            if state not in [
                State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE,
                State.TRACKING_DECIDED_CLEAN,
            ]:
                cv2.putText(
                    frame_resized,
                    f"Cleanliness: {'Clean' if clean_percent >= CLEAN_THRESHOLD else 'Not Clean'}",
                    (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if clean_percent >= CLEAN_THRESHOLD else (0, 0, 255),
                    2,
                )

        # Display Starting Edge if active
        if state == State.TRACKING_SCANNING and y1_positions:
            y1_min = min(y1_positions)
            if y1_min > frame_height * 0.05:
                cv2.putText(
                    frame_resized,
                    "Starting Edge",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )
                log_print("Starting Edge")

        # Display Ending Edge if active
        if state in [State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE, State.TRACKING_DECIDED_CLEAN]:
            cv2.putText(
                frame_resized,
                "Ending Edge",
                (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2,
            )
            log_print("Ending Edge")

        # Display Accumulated Area
        cv2.putText(
            frame_resized,
            f"Accumulated Area: {int(total_bedsheet_area)}",
            (10, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        # Show frame even when no bedsheet is detected
        cv2.imshow("Video with FPS and Detection Status", frame_resized)

        # Handle display of "Not Clean" message
        if display_not_clean and state == State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE:
            cv2.putText(
                frame_resized,
                "Cleanliness: Not Clean",
                (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            log_print("Cleanliness: Not Clean")
            input_clean_log_file.write("Cleanliness: Not Clean\n")
            input_clean_log_file.flush()
            # Transition to IDLE after logging
            state = State.IDLE
            display_not_clean = False
            await_ending_edge = False  # Reset await flag for next bedsheet

        # Handle display of "Clean" message
        if state == State.TRACKING_DECIDED_CLEAN:
            cv2.putText(
                frame_resized,
                "Cleanliness: Clean",
                (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            log_print("Cleanliness: Clean")
            input_clean_log_file.write("Cleanliness: Clean\n")
            input_clean_log_file.flush()
            # Transition to IDLE after logging
            state = State.IDLE
            await_ending_edge = False  # Reset await flag for next bedsheet

        # Exit if 'q' is pressed
        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break

except Exception as e:
    log_print(f"An error occurred: {e}")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    defect_log_file.close()
    defect_coordinates_log_file.close()
    defect_area_log_file.close()
    total_defect_area_log_file.close()
    defect_percent_log_file.close()
    input_clean_log_file.close()
