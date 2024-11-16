# main.py

import signal
import sys
import cv2
from datetime import datetime, date, timezone, timedelta
from config import *
from database import connect_to_mongo, get_daily_collection, get_last_bedsheet_number, log_to_mongo, log_threshold_change, add_threshold_entry, get_current_date_str, update_history, initialize_history_document
from logger import log_bug, log_print
from video_processing import initialize_video_capture, release_video_resources
from models_and_states import State, bedsheet_model, defect_model
from csv_saving import save_collection_to_csv, save_history_to_csv
import time
import pytz
import numpy as np
import pandas as pd

# Global variables for video capture
cap = None

# Signal handler for graceful exit
def signal_handler(sig, frame):
    log_print("Interrupt received. Exiting gracefully...")
    release_video_resources(cap)
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Helper function to reset defect tracking variables
def reset_defect_tracking_variables():
    global total_defect_area, unique_defect_ids, defect_max_areas
    total_defect_area = 0
    unique_defect_ids.clear()
    defect_max_areas.clear()

# Initialize the script
def initialize():
    global db, collection, threshold_collection, history_collection
    global bedsheet_count, CLEAN_THRESHOLD

    # Connect to MongoDB
    try:
        db = connect_to_mongo()
        threshold_collection = db['threshold_changes']
        history_collection = db['history']
        collection = get_daily_collection(db)
    except SystemExit as e:
        log_bug(f"System exit triggered: {e}")
        sys.exit(1)

    # Fetch the last bedsheet number from the logs
    bedsheet_count = get_last_bedsheet_number(collection)

    # Initialize cleanliness threshold from the database
    CLEAN_THRESHOLD = get_last_threshold()

    # Initialize history document
    current_date = get_current_date_str()
    initialize_history_document(history_collection, current_date, CLEAN_THRESHOLD)

# Function to get the last threshold or set default
def get_last_threshold():
    # Fetch the last threshold entry
    last_entry = threshold_collection.find_one(sort=[("timestamp", -1)])
    if last_entry and "set_threshold" in last_entry:
        return last_entry["set_threshold"]
    else:
        # If no threshold is found, use default and log it
        threshold_value = DEFAULT_THRESHOLD
        threshold_changed = log_threshold_change(threshold_collection, threshold_value)
        if threshold_changed:
            add_threshold_entry(history_collection, get_current_date_str(), threshold_value)
        return threshold_value

# Trackbar callback to update threshold in real-time and log changes
def update_threshold(val):
    global CLEAN_THRESHOLD
    threshold_changed = log_threshold_change(threshold_collection, val)  # Log and check for change
    if threshold_changed:
        CLEAN_THRESHOLD = val
        initialize_history_document(history_collection, get_current_date_str(), CLEAN_THRESHOLD)  # Initialize history only on change
        log_print("Clean Threshold changed to {}".format(CLEAN_THRESHOLD))

# Main function
def main():
    global cap, bedsheet_count, CLEAN_THRESHOLD
    global total_defect_area, unique_defect_ids, defect_max_areas
    global await_ending_edge, display_not_clean, defect_tracking_error

    # Initialize variables
    total_defect_area = 0  # Initialize total defect area
    unique_defect_ids = set()  # Track unique defect IDs across the bedsheet
    defect_max_areas = {}  # Dictionary to store maximum area of each unique defect ID

    await_ending_edge = False  # Flag to await ending edge after premature decision
    display_not_clean = False
    defect_tracking_error = False

    # Initialize MongoDB and other settings
    initialize()

    # Initialize video capture
    try:
        cap = initialize_video_capture()
    except Exception as e:
        log_bug(f"Video capture initialization failed. Exception: {e}")
        sys.exit(1)

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    wait_time = int(1000 / video_fps)
    original_width, original_height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    half_width, half_height = original_width // 2, original_height // 2
    resized_dimensions = (half_width, half_height)
    original_dimensions = (original_width, original_height)
    # Initialize the display window and trackbar
    cv2.namedWindow("Video with FPS and Detection Status")
    cv2.createTrackbar("Clean Threshold", "Video with FPS and Detection Status", int(CLEAN_THRESHOLD), 100, update_threshold)

    # Initialize state
    state = State.IDLE

    # Main loop
    try:
        error_occurred = False  # Flag to track errors
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    log_bug("Failed to read frame from camera.")
                    break

                # Resize frame for faster processing
                frame_resized = cv2.resize(frame, original_dimensions)
                frame_height = frame_resized.shape[0]

                # Initialize bedsheet presence flags for display
                bedsheet_present = False
                y1_positions = []
                y2_positions = []

                # FSM Logic
                if state == State.IDLE:
                    if bedsheet_model:  # Check if bedsheet_model is loaded
                        try:
                            # Detect starting edge to transition from IDLE to TRACKING_SCANNING
                            bedsheet_results = bedsheet_model.predict(source=frame_resized, conf=CONF_THRESHOLD, verbose=False)

                            for result in bedsheet_results:
                                if result.boxes:
                                    boxes, classes, confidences = result.boxes.xyxy, result.boxes.cls, result.boxes.conf
                                    for idx, class_id in enumerate(classes):
                                        if int(class_id) == 0 and confidences[idx] > CONF_THRESHOLD:
                                            bedsheet_present = True
                                            x1, y1, x2, y2 = map(int, boxes[idx])
                                            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            y1_positions.append(y1)
                                            y2_positions.append(y2)

                                            if y1 > frame_height * 0.75:  # Starting edge detected
                                                state = State.TRACKING_SCANNING
                                                reset_defect_tracking_variables()
                                                await_ending_edge = False  # Reset await flag
                                                display_not_clean = False
                                                log_print("Transitioned to TRACKING_SCANNING: Starting edge detected.")
                                                break  # Assuming one bedsheet per frame

                            log_print("Bedsheet Present" if bedsheet_present else "Bedsheet Not Present")
                        except Exception as e:
                            log_bug(f"Error during bedsheet detection. Exception: {e}")
                            log_print("Skipping bedsheet detection due to an error.")
                    else:
                        log_print("Bedsheet detection skipped. Model not loaded.")

                elif state == State.TRACKING_SCANNING and defect_model:
                    if defect_model:  # Check if defect_model is loaded                
                        # Handle defect tracking only if no error occurred
                        if not defect_tracking_error:
                            try:
                                # Perform defect tracking
                                defect_results = defect_model.track(
                                    source=frame_resized,
                                    conf=DEFECT_CONF_THRESHOLD,
                                    verbose=False,
                                    persist=True,
                                    tracker=TRACKER_PATH,
                                )
                            except Exception as e:
                                defect_tracking_error = True
                                defect_results = None  # Ensure defect_results is defined
                                log_bug(f"Defect tracking error occurred. Exception: {e}")
                                log_print("Skipping defect detection due to an error. Feed will continue running.")
                        else:
                            defect_results = None  # Ensure defect_results is defined
                            log_print("Skipping defect detection as an error was previously encountered.")
                    else:
                        defect_results = None  # Ensure defect_results is defined
                        log_print("Defect detection skipped. Model not loaded.")
                        
                    if defect_results and (state == State.TRACKING_SCANNING):
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
                                    # del defect_mask                    

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

                                    # Immediate rejection if dirty percentage exceeds 100%
                                    if defect_percent_real_time >= 100:
                                        state = State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                                        log_print(f"Bedsheet {bedsheet_count + 1}: Rejected due to defect percent >= 100%")
                                        decision = "Rejected"
                                        log_to_mongo(collection, bedsheet_count + 1, 100, CLEAN_THRESHOLD, decision)
                                        update_history(history_collection, get_current_date_str(), CLEAN_THRESHOLD, decision)
                                        log_print(f"Bedsheet {bedsheet_count + 1} logged as 'Not Clean'")
                                        bedsheet_count += 1
                                        reset_defect_tracking_variables()

                                        # Transition to IDLE after logging
                                        state = State.IDLE
                                        log_print("Transitioned to IDLE after rejection due to defect percent >= 100%")
                                        break  # Exit further processing for this frame

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

                                        # Decision for "Not Clean"
                                        decision = "Rejected"
                                        log_to_mongo(collection, bedsheet_count + 1, defect_percent_real_time, CLEAN_THRESHOLD, decision)
                                        update_history(history_collection, get_current_date_str(), CLEAN_THRESHOLD, decision)  # Update history here
                                        log_print(f"Bedsheet {bedsheet_count + 1} logged as 'Not Clean'")
                                        bedsheet_count += 1  # Increment bedsheet number

                                        # Reset area calculations but continue tracking until ending edge
                                        reset_defect_tracking_variables()

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

                # Detect ending edge to transition to IDLE or other states
                if state == State.TRACKING_SCANNING:
                    bedsheet_results = bedsheet_model.predict(source=frame_resized, conf=CONF_THRESHOLD, verbose=False)
                    bedsheet_present = False
                    y2_positions = []

                    for result in bedsheet_results:
                        if result.boxes:
                            boxes, classes, confidences = result.boxes.xyxy, result.boxes.cls, result.boxes.conf
                            for idx, class_id in enumerate(classes):
                                if int(class_id) == 0 and confidences[idx] > CONF_THRESHOLD:
                                    bedsheet_present = True
                                    x1, y1, x2, y2 = map(int, boxes[idx])
                                    y2_positions.append(y2)

                    if y2_positions:
                        y2_max = max(y2_positions)
                        if y2_max < frame_height * 0.90:  # Ending edge detected
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

                                # Decision for "Clean"
                                decision = "Accepted"
                                log_to_mongo(collection, bedsheet_count + 1, defect_percent_real_time, CLEAN_THRESHOLD, decision)
                                update_history(history_collection, get_current_date_str(), CLEAN_THRESHOLD, decision)  # Update history here
                                log_print(f"Bedsheet {bedsheet_count + 1} logged as 'Clean'")
                                bedsheet_count += 1  # Increment bedsheet number

                                # Reset area calculations but continue tracking until ending edge
                                reset_defect_tracking_variables()

                                log_print("Ending Edge Detected and Counted as Clean")

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

                                # Decision for "Not Clean"
                                decision = "Rejected"
                                log_to_mongo(collection, bedsheet_count + 1, defect_percent_real_time, CLEAN_THRESHOLD, decision)
                                update_history(history_collection, get_current_date_str(), CLEAN_THRESHOLD, decision)  # Update history here
                                log_print(f"Bedsheet {bedsheet_count + 1} logged as 'Not Clean'")
                                bedsheet_count += 1  # Increment bedsheet number

                                # Reset area calculations but continue tracking until ending edge
                                reset_defect_tracking_variables()

                elif state == State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE:
                    # Await ending edge detection
                    bedsheet_results = bedsheet_model.predict(source=frame_resized, conf=CONF_THRESHOLD, verbose=False)
                    bedsheet_present = False
                    y2_positions = []

                    for result in bedsheet_results:
                        if result.boxes:
                            boxes, classes, confidences = result.boxes.xyxy, result.boxes.cls, result.boxes.conf
                            for idx, class_id in enumerate(classes):
                                if int(class_id) == 0 and confidences[idx] > CONF_THRESHOLD:
                                    bedsheet_present = True
                                    x1, y1, x2, y2 = map(int, boxes[idx])
                                    y2_positions.append(y2)

                    if y2_positions:
                        y2_max = max(y2_positions)
                        if y2_max < frame_height * 0.90:  # Ending edge detected
                            state = State.IDLE
                            await_ending_edge = False
                            display_not_clean = False
                            log_print("Transitioned to IDLE: Ending edge detected after Not Clean decision.")

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
                        log_print("Starting Edge")

                # Display Ending Edge if active
                if state in [State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE, State.TRACKING_DECIDED_CLEAN]:
                    log_print("Ending Edge")

                # Show frame even when no bedsheet is detected
                cv2.imshow("Video with FPS and Detection Status", frame_resized)

                # Handle display of "Not Clean" message
                if display_not_clean and state == State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE:
                    log_print("Cleanliness: Not Clean")
                    # Transition to IDLE after logging
                    state = State.IDLE
                    display_not_clean = False
                    await_ending_edge = False  # Reset await flag for next bedsheet

                # Handle display of "Clean" message
                if state == State.TRACKING_DECIDED_CLEAN:
                    log_print("Cleanliness: Clean")
                    # Transition to IDLE after logging
                    state = State.IDLE
                    await_ending_edge = False  # Reset await flag for next bedsheet

                # Exit if 'q' is pressed
                if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                    break

            except Exception as e:
                log_bug(f"Error during main loop processing. Exception: {e}")
                error_occurred = True
                break  # Exit the loop on error

    except Exception as e:
        log_bug(f"Fatal error in main loop. Exception: {e}")
    finally:
        try:
            # Save collections to CSV
            #save_collection_to_csv(collection, LOGS_FILENAME_TEMPLATE.format(date=datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y%m%d")))
            #save_collection_to_csv(threshold_collection, THRESHOLD_FILENAME)
            #save_history_to_csv(history_collection, HISTORY_FILENAME)

            # Release resources
            release_video_resources(cap)
        except Exception as e:
            log_bug(f"Failed to release resources or save CSVs. Exception: {e}")

if __name__ == "__main__":
    main()
