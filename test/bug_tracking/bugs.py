import cv2
import time
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os
from pymongo import MongoClient
from datetime import datetime, date, timezone, timedelta
import pytz
from enum import Enum
import signal
import sys
from pymongo.errors import PyMongoError  # Use PyMongoError for generic exceptions

#Finite State Machine
# Define FSM States
class State(Enum):
    IDLE = 0
    TRACKING_SCANNING = 1
    TRACKING_DECIDED_NOT_CLEAN_PREMATURE = 2
    TRACKING_DECIDED_CLEAN = 3

# Define a bug log file
BUG_LOG_FILE = "bug_log.txt"

# Function to log bugs into the bug log file
def log_bug(bug_message):
    with open(BUG_LOG_FILE, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} - BUG: {bug_message}\n")
    print(f"BUG LOGGED: {bug_message}")

# Define a helper function to log and print simultaneously
def log_print(message):
    print(message)  # Print to console

# Signal handler for graceful exit
def signal_handler(sig, frame):
    log_print("Interrupt received. Exiting gracefully...")
    # Release resources here if needed
    if 'cap' in globals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)



#Database

# MongoDB setup with bug tracking
def connect_to_mongo():
    max_retries = 5
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)  # 5 seconds timeout
            # Force connection to verify the MongoDB server is reachable
            client.server_info()  # This will raise an exception if the server is unreachable
            db = client['lisa_db']
            log_print(f"Connected to MongoDB on attempt {attempt + 1}")
            return db
        except PyMongoError as e:
            log_bug(f"Attempt {attempt + 1} failed to connect to MongoDB. Exception: {e}")
            if attempt < max_retries - 1:
                log_print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise SystemExit("Unable to connect to MongoDB after multiple attempts.")

# Connect to MongoDB
try:
    db = connect_to_mongo()
    threshold_collection = db['threshold_changes']
    history_collection = db['history']
except SystemExit as e:
    log_bug(f"System exit triggered: {e}")
    sys.exit(1)

# Daily collection
def get_daily_collection():
    try:
        local_timezone = pytz.timezone('Asia/Kolkata')
        date_str = datetime.now(local_timezone).strftime("%Y%m%d")
        return db[f'logs_{date_str}']
    except PyMongoError as e:
        log_bug(f"Failed to create or access daily collection. Exception: {e}")
        raise

try:
    collection = get_daily_collection()
except Exception as e:
    log_bug(f"Error initializing daily collection: {e}")
    sys.exit(1)


# Fetch the last bedsheet number from the logs
last_entry = collection.find_one(sort=[("bedsheet_number", -1)])
bedsheet_count = last_entry.get("bedsheet_number", 0) if last_entry else 0  # Start from the last logged number or 0 if empty

def log_to_mongo(bedsheet_number, detected_threshold, set_threshold, decision):
    try:
        # Convert np.float32 to float and set_threshold to int for consistency
        local_timezone = pytz.timezone('Asia/Kolkata')  # Replace with your local time zone
        timestamp = datetime.now(local_timezone)
        document = {
            "bedsheet_number": int(bedsheet_number),
            "detected_threshold": float(detected_threshold),
            "set_threshold": int(set_threshold),  # Store as int
            "decision": decision,
            "timestamp": timestamp
        }
        # Insert document into MongoDB
        collection.insert_one(document)
    except Exception as e:
        log_bug(f"Failed to log to MongoDB. Document: {document}. Exception: {e}")



def log_threshold_change(threshold_value):
    # Convert to int to ensure consistency
    threshold_value = int(threshold_value)

    local_timezone = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(local_timezone)

    # Get the last logged threshold entry
    last_entry = threshold_collection.find_one(sort=[("timestamp", -1)])
    
    # Only log if the threshold value is different from the last logged value
    if last_entry is None or last_entry.get("set_threshold") != threshold_value:
        document = {
            "set_threshold": threshold_value,  # Store as int
            "timestamp": timestamp
        }
        threshold_collection.insert_one(document)
        return True  # Indicates that a new threshold was logged
    return False  # No change in threshold

# Helper function to get the current date as a string
def get_current_date_str():
    local_timezone = pytz.timezone('Asia/Kolkata')  # Replace with your local time zone
    return datetime.now(local_timezone).strftime("%Y-%m-%d")

# Add a new threshold entry with counters for accepted and rejected
def add_threshold_entry(date, threshold):
    history_collection.update_one(
        {"date": date},
        {
            "$push": {
                "thresholds": {
                    "set_threshold": threshold,
                    "accepted": 0,
                    "rejected": 0,
                }
            }
        }
    )

# Initialize the history document for the current date if it doesn't exist
def initialize_history_document(date, threshold):
    history_collection.update_one(
        {"date": date},
        {
            "$setOnInsert": {
                "date": date,
                "total_bedsheets": 0,
                "total_accepted": 0,
                "total_rejected": 0,
                "thresholds": []
            }
        },
        upsert=True
    )
    # Removed the automatic addition of a threshold entry


def update_history(date, threshold, decision):
    try:

        # Increment total bedsheets count
        result = history_collection.update_one({"date": date}, {"$inc": {"total_bedsheets": 1}})
        
        # Increment total accepted/rejected for the day based on decision
        if decision == "Accepted":
            result = history_collection.update_one({"date": date}, {"$inc": {"total_accepted": 1}})
        else:
            result = history_collection.update_one({"date": date}, {"$inc": {"total_rejected": 1}})

        # Fetch the current document for the specified date
        doc = history_collection.find_one({"date": date})
        # Ensure the document exists before accessing its fields
        if doc is not None:
            thresholds = doc.get("thresholds", [])

            # Check if there is an existing entry for the current threshold
            if thresholds and thresholds[-1]["set_threshold"] == threshold:
                log_print(f"Matching threshold found: {threshold}")
                # Update the accepted or rejected count for the most recent threshold entry
                if decision == "Accepted":
                    result = history_collection.update_one(
                        {"date": date, "thresholds.set_threshold": threshold},
                        {"$inc": {"thresholds.$.accepted": 1}}
                    )
                    log_print(f"Incremented thresholds.$.accepted: {result.modified_count} document(s) updated.")
                else:
                    result = history_collection.update_one(
                        {"date": date, "thresholds.set_threshold": threshold},
                        {"$inc": {"thresholds.$.rejected": 1}}
                    )
                    log_print(f"Incremented thresholds.$.rejected: {result.modified_count} document(s) updated.")
            else:
                log_print(f"No matching threshold found for {threshold}, adding a new threshold entry.")
                # If the threshold has changed or if there is no previous threshold, add a new threshold entry with the current timestamp
                add_threshold_entry(date, threshold)

                # Initialize accepted or rejected count for the new threshold entry
                if decision == "Accepted":
                    result = history_collection.update_one(
                        {"date": date, "thresholds.set_threshold": threshold},
                        {"$inc": {"thresholds.$.accepted": 1}}
                    )
                    log_print(f"Added and incremented thresholds.$.accepted: {result.modified_count} document(s) updated.")
                else:
                    result = history_collection.update_one(
                        {"date": date, "thresholds.set_threshold": threshold},
                        {"$inc": {"thresholds.$.rejected": 1}}
                    )
                    log_print(f"Added and incremented thresholds.$.rejected: {result.modified_count} document(s) updated.")
        else:
            # If the document is missing, initialize it
            initialize_history_document(date, threshold)
    except Exception as e:
        log_bug(f"Failed to update history. Date: {date}, Threshold: {threshold}, Decision: {decision}. Exception: {e}")



#CSV saving

# Function to save MongoDB collection to a CSV file
def save_collection_to_csv(collection, filename):
    data = list(collection.find())
    if data:
        df = pd.DataFrame(data)
        # Drop MongoDB-specific '_id' column if it exists
        if '_id' in df.columns:
            df.drop(columns=['_id'], inplace=True)
        
        # Handle unhashable types by excluding them from duplicates check
        unhashable_columns = [col for col in df.columns if isinstance(df[col].iloc[0], list)]
        if os.path.exists(filename):
            # If file exists, update it by appending new data
            df_existing = pd.read_csv(filename)
            # Avoid duplicating rows by checking for existing data, ignoring unhashable columns
            combined_df = pd.concat([df_existing, df])
            combined_df = combined_df.drop_duplicates(subset=[col for col in df.columns if col not in unhashable_columns])
        else:
            combined_df = df
        
        # Save the combined DataFrame back to CSV
        combined_df.to_csv(filename, index=False)

# Save 'logs' collection as a CSV with daily naming
local_timezone = pytz.timezone('Asia/Kolkata')  # Define at the top if not already
logs_filename = f"logs_{datetime.now(local_timezone).strftime('%Y%m%d')}.csv"

# Save 'threshold_changes' collection to a CSV file
threshold_filename = "threshold_changes.csv"

# Function to save the 'history' collection in a table-like format
def save_history_to_csv(history_collection, filename):
    data = list(history_collection.find())
    rows = []

    for entry in data:
        date = entry.get("date")
        total_bedsheets = entry.get("total_bedsheets", 0)
        total_accepted = entry.get("total_accepted", 0)
        total_rejected = entry.get("total_rejected", 0)
        
        # Main row with the date and total values
        rows.append({
            "Date": date,
            "Total Bedsheets": total_bedsheets,
            "Total Accepted": total_accepted,
            "Total Rejected": total_rejected,
            "Threshold": "",  # Empty for the main row
            "Accepted": "",
            "Rejected": "",
        })
        
        # Subsequent rows for each threshold entry within the date
        for threshold_entry in entry.get("thresholds", []):
            rows.append({
                "Date": "",  # Empty to avoid repeating the date
                "Total Bedsheets": "",  # Empty for threshold sub-row
                "Total Accepted": "",  # Empty for threshold sub-row
                "Total Rejected": "",  # Empty for threshold sub-row
                "Threshold": threshold_entry.get("set_threshold"),
                "Accepted": threshold_entry.get("accepted", 0),
                "Rejected": threshold_entry.get("rejected", 0),
            })
    
    # Create DataFrame from formatted data
    new_df = pd.DataFrame(rows)

    if os.path.exists(filename):
        # Load existing CSV data
        existing_df = pd.read_csv(filename)
        
        # Remove any rows in existing_df that have the same date as in new_df
        unique_dates = new_df['Date'].dropna().unique()
        existing_df = existing_df[~existing_df['Date'].isin(unique_dates)]
        
        # Concatenate and remove duplicates based on all columns to keep only latest
        combined_df = pd.concat([existing_df, new_df]).reset_index(drop=True)
    else:
        combined_df = new_df

    # Save the combined DataFrame to CSV
    combined_df.to_csv(filename, index=False)

# Usage:
history_filename = "history.csv"


# Set up logging with timestamps for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#Defaults

# Define cleanliness threshold and default bedsheet area
DEFAULT_BEDSHEET_AREA = 70000  # Predefined bedsheet area in pixels

# Models

# Model loading with error handling
try:
    bedsheet_model = YOLO("/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/bedsheet.pt")
except Exception as e:
    bedsheet_model = None  # Assign None if loading fails
    log_bug(f"Failed to load bedsheet model. Exception: {e}")
    log_print("Bedsheet model not loaded. Detection will be skipped.")

try:
    defect_model = YOLO("/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/defect.pt")
except Exception as e:
    defect_model = None  # Assign None if loading fails
    log_bug(f"Failed to load defect model. Exception: {e}")
    log_print("Defect model not loaded. Detection will be skipped.")



# Initialize cleanliness threshold from the database
def get_last_threshold():
    # Fetch the last threshold entry
    last_entry = threshold_collection.find_one(sort=[("timestamp", -1)])
    if last_entry and "set_threshold" in last_entry:
        return last_entry["set_threshold"]
    else:
        # If no threshold is found, use default and log it
        default_threshold = 95.0
        log_threshold_change(default_threshold)
        return default_threshold

CLEAN_THRESHOLD = get_last_threshold()

# Trackbar callback to update threshold in real-time and log changes
def update_threshold(val):
    global CLEAN_THRESHOLD
    threshold_changed = log_threshold_change(val)  # Log and check for change
    if threshold_changed:
        CLEAN_THRESHOLD = val
        initialize_history_document(get_current_date_str(), CLEAN_THRESHOLD)  # Initialize history only on change
        print("Clean Threshold changed to ", CLEAN_THRESHOLD)

# Initialize the display window
cv2.namedWindow("Video with FPS and Detection Status")
cv2.createTrackbar("Clean Threshold", "Video with FPS and Detection Status", int(CLEAN_THRESHOLD), 100, update_threshold)



# Video capture initialization
try:
    cap = cv2.VideoCapture("/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/media/video001.avi")
    if not cap.isOpened():
        raise Exception("Could not open the camera.")
except Exception as e:
    log_bug(f"Camera initialization failed. Exception: {e}")
    raise

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
unique_defect_ids = set()  # Track unique defect IDs across the bedsheet

# Dictionary to store maximum area of each unique defect ID
defect_max_areas = {}

# Initialize variables to track the visible bedsheet area
total_defect_area = 0  # Initialize total defect area

# Flags for state management
await_ending_edge = False  # Flag to await ending edge after premature decision
display_not_clean = False

# Flag to track defect tracking status
defect_tracking_error = False

# Main loop
try:
    error_occurred = False  # Flag to track errors
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                log_bug("Failed to read frame from camera.")
                break

            # Resize frame for faster processing
            frame_resized = cv2.resize(frame, (half_width, half_height))
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
                        bedsheet_results = bedsheet_model.predict(source=frame_resized, conf=conf_threshold, verbose=False)

                        for result in bedsheet_results:
                            if result.boxes:
                                boxes, classes, confidences = result.boxes.xyxy, result.boxes.cls, result.boxes.conf
                                for idx, class_id in enumerate(classes):
                                    if int(class_id) == 0 and confidences[idx] > conf_threshold:
                                        bedsheet_present = True
                                        x1, y1, x2, y2 = map(int, boxes[idx])
                                        #cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        y1_positions.append(y1)
                                        y2_positions.append(y2)

                                        if y1 > frame_height * 0.75:  # Starting edge detected
                                            state = State.TRACKING_SCANNING
                                            total_defect_area = 0
                                            unique_defect_ids.clear()
                                            defect_max_areas.clear()
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

            elif state == State.TRACKING_SCANNING:
                if defect_model:  # Check if defect_model is loaded                
                    # Handle defect tracking only if no error occurred
                    if not defect_tracking_error:
                        try:
                            # Perform defect tracking
                            defect_results = defect_model.track(
                                source=frame_resized,
                                conf=defect_conf_threshold,
                                verbose=False,
                                persist=True,
                                tracker="/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/botsort_defect.yaml",
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

                                    # Increment bedsheet count only because it's classified as "Not Clean"
                                    # Decision for "Not Clean"
                                    decision = "Rejected"
                                    log_to_mongo(bedsheet_count + 1, defect_percent_real_time, CLEAN_THRESHOLD, decision)
                                    update_history(get_current_date_str(), CLEAN_THRESHOLD, decision)  # Update history here
                                    log_print(f"Bedsheet {bedsheet_count + 1} logged as 'Not Clean'")
                                    bedsheet_count += 1  # Increment bedsheet number

                                    # Reset area calculations but continue tracking until ending edge
                                    total_defect_area = 0  # Reset total defect area
                                    unique_defect_ids.clear()  # Clear tracked defects for the next bedsheet
                                    defect_max_areas.clear()  # Reset defect area tracking for the next bedsheet

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

                                # Increment bedsheet count because it's classified as "Clean"
                                # Decision for "Clean"
                                decision = "Accepted"
                                log_to_mongo(bedsheet_count + 1, defect_percent_real_time, CLEAN_THRESHOLD, decision)
                                update_history(get_current_date_str(), CLEAN_THRESHOLD, decision)  # Update history here
                                log_print(f"Bedsheet {bedsheet_count + 1} logged as 'Clean'")
                                bedsheet_count += 1  # Increment bedsheet number

                                # Reset area calculations but continue tracking until ending edge
                                total_defect_area = 0  # Reset total defect area
                                unique_defect_ids.clear()  # Clear tracked defects for the next bedsheet
                                defect_max_areas.clear()  # Reset defect area tracking for the next bedsheet

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

                                # Increment bedsheet count only because it's classified as "Not Clean"
                                # Decision for "Not Clean"
                                decision = "Rejected"
                                log_to_mongo(bedsheet_count + 1, defect_percent_real_time, CLEAN_THRESHOLD, decision)
                                update_history(get_current_date_str(), CLEAN_THRESHOLD, decision)  # Update history here
                                log_print(f"Bedsheet {bedsheet_count + 1} logged as 'Not Clean'")
                                bedsheet_count += 1  # Increment bedsheet number

                                # Reset area calculations but continue tracking until ending edge
                                total_defect_area = 0  # Reset total defect area
                                unique_defect_ids.clear()  # Clear tracked defects for the next bedsheet
                                defect_max_areas.clear()  # Reset defect area tracking for the next bedsheet

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
        # Release resources
    #    save_collection_to_csv(collection, logs_filename)
    #    save_collection_to_csv(threshold_collection, threshold_filename)
    #    save_history_to_csv(history_collection, history_filename)

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        log_bug(f"Failed to release resources. Exception: {e}")