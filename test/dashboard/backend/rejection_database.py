from fastapi import FastAPI, Response, WebSocket, HTTPException, Request, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import time
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os
import pymongo
from pymongo import MongoClient
import json
import asyncio
import threading  # Import threading module
from datetime import datetime, date, timezone, timedelta
import pytz


from enum import Enum

app = FastAPI()

# Allow CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing purposes; specify exact origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Finite State Machine
# Define FSM States
class State(Enum):
    IDLE = 0
    TRACKING_SCANNING = 1
    TRACKING_DECIDED_NOT_CLEAN_PREMATURE = 2
    TRACKING_DECIDED_CLEAN = 3



#Database

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['lisa_db']
threshold_collection = db['threshold_changes']  # New collection for threshold changes
history_collection = db['history']

# Create a new collection for each day based on the current date
def get_daily_collection():
    local_timezone = pytz.timezone('Asia/Kolkata')  # Replace with your local time zone
    date_str = datetime.now(local_timezone).strftime("%Y%m%d")  # Format: YYYYMMDD
    daily_collection = db[f'logs_{date_str}']

    # Check if the collection already has entries; if not, start bedsheet count at 1
    global bedsheet_count
    last_entry = daily_collection.find_one(sort=[("bedsheet_number", -1)])
    bedsheet_count = last_entry.get("bedsheet_number", 0) + 1 if last_entry else 1

    return daily_collection

# Update collection to today's date and initialize bedsheet_count
collection = get_daily_collection()  # Ensure it is refreshed daily

# Fetch the last bedsheet number from the logs
last_entry = collection.find_one(sort=[("bedsheet_number", -1)])
bedsheet_count = last_entry.get("bedsheet_number", 0) if last_entry else 0  # Start from the last logged number or 0 if empty

def log_to_mongo(bedsheet_number, detected_threshold, set_threshold, decision):
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



#Logging

# Set up logging with timestamps for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define a helper function to log and print simultaneously
def log_print(message):
    print(message)  # Print to console



#Defaults

# Define cleanliness threshold and default bedsheet area
DEFAULT_BEDSHEET_AREA = 70000  # Predefined bedsheet area in pixels

# Models

# Load the trained YOLOv8 models
bedsheet_model = YOLO(
    "/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/bedsheet.pt"
)
defect_model = YOLO(
    "/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/defect.pt"
)



#App

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

# New endpoint to update CLEAN_THRESHOLD from the React frontend
@app.post("/update_threshold")
async def update_threshold(request: Request):
    global CLEAN_THRESHOLD
    data = await request.json()
    new_threshold = data.get("threshold")
    print(f"Received new threshold: {new_threshold}")  # Logging

    if isinstance(new_threshold, (int, float)) and 0 <= new_threshold <= 100:
        CLEAN_THRESHOLD = new_threshold
        initialize_history_document(get_current_date_str(), CLEAN_THRESHOLD)
        threshold_logged = log_threshold_change(CLEAN_THRESHOLD)
        if threshold_logged:
            print(f"Threshold {CLEAN_THRESHOLD} logged in the database.")
        else:
            print(f"Threshold {CLEAN_THRESHOLD} is the same as the last logged threshold. Not logging.")
        return JSONResponse(content={"message": "Threshold updated successfully."})
    else:
        return JSONResponse(content={"error": "Invalid threshold value."}, status_code=400)

@app.get("/get_current_threshold")
async def get_current_threshold():
    global CLEAN_THRESHOLD
    return {"threshold": CLEAN_THRESHOLD}

@app.websocket("/ws/todays_counts")
async def websocket_todays_counts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Get today's date in 'YYYY-MM-DD' format
            local_timezone = pytz.timezone('Asia/Kolkata')  # Replace with your local time zone
            today = datetime.now(local_timezone).strftime('%Y-%m-%d')

            # Fetch today's data from history_collection
            item = history_collection.find_one({"date": today})

            if item:
                response_data = {
                    "date": item.get("date", "Unknown"),
                    "total_bedsheets": item.get("total_bedsheets", 0),
                    "total_accepted": item.get("total_accepted", 0),
                    "total_rejected": item.get("total_rejected", 0),
                }
            else:
                response_data = {
                    "date": today,
                    "total_bedsheets": 0,
                    "total_accepted": 0,
                    "total_rejected": 0,
                }

            # Send the data to the client
            await websocket.send_text(json.dumps(response_data))
            await asyncio.sleep(1)  # Adjust the interval as needed
    except WebSocketDisconnect:
        print("Client disconnected")

# New endpoint to fetch analytics data
@app.get("/analytics")
async def get_analytics(date: str = None):
    query = {}
    if date:
        try:
            # Parse the date string into a datetime object
            date_obj = datetime.strptime(date, "%Y-%m-%d")

            # Define UTC start and end of the day
            start_of_day = datetime.combine(date_obj, time.min).replace(tzinfo=timezone.utc)
            end_of_day = datetime.combine(date_obj, time.max).replace(tzinfo=timezone.utc)

            query["timestamp"] = {"$gte": start_of_day, "$lte": end_of_day}
        except ValueError:
            return JSONResponse(content={"error": "Invalid date format. Use YYYY-MM-DD."}, status_code=400)
    
    # Fetch data from MongoDB
    data = list(collection.find(query).sort("timestamp", -1))

    # Format data for JSON response
    response_data = []
    for item in data:
        response_data.append({
            "date": item["timestamp"].strftime("%Y-%m-%d"),
            "bedsheet_number": item["bedsheet_number"],
            "detected_threshold": item["detected_threshold"],
            "set_threshold": item["set_threshold"],
            "decision": item["decision"]
        })
    return JSONResponse(content=response_data)


@app.get("/daily_analytics")
async def get_daily_analytics():
    # Fetch all documents from the 'history_collection'
    data = list(history_collection.find())

    # Format data for JSON response
    response_data = []
    for item in data:
        # Ensure the date is in 'YYYY-MM-DD' format
        date_str = item.get("date", "Unknown")
        if isinstance(date_str, datetime):
            date_str = date_str.strftime('%Y-%m-%d')
        response_data.append({
            "date": item.get("date", "Unknown"),
            "total_bedsheets": item.get("total_bedsheets", 0),
            "total_accepted": item.get("total_accepted", 0),
            "total_rejected": item.get("total_rejected", 0),
        })
    return JSONResponse(content=response_data)

@app.get("/monthly_analytics")
async def get_monthly_analytics():
    # Fetch all documents from the 'history_collection'
    data = list(history_collection.find())

    # Prepare a dictionary to aggregate monthly data
    monthly_aggregated = {}

    for item in data:
        date_str = item.get('date', 'Unknown')
        if date_str == 'Unknown':
            continue  # Skip if date is unknown

        # Parse the date string to get the month and year
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        month_year = date_obj.strftime('%B %Y')  # E.g., "November 2024"

        if month_year not in monthly_aggregated:
            monthly_aggregated[month_year] = {
                'total_bedsheets': 0,
                'accepted': 0,
                'rejected': 0,
            }

        monthly_aggregated[month_year]['total_bedsheets'] += item.get('total_bedsheets', 0)
        monthly_aggregated[month_year]['accepted'] += item.get('total_accepted', 0)
        monthly_aggregated[month_year]['rejected'] += item.get('total_rejected', 0)

    # Convert aggregated data into a list sorted by date
    result_data = []
    for month_year, counts in monthly_aggregated.items():
        result_data.append({
            'month': month_year,
            'total_bedsheets': counts['total_bedsheets'],
            'accepted': counts['accepted'],
            'rejected': counts['rejected']
        })

    # Sort the result data by date
    result_data.sort(key=lambda x: datetime.strptime(x['month'], '%B %Y'))

    return JSONResponse(content=result_data)

# Open the camera feed (camera index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Shared variables for frame processing
latest_frame = None
frame_lock = threading.Lock()

@app.get("/video_feed")
async def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            # Yield the frame as an HTTP response for MJPEG streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#            time.sleep(0.05)  # Adjust as needed

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# WebSocket for analytics data
@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Get all collection names that start with 'logs_'
        collection_names = [name for name in db.list_collection_names() if name.startswith('logs_')]
        all_data = []
        for col_name in collection_names:
            col = db[col_name]
            latest_logs = list(col.find().sort("timestamp", -1).limit(10))
            all_data.extend(latest_logs)
        # Format data for JSON response
        response_data = []
        for item in all_data:
            response_data.append({
                "date": item["timestamp"].strftime("%Y-%m-%d"),
                "bedsheet_number": item["bedsheet_number"],
                "detected_threshold": item["detected_threshold"],
                "set_threshold": item["set_threshold"],
                "decision": item["decision"]
            })
        await websocket.send_text(json.dumps(response_data))
        await asyncio.sleep(2)  # Adjust as needed

def process_frames():
    global latest_frame, cap, bedsheet_count  # Use global variables
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    wait_time = int(1000 / video_fps)
    original_width, original_height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

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

    # Main processing loop with new FSM structure
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                log_print("End of video.")
                break

            # Resize frame for faster processing
            frame_resized = cv2.resize(frame, (original_width, original_height))
            frame_height = frame_resized.shape[0]

            # Initialize bedsheet presence flags for display
            bedsheet_present = False
            y1_positions = []
            y2_positions = []

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
                                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
    #        cv2.imshow("Video with FPS and Detection Status", frame_resized)

            # Update the latest frame to be used by the video feed endpoint
            with frame_lock:
                latest_frame = frame_resized.copy()

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
        log_print(f"An error occurred: {e}")

    finally:
        # Release resources
    #    save_collection_to_csv(collection, logs_filename)
    #    save_collection_to_csv(threshold_collection, threshold_filename)
    #    save_history_to_csv(history_collection, history_filename)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start the processing in a separate thread
    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True  # Allow thread to exit when main program exits
    processing_thread.start()

    # Start the Uvicorn server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
