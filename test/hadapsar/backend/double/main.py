# main.py

import signal
import sys
import cv2
from datetime import datetime, date, timezone, timedelta
from config import *
from database import (
    connect_to_mongo,
    get_daily_collection,
    get_last_bedsheet_number,
    log_to_mongo,
    add_threshold_entry,
    log_threshold_change,
    get_current_date_str,
    update_history,
    initialize_history_document,
)
from logger import log_bug, log_print
from video_processing import initialize_camera, release_video_resources, capture_frames
from models_and_states import State, bedsheet_model, defect_model
from csv_saving import save_collection_to_csv, save_history_to_csv
import time
import pytz
import numpy as np
from fastapi import (
    FastAPI,
    Response,
    WebSocket,
    HTTPException,
    Request,
    WebSocketDisconnect,
)
import logging
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import asyncio
import threading  # Import threading module
from threading import Lock
import uvicorn
import queue
import neoapi


class CamBuffer(neoapi.BufferBase):
    def __init__(self, size):
        neoapi.BufferBase.__init__(self)
        self.cpu_mat = np.ndarray(size, np.uint8)
        self.RegisterMemory(self.cpu_mat, size)

    def FreeCamBuffers(self):
        while self._buffers:
            self._camera.RevokeUserBuffer(self._buffers.pop())

    def __del__(self):
        self.UnregisterMemory()
        
        
# Global Variables
stop_event = threading.Event()

def write_decision_to_file(decision):
    # Create the file if it does not exist
    if not os.path.exists("decision.txt"):
        with open("decision.txt", "w") as file:
            pass  # Create an empty file
    
    # Write the decision to the file
    with open("decision.txt", "w") as file:
        file.write(str(decision))

app = FastAPI()

# Allow CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing purposes; specify exact origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for video capture
camera = None

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
        threshold_logged = log_threshold_change(threshold_collection, threshold_value)
        if threshold_logged:
            add_threshold_entry(history_collection, get_current_date_str(), threshold_value)
        return threshold_value

# New endpoint to update CLEAN_THRESHOLD from the React frontend
@app.post("/update_threshold")
async def update_threshold(request: Request):
    global CLEAN_THRESHOLD
    data = await request.json()
    new_threshold = data.get("threshold")
    print(f"Received new threshold: {new_threshold}")  # Logging

    if isinstance(new_threshold, (int, float)) and 0 <= new_threshold <= 100:
        CLEAN_THRESHOLD = new_threshold
        initialize_history_document(history_collection, get_current_date_str(), CLEAN_THRESHOLD)
        threshold_logged = log_threshold_change(threshold_collection, CLEAN_THRESHOLD)
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
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            # Yield the frame as an HTTP response for MJPEG streaming
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


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

# Register the signal handler for shutdown
def handle_shutdown(signal, frame):
    cv2.destroyAllWindows()
    print("Cleanup done. Exiting...")
    sys.exit(0)


# Main function
def detect(frame):
    global bedsheet_count, CLEAN_THRESHOLD
    global total_defect_area, unique_defect_ids, defect_max_areas
    global await_ending_edge, display_not_clean, defect_tracking_error
    global latest_frame  # Add this line

    # Initialize variables
    total_defect_area = 0  # Initialize total defect area
    unique_defect_ids = set()  # Track unique defect IDs across the bedsheet
    defect_max_areas = {}  # Dictionary to store maximum area of each unique defect ID

    await_ending_edge = False  # Flag to await ending edge after premature decision
    display_not_clean = False
    defect_tracking_error = False
    # Get video properties
    original_height, original_width = frame.shape[:2]    
    # Resize frame for faster processing
    half_width, half_height = original_width // 2, original_height // 2
    resized_dimensions = (half_width, half_height)
    original_dimensions = (original_width, original_height)

    # Initialize the display window and trackbar
    #cv2.namedWindow("Video with FPS and Detection Status")

    # Initialize state
    state = State.IDLE
    # Main loop
    try:
        error_occurred = False  # Flag to track errors
        # Resize frame for faster processing
        frame_resized = frame
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
                                write_decision_to_file(True)  # Write 'True' for Not Clean decision
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
                                write_decision_to_file(True)  # Write 'True' for Not Clean decision

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
                        write_decision_to_file(False)  # Write 'False' for Clean decision

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
                        write_decision_to_file(True)  # Write 'True' for Not Clean decision

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
        #cv2.imshow("Video with FPS and Detection Status", frame_resized)

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

        # Update latest_frame for streaming
        with frame_lock:
            latest_frame = frame_resized.copy()

    except Exception as e:
        log_bug(f"Error during main loop processing. Exception: {e}")
        error_occurred = True
    return frame_resized, bedsheet_count

def process_frame(img):
    # Crop the image from the top and bottom instead of left and right
    height, width, _ = img.shape
    # Crop the image
    cropped_img = img[CROP_TOP:height - CROP_BOTTOM, :]  # Crop top and bottom
    #cropped_img = img[:, CROP_LEFT:width - CROP_RIGHT]
    frame_resized, bedsheet_count = detect(cropped_img)

# Main Processing Loop
def main_loop(camera, frame, buf):
    global bedsheet_count
    # Initialize MongoDB and other settings
    initialize()

    print("in main loop")
    if camera is None:
        print("No camera initialized, exiting main loop.")
        return  # Exit the loop if the camera is not available
    # Make sure buf is initialized
    if buf is None:
        payloadsize = camera.f.PayloadSize.Get()
        buf = CamBuffer(payloadsize)
    while not stop_event.is_set():
        if not frame_queue.empty():
            img = frame_queue.get()
            try:
                # Process each frame
                process_frame(img)
                # Resize the image to reduce window size
    #            cv2.imshow("Webcam", resized_img)
            except Exception as e:
                logging.error(f"We have an error processing frame: {e}")
            finally:
                if camera:  # Ensure the camera is not None before revoking the buffer
                    camera.RevokeUserBuffer(buf)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            print("Stopping...")

    # Clean up
    if camera:
        camera.RevokeUserBuffer(buf)
        camera.Disconnect()
    cv2.destroyAllWindows()
    print(f"Final total bedsheets counted: {bedsheet_count}")

# Register the signal handler for shutdown
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# Signal Handling
def signal_handler(sig, frame):
    print('Signal received, stopping...')
    stop_event.set()

# Function to start the Uvicorn server
def start_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    initialize()
    # Start the Uvicorn server in a separate thread
    uvicorn_thread = threading.Thread(target=start_uvicorn, daemon=True)
    uvicorn_thread.start()
    # Initialize video capture
    try:
        
        camera = initialize_camera()
        # Initialize Frame Queue and Stop Event
        frame_queue = queue.Queue(maxsize=200)
        stop_event = threading.Event()
    except Exception as e:
        log_bug(f"Video capture initialization failed. Exception: {e}")
        sys.exit(1)
        
    # Start the frame capture thread
    capture_thread = threading.Thread(target=capture_frames, args=(camera, frame_queue, stop_event))
    capture_thread.start()
    
    try:
        # Run the main detection loop
        main_loop()
    finally:
        try:
            # Save collections to CSV
            #save_collection_to_csv(collection, LOGS_FILENAME_TEMPLATE.format(date=datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y%m%d")))
            #save_collection_to_csv(threshold_collection, THRESHOLD_FILENAME)
            #save_history_to_csv(history_collection, HISTORY_FILENAME)

            # Release resources
            # Stop all threads and clean up
            stop_event.set()
            capture_thread.join()
            release_video_resources(camera)
            print("All threads stopped. Cleanup done.")
        except Exception as e:
            log_bug(f"Failed to release resources or save CSVs. Exception: {e}")