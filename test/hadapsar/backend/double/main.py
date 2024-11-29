# main.py

import signal
import sys
import cv2
from datetime import datetime, date, timezone
from config import *
from database import *
from logger import log_bug, log_print
from video_processing import CameraManager
from models_and_states import State, bedsheet_model, defect_model
import pytz
import numpy as np
from fastapi import (
    FastAPI,
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
import threading
from threading import Lock
import uvicorn
import queue
import time

app = FastAPI()

# Allow CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing purposes; specify exact origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to hold camera processors
camera_processors = {}

# Initialize the CameraManager
camera_manager = CameraManager()
camera_manager.start()


class CameraProcessor:
    def __init__(self, side, serial_number):
        self.side = side  # 'left' or 'right'
        self.serial_number = serial_number

        # Variables for frame processing
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        # Initialize state
        self.state = State.IDLE
        # Initialize per-camera variables
        self.total_defect_area = 0
        self.unique_defect_ids = set()
        self.defect_max_areas = {}
        self.await_ending_edge = False
        self.display_not_clean = False
        self.defect_tracking_error = False

        # Initialize database connections
        self.db = connect_to_mongo()
        self.threshold_collection = self.db[f"threshold_changes_{self.side}"]
        self.history_collection = self.db[f"history_{self.side}"]
        self.collection = get_daily_collection(self.db, self.side)
        # Fetch last bedsheet number and threshold
        self.bedsheet_count = get_last_bedsheet_number(self.collection)
        self.CLEAN_THRESHOLD = get_last_threshold(self.threshold_collection, self.history_collection)
        initialize_history_document(
            self.history_collection, get_current_date_str(), self.CLEAN_THRESHOLD
        )

        self.stop_event = threading.Event()

    def start(self):
        # Start main loop thread
        self.main_loop_thread = threading.Thread(target=self.main_loop)
        self.main_loop_thread.start()

    def stop(self):
        # Stop the threads and clean up
        self.stop_event.set()
        self.main_loop_thread.join()

    def reset_defect_tracking_variables(self):
        self.total_defect_area = 0
        self.unique_defect_ids.clear()
        self.defect_max_areas.clear()

    def main_loop(self):
        while not self.stop_event.is_set():
            frame = camera_manager.get_frame(self.serial_number)
            if frame is None:
                # Camera might not be connected or no frame available
                time.sleep(0.1)
                continue
            try:
                self.process_frame(frame)
            except Exception as e:
                logging.error(f"Error processing frame for {self.side}: {e}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop_event.set()
                print(f"Stopping {self.side} camera.")
        cv2.destroyAllWindows()
        print(f"Final total bedsheets counted for {self.side}: {self.bedsheet_count}")

    def process_frame(self, img):
        # Crop the image from the top and bottom instead of left and right
        height, width, _ = img.shape
        cropped_img = img[:, CROP_LEFT:width - CROP_RIGHT]  # Crop left and right
        self.detect(cropped_img)

    def write_decision_to_file(self, decision):
        # Create the file if it does not exist
        decision_file = f"decision_{self.side}.txt"
        if not os.path.exists(decision_file):
            with open(decision_file, "w") as file:
                pass  # Create an empty file

        # Write the decision to the file
        with open(decision_file, "w") as file:
            file.write(str(decision))

    def detect(self, frame):
        # Get video properties
        original_height, original_width = frame.shape[:2]
        frame_resized = frame
        frame_height = frame_resized.shape[0]

        # Initialize bedsheet presence flags for display
        bedsheet_present = False
        y1_positions = []
        y2_positions = []

        # FSM Logic
        try:
            if self.state == State.IDLE:
                if bedsheet_model:  # Check if bedsheet_model is loaded
                    try:
                        # Detect starting edge to transition from IDLE to TRACKING_SCANNING
                        bedsheet_results = bedsheet_model.predict(
                            source=frame_resized, conf=CONF_THRESHOLD, verbose=False
                        )

                        for result in bedsheet_results:
                            if result.boxes:
                                boxes, classes, confidences = (
                                    result.boxes.xyxy,
                                    result.boxes.cls,
                                    result.boxes.conf,
                                )
                                for idx, class_id in enumerate(classes):
                                    if (
                                        int(class_id) == 0
                                        and confidences[idx] > CONF_THRESHOLD
                                    ):
                                        bedsheet_present = True
                                        x1, y1, x2, y2 = map(int, boxes[idx])
                                        cv2.rectangle(
                                            frame_resized,
                                            (x1, y1),
                                            (x2, y2),
                                            (0, 255, 0),
                                            2,
                                        )
                                        y1_positions.append(y1)
                                        y2_positions.append(y2)

                                        if (
                                            y1 > frame_height * 0.75
                                        ):  # Starting edge detected
                                            self.state = State.TRACKING_SCANNING
                                            self.reset_defect_tracking_variables()
                                            self.await_ending_edge = (
                                                False  # Reset await flag
                                            )
                                            self.display_not_clean = False
                                            log_print(
                                                f"{self.side} camera: Transitioned to TRACKING_SCANNING: Starting edge detected."
                                            )
                                            break  # Assuming one bedsheet per frame

                        log_print(
                            f"{self.side} camera: Bedsheet Present"
                            if bedsheet_present
                            else f"{self.side} camera: Bedsheet Not Present"
                        )
                    except Exception as e:
                        log_bug(
                            f"{self.side} camera: Error during bedsheet detection. Exception: {e}"
                        )
                        log_print(
                            f"{self.side} camera: Skipping bedsheet detection due to an error."
                        )
                else:
                    log_print(
                        f"{self.side} camera: Bedsheet detection skipped. Model not loaded."
                    )

            elif self.state == State.TRACKING_SCANNING and defect_model:
                if defect_model:  # Check if defect_model is loaded
                    if not self.defect_tracking_error:
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
                            self.defect_tracking_error = True
                            defect_results = None  # Ensure defect_results is defined
                            log_bug(
                                f"{self.side} camera: Defect tracking error occurred. Exception: {e}"
                            )
                            log_print(
                                f"{self.side} camera: Skipping defect detection due to an error. Feed will continue running."
                            )
                    else:
                        defect_results = None  # Ensure defect_results is defined
                        log_print(
                            f"{self.side} camera: Skipping defect detection as an error was previously encountered."
                        )
                else:
                    defect_results = None  # Ensure defect_results is defined
                    log_print(
                        f"{self.side} camera: Defect detection skipped. Model not loaded."
                    )

                if defect_results and (self.state == State.TRACKING_SCANNING):
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
                                defect_area = np.sum(
                                    defect_mask
                                )  # Calculate defect area as the sum of mask pixels

                                # Track unique defect IDs for the current bedsheet
                                self.unique_defect_ids.add(defect_id)

                                # Check if this defect ID already exists in defect_max_areas
                                if defect_id in self.defect_max_areas:
                                    # Only update if the new area is larger than the last maximum area
                                    if defect_area > self.defect_max_areas[defect_id]:
                                        # Adjust total_defect_area to account for the increase
                                        self.total_defect_area += (
                                            defect_area
                                            - self.defect_max_areas[defect_id]
                                        )
                                        # Update the maximum area for this defect ID
                                        self.defect_max_areas[defect_id] = defect_area
                                else:
                                    # New defect ID: add its area to total_defect_area and store it
                                    self.defect_max_areas[defect_id] = defect_area
                                    self.total_defect_area += defect_area

                                # Calculate real-time clean percent based on DEFAULT_BEDSHEET_AREA
                                defect_percent_real_time = (
                                    self.total_defect_area / DEFAULT_BEDSHEET_AREA
                                ) * 100
                                clean_percent_real_time = 100 - defect_percent_real_time

                            
                                # **Draw Bounding Boxes Around Defects**
                                # Check if bounding box coordinates are available
                                if (
                                    hasattr(defect_result.boxes, "xyxy")
                                    and len(defect_result.boxes.xyxy) > j
                                ):
                                    x1_d, y1_d, x2_d, y2_d = (
                                        defect_result.boxes.xyxy[j].int().tolist()
                                    )
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

                                # Immediate rejection if dirty percentage exceeds 100%
                                if defect_percent_real_time >= 100:
                                    self.state = (
                                        State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                                    )
                                    log_print(
                                        f"{self.side} camera: Bedsheet {self.bedsheet_count + 1}: Rejected due to defect percent >= 100%"
                                    )
                                    self.write_decision_to_file(True)
                                    decision = "Rejected"
                                    log_to_mongo(
                                        self.collection,
                                        self.bedsheet_count + 1,
                                        100,
                                        self.CLEAN_THRESHOLD,
                                        decision,
                                    )
                                    update_history(
                                        self.history_collection,
                                        get_current_date_str(),
                                        self.CLEAN_THRESHOLD,
                                        decision,
                                    )
                                    log_print(
                                        f"{self.side} camera: Bedsheet {self.bedsheet_count + 1} logged as 'Not Clean'"
                                    )
                                    self.bedsheet_count += 1
                                    self.reset_defect_tracking_variables()

                                    # Transition to IDLE after logging
                                    self.state = State.IDLE
                                    log_print(
                                        f"{self.side} camera: Transitioned to IDLE after rejection due to defect percent >= 100%"
                                    )
                                    break  # Exit further processing for this frame

                                # Check cleanliness threshold
                                if clean_percent_real_time < self.CLEAN_THRESHOLD:
                                    self.state = (
                                        State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                                    )
                                    self.await_ending_edge = True
                                    self.display_not_clean = True
                                    self.write_decision_to_file(True)

                                    # Log cleanliness analysis
                                    analysis_message = (
                                        f"Threshold: {self.CLEAN_THRESHOLD}%, "
                                        f"Bedsheet {self.bedsheet_count + 1}: Not Clean Prematurely. "
                                        f"Dirty Percent: {defect_percent_real_time:.2f}%, "
                                        f"Clean Percent: {clean_percent_real_time:.2f}%"
                                    )
                                    log_print(f"{self.side} camera: {analysis_message}")

                                    # Decision for "Not Clean"
                                    decision = "Rejected"
                                    log_to_mongo(
                                        self.collection,
                                        self.bedsheet_count + 1,
                                        defect_percent_real_time,
                                        self.CLEAN_THRESHOLD,
                                        decision,
                                    )
                                    update_history(
                                        self.history_collection,
                                        get_current_date_str(),
                                        self.CLEAN_THRESHOLD,
                                        decision,
                                    )
                                    log_print(
                                        f"{self.side} camera: Bedsheet {self.bedsheet_count + 1} logged as 'Not Clean'"
                                    )
                                    self.bedsheet_count += (
                                        1  # Increment bedsheet number
                                    )

                                    # Reset area calculations but continue tracking until ending edge
                                    self.reset_defect_tracking_variables()

                                    # **Important:** Break out of defect processing to avoid further detections in this frame
                                    break

            # Detect ending edge to transition to IDLE or other states
            if self.state == State.TRACKING_SCANNING:
                bedsheet_results = bedsheet_model.predict(
                    source=frame_resized, conf=CONF_THRESHOLD, verbose=False
                )
                bedsheet_present = False
                y2_positions = []

                for result in bedsheet_results:
                    if result.boxes:
                        boxes, classes, confidences = (
                            result.boxes.xyxy,
                            result.boxes.cls,
                            result.boxes.conf,
                        )
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
                            self.total_defect_area / DEFAULT_BEDSHEET_AREA
                        ) * 100
                        clean_percent_real_time = 100 - defect_percent_real_time

                        if clean_percent_real_time >= self.CLEAN_THRESHOLD:
                            self.state = State.TRACKING_DECIDED_CLEAN

                            self.display_not_clean = (
                                False  # No need to display "Not Clean"
                            )
                            self.write_decision_to_file(False)

                            # Log cleanliness analysis
                            analysis_message = (
                                f"Threshold: {self.CLEAN_THRESHOLD}%, "
                                f"Bedsheet {self.bedsheet_count + 1}: Clean. "
                                f"Dirty Percent: {defect_percent_real_time:.2f}%, "
                                f"Clean Percent: {clean_percent_real_time:.2f}%"
                            )
                            log_print(f"{self.side} camera: {analysis_message}")

                            # Decision for "Clean"
                            decision = "Accepted"
                            log_to_mongo(
                                self.collection,
                                self.bedsheet_count + 1,
                                defect_percent_real_time,
                                self.CLEAN_THRESHOLD,
                                decision,
                            )
                            update_history(
                                self.history_collection,
                                get_current_date_str(),
                                self.CLEAN_THRESHOLD,
                                decision,
                            )
                            log_print(
                                f"{self.side} camera: Bedsheet {self.bedsheet_count + 1} logged as 'Clean'"
                            )
                            self.bedsheet_count += 1  # Increment bedsheet number

                            # Reset area calculations but continue tracking until ending edge
                            self.reset_defect_tracking_variables()

                            log_print(
                                f"{self.side} camera: Ending Edge Detected and Counted as Clean"
                            )

                        else:
                            # If clean percent is still below threshold upon ending edge
                            self.state = State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE

                            self.await_ending_edge = True
                            self.display_not_clean = True
                            self.write_decision_to_file(True)

                            # Log cleanliness analysis
                            analysis_message = (
                                f"Threshold: {self.CLEAN_THRESHOLD}%, "
                                f"Bedsheet {self.bedsheet_count + 1}: Not Clean at Ending Edge. "
                                f"Dirty Percent: {defect_percent_real_time:.2f}%, "
                                f"Clean Percent: {clean_percent_real_time:.2f}%"
                            )
                            log_print(f"{self.side} camera: {analysis_message}")

                            # Decision for "Not Clean"
                            decision = "Rejected"
                            log_to_mongo(
                                self.collection,
                                self.bedsheet_count + 1,
                                defect_percent_real_time,
                                self.CLEAN_THRESHOLD,
                                decision,
                            )
                            update_history(
                                self.history_collection,
                                get_current_date_str(),
                                self.CLEAN_THRESHOLD,
                                decision,
                            )
                            log_print(
                                f"{self.side} camera: Bedsheet {self.bedsheet_count + 1} logged as 'Not Clean'"
                            )
                            self.bedsheet_count += 1  # Increment bedsheet number

                            # Reset area calculations but continue tracking until ending edge
                            self.reset_defect_tracking_variables()

            elif self.state == State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE:
                # Await ending edge detection
                bedsheet_results = bedsheet_model.predict(
                    source=frame_resized, conf=CONF_THRESHOLD, verbose=False
                )
                bedsheet_present = False
                y2_positions = []

                for result in bedsheet_results:
                    if result.boxes:
                        boxes, classes, confidences = (
                            result.boxes.xyxy,
                            result.boxes.cls,
                            result.boxes.conf,
                        )
                        for idx, class_id in enumerate(classes):
                            if int(class_id) == 0 and confidences[idx] > CONF_THRESHOLD:
                                bedsheet_present = True
                                x1, y1, x2, y2 = map(int, boxes[idx])
                                y2_positions.append(y2)

                if y2_positions:
                    y2_max = max(y2_positions)
                    if y2_max < frame_height * 0.90:  # Ending edge detected
                        self.state = State.IDLE
                        self.await_ending_edge = False
                        self.display_not_clean = False
                        log_print(
                            f"{self.side} camera: Transitioned to IDLE: Ending edge detected after Not Clean decision."
                        )

            # Display defect percentage and clean percentage if active
            if self.state in [
                State.TRACKING_SCANNING,
                State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE,
            ]:
                # Calculate defect percent and clean percent
                if DEFAULT_BEDSHEET_AREA > 0:
                    defect_percent = (
                        self.total_defect_area / DEFAULT_BEDSHEET_AREA
                    ) * 100
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
                if self.state not in [
                    State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE,
                    State.TRACKING_DECIDED_CLEAN,
                ]:
                    cv2.putText(
                        frame_resized,
                        f"Cleanliness: {'Clean' if clean_percent >= self.CLEAN_THRESHOLD else 'Not Clean'}",
                        (10, 390),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (
                            (0, 255, 0)
                            if clean_percent >= self.CLEAN_THRESHOLD
                            else (0, 0, 255)
                        ),
                        2,
                    )

            # Handle display of "Not Clean" message
            if (
                self.display_not_clean
                and self.state == State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
            ):
                log_print(f"{self.side} camera: Cleanliness: Not Clean")
                # Transition to IDLE after logging
                self.state = State.IDLE
                self.display_not_clean = False
                self.await_ending_edge = False  # Reset await flag for next bedsheet

            # Handle display of "Clean" message
            if self.state == State.TRACKING_DECIDED_CLEAN:
                log_print(f"{self.side} camera: Cleanliness: Clean")
                # Transition to IDLE after logging
                self.state = State.IDLE
                self.await_ending_edge = False  # Reset await flag for next bedsheet

            # Update latest_frame for streaming
            with self.frame_lock:
                self.latest_frame = frame_resized.copy()

        except Exception as e:
            log_bug(
                f"{self.side} camera: Error during detect processing. Exception: {e}"
            )

    def release_video_resources(self, buf):
        try:
            with self.camera_lock:
                if self.camera:
                    self.camera.RevokeUserBuffer(buf)
                    self.camera.Disconnect()
                    print(f"{self.side.capitalize()} camera disconnected.")
        except Exception as e:
            log_bug(
                f"Failed to release video resources for {self.side} camera. Exception: {e}"
            )


# Function to start the Uvicorn server
def start_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000)


# FastAPI endpoints
@app.post("/update_threshold")
async def update_threshold(request: Request):
    data = await request.json()
    new_threshold = data.get("threshold")
    side = data.get("side", "left")  # Default to 'left' if side not specified

    if side not in camera_processors:
        return JSONResponse(content={"error": "Invalid side."}, status_code=400)
    camera_processor = camera_processors[side]

    if isinstance(new_threshold, (int, float)) and 0 <= new_threshold <= 100:
        camera_processor.CLEAN_THRESHOLD = new_threshold
        initialize_history_document(
            camera_processor.history_collection,
            get_current_date_str(),
            camera_processor.CLEAN_THRESHOLD,
        )
        threshold_logged = log_threshold_change(
            camera_processor.threshold_collection, camera_processor.CLEAN_THRESHOLD
        )
        if threshold_logged:
            print(
                f"Threshold {camera_processor.CLEAN_THRESHOLD} logged in the database for {side} camera."
            )
        else:
            print(
                f"Threshold {camera_processor.CLEAN_THRESHOLD} is the same as the last logged threshold for {side} camera. Not logging."
            )
        return JSONResponse(content={"message": "Threshold updated successfully."})
    else:
        return JSONResponse(
            content={"error": "Invalid threshold value."}, status_code=400
        )


@app.get("/get_current_threshold/{side}")
async def get_current_threshold(side: str):
    if side not in camera_processors:
        return JSONResponse(content={"error": "Invalid side."}, status_code=400)
    camera_processor = camera_processors[side]
    return {"threshold": camera_processor.CLEAN_THRESHOLD}


@app.get("/video_feed/{side}")
async def video_feed(side: str):
    if side not in camera_processors:
        raise HTTPException(status_code=404, detail="Camera not found")
    camera_processor = camera_processors[side]

    def generate():
        while True:
            with camera_processor.frame_lock:
                if camera_processor.latest_frame is None:
                    continue
                frame = camera_processor.latest_frame.copy()
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
            time.sleep(0.01)

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/analytics/{side}")
async def get_analytics(side: str, date: str = None):
    if side not in camera_processors:
        return JSONResponse(content={"error": "Invalid side."}, status_code=400)
    camera_processor = camera_processors[side]
    collection = camera_processor.collection

    query = {}
    if date:
        try:
            # Parse the date string into a datetime object
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            local_timezone = pytz.timezone(TIMEZONE)
            start_of_day = local_timezone.localize(datetime.combine(date_obj, time.min))
            end_of_day = local_timezone.localize(datetime.combine(date_obj, time.max))
            query["timestamp"] = {"$gte": start_of_day, "$lte": end_of_day}
        except ValueError:
            return JSONResponse(
                content={"error": "Invalid date format. Use YYYY-MM-DD."},
                status_code=400,
            )

    # Fetch data from MongoDB
    data = list(collection.find(query).sort("timestamp", -1))

    # Format data for JSON response
    response_data = []
    for item in data:
        response_data.append(
            {
                "date": item["timestamp"].strftime("%Y-%m-%d"),
                "bedsheet_number": item["bedsheet_number"],
                "detected_threshold": item["detected_threshold"],
                "set_threshold": item["set_threshold"],
                "decision": item["decision"],
            }
        )
    return JSONResponse(content=response_data)


@app.get("/daily_analytics/{side}")
async def get_daily_analytics(side: str):
    if side not in camera_processors:
        return JSONResponse(content={"error": "Invalid side."}, status_code=400)
    camera_processor = camera_processors[side]
    history_collection = camera_processor.history_collection

    # Fetch all documents from the 'history_collection'
    data = list(history_collection.find())

    # Format data for JSON response
    response_data = []
    for item in data:
        # Ensure the date is in 'YYYY-MM-DD' format
        date_str = item.get("date", "Unknown")
        if isinstance(date_str, datetime):
            date_str = date_str.strftime("%Y-%m-%d")
        response_data.append(
            {
                "date": date_str,
                "total_bedsheets": item.get("total_bedsheets", 0),
                "total_accepted": item.get("total_accepted", 0),
                "total_rejected": item.get("total_rejected", 0),
            }
        )
    return JSONResponse(content=response_data)


@app.get("/monthly_analytics/{side}")
async def get_monthly_analytics(side: str):
    if side not in camera_processors:
        return JSONResponse(content={"error": "Invalid side."}, status_code=400)
    camera_processor = camera_processors[side]
    history_collection = camera_processor.history_collection

    # Fetch all documents from the 'history_collection'
    data = list(history_collection.find())

    # Prepare a dictionary to aggregate monthly data
    monthly_aggregated = {}

    for item in data:
        date_str = item.get("date", "Unknown")
        if date_str == "Unknown":
            continue  # Skip if date is unknown

        # Parse the date string to get the month and year
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        month_year = date_obj.strftime("%B %Y")  # E.g., "November 2024"

        if month_year not in monthly_aggregated:
            monthly_aggregated[month_year] = {
                "total_bedsheets": 0,
                "accepted": 0,
                "rejected": 0,
            }

        monthly_aggregated[month_year]["total_bedsheets"] += item.get(
            "total_bedsheets", 0
        )
        monthly_aggregated[month_year]["accepted"] += item.get("total_accepted", 0)
        monthly_aggregated[month_year]["rejected"] += item.get("total_rejected", 0)

    # Convert aggregated data into a list sorted by date
    result_data = []
    for month_year, counts in monthly_aggregated.items():
        result_data.append(
            {
                "month": month_year,
                "total_bedsheets": counts["total_bedsheets"],
                "accepted": counts["accepted"],
                "rejected": counts["rejected"],
            }
        )

    # Sort the result data by date
    result_data.sort(key=lambda x: datetime.strptime(x["month"], "%B %Y"))

    return JSONResponse(content=result_data)


# WebSocket endpoint for today's counts
@app.websocket("/ws/todays_counts/{side}")
async def websocket_todays_counts(websocket: WebSocket, side: str):
    if side not in camera_processors:
        await websocket.close(code=1003)  # Close with error code
        return
    camera_processor = camera_processors[side]
    history_collection = camera_processor.history_collection

    await websocket.accept()
    try:
        while True:
            # Get today's date in 'YYYY-MM-DD' format
            local_timezone = pytz.timezone(TIMEZONE)
            today = datetime.now(local_timezone).strftime("%Y-%m-%d")

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
        print(f"Client disconnected from {side} camera WebSocket")


if __name__ == "__main__":
    # Start the Uvicorn server in a separate thread
    uvicorn_thread = threading.Thread(target=start_uvicorn, daemon=True)
    uvicorn_thread.start()

    # Create CameraProcessor instances
    camera_processors["left"] = CameraProcessor("left", SERIAL_NUMBER_LEFT)
    camera_processors["right"] = CameraProcessor("right", SERIAL_NUMBER_RIGHT)

    # Start the camera processors
    camera_processors["left"].start()
    camera_processors["right"].start()

    try:
        # Wait for both camera processors to finish
        camera_processors["left"].main_loop_thread.join()
        camera_processors["right"].main_loop_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the camera processors
        camera_processors["left"].stop()
        camera_processors["right"].stop()
        # Stop the camera manager
        camera_manager.stop()
