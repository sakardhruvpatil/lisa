# main.py
# Full machine vision Logic

import cv2
from datetime import datetime, date, timezone
from config.config import *
from database.database import *
from utils.logger import log_bug, log_print
from utils.video_processing import CameraManager
from utils.models_and_states import (
    State,
    bedsheet_model,
    defect_model,
    hor_bedsheet_model,
    tear_model,
)
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
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import asyncio
import threading
from threading import Lock
import uvicorn
import time

app = FastAPI()

# Allow CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    # For testing purposes; specify exact origins in production
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to hold camera processors
camera_processors = {}

# Initialize the stitched camera processor but do not start it
stitched_camera_processor = None

# Shared lock for horizontal mode
horizontal_processing_lock = threading.Lock()

# Global variable for the threshold
CLEAN_THRESHOLD = 0


class CameraProcessor:
    def __init__(self, side, camera_manager, process_mode="vertical"):
        self.side = side  # 'left' or 'right'
        self.camera_manager = camera_manager
        self.process_mode = (
            process_mode  # Mode to determine vertical or horizontal processing
        )
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.state = State.IDLE
        self.is_active = True  # Indicates if the processor is active
        self.stop_event = threading.Event()  # Initialize stop_event here
        self.detection_enabled = True  # Flag to control detection
        # Initialize additional attributes
        self.previous_frame = None  # To store the previous frame for optical flow
        self.frame_counter = 0  # Counter to manage error state resets
        # Initialize per-camera variables
        self.total_defect_area = 0
        self.unique_defect_ids = set()
        self.defect_max_areas = {}
        self.await_ending_edge = False
        self.display_not_clean = False
        self.defect_tracking_error = False

        # Initialize database connections
        self.db = connect_to_mongo()
        self.threshold_collection = self.db["threshold_changes"]
        self.history_collection = self.db[f"history_{self.side}"]
        self.collection = get_daily_collection(self.db, self.side)
        self.horizontal_collection = get_daily_collection_hor(self.db)
        self.history_collection_hor = self.db[
            "history_horizontal"
        ]  # Fetch last bedsheet number and threshold
        self.bedsheet_count = get_last_bedsheet_number(self.collection)

        # Initialize threshold
        global CLEAN_THRESHOLD
        CLEAN_THRESHOLD = get_last_threshold(
            self.threshold_collection, self.history_collection
        )
        print(f"Initialized {self.side} camera with threshold: {CLEAN_THRESHOLD}")

        initialize_history_document(
            self.history_collection, get_current_date_str(), CLEAN_THRESHOLD
        )
        initialize_history_document_hor(
            self.history_collection_hor, get_current_date_str(), CLEAN_THRESHOLD
        )

    def set_process_mode(self, mode):
        """Set process mode: 'vertical' or 'horizontal'"""
        if mode not in ["vertical", "horizontal"]:
            raise ValueError("Invalid process mode")

        self.process_mode = mode
        self.detection_enabled = (
            mode == "vertical"
        )  # Enable detection only in vertical mode

    def update_threshold(self, new_threshold):
        global CLEAN_THRESHOLD  # Access the global variable
        if not (0 <= new_threshold <= 100):
            raise ValueError("Threshold must be between 0 and 100.")
        CLEAN_THRESHOLD = new_threshold  # Update the global threshold
        print(f"Updated {self.side} threshold to: {CLEAN_THRESHOLD}")

    def start(self):
        self.is_active = True
        print(f"Starting vertical camera with initial threshold: {CLEAN_THRESHOLD}")
        self.main_loop_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.main_loop_thread.start()

    def stop(self):
        self.stop_event.set()
        self.is_active = False
        if self.main_loop_thread.is_alive():
            self.main_loop_thread.join()

    def reset_defect_tracking_variables(self):
        self.total_defect_area = 0
        self.unique_defect_ids.clear()
        self.defect_max_areas.clear()

    def main_loop(self):
        """
        Continuously capture frames from the camera. If the camera is disconnected,
        repeatedly attempt to reconnect until a valid frame is retrieved and then
        continue normal execution.
        """
        try:
            while self.is_active and not self.stop_event.is_set():
                # If in horizontal mode, do not process individual frames
                if self.process_mode == "horizontal":
                    time.sleep(0.1)  # Sleep to avoid busy waiting
                    continue

                # Attempt to capture a frame
                frame = self.camera_manager.get_frame(self.side)

                if frame is None:
                    # Briefly wait if a frame isn't available (e.g., transient network delay)
                    time.sleep(0.1)

                    # Check again to see if it's just a temporary issue
                    frame = self.camera_manager.get_frame(self.side)
                    if frame is None:
                        # If still None, treat it as a potential disconnect
                        log_print(
                            f"{self.side.capitalize()} camera: No frame received. Attempting reconnection indefinitely..."
                        )

                        # Keep trying until the camera is successfully reconnected
                        while True:
                            # Attempt to reconnect
                            self.camera_manager.reconnect_camera(self.side)
                            time.sleep(self.reconnect_delay)

                            # Test capture after attempting to reconnect
                            test_frame = self.camera_manager.get_frame(self.side)
                            if test_frame is not None:
                                log_print(
                                    f"{self.side.capitalize()} camera: Reconnection successful."
                                )
                                break  # Exit reconnect loop and continue main loop

                        continue  # Once reconnected, continue from the top of the loop

                # If we have a valid frame, proceed with detection
                if self.detection_enabled:
                    try:
                        self.process_frame(frame)
                    except Exception as e:
                        logging.error(f"Error processing frame for {self.side}: {e}")

        except KeyboardInterrupt:
            self.stop()
            print(f"Stopping {self.side} camera.")
            print(f"Final bedsheet count for {self.side}: {self.bedsheet_count}")
            raise

    def process_frame(self, img):
        # Process each side independently
        height, width, _ = img.shape
        # cropped = img[:, CROP_LEFT : width - CROP_RIGHT]  # Crop left and right
        # print(f"Cropped image shape: {img.shape}")  # Log the shape
        self.detect(img)

    # Replace the existing write_decision_to_file method with the one below:
    def write_decision_to_file(self, decision):
        """
        Write the decision to the specific decision file for the camera (left or right).
        Uses ACCEPT and REJECT from config.
        """
        # Define the writable directory for bug logs
        log_dir = os.path.join(os.getenv("HOME"), "LISA_LOGS")
        os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists

        # Determine the decision value from config
        decision_value = REJECT if decision == REJECT else ACCEPT

        # Determine the decision file based on the camera side
        decision_file = os.path.join(log_dir, f"decision_{self.side}.txt")
        try:
            # Write the decision to the file
            with open(decision_file, "w") as file:
                file.write(
                    str(decision_value)
                )  # Write the corresponding value (True/False or 1/0)
            print(f"Decision for {self.side} camera written to {decision_file}.")
        except Exception as e:
            print(f"Failed to write decision for {self.side} camera: {e}")

    def detect(self, frame):
        global CLEAN_THRESHOLD  # Access the global variable

        # Check if the previous frame is None
        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return  # Skip processing until we have a previous frame

        # Ensure both frames are of the same size
        if self.previous_frame.shape != frame.shape:
            error_code = 1014
            log_bug(
                f"Frame size mismatch: previous_frame {self.previous_frame.shape}, current_frame {frame.shape} (Error code: {error_code})"
            )
            return

        # Get video properties
        frame_resized = frame
        frame_height = frame_resized.shape[0]

        # Initialize bedsheet presence flags for display
        y1_positions = []
        y2_positions = []

        # Reset error state after 100 frames
        if self.defect_tracking_error and self.frame_counter >= 100:
            self.defect_tracking_error = False
            self.frame_counter = 0

        # ----------------------------------------------------------------
        # Create or reset a tear_detected flag if not present.
        # Once this is True, skip any further checks until ending edge.
        # ----------------------------------------------------------------
        if not hasattr(self, "tear_detected"):
            self.tear_detected = False

        # Perform normal tracking if no error
        if not self.defect_tracking_error:
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

                                            # Starting edge detected
                                            if y1 > frame_height * 0.75:
                                                self.state = State.TRACKING_SCANNING
                                                self.reset_defect_tracking_variables()
                                                self.await_ending_edge = False
                                                self.display_not_clean = False
                                                self.tear_detected = (
                                                    False  # Reset tear flag
                                                )
                                                log_print(
                                                    f"{self.side} camera: Transitioned to TRACKING_SCANNING (bedsheet start)."
                                                )
                                                break
                        except Exception as e:
                            error_code = 1015
                            log_bug(
                                f"{self.side} camera: Bedsheet detection error: {e} (Error code: {error_code})"
                            )
                            log_print(
                                f"{self.side} camera: Skipping bedsheet detection due to an error."
                            )
                    else:
                        log_print(
                            f"{self.side} camera: Bedsheet detection skipped (model not loaded)."
                        )

                elif self.state == State.TRACKING_SCANNING:
                    # ---------------------------------------------------
                    # 1) First check if tear is already detected.
                    #    If not, run the tear model.
                    # ---------------------------------------------------
                    if not self.tear_detected and tear_model:
                        try:
                            tear_results = tear_model.predict(
                                source=frame_resized,
                                conf=TEAR_CONF_THRESHOLD,
                                verbose=False,
                            )
                            for result in tear_results:
                                if result.boxes:
                                    boxes, classes, confidences = (
                                        result.boxes.xyxy,
                                        result.boxes.cls,
                                        result.boxes.conf,
                                    )
                                    for idx, class_id in enumerate(classes):
                                        if (
                                            int(class_id) == 0
                                            and confidences[idx] > TEAR_CONF_THRESHOLD
                                        ):
                                            self.tear_detected = True
                                            log_print(
                                                f"{self.side} camera: Tear detected!"
                                            )
                                            break
                        except Exception as e:
                            error_code = 1018
                            log_bug(
                                f"{self.side} camera: Tear detection error: {e} (Error code: {error_code})"
                            )

                    # --------------------------------------
                    # 2) If tear not detected, do defect checks
                    # --------------------------------------
                    if not self.tear_detected and defect_model:
                        if not self.defect_tracking_error:
                            try:
                                defect_results = defect_model.track(
                                    source=frame_resized,
                                    conf=DEFECT_CONF_THRESHOLD,
                                    verbose=False,
                                    persist=True,
                                    tracker=TRACKER_PATH,
                                )
                            except Exception as e:
                                self.defect_tracking_error = True
                                defect_results = None
                                error_code = 1016
                                log_bug(
                                    f"{self.side} camera: Defect tracking error: {e} (Error code: {error_code})"
                                )
                                log_print(
                                    f"{self.side} camera: Skipping defect detection due to an error."
                                )
                        else:
                            defect_results = None
                            log_print(
                                f"{self.side} camera: Skipping defect detection because an error was encountered."
                            )
                    else:
                        defect_results = None

                    # ------------------------------------
                    # 3) Update defect area if no tear
                    # ------------------------------------
                    if defect_results and not self.tear_detected:
                        for defect_result in defect_results:
                            masks = defect_result.masks
                            tracks = (
                                defect_result.boxes.id.cpu().numpy()
                                if defect_result.boxes.id is not None
                                else None
                            )
                            if masks is not None and tracks is not None:
                                for j, mask in enumerate(masks.data):
                                    defect_mask = mask.cpu().numpy()
                                    defect_id = tracks[j]
                                    defect_area = np.sum(defect_mask)

                                    # Track unique defect IDs
                                    self.unique_defect_ids.add(defect_id)
                                    if defect_id in self.defect_max_areas:
                                        if (
                                            defect_area
                                            > self.defect_max_areas[defect_id]
                                        ):
                                            self.total_defect_area += (
                                                defect_area
                                                - self.defect_max_areas[defect_id]
                                            )
                                            self.defect_max_areas[defect_id] = (
                                                defect_area
                                            )
                                    else:
                                        self.defect_max_areas[defect_id] = defect_area
                                        self.total_defect_area += defect_area

                                    # Optionally draw bounding boxes for defects
                                    if (
                                        hasattr(defect_result.boxes, "xyxy")
                                        and len(defect_result.boxes.xyxy) > j
                                    ):
                                        x1_d, y1_d, x2_d, y2_d = (
                                            defect_result.boxes.xyxy[j].int().tolist()
                                        )
                                        cv2.rectangle(
                                            frame_resized,
                                            (x1_d, y1_d),
                                            (x2_d, y2_d),
                                            (0, 0, 255),
                                            2,
                                        )
                                        cv2.putText(
                                            frame_resized,
                                            f"ID: {defect_id}",
                                            (x1_d, y1_d - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (0, 0, 255),
                                            1,
                                        )

                    # ------------------------
                    # 4) Detect ending edge
                    # ------------------------
                    if self.state == State.TRACKING_SCANNING:
                        bedsheet_results = bedsheet_model.predict(
                            source=frame_resized, conf=CONF_THRESHOLD, verbose=False
                        )
                        y2_positions = []
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
                                        x1, y1, x2, y2 = map(int, boxes[idx])
                                        y2_positions.append(y2)

                        if y2_positions:
                            y2_max = max(y2_positions)
                            if y2_max < frame_height * 0.90:  # Ending edge detected
                                # If tear_detected is True, auto reject
                                if self.tear_detected:
                                    self.state = (
                                        State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                                    )
                                    self.await_ending_edge = True
                                    self.display_not_clean = True
                                    self.write_decision_to_file(REJECT)
                                    log_print(
                                        f"{self.side} camera: Tear present at ending edge -> Rejected."
                                    )

                                    # Optional: Log tear event as you do with defect data
                                    defect_percent_real_time = (
                                        (self.total_defect_area / DEFAULT_BEDSHEET_AREA)
                                        * 100
                                        if DEFAULT_BEDSHEET_AREA > 0
                                        else 0
                                    )
                                    decision = "Rejected"
                                    log_to_mongo(
                                        self.collection,
                                        self.bedsheet_count + 1,
                                        defect_percent_real_time,
                                        CLEAN_THRESHOLD,
                                        decision,
                                    )
                                    update_history(
                                        self.history_collection,
                                        get_current_date_str(),
                                        CLEAN_THRESHOLD,
                                        decision,
                                    )
                                    self.bedsheet_count += 1
                                    self.reset_defect_tracking_variables()

                                else:
                                    # Usual checks if no tear detected
                                    defect_percent_real_time = (
                                        (self.total_defect_area / DEFAULT_BEDSHEET_AREA)
                                        * 100
                                        if DEFAULT_BEDSHEET_AREA > 0
                                        else 0
                                    )
                                    clean_percent_real_time = (
                                        100 - defect_percent_real_time
                                    )
                                    if clean_percent_real_time >= CLEAN_THRESHOLD:
                                        self.state = State.TRACKING_DECIDED_CLEAN
                                        self.display_not_clean = False
                                        self.write_decision_to_file(ACCEPT)
                                        analysis = (
                                            f"Threshold: {CLEAN_THRESHOLD}%, "
                                            f"Bedsheet {self.bedsheet_count + 1}: Clean. "
                                            f"Dirty Percent: {defect_percent_real_time:.2f}%, "
                                            f"Clean Percent: {clean_percent_real_time:.2f}%"
                                        )
                                        log_print(f"{self.side} camera: {analysis}")
                                        decision = "Accepted"
                                        log_to_mongo(
                                            self.collection,
                                            self.bedsheet_count + 1,
                                            defect_percent_real_time,
                                            CLEAN_THRESHOLD,
                                            decision,
                                        )
                                        update_history(
                                            self.history_collection,
                                            get_current_date_str(),
                                            CLEAN_THRESHOLD,
                                            decision,
                                        )
                                        self.bedsheet_count += 1
                                        self.reset_defect_tracking_variables()
                                        log_print(
                                            f"{self.side} camera: Bedsheet Counted as Clean at Ending Edge"
                                        )
                                    else:
                                        self.state = (
                                            State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                                        )
                                        self.await_ending_edge = True
                                        self.display_not_clean = True
                                        self.write_decision_to_file(REJECT)
                                        analysis = (
                                            f"Threshold: {CLEAN_THRESHOLD}%, "
                                            f"Bedsheet {self.bedsheet_count + 1}: Not Clean. "
                                            f"Dirty Percent: {defect_percent_real_time:.2f}%"
                                        )
                                        log_print(f"{self.side} camera: {analysis}")
                                        decision = "Rejected"
                                        log_to_mongo(
                                            self.collection,
                                            self.bedsheet_count + 1,
                                            defect_percent_real_time,
                                            CLEAN_THRESHOLD,
                                            decision,
                                        )
                                        update_history(
                                            self.history_collection,
                                            get_current_date_str(),
                                            CLEAN_THRESHOLD,
                                            decision,
                                        )
                                        self.bedsheet_count += 1
                                        self.reset_defect_tracking_variables()

                elif self.state == State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE:
                    # Await ending edge detection
                    bedsheet_results = bedsheet_model.predict(
                        source=frame_resized, conf=CONF_THRESHOLD, verbose=False
                    )
                    y2_positions = []
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
                                    x1, y1, x2, y2 = map(int, boxes[idx])
                                    y2_positions.append(y2)

                    if y2_positions:
                        y2_max = max(y2_positions)
                        if y2_max < frame_height * 0.90:  # Ending edge detected
                            self.state = State.IDLE
                            self.await_ending_edge = False
                            self.display_not_clean = False
                            log_print(
                                f"{self.side} camera: Transitioned to IDLE after Not Clean decision."
                            )

                # Display cleanliness info if actively scanning
                if self.state in [
                    State.TRACKING_SCANNING,
                    State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE,
                ]:
                    if DEFAULT_BEDSHEET_AREA > 0:
                        defect_percent = (
                            self.total_defect_area / DEFAULT_BEDSHEET_AREA
                        ) * 100
                    else:
                        defect_percent = 0.0
                    clean_percent = 100 - defect_percent

                    if self.state not in [
                        State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE,
                        State.TRACKING_DECIDED_CLEAN,
                    ]:
                        cv2.putText(
                            frame_resized,
                            f"Cleanliness: {'Clean' if clean_percent >= CLEAN_THRESHOLD else 'Not Clean'}",
                            (10, 390),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (
                                (0, 255, 0)
                                if clean_percent >= CLEAN_THRESHOLD
                                else (0, 0, 255)
                            ),
                            2,
                        )

                # Handle "Not Clean" message
                if (
                    self.display_not_clean
                    and self.state == State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                ):
                    log_print(f"{self.side} camera: Cleanliness: Not Clean")
                    self.state = State.IDLE
                    self.display_not_clean = False
                    self.await_ending_edge = False

                # Handle "Clean" message
                if self.state == State.TRACKING_DECIDED_CLEAN:
                    log_print(f"{self.side} camera: Cleanliness: Clean")
                    self.state = State.IDLE
                    self.await_ending_edge = False

                # Update latest_frame for streaming
                with self.frame_lock:
                    self.latest_frame = frame_resized.copy()

            except Exception as e:
                self.defect_tracking_error = True
                error_code = 1017
                log_bug(
                    f"{self.side} camera: Error during detect processing. Exception: {e} (Error code: {error_code})"
                )
                self.frame_counter += 1

        # Update the previous frame for the next iteration
        self.previous_frame = frame.copy()


class StitchedCameraProcessor:
    def __init__(self, camera_manager, process_mode="horizontal"):
        self.camera_manager = camera_manager
        self.process_mode = process_mode
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.is_active = False
        self.stop_event = threading.Event()
        self.state = State.IDLE  # State management
        self.await_ending_edge = False
        self.display_not_clean = False

        # Initialize defect tracking variables
        self.total_defect_area = 0
        self.unique_defect_ids = set()
        self.defect_max_areas = {}
        self.defect_tracking_error = False

        # Initialize database connections
        self.db = connect_to_mongo()
        self.threshold_collection = self.db["threshold_changes"]
        self.history_collection_hor = self.db["history_horizontal"]
        self.horizontal_collection = get_daily_collection_hor(self.db)
        # Fetch last bedsheet number and threshold
        self.bedsheet_count = get_last_bedsheet_number_hor(self.horizontal_collection)

        global CLEAN_THRESHOLD
        CLEAN_THRESHOLD = get_last_threshold_hor(
            self.threshold_collection, self.history_collection_hor
        )
        print(f"Initialized horizontal camera with threshold: {CLEAN_THRESHOLD}")

        initialize_history_document_hor(
            self.history_collection_hor, get_current_date_str(), CLEAN_THRESHOLD
        )

        # Initialize additional attributes
        self.previous_frame = None  # To store the previous frame for optical flow
        self.frame_counter = 0  # Counter to manage error state resets
        self.detection_enabled = True  # Flag to control detection

    def set_process_mode(self, mode):
        """Set process mode: 'horizontal'"""
        if mode != "horizontal":
            raise ValueError("Invalid process mode for stitched processor")

        self.process_mode = mode
        self.detection_enabled = True  # Enable detection in horizontal mode

    def update_threshold(self, new_threshold):
        global CLEAN_THRESHOLD  # Access the global variable
        if not (0 <= new_threshold <= 100):
            raise ValueError("Threshold must be between 0 and 100.")
        CLEAN_THRESHOLD = new_threshold  # Update the global threshold
        print(f"Updated stitched camera threshold to: {CLEAN_THRESHOLD}")

    def start(self):
        """Start the main loop for stitching frames."""
        self.is_active = True
        self.main_loop_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.main_loop_thread.start()

    def stop(self):
        """Stop the main loop and clean up resources."""
        self.is_active = False
        self.stop_event.set()
        if self.main_loop_thread.is_alive():
            self.main_loop_thread.join()

    def reset_defect_tracking_variables(self):
        self.total_defect_area = 0
        self.unique_defect_ids.clear()
        self.defect_max_areas.clear()

    def main_loop(self):
        """
        Main loop handling horizontal-mode processing, including attempts to reconnect
        if either left or right camera is disconnected.
        """
        while self.is_active and not self.stop_event.is_set():
            try:
                # Run horizontal processing only if process_mode is set to 'horizontal'
                if self.process_mode == "horizontal":
                    if self.detection_enabled:
                        # Attempt to get frames from both left and right cameras
                        left_frame = self.camera_manager.get_frame("left")
                        right_frame = self.camera_manager.get_frame("right")

                        # Briefly wait if any frame is None (possible transient delay)
                        if left_frame is None or right_frame is None:
                            time.sleep(0.1)

                            # Try one more time
                            if left_frame is None:
                                left_frame = self.camera_manager.get_frame("left")
                            if right_frame is None:
                                right_frame = self.camera_manager.get_frame("right")

                            # If still None, treat it as a disconnect and reconnect indefinitely
                            while left_frame is None or right_frame is None:
                                if left_frame is None:
                                    log_print(
                                        "Left camera: No frame received. Attempting reconnection..."
                                    )
                                    self.camera_manager.reconnect_camera("left")
                                if right_frame is None:
                                    log_print(
                                        "Right camera: No frame received. Attempting reconnection..."
                                    )
                                    self.camera_manager.reconnect_camera("right")

                                time.sleep(self.reconnect_delay)

                                # Test frames again after reconnect
                                left_frame = self.camera_manager.get_frame("left")
                                right_frame = self.camera_manager.get_frame("right")

                        # Now that both frames are presumably available, stitch and detect
                        if left_frame is not None and right_frame is not None:
                            stitched_frame = self.stitch_frames(left_frame, right_frame)
                            if stitched_frame is not None:
                                # Update the latest frame reference (thread-safe)
                                with self.frame_lock:
                                    self.latest_frame = stitched_frame

                                # Run horizontal detection on the stitched frame
                                self.detect_horizontal(stitched_frame)

                else:
                    # If not in horizontal mode, or detection is not needed, avoid busy-wait
                    time.sleep(0.1)

            except Exception as e:
                error_code = 1018
                print(
                    f"Error in stitched camera processing: {e} (Error code: {error_code})"
                )

    def stitch_frames(self, left_frame, right_frame):
        """Stitch two frames horizontally after resizing."""
        try:
            # left_frame_resized = cv2.resize(left_frame, (640, 480))
            # right_frame_resized = cv2.resize(right_frame, (640, 480))
            stitched_frame = np.concatenate([left_frame, right_frame], axis=1)
            return stitched_frame
        except Exception as e:
            logging.error(f"Error stitching frames: {e}")
            return None

    # Replace the existing write_decision_to_file method with the one below:
    def write_decision_to_file(self, decision):
        """
        Write the decision to both "decision_left.txt" and "decision_right.txt"
        in a directory outside the AppImage, ensuring it is writable.
        Uses ACCEPT and REJECT from config.
        """
        # Define the writable directory for bug logs
        log_dir = os.path.join(os.getenv("HOME"), "LISA_LOGS")
        os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists

        # Determine the decision value from config
        decision_value = REJECT if decision == REJECT else ACCEPT

        for side in ["left", "right"]:
            decision_file = os.path.join(log_dir, f"decision_{side}.txt")
            try:
                # Write the decision to the file
                with open(decision_file, "w") as file:
                    file.write(
                        str(decision_value)
                    )  # Write the corresponding value (True/False or 1/0)
                print(f"Decision for {side} camera written to {decision_file}.")
            except Exception as e:
                print(f"Failed to write decision for {side} camera: {e}")


    def detect_horizontal(self, stitched_frame):
        """
        Detects the presence of a horizontally laid bedsheet, tracks defects, and
        flags a tear if found. If a tear is detected after the bedsheet start,
        the bedsheet will be automatically rejected at the ending edge.

        Global variables, helper functions, and logging mechanisms (e.g., CLEAN_THRESHOLD,
        DEFAULT_BEDSHEET_AREA, log_print, log_bug) are assumed to exist.
        """

        global CLEAN_THRESHOLD  # Use a global threshold variable for cleanliness

        # 1. Initialize previous frame if needed
        if self.previous_frame is None:
            self.previous_frame = stitched_frame.copy()
            return  # Skip detection until we have an initial previous frame

        # 2. Check for frame size mismatch
        if self.previous_frame.shape != stitched_frame.shape:
            error_code = 1014
            log_bug(
                f"Frame size mismatch: previous_frame {self.previous_frame.shape}, "
                f"current_frame {stitched_frame.shape}(Error code: {error_code})"
            )
            return

        # 3. Prepare the current frame
        frame_resized = stitched_frame
        frame_height = frame_resized.shape[0]

        # 4. Initialize flags and counters
        bedsheet_present = False
        y1_positions, y2_positions = [], []

        # ----------------------------------------------------------------
        # Tear detection setup: once this is True, skip further defect checks
        # ----------------------------------------------------------------
        if not hasattr(self, "tear_detected"):
            self.tear_detected = False  # Create tear_detected attribute if missing

        # Reset error state after 100 frames
        if self.defect_tracking_error and self.frame_counter >= 100:
            self.defect_tracking_error = False
            self.frame_counter = 0

        # 5. Main logic only runs if there's no active defect-tracking error
        if not self.defect_tracking_error:
            try:
                # ----------------------------------------------------
                # STATE: IDLE (Detect Start of Bedsheet)
                # ----------------------------------------------------
                if self.state == State.IDLE and hor_bedsheet_model:
                    try:
                        bedsheet_results = hor_bedsheet_model.predict(
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
                                    # 'class_id == 0' means bedsheet
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

                                        if y1 > frame_height * 0.75:
                                            # Transition to TRACKING_SCANNING
                                            self.state = State.TRACKING_SCANNING
                                            self.reset_defect_tracking_variables()
                                            self.await_ending_edge = False
                                            self.display_not_clean = False
                                            self.tear_detected = False  # Reset tear flag
                                            log_print(
                                                "Transitioned to TRACKING_SCANNING: Starting edge detected."
                                            )
                                            break

                    except Exception as e:
                        error_code = 1015
                        log_bug(
                            f"Error during bedsheet detection. Exception: {e}(Error code: {error_code})"
                        )
                        log_print("Skipping bedsheet detection due to an error.")

                # ----------------------------------------------------
                # STATE: TRACKING_SCANNING (Detect Defects + Tears)
                # ----------------------------------------------------
                elif self.state == State.TRACKING_SCANNING:

                    # 1) Check for tear first (if not already detected)
                    if not self.tear_detected and tear_model:  # Ensure tear_model is loaded
                        try:
                            tear_results = tear_model.predict(
                                source=frame_resized,
                                conf=TEAR_CONF_THRESHOLD,
                                verbose=False,
                            )
                            for result in tear_results:
                                if result.boxes:
                                    boxes, classes, confidences = (
                                        result.boxes.xyxy,
                                        result.boxes.cls,
                                        result.boxes.conf,
                                    )
                                    for idx, class_id in enumerate(classes):
                                        # Suppose tear class is 0 (adjust as needed for your model)
                                        if (
                                            int(class_id) == 0
                                            and confidences[idx] > TEAR_CONF_THRESHOLD
                                        ):
                                            self.tear_detected = True
                                            log_print("Tear detected!")
                                            break
                        except Exception as e:
                            error_code = 1018
                            log_bug(
                                f"Tear detection error. Exception: {e}(Error code: {error_code})"
                            )

                    # 2) If no tear is detected yet, run defect detection
                    defect_results = None
                    if not self.tear_detected and defect_model:
                        if not self.defect_tracking_error:
                            try:
                                defect_results = defect_model.track(
                                    source=frame_resized,
                                    conf=DEFECT_CONF_THRESHOLD,
                                    verbose=False,
                                    persist=True,
                                    tracker=TRACKER_PATH,
                                )
                            except Exception as e:
                                self.defect_tracking_error = True
                                defect_results = None
                                error_code = 1016
                                log_bug(
                                    f"Defect tracking error occurred. Exception: {e}(Error code: {error_code})"
                                )
                                log_print("Skipping defect detection due to an error.")
                        else:
                            log_print("Skipping defect detection due to a prior error.")
                    # else: no defect detection if tear was found

                    # 3) Update defect tracking if no tear was found
                    if defect_results and not self.tear_detected:
                        for defect_result in defect_results:
                            masks = defect_result.masks
                            tracks = (
                                defect_result.boxes.id.cpu().numpy()
                                if defect_result.boxes.id is not None
                                else None
                            )

                            if masks is not None and tracks is not None:
                                for j, mask in enumerate(masks.data):
                                    defect_mask = mask.cpu().numpy()
                                    defect_id = tracks[j]
                                    defect_area = np.sum(defect_mask)

                                    self.unique_defect_ids.add(defect_id)
                                    if defect_id in self.defect_max_areas:
                                        if defect_area > self.defect_max_areas[defect_id]:
                                            self.total_defect_area += (
                                                defect_area
                                                - self.defect_max_areas[defect_id]
                                            )
                                            self.defect_max_areas[defect_id] = defect_area
                                    else:
                                        self.defect_max_areas[defect_id] = defect_area
                                        self.total_defect_area += defect_area

                                    # Draw bounding box if available
                                    if (
                                        hasattr(defect_result.boxes, "xyxy")
                                        and len(defect_result.boxes.xyxy) > j
                                    ):
                                        x1_d, y1_d, x2_d, y2_d = (
                                            defect_result.boxes.xyxy[j].int().tolist()
                                        )
                                        cv2.rectangle(
                                            frame_resized,
                                            (x1_d, y1_d),
                                            (x2_d, y2_d),
                                            (0, 0, 255),
                                            2,
                                        )
                                        cv2.putText(
                                            frame_resized,
                                            f"ID: {defect_id}",
                                            (x1_d, y1_d - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (0, 0, 255),
                                            1,
                                        )

                # ----------------------------------------------------
                # DETECT ENDING EDGE IF STILL TRACKING
                # ----------------------------------------------------
                if self.state == State.TRACKING_SCANNING:
                    bedsheet_results = hor_bedsheet_model.predict(
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
                        # If ending edge is detected
                        if y2_max < frame_height * 0.90:
                            # If tear_detected is True, auto reject
                            if self.tear_detected:
                                self.state = State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                                self.await_ending_edge = True
                                self.display_not_clean = True
                                self.write_decision_to_file(REJECT)

                                defect_percent_real_time = (
                                    (self.total_defect_area / DEFAULT_BEDSHEET_AREA) * 100
                                    if DEFAULT_BEDSHEET_AREA > 0
                                    else 0
                                )
                                clean_percent_real_time = 100 - defect_percent_real_time

                                log_print(
                                    f"Tear present at ending edge -> Rejected. Defect %: {defect_percent_real_time:.2f}%, Clean %: {clean_percent_real_time:.2f}%"
                                )

                                decision = "Rejected"
                                log_to_horizontal(
                                    self.horizontal_collection,
                                    self.bedsheet_count + 1,
                                    defect_percent_real_time,
                                    CLEAN_THRESHOLD,
                                    decision,
                                )
                                update_history_hor(
                                    self.history_collection_hor,
                                    get_current_date_str(),
                                    CLEAN_THRESHOLD,
                                    decision,
                                )
                                log_print(
                                    f"Bedsheet {self.bedsheet_count + 1} logged as 'Not Clean (Tear)'"
                                )
                                self.bedsheet_count += 1
                                self.reset_defect_tracking_variables()
                            else:
                                # Normal clean vs. not clean decision
                                defect_percent_real_time = (
                                    (self.total_defect_area / DEFAULT_BEDSHEET_AREA) * 100
                                    if DEFAULT_BEDSHEET_AREA > 0
                                    else 0
                                )
                                clean_percent_real_time = 100 - defect_percent_real_time

                                if clean_percent_real_time >= CLEAN_THRESHOLD:
                                    self.state = State.TRACKING_DECIDED_CLEAN
                                    self.display_not_clean = False
                                    self.write_decision_to_file(ACCEPT)

                                    analysis_message = (
                                        f"Threshold: {CLEAN_THRESHOLD}%, "
                                        f"Bedsheet {self.bedsheet_count + 1}: Clean. "
                                        f"Dirty Percent: {defect_percent_real_time:.2f}%, "
                                        f"Clean Percent: {clean_percent_real_time:.2f}%"
                                    )
                                    log_print(analysis_message)

                                    decision = "Accepted"
                                    log_to_horizontal(
                                        self.horizontal_collection,
                                        self.bedsheet_count + 1,
                                        defect_percent_real_time,
                                        CLEAN_THRESHOLD,
                                        decision,
                                    )
                                    update_history_hor(
                                        self.history_collection_hor,
                                        get_current_date_str(),
                                        CLEAN_THRESHOLD,
                                        decision,
                                    )
                                    log_print(
                                        f"Bedsheet {self.bedsheet_count + 1} logged as 'Clean'"
                                    )
                                    self.bedsheet_count += 1
                                    self.reset_defect_tracking_variables()
                                    log_print("Ending Edge Detected and Counted as Clean")
                                else:
                                    self.state = State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                                    self.await_ending_edge = True
                                    self.display_not_clean = True
                                    self.write_decision_to_file(REJECT)

                                    analysis_message = (
                                        f"Threshold: {CLEAN_THRESHOLD}%, "
                                        f"Bedsheet {self.bedsheet_count + 1}: Not Clean at Ending Edge. "
                                        f"Dirty Percent: {defect_percent_real_time:.2f}%, "
                                        f"Clean Percent: {clean_percent_real_time:.2f}%"
                                    )
                                    log_print(analysis_message)

                                    decision = "Rejected"
                                    log_to_horizontal(
                                        self.horizontal_collection,
                                        self.bedsheet_count + 1,
                                        defect_percent_real_time,
                                        CLEAN_THRESHOLD,
                                        decision,
                                    )
                                    update_history_hor(
                                        self.history_collection_hor,
                                        get_current_date_str(),
                                        CLEAN_THRESHOLD,
                                        decision,
                                    )
                                    log_print(
                                        f"Bedsheet {self.bedsheet_count + 1} logged as 'Not Clean'"
                                    )
                                    self.bedsheet_count += 1
                                    self.reset_defect_tracking_variables()

                # ----------------------------------------------------
                # STATE: TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                # ----------------------------------------------------
                elif self.state == State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE:
                    # Wait for the bedsheet to fully exit
                    bedsheet_results = hor_bedsheet_model.predict(
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
                        if y2_max < frame_height * 0.90:
                            self.state = State.IDLE
                            self.await_ending_edge = False
                            self.display_not_clean = False
                            log_print(
                                "Transitioned to IDLE: Ending edge detected after Not Clean decision."
                            )

                # ----------------------------------------------------
                # DISPLAY CLEANLINESS INFO IF STILL ACTIVE
                # ----------------------------------------------------
                if self.state in [
                    State.TRACKING_SCANNING,
                    State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE,
                ]:
                    if DEFAULT_BEDSHEET_AREA > 0:
                        defect_percent = (
                            self.total_defect_area / DEFAULT_BEDSHEET_AREA
                        ) * 100
                        clean_percent = 100 - defect_percent
                    else:
                        defect_percent = 0.0
                        clean_percent = 100.0

                    if self.state not in [
                        State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE,
                        State.TRACKING_DECIDED_CLEAN,
                    ]:
                        cv2.putText(
                            frame_resized,
                            f"Cleanliness: {'Clean' if clean_percent >= CLEAN_THRESHOLD else 'Not Clean'}",
                            (10, 390),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (
                                (0, 255, 0)
                                if clean_percent >= CLEAN_THRESHOLD
                                else (0, 0, 255)
                            ),
                            2,
                        )

                # ----------------------------------------------------
                # HANDLE "Not Clean" & "Clean" MESSAGES
                # ----------------------------------------------------
                if (
                    self.display_not_clean
                    and self.state == State.TRACKING_DECIDED_NOT_CLEAN_PREMATURE
                ):
                    log_print("Cleanliness: Not Clean")
                    self.state = State.IDLE
                    self.display_not_clean = False
                    self.await_ending_edge = False

                if self.state == State.TRACKING_DECIDED_CLEAN:
                    log_print("Cleanliness: Clean")
                    self.state = State.IDLE
                    self.await_ending_edge = False

                # Update latest_frame for streaming
                with self.frame_lock:
                    self.latest_frame = frame_resized.copy()

            except Exception as e:
                self.defect_tracking_error = True
                error_code = 1016
                log_bug(
                    f"Defect tracking error occurred. Exception: {e}(Error code: {error_code})"
                )
                self.frame_counter += 1

        # 6. Update the previous frame
        self.previous_frame = stitched_frame.copy()


# Function to start the Uvicorn server
def start_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000)


# FastAPI endpoints
@app.post("/update_threshold")
async def update_threshold(request: Request):
    global CLEAN_THRESHOLD  # Declare the global variable
    data = await request.json()
    new_threshold = data.get("threshold")
    side = data.get("side", "left")  # Default to 'left' if side not specified

    # Validate the side
    if side not in camera_processors and side != "horizontal":
        return JSONResponse(content={"error": "Invalid side."}, status_code=400)

    # Ensure the new threshold is valid
    if isinstance(new_threshold, (int, float)) and 0 <= new_threshold <= 100:
        print(f"Updating threshold to {new_threshold}")
        CLEAN_THRESHOLD = new_threshold  # Update the global threshold

        # Update all camera processors to use the new threshold
        for camera_processor in camera_processors.values():
            camera_processor.update_threshold(new_threshold)

        # Log the change in the database
        if side == "horizontal" and stitched_camera_processor:
            initialize_history_document_hor(
                stitched_camera_processor.history_collection_hor,
                get_current_date_str(),
                CLEAN_THRESHOLD,
            )
            log_threshold_change(
                stitched_camera_processor.threshold_collection, CLEAN_THRESHOLD
            )
        else:
            camera_processor = camera_processors[side]
            initialize_history_document(
                camera_processor.history_collection,
                get_current_date_str(),
                CLEAN_THRESHOLD,
            )
            log_threshold_change(camera_processor.threshold_collection, CLEAN_THRESHOLD)

        return JSONResponse(content={"message": "Threshold updated successfully."})
    else:
        return JSONResponse(
            content={"error": "Invalid threshold value."}, status_code=400
        )


@app.get("/get_current_threshold/{mode}")
async def get_current_threshold(mode: str):
    if mode == "horizontal":
        return {"threshold": CLEAN_THRESHOLD}
    elif mode == "vertical":
        threshold = CLEAN_THRESHOLD
        return {"threshold": threshold}
    else:
        camera_processor = camera_processors[mode]
        return {"threshold": CLEAN_THRESHOLD}


# Define the available processing modes
PROCESS_MODES = ["vertical", "horizontal"]


@app.get("/current_feed")
async def get_current_feed():
    if camera_processors["left"].is_active and camera_processors["right"].is_active:
        active_feed = "horizontal"  # Both cameras active, horizontal mode
    elif camera_processors["left"].is_active:
        active_feed = "left"  # Only left camera active, show left feed
    elif camera_processors["right"].is_active:
        active_feed = "right"  # Only right camera active, show right feed
    else:
        active_feed = "none"  # No active feed
    return {"activeFeed": active_feed}


@app.get("/set_process_mode/{mode}")
async def set_process_mode(mode: str):
    global stitched_camera_processor

    if mode not in ["vertical", "horizontal"]:
        raise HTTPException(status_code=400, detail="Invalid process mode")

    if mode == "horizontal":
        # Create and start the stitched camera processor if it doesn't exist
        if stitched_camera_processor is None:
            stitched_camera_processor = StitchedCameraProcessor(camera_manager)
            stitched_camera_processor.start()

        # Set horizontal mode for both camera processors
        for side in camera_processors:
            camera_processors[side].set_process_mode("horizontal")
        log_print("Switched to horizontal mode.")
    else:
        # Stop the stitched camera processor if it exists
        if stitched_camera_processor is not None:
            stitched_camera_processor.stop()
            stitched_camera_processor = None  # Clear the reference

        # Set vertical mode for both camera processors
        for side in camera_processors:
            camera_processors[side].set_process_mode("vertical")
        log_print("Switched to vertical mode.")

    return {"message": f"Process mode set to {mode}"}


@app.get("/frame_status/{mode}")
async def frame_status(mode: str):
    if mode == "horizontal":
        if stitched_camera_processor is None:
            return JSONResponse({"frame_available": False})
        camera_processor = stitched_camera_processor
    else:
        if mode not in camera_processors:
            return JSONResponse({"frame_available": False})
        camera_processor = camera_processors[mode]

    # Check if the latest frame is available
    frame_available = camera_processor.latest_frame is not None
    return JSONResponse({"frame_available": frame_available})


@app.get("/video_feed/{mode}")
async def video_feed(mode: str):
    if mode == "horizontal":
        if stitched_camera_processor is None:
            raise HTTPException(
                status_code=404, detail="Stitched camera processor not found"
            )
        camera_processor = stitched_camera_processor
    else:
        if mode not in camera_processors:
            raise HTTPException(status_code=404, detail="Camera not found")
        camera_processor = camera_processors[mode]

    placeholder_frame = cv2.imencode(
        ".jpg",
        cv2.putText(
            np.zeros((480, 640, 3), dtype=np.uint8),
            "Waiting for frames...",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        ),
    )[1].tobytes()

    async def generate():
        last_frame = None  # Store the last frame
        while True:
            frame = camera_processor.latest_frame
            if frame is None:
                # Send placeholder frame if no frame is available
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + placeholder_frame + b"\r\n"
                )
                await asyncio.sleep(0.1)  # Retry after delay
                continue

            # Check if the frame has changed since the last one using np.array_equal
            if last_frame is None or not np.array_equal(frame, last_frame):
                ret, buffer = cv2.imencode(".jpg", frame)
                if not ret:
                    # Handle frame encoding failure gracefully
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + placeholder_frame
                        + b"\r\n"
                    )
                    await asyncio.sleep(0.1)
                    continue

                # Send the actual frame if it's different from the last one
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                last_frame = frame  # Update last frame
            await asyncio.sleep(0.01)  # Adjust for performance if necessary

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/daily_analytics")
async def get_combined_daily_analytics():
    response_data = []
    aggregated_data = {}

    # Process both cameras (left and right)
    for side, camera_processor in camera_processors.items():
        # Access both vertical and horizontal history collections
        history_collection = camera_processor.history_collection
        horizontal_history_collection = camera_processor.history_collection_hor

        # Aggregate data from vertical history collection
        data = list(history_collection.find())
        for item in data:
            date_str = item.get("date", "Unknown")
            if isinstance(date_str, str) and date_str != "Unknown":
                # Use the date string directly
                date_str = date_str
            else:
                continue  # Skip this entry if the date is unknown or not a string

            if date_str not in aggregated_data:
                aggregated_data[date_str] = {
                    "total_bedsheets": 0,
                    "total_accepted": 0,
                    "total_rejected": 0,
                }

            aggregated_data[date_str]["total_bedsheets"] += item.get(
                "total_bedsheets", 0
            )
            aggregated_data[date_str]["total_accepted"] += item.get("total_accepted", 0)
            aggregated_data[date_str]["total_rejected"] += item.get("total_rejected", 0)

        # Aggregate data from horizontal history collection
        horizontal_data = list(horizontal_history_collection.find())
        for item in horizontal_data:
            date_str = item.get("date", "Unknown")
            if isinstance(date_str, str) and date_str != "Unknown":
                # Use the date string directly
                date_str = date_str
            else:
                continue  # Skip this entry if the date is unknown or not a string

            if date_str not in aggregated_data:
                aggregated_data[date_str] = {
                    "total_bedsheets": 0,
                    "total_accepted": 0,
                    "total_rejected": 0,
                }

            aggregated_data[date_str]["total_bedsheets"] += item.get(
                "total_bedsheets", 0
            )
            aggregated_data[date_str]["total_accepted"] += item.get("total_accepted", 0)
            aggregated_data[date_str]["total_rejected"] += item.get("total_rejected", 0)

    # Format the aggregated data
    for date_str, counts in aggregated_data.items():
        # Convert the date string to a datetime object
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # Format the date to include the weekday
        formatted_date = date_obj.strftime("%Y-%m-%d")

        response_data.append(
            {
                "date": formatted_date,  # Use the formatted date
                "total_bedsheets": counts["total_bedsheets"],
                "total_accepted": counts["total_accepted"],
                "total_rejected": counts["total_rejected"],
            }
        )

    return JSONResponse(content=response_data)


@app.get("/monthly_analytics")
async def get_combined_monthly_analytics():
    aggregated_data = {}

    # Process both cameras (left and right)
    for side, camera_processor in camera_processors.items():
        # Access both vertical and horizontal history collections
        history_collection = camera_processor.history_collection
        horizontal_history_collection = camera_processor.history_collection_hor

        # Aggregate data from vertical history collection
        data = list(history_collection.find())
        for item in data:
            date_str = item.get("date", "Unknown")
            if date_str == "Unknown":
                continue

            try:
                # Parse the date string to get the month and year
                date_obj = datetime.strptime(
                    date_str.strip(), "%Y-%m-%d"
                )  # Strip any whitespace
                month_year = date_obj.strftime("%B %Y")  # Get month name and year

                if month_year not in aggregated_data:
                    aggregated_data[month_year] = {
                        "total_bedsheets": 0,
                        "accepted": 0,
                        "rejected": 0,
                    }

                aggregated_data[month_year]["total_bedsheets"] += item.get(
                    "total_bedsheets", 0
                )
                aggregated_data[month_year]["accepted"] += item.get("total_accepted", 0)
                aggregated_data[month_year]["rejected"] += item.get("total_rejected", 0)

            except ValueError as e:
                print(f"Error parsing date '{date_str}': {e}")

        # Aggregate data from horizontal history collection
        horizontal_data = list(horizontal_history_collection.find())
        for item in horizontal_data:
            date_str = item.get("date", "Unknown")
            if date_str == "Unknown":
                continue

            try:
                # Parse the date string to get the month and year
                date_obj = datetime.strptime(
                    date_str.strip(), "%Y-%m-%d"
                )  # Strip any whitespace
                month_year = date_obj.strftime("%B %Y")  # Get month name and year

                if month_year not in aggregated_data:
                    aggregated_data[month_year] = {
                        "total_bedsheets": 0,
                        "accepted": 0,
                        "rejected": 0,
                    }

                aggregated_data[month_year]["total_bedsheets"] += item.get(
                    "total_bedsheets", 0
                )
                aggregated_data[month_year]["accepted"] += item.get("total_accepted", 0)
                aggregated_data[month_year]["rejected"] += item.get("total_rejected", 0)

            except ValueError as e:
                print(f"Error parsing horizontal date '{date_str}': {e}")

    # Format the aggregated data
    response_data = []
    for month_year, counts in aggregated_data.items():
        response_data.append(
            {
                "month_year": month_year,
                "total_bedsheets": counts["total_bedsheets"],
                "accepted": counts["accepted"],
                "rejected": counts["rejected"],
            }
        )

    return JSONResponse(content=response_data)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming data and respond accordingly
            if data == "get_counts":
                counts = {
                    "left": camera_processors["left"].bedsheet_count,
                    "right": camera_processors["right"].bedsheet_count,
                }
                await websocket.send_json(counts)
            elif data == "get_horizontal_counts" and "horizontal" in camera_processors:
                horizontal_counts = {
                    "horizontal": camera_processors["horizontal"].bedsheet_count,
                }
                await websocket.send_json(horizontal_counts)
    except WebSocketDisconnect:
        print("Client disconnected")


# WebSocket endpoint for today's counts
@app.websocket("/ws/todays_counts/{side}")
async def websocket_todays_counts(websocket: WebSocket, side: str):
    await websocket.accept()
    try:
        while True:
            local_timezone = pytz.timezone(TIMEZONE)
            today = datetime.now(local_timezone).strftime("%Y-%m-%d")

            if side == "horizontal":
                # Handle horizontal processor
                if stitched_camera_processor is None:
                    await websocket.close(code=1003)  # Close with error code
                    return

                # Fetch today's counts for horizontal mode from the history collection
                history_collection_hor = (
                    stitched_camera_processor.history_collection_hor
                )
                item = history_collection_hor.find_one({"date": today})

                if item:
                    response_data = {
                        "date": item.get("date", "Unknown"),
                        "total_bedsheets": item.get("total_bedsheets", 0),  # Example
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
            else:
                camera_processor = camera_processors.get(side)
                if camera_processor is None:
                    await websocket.close(code=1003)  # Close with error code
                    return

                history_collection = camera_processor.history_collection
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

            await websocket.send_text(json.dumps(response_data))
            await asyncio.sleep(1)  # Adjust the interval as needed
    except WebSocketDisconnect:
        print(f"Client disconnected from {side} camera WebSocket")


if __name__ == "__main__":
    # Start the Uvicorn server in a separate thread
    uvicorn_thread = threading.Thread(target=start_uvicorn, daemon=True)
    uvicorn_thread.start()

    # Initialize the CameraManager
    camera_manager = (
        CameraManager()
    )  # One CameraManager instance to handle both cameras
    camera_manager.initialize_video_capture("left", VIDEO_SOURCE_LEFT)
    camera_manager.initialize_video_capture("right", VIDEO_SOURCE_RIGHT)

    # Create CameraProcessor instances
    camera_processors["left"] = CameraProcessor("left", camera_manager)
    camera_processors["right"] = CameraProcessor("right", camera_manager)

    # Start the camera processors
    camera_processors["left"].start()
    camera_processors["right"].start()

    try:
        # Wait for both camera processors to finish
        camera_processors["left"].main_loop_thread.join()
        camera_processors["right"].main_loop_thread.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Stopping cameras and cleaning up...")
    finally:
        # Stop the camera processors and release resources
        camera_processors["left"].stop()
        camera_processors["right"].stop()
        if stitched_camera_processor is not None:
            stitched_camera_processor.stop()
        # Release video resources from the camera manager
        camera_manager.release_video_resources()

        # Optionally, you can also destroy OpenCV windows
        print("All OpenCV windows destroyed.")
