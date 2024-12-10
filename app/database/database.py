# database.py
# Logging Data to MongoDB

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from datetime import datetime, timezone
import pytz
import time
import sys
sys.path.insert(0, '/home/dp/lisa/app/utils')
sys.path.insert(0, '/home/dp/lisa/app/config')
from logger import log_bug, log_print
import numpy as np
from config import MONGO_URI, DB_NAME, TIMEZONE, DEFAULT_THRESHOLD


# Helper function to get the current date as a string
def get_current_date_str():
    local_timezone = pytz.timezone(TIMEZONE)
    return datetime.now(local_timezone).strftime("%Y-%m-%d")


# MongoDB setup with bug tracking
def connect_to_mongo():
    max_retries = 5
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            client.server_info()
            db = client[DB_NAME]
            log_print(f"Connected to MongoDB on attempt {attempt + 1}")
            return db
        except PyMongoError as e:
            log_bug(
                f"Attempt {attempt + 1} failed to connect to MongoDB. Exception: {e}"
            )
            if attempt < max_retries - 1:
                log_print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise SystemExit(
                    "Unable to connect to MongoDB after multiple attempts."
                )


# Get daily collection
def get_daily_collection(db, side):
    try:
        local_timezone = pytz.timezone(TIMEZONE)
        date_str = datetime.now(local_timezone).strftime("%Y%m%d")
        return db[f"logs_{side}_{date_str}"]
    except PyMongoError as e:
        log_bug(f"Failed to access daily collection for {side}. Exception: {e}")
        raise

# Get daily collection
def get_daily_collection_hor(db):
    try:
        local_timezone = pytz.timezone(TIMEZONE)
        date_str = datetime.now(local_timezone).strftime("%Y%m%d")
        return db[f"logs_hor_{date_str}"]
    except PyMongoError as e:
        log_bug(f"Failed to access daily collection for hor. Exception: {e}")
        raise


# Fetch the last bedsheet number from the logs
def get_last_bedsheet_number(collection):
    last_entry = collection.find_one(sort=[("bedsheet_number", -1)])
    return last_entry.get("bedsheet_number", 0) if last_entry else 0

# Fetch the last bedsheet number from the logs
def get_last_bedsheet_number_hor(horizontal_collection):
    last_entry = horizontal_collection.find_one(sort=[("bedsheet_number", -1)])
    return last_entry.get("bedsheet_number", 0) if last_entry else 0


# Log a document to MongoDB
def log_to_mongo(
    collection, bedsheet_number, detected_threshold, set_threshold, decision
):
    try:
        local_timezone = pytz.timezone(TIMEZONE)
        timestamp = datetime.now(local_timezone)
        document = {
            "bedsheet_number": int(bedsheet_number),
            "detected_threshold": float(detected_threshold),
            "set_threshold": int(set_threshold),
            "decision": decision,
            "timestamp": timestamp,
        }
        collection.insert_one(document)
    except Exception as e:
        log_bug(f"Failed to log to MongoDB. Document: {document}. Exception: {e}")


# Function to log to the horizontal collection
def log_to_horizontal(
    horizontal_collection, bedsheet_number, detected_threshold, set_threshold, decision
):
    try:
        local_timezone = pytz.timezone(TIMEZONE)
        timestamp = datetime.now(local_timezone)
        document = {
            "bedsheet_number": int(bedsheet_number),
            "detected_threshold": float(detected_threshold),
            "set_threshold": int(set_threshold),
            "decision": decision,
            "timestamp": timestamp,
        }
        horizontal_collection.insert_one(document)
    except Exception as e:
        log_bug(
            f"Failed to log to horizontal MongoDB. Document: {document}. Exception: {e}"
        )


# Log threshold changes to MongoDB
def log_threshold_change(threshold_collection, threshold_value):
    threshold_value = int(threshold_value)
    local_timezone = pytz.timezone(TIMEZONE)
    timestamp = datetime.now(local_timezone)
    last_entry = threshold_collection.find_one(sort=[("timestamp", -1)])
    if last_entry is None or last_entry.get("set_threshold") != threshold_value:
        document = {"set_threshold": threshold_value, "timestamp": timestamp}
        threshold_collection.insert_one(document)
        return True
    return False


# Initialize history document
def initialize_history_document(history_collection, date, threshold):
    history_collection.update_one(
        {"date": date},
        {
            "$setOnInsert": {
                "date": date,
                "total_bedsheets": 0,
                "total_accepted": 0,
                "total_rejected": 0,
                "thresholds": [],
            }
        },
        upsert=True,
    )

# Initialize history document
def initialize_history_document_hor(history_collection_hor, date, threshold):
    history_collection_hor.update_one(
        {"date": date},
        {
            "$setOnInsert": {
                "date": date,
                "total_bedsheets": 0,
                "total_accepted": 0,
                "total_rejected": 0,
                "thresholds": [],
            }
        },
        upsert=True,
    )

# Add a new threshold entry with counters for accepted and rejected
def add_threshold_entry(history_collection, date, threshold):
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
        },
    )

# Add a new threshold entry with counters for accepted and rejected
def add_threshold_entry_hor(history_collection_hor, date, threshold):
    history_collection_hor.update_one(
        {"date": date},
        {
            "$push": {
                "thresholds": {
                    "set_threshold": threshold,
                    "accepted": 0,
                    "rejected": 0,
                }
            }
        },
    )


# Update history document
def update_history(history_collection, date, threshold, decision):
    try:
        # Increment total bedsheets count
        history_collection.update_one({"date": date}, {"$inc": {"total_bedsheets": 1}})

        # Increment total accepted/rejected for the day based on decision
        if decision == "Accepted":
            history_collection.update_one(
                {"date": date}, {"$inc": {"total_accepted": 1}}
            )
        else:
            history_collection.update_one(
                {"date": date}, {"$inc": {"total_rejected": 1}}
            )

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
                        {"$inc": {"thresholds.$.accepted": 1}},
                    )
                    log_print(
                        f"Incremented thresholds.$.accepted: {result.modified_count} document(s) updated."
                    )
                else:
                    result = history_collection.update_one(
                        {"date": date, "thresholds.set_threshold": threshold},
                        {"$inc": {"thresholds.$.rejected": 1}},
                    )
                    log_print(
                        f"Incremented thresholds.$.rejected: {result.modified_count} document(s) updated."
                    )
            else:
                log_print(
                    f"No matching threshold found for {threshold}, adding a new threshold entry."
                )
                # If the threshold has changed or if there is no previous threshold, add a new threshold entry
                add_threshold_entry(history_collection, date, threshold)

                # Initialize accepted or rejected count for the new threshold entry
                if decision == "Accepted":
                    result = history_collection.update_one(
                        {"date": date, "thresholds.set_threshold": threshold},
                        {"$inc": {"thresholds.$.accepted": 1}},
                    )
                    log_print(
                        f"Added and incremented thresholds.$.accepted: {result.modified_count} document(s) updated."
                    )
                else:
                    result = history_collection.update_one(
                        {"date": date, "thresholds.set_threshold": threshold},
                        {"$inc": {"thresholds.$.rejected": 1}},
                    )
                    log_print(
                        f"Added and incremented thresholds.$.rejected: {result.modified_count} document(s) updated."
                    )
        else:
            # If the document is missing, initialize it
            initialize_history_document(history_collection, date, threshold)
    except Exception as e:
        log_bug(
            f"Failed to update history. Date: {date}, Threshold: {threshold}, Decision: {decision}. Exception: {e}"
        )

# Update history document
def update_history_hor(history_collection_hor, date, threshold, decision):
    try:
        # Increment total bedsheets count
        history_collection_hor.update_one({"date": date}, {"$inc": {"total_bedsheets": 1}})

        # Increment total accepted/rejected for the day based on decision
        if decision == "Accepted":
            history_collection_hor.update_one(
                {"date": date}, {"$inc": {"total_accepted": 1}}
            )
        else:
            history_collection_hor.update_one(
                {"date": date}, {"$inc": {"total_rejected": 1}}
            )

        # Fetch the current document for the specified date
        doc = history_collection_hor.find_one({"date": date})
        # Ensure the document exists before accessing its fields
        if doc is not None:
            thresholds = doc.get("thresholds", [])

            # Check if there is an existing entry for the current threshold
            if thresholds and thresholds[-1]["set_threshold"] == threshold:
                log_print(f"Matching threshold found: {threshold}")
                # Update the accepted or rejected count for the most recent threshold entry
                if decision == "Accepted":
                    result = history_collection_hor.update_one(
                        {"date": date, "thresholds.set_threshold": threshold},
                        {"$inc": {"thresholds.$.accepted": 1}},
                    )
                    log_print(
                        f"Incremented thresholds.$.accepted: {result.modified_count} document(s) updated."
                    )
                else:
                    result = history_collection_hor.update_one(
                        {"date": date, "thresholds.set_threshold": threshold},
                        {"$inc": {"thresholds.$.rejected": 1}},
                    )
                    log_print(
                        f"Incremented thresholds.$.rejected: {result.modified_count} document(s) updated."
                    )
            else:
                log_print(
                    f"No matching threshold found for {threshold}, adding a new threshold entry."
                )
                # If the threshold has changed or if there is no previous threshold, add a new threshold entry
                add_threshold_entry_hor(history_collection_hor, date, threshold)

                # Initialize accepted or rejected count for the new threshold entry
                if decision == "Accepted":
                    result = history_collection_hor.update_one(
                        {"date": date, "thresholds.set_threshold": threshold},
                        {"$inc": {"thresholds.$.accepted": 1}},
                    )
                    log_print(
                        f"Added and incremented thresholds.$.accepted: {result.modified_count} document(s) updated."
                    )
                else:
                    result = history_collection_hor.update_one(
                        {"date": date, "thresholds.set_threshold": threshold},
                        {"$inc": {"thresholds.$.rejected": 1}},
                    )
                    log_print(
                        f"Added and incremented thresholds.$.rejected: {result.modified_count} document(s) updated."
                    )
        else:
            # If the document is missing, initialize it
            initialize_history_document_hor(history_collection_hor, date, threshold)
    except Exception as e:
        log_bug(
            f"Failed to update history. Date: {date}, Threshold: {threshold}, Decision: {decision}. Exception: {e}"
        )


# Function to get the last threshold or set default
def get_last_threshold(threshold_collection, history_collection):
    # Fetch the last threshold entry
    last_entry = threshold_collection.find_one(sort=[("timestamp", -1)])
    if last_entry and "set_threshold" in last_entry:
        return last_entry["set_threshold"]
    else:
        # If no threshold is found, use default and log it
        threshold_value = DEFAULT_THRESHOLD
        threshold_logged = log_threshold_change(threshold_collection, threshold_value)
        if threshold_logged:
            add_threshold_entry(
                history_collection, get_current_date_str(), threshold_value
            )
        return threshold_value


# Function to get the last threshold or set default
def get_last_threshold_hor(threshold_collection, history_collection_hor):
    # Fetch the last threshold entry
    last_entry = threshold_collection.find_one(sort=[("timestamp", -1)])
    if last_entry and "set_threshold" in last_entry:
        return last_entry["set_threshold"]
    else:
        # If no threshold is found, use default and log it
        threshold_value = DEFAULT_THRESHOLD
        threshold_logged = log_threshold_change(threshold_collection, threshold_value)
        if threshold_logged:
            add_threshold_entry_hor(
                history_collection_hor, get_current_date_str(), threshold_value
            )
        return threshold_value