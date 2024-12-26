# config.py
#Configuring Parameters like Model Path, Camera IP, MongoDB Configurations.
import os
from datetime import datetime

# Dynamically calculate the base directory as the app directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "lisa_db"

# Log file: Ensure writable directory outside the AppImage
BUG_LOG_DIR = os.path.join(os.getenv('HOME'), "LISA_LOGS")
os.makedirs(BUG_LOG_DIR, exist_ok=True)  # Ensure the directory exists

# Dynamic bug log file path
def get_bug_log_file():
    current_date = datetime.now().strftime("%Y-%m-%d")  # Get today's date
    return os.path.join(BUG_LOG_DIR, f"bug_log_{current_date}.txt")  # Generate file path for the date

# Video configuration

DEFAULT_BEDSHEET_AREA = 50000  # Predefined bedsheet area in pixels
CONF_THRESHOLD = 0.8
DEFECT_CONF_THRESHOLD = 0.01
TEAR_CONF_THRESHOLD = 0.5

# Update video sources to use IP addresses for Basler cameras
VIDEO_SOURCE_LEFT = "192.168.1.20"  # IP address of the left Basler camera
VIDEO_SOURCE_RIGHT = "192.168.1.10"  # IP address of the right Basler camera

# Paths to your PFS files
LEFT_CAMERA_PFS = os.path.join(BASE_DIR, "config/left_camera_config.pfs")
RIGHT_CAMERA_PFS =  os.path.join(BASE_DIR, "config/right_camera_config.pfs")


# Model paths
BEDSHEET_MODEL_PATH = os.path.join(BASE_DIR, "models/bedsheet_v11.engine")
DEFECT_MODEL_PATH = os.path.join(BASE_DIR, "models/defect.engine")
HOR_BEDSHEET_MODEL_PATH = os.path.join(BASE_DIR, "models/bedsheet_v11_hor.engine")
TEAR_MODEL_PATH = os.path.join(BASE_DIR, "models/tear.engine")

# Tracker path
TRACKER_PATH = os.path.join(BASE_DIR, "models/botsort_defect.yaml")

# Threshold
DEFAULT_THRESHOLD = 95.0

# Configuring Parameters for Accept/Reject decision
ACCEPT = 0  # Can also be True, False, 1, or 0 as per your requirement
REJECT = 1  # Can also be True, False, 1, or 0 as per your requirement

# Date/Time configuration
TIMEZONE = "Asia/Kolkata"

# CSV filenames
LOGS_FILENAME_TEMPLATE = os.path.join(BASE_DIR, "logs/logs_{date}.csv")
THRESHOLD_FILENAME = os.path.join(BASE_DIR, "logs/threshold_changes.csv")
HISTORY_FILENAME = os.path.join(BASE_DIR, "logs/history.csv")

# Define the amount to crop from the left and right
CROP_LEFT = 1  # Pixels to crop from the left
CROP_RIGHT = 1  # Pixels to crop from the right
CROP_TOP = 1  # Pixels to crop from the top
CROP_BOTTOM = 1  # Pixels to crop from the bottom


