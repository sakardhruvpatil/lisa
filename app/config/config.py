# config.py
#Configuring Parameters like Model Path, Camera IP, MongoDB Configurations.
import os

# Dynamically calculate the base directory as the app directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "lisa_db"

# Log file
BUG_LOG_FILE = os.path.join(BASE_DIR, "bug_log.txt")

# Video configuration

DEFAULT_BEDSHEET_AREA = 10000  # Predefined bedsheet area in pixels
CONF_THRESHOLD = 0.8
DEFECT_CONF_THRESHOLD = 0.01

# Update video sources to use IP addresses for Basler cameras
VIDEO_SOURCE_LEFT = "192.168.1.11"  # IP address of the left Basler camera
VIDEO_SOURCE_RIGHT = "192.168.1.21"  # IP address of the right Basler camera

# Paths to your PFS files
LEFT_CAMERA_PFS = os.path.join(BASE_DIR, "config/left_camera_config.pfs")
RIGHT_CAMERA_PFS =  os.path.join(BASE_DIR, "config/right_camera_config.pfs")


# Model paths
BEDSHEET_MODEL_PATH = os.path.join(BASE_DIR, "models/bedsheet_v11.engine")
DEFECT_MODEL_PATH = os.path.join(BASE_DIR, "models/defect.engine")

# Tracker path
TRACKER_PATH = os.path.join(BASE_DIR, "models/botsort_defect.yaml")

# Threshold
DEFAULT_THRESHOLD = 95.0

# Date/Time configuration
TIMEZONE = "Asia/Kolkata"

# CSV filenames
LOGS_FILENAME_TEMPLATE = os.path.join(BASE_DIR, "logs/logs_{date}.csv")
THRESHOLD_FILENAME = os.path.join(BASE_DIR, "logs/threshold_changes.csv")
HISTORY_FILENAME = os.path.join(BASE_DIR, "logs/history.csv")

# Video properties
VIDEO_FPS = 25  # Default FPS if not available

# Define the amount to crop from the left and right
CROP_LEFT = 1  # Pixels to crop from the left
CROP_RIGHT = 1  # Pixels to crop from the right
CROP_TOP = 130  # Pixels to crop from the top
CROP_BOTTOM = 130  # Pixels to crop from the bottom


