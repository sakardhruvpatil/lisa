# config.py

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "lisa_db"

# Log file
BUG_LOG_FILE = "bug_log.txt"

# Video configuration

DEFAULT_BEDSHEET_AREA = 10000  # Predefined bedsheet area in pixels
CONF_THRESHOLD = 0.8
DEFECT_CONF_THRESHOLD = 0.01

VIDEO_SOURCE_LEFT = 0
VIDEO_SOURCE_RIGHT = 2

# Model paths
BEDSHEET_MODEL_PATH = "/home/sakarws03/Documents/Sarthak/SakarRobotics/lisa/test/models/bedsheet_v11.pt"
DEFECT_MODEL_PATH = "/home/sakarws03/Documents/Sarthak/SakarRobotics/lisa/test/models/defect.pt"

# Tracker path
TRACKER_PATH = "/home/sakarws03/Documents/Sarthak/SakarRobotics/lisa/test/models/botsort_defect.yaml"

# Threshold
DEFAULT_THRESHOLD = 95.0

# Date/Time configuration
TIMEZONE = "Asia/Kolkata"

# CSV filenames
LOGS_FILENAME_TEMPLATE = "logs_{date}.csv"
THRESHOLD_FILENAME = "threshold_changes.csv"
HISTORY_FILENAME = "history.csv"

# Video properties
VIDEO_FPS = 25  # Default FPS if not available

# Define the amount to crop from the left and right
CROP_LEFT = 1  # Pixels to crop from the left
CROP_RIGHT = 1  # Pixels to crop from the right
CROP_TOP = 130  # Pixels to crop from the top
CROP_BOTTOM = 130  # Pixels to crop from the bottom


