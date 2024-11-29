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

# Model paths
BEDSHEET_MODEL_PATH = "/home/sr10/Documents/lisa/test/models/bedsheet_v11.pt"
DEFECT_MODEL_PATH = "/home/sr10/Documents/lisa/test/models/defect.pt"

# Tracker path
TRACKER_PATH = "/home/sr10/Documents/lisa/test/models/botsort_defect.yaml"

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
CROP_LEFT = 150  # Pixels to crop from the left
CROP_RIGHT = 80  # Pixels to crop from the right
CROP_TOP = 130  # Pixels to crop from the top
CROP_BOTTOM = 130  # Pixels to crop from the bottom

# Camera serial numbers
SERIAL_NUMBER_LEFT = "700008707086"  # Replace with your left camera's serial number
SERIAL_NUMBER_RIGHT = "700009740793"  # Replace with your right camera's serial number
