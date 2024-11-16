# config.py

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "lisa_db"

# Log file
BUG_LOG_FILE = "bug_log.txt"

# Video configuration
video_path = ("/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/media/video001.avi")
VIDEO_SOURCE = 0  # Camera source
DEFAULT_BEDSHEET_AREA = 70000  # Predefined bedsheet area in pixels
CONF_THRESHOLD = 0.8
DEFECT_CONF_THRESHOLD = 0.01

# Model paths
BEDSHEET_MODEL_PATH = "/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/bedsheet_v11.engine"
DEFECT_MODEL_PATH = "/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/defect.engine"

# Tracker path
TRACKER_PATH = "/home/sakar03/Documents/Sarthak/SakarRobotics/lisa/test/models/botsort_defect.yaml"

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
