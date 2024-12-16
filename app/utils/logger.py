# logger.py
# Bud logging
from datetime import datetime
from config.config import get_bug_log_file  # Import the function dynamically

def log_bug(bug_message):
    try:
        log_file = get_bug_log_file()  # Get the current day's log file
        with open(log_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - BUG: {bug_message}\n")
        print(f"BUG LOGGED: {bug_message}")
    except OSError as e:
        print(f"Failed to log bug due to a file system error: {e}")

def log_print(message):
    print(message)  # Print to console
