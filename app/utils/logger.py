# logger.py
# Bud logging
from datetime import datetime
from config.config import BUG_LOG_FILE  # Import from config directory


def log_bug(bug_message):
    with open(BUG_LOG_FILE, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} - BUG: {bug_message}\n")
    print(f"BUG LOGGED: {bug_message}")

def log_print(message):
    print(message)  # Print to console
