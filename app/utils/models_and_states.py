# models_and_states.py
# Model Loading and FSM Defining
import sys
sys.path.insert(0, '/home/dp/lisa/app/utils')
sys.path.insert(0, '/home/dp/lisa/app/config')
from enum import Enum
from ultralytics import YOLO
from config import BEDSHEET_MODEL_PATH, DEFECT_MODEL_PATH
from logger import log_bug, log_print
import torch

# Finite State Machine
class State(Enum):
    IDLE = 0
    TRACKING_SCANNING = 1
    TRACKING_DECIDED_NOT_CLEAN_PREMATURE = 2
    TRACKING_DECIDED_CLEAN = 3

# Model loading with error handling
def load_model(model_path, model_name):
    try:
        model = YOLO(model_path, task='segment')
        log_print(f"{model_name} model loaded successfully.")
        return model
    except Exception as e:
        log_bug(f"Failed to load {model_name} model from {model_path}. Exception: {e}")
        log_print(f"{model_name} model not loaded. Detection will be skipped.")
        return None

# Load models
bedsheet_model = load_model(BEDSHEET_MODEL_PATH, "Bedsheet")
defect_model = load_model(DEFECT_MODEL_PATH, "Defect")
