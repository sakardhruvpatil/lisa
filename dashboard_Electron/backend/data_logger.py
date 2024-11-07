import logging
import sys

# Configure logging
logging.basicConfig(
   
    level=logging.INFO,        # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_data(action, current_speed, new_speed):
    logging.info(f'Action: {action}, Current Speed: {current_speed}, New Speed: {new_speed}')

def change_speed(action, current_speed):
    if action == 'increase':
        new_speed = min(current_speed + 10, 100)  # Increase speed by 10, max 100
    elif action == 'decrease':
        new_speed = max(current_speed - 10, 0)    # Decrease speed by 10, min 0
    else:
        new_speed = current_speed  # No change if action is invalid
    log_data(action, current_speed, new_speed)
    return new_speed

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_logger.py <action> <current_speed>")
        sys.exit(1)

    action = sys.argv[1]
    current_speed = int(sys.argv[2])
    new_speed = change_speed(action, current_speed)
    print(new_speed)


