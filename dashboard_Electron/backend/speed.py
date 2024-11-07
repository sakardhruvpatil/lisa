
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def main(action, current_speed):
    current_speed = int(current_speed)
    if action == 'increase':
        new_speed = current_speed + 10  # Example increment
    elif action == 'decrease':
        new_speed = max(0, current_speed - 10)  # Prevent negative speed
    else:
        new_speed = current_speed  # No change for invalid action

    # Log information (you can choose to log to stdout if necessary)
    logging.info(f"Action: {action}, Current Speed: {current_speed}, New Speed: {new_speed}")

    print(new_speed)  # Output the new speed to stdout

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Invalid arguments")
        sys.exit(1)
    
    action = sys.argv[1]
    current_speed = sys.argv[2]
    main(action, current_speed)

