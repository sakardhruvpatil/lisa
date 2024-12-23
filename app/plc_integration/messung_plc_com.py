from pymodbus.client import ModbusTcpClient
import os
import time
import sys

# Add the necessary directories to sys.path dynamically
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(APP_DIR)

from utils.logger import log_bug, log_print  # Now you can import from utils.logger

# Define the directory for external files (must match the earlier implementation)
LOG_DIR = os.path.join(os.getenv("HOME"), "LISA_LOGS")

client = ModbusTcpClient(host="192.168.1.30", port=502)


def connect_modbus():
    """Connect to Modbus server and keep retrying if it fails."""
    while not client.connect():
        error_code = 1019
        log_bug(
            f"Failed to connect to Modbus server. Retrying... (Error code: {error_code})"
        )
        time.sleep(5)  # Wait for 5 seconds before retrying
    log_print("Connected to Modbus server successfully.")
    return True


def read_signal_from_file(filename):
    """
    Reads the signal from a specified file in the LOG_DIR directory.
    Expects the file to contain either '0' or '1'.
    """
    filepath = os.path.join(LOG_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            signal = f.read().strip()
            if signal in ["0", "1"]:
                return int(signal)  # Convert to integer 0 or 1
            else:
                error_code = 1020
                log_bug(
                    f"Invalid signal value in {filename}. Expected '0' or '1', got '{signal}'. (Error code: {error_code})"
                )
    return None  # Return None if the file doesn't exist or has invalid content


def write_signal_to_file(filename, value):
    """
    Writes a specified value ('0' or '1') to the given file in the LOG_DIR directory.
    """
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w") as f:
        f.write(str(value))  # Write '0' or '1' to the file


def read_current_speed(logFilePath, log_file_name):
    """Reads the current speed from the log file."""
    logFilePath = os.path.join(LOG_DIR, log_file_name)
    try:
        with open(logFilePath, "r") as file:
            lines = file.readlines()
            last_line = lines[-1] if lines else None
            if last_line and "Current Speed:" in last_line:
                try:
                    speed = int(last_line.split(":")[1].strip())
                    return speed
                except ValueError:
                    error_code = 1024
                    log_bug(
                        f"Error parsing speed from line '{last_line}'. (Error code: {error_code})"
                    )
                    return None
            else:
                error_code = 1025
                log_bug(
                    f"Last line in {logFilePath} does not contain 'Current Speed:'. (Error code: {error_code})"
                )
    except Exception as e:
        error_code = 1021
        log_bug(f"Error reading from {logFilePath}: {e}. (Error code: {error_code})")
    return None


def write_registers(signal_code1, signal_code2, current_speed):
    """Writes signal codes and current speed to the Modbus registers."""
    if signal_code1 is not None:
        client.write_register(0, int(signal_code1))  # Ensure it's an integer
    if signal_code2 is not None:
        client.write_register(2, int(signal_code2))  # Ensure it's an integer
    if current_speed is not None:
        client.write_register(1, int(current_speed))  # Example: writing to register 2 for speed


# Main execution
if connect_modbus():
    try:
        while True:
            try:
                # Read signals and current speed
                signal_code1 = read_signal_from_file("decision_left.txt")
                signal_code2 = read_signal_from_file("decision_right.txt")
                current_speed = read_current_speed(LOG_DIR, "log.txt")

                # Write the read values to the Modbus registers
                write_registers(
                    signal_code1, signal_code2, current_speed
                )

                # If signal is 1, overwrite the file with 0 after 1 second
                if signal_code1 == 1:
                    time.sleep(1)  # Wait for 1 second before updating the file
                    write_signal_to_file("decision_left.txt", 0)  # Overwrite with 0
                if signal_code2 == 1:
                    time.sleep(1)  # Wait for 1 second before updating the file
                    write_signal_to_file("decision_right.txt", 0)  # Overwrite with 0

                # Add a delay to avoid overloading the system
                time.sleep(1)  # Sleep for 1 second (adjust as necessary)

                # Check if the connection is still alive, if not, reconnect
                if not client.is_socket_open():
                    error_code = 1022
                    log_bug(
                        f"Lost connection to Modbus server. Reconnecting... (Error code: {error_code})"
                    )
                    connect_modbus()  # Try to reconnect

            except Exception as e:
                error_code = 1026
                log_bug(f"An error occurred: {e}. (Error code: {error_code})")
                time.sleep(5)  # Wait before retrying in case of an error

    except KeyboardInterrupt:
        error_code = 1023
        log_bug(f"Process interrupted. Closing connection. (Error code: {error_code})")
        # Gracefully close the client connection before exiting
        if client.is_socket_open():
            client.close()  # Close the Modbus connection properly
        print("Modbus connection closed, exiting...")
    finally:
        if client.is_socket_open():
            client.close()  # Ensure the connection is closed when done
        print("Process terminated successfully.")