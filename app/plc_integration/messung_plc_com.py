from pymodbus.client import ModbusTcpClient
import os
import time

client = ModbusTcpClient(host='127.0.0.1', port=502)

def connect_modbus():
    """ Connect to Modbus server and check connection. """
    if not client.connect():
        print("Failed to connect to Modbus server.")
        return False
    return True

def read_signal_from_file(filepath):
    """Reads the signal from the specified file path."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return f.read().strip()
    return None  # Return None if the file doesn't exist yet

def read_current_speed(log_filepath):
    """Reads the current speed from the log file."""
    try:
        with open(log_filepath, "r") as log_file:
            lines = log_file.readlines()
            last_line = lines[-1] if lines else None
            if last_line and "Current Speed:" in last_line:
                return int(last_line.split(":")[1].strip())
    except Exception as e:
        print(f"Error reading from log.txt: {e}")
    return None

def write_registers(signal_code1, signal_code2, current_speed):
    """Writes signal codes and current speed to the Modbus registers."""
    if signal_code1 is not None:
        client.write_register(0, int(signal_code1))  # Ensure it's an integer
    if signal_code2 is not None:
        client.write_register(1, int(signal_code2))  # Ensure it's an integer
    if current_speed is not None:
        client.write_register(2, int(current_speed))  # Example: writing to register 2 for speed

# Main execution
if connect_modbus():
    try:
        while True:
            # Read signals and current speed
            signal_code1 = read_signal_from_file('decision_left.txt')
            signal_code2 = read_signal_from_file('decision_right.txt')
            current_speed = read_current_speed("log.txt")
            
            # Write the read values to the Modbus registers
            write_registers(signal_code1, signal_code2, current_speed)
            
            # Add a delay to avoid overloading the system
            time.sleep(1)  # Sleep for 1 second (adjust as necessary)

    except KeyboardInterrupt:
        print("Process interrupted. Closing connection.")
    finally:
        client.close()  # Ensure the connection is closed when done