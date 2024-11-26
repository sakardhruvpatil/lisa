import snap7
from snap7.util import set_bool, set_int
import time
import os

class PLCController:
    def __init__(self, ip, rack=0, slot=1):
        self.plc = snap7.client.Client()
        try:
            print(f"Connecting to PLC @{ip}...")
            self.plc.connect(ip, rack, slot)
            if self.plc.get_connected():
                print("Successfully connected to the PLC.")
            else:
                print("Failed to connect to the PLC.")
        except Exception as e:
            print(f"Error connecting to PLC: {e}")

    def write_boolean_to_plc(self, db_number, byte_address, bit_number, bool_value):
        try:
            data = self.plc.db_read(db_number, byte_address, 1)
            set_bool(data, 0, bit_number, bool_value)
            self.plc.db_write(db_number, byte_address, data)
            print(f"Wrote boolean to DB{db_number}, Byte {byte_address}, Bit {bit_number}: {bool_value}")
        except Exception as e:
            print(f"Error writing boolean to PLC: {e}")

    def write_integer_to_plc(self, db_number, start_byte, value):
        try:
            data = self.plc.db_read(db_number, start_byte, 2)
            set_int(data, 0, value)
            self.plc.db_write(db_number, start_byte, data)
            print(f"Wrote integer to DB{db_number}, Byte {start_byte}: {value}")
        except Exception as e:
            print(f"Error writing integer to PLC: {e}")

    def change_speed(self, action):
        db_number = 1
        byte_address = 0
        db_number2 = 2
        start_byte = 0

        if action == 'increase':
            self.write_boolean_to_plc(db_number, byte_address, 0, True)  # DBX0.0
            self.write_integer_to_plc(db_number2, start_byte, 70)  # Example value for increase
        elif action == 'decrease':
            self.write_boolean_to_plc(db_number, byte_address, 0, False)  # DBX0.0
            self.write_integer_to_plc(db_number2, start_byte, 50)  # Example value for decrease

    def disconnect(self):
        if self.plc.get_connected():
            self.plc.disconnect()
            print("Disconnected from PLC.")

def read_signal_from_file(filepath):
    """Reads the signal from the specified file path."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return f.read().strip()
    return None  # Return None if the file doesn't exist yet

def str_to_bool(s):
    """Converts a string to boolean based on common true/false string values."""
    return s.lower() in ['true', '1', 'yes', 'y']

if __name__ == "__main__":
    plc_ip = '192.168.1.200'  # Replace with your PLC's IP address
    decision_file = 'decision.txt'  # Path to your decision file
    plc_controller = PLCController(plc_ip)
    
    try:
        while True:
            signal = read_signal_from_file(decision_file)
            if signal is not None:
                bool_signal = str_to_bool(signal)
                print(f"Read signal from file: {bool_signal}")
                
                if bool_signal:
                    action = 'increase'
                else:
                    action = 'decrease'
                
                plc_controller.change_speed(action)
            else:
                print(f"No decision found in {decision_file}. Waiting for input...")
            
            time.sleep(5)  # Wait for 5 seconds before checking again
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        plc_controller.disconnect()