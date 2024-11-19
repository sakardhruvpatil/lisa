import snap7
from snap7.util import set_bool
import time

import os

# PLC connection parameters
PLC_IP = '192.168.1.200'  # Replace with your PLC's IP address
RACK = 0
SLOT = 1

# Create a client to connect to the PLC
plc = snap7.client.Client()
try:
    # Connect to the PLC
    print(f"Connecting to PLC @{PLC_IP}...")
    plc.connect(PLC_IP, RACK, SLOT)
    
    if not plc.get_connected():
        raise RuntimeError("Failed to connect to the PLC. Check IP address and network configuration.")

    def write_boolean_to_plc(db_number, byte_address, bit_number, bool_value):

        # Read the current byte at the specified address
        data = plc.db_read(db_number, byte_address, 1)
        
        # Set the boolean value into the byte buffer
        set_bool(data, 0, bit_number, bool_value)
        
        # Write the modified byte back to the PLC
        try:
            plc.db_write(db_number, byte_address, data)
            print(f"PLC Write: DB{db_number}, Byte {byte_address}, Bit {bit_number}, Value {bool_value}")
        except Exception as e:
            print(f"Failed to write to the PLC: {e}")
            raise

    # Example DB settings
    db_number = 1  # The DB number to write to (ensure this DB exists in the PLC)
    byte_address = 0  # Start at byte 0 (DBX0.x)
    
    def read_signal_from_file(filepath):
        """ Reads the signal from the specified file path. """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return f.read().strip()
        return None  # Return None if the file doesn't exist yet

    def str_to_bool(s):
        """Converts a string to boolean based on common true/false string values."""
        return s.lower() in ['true', '1', 'yes', 'y']


    while True:
    # Read signals from files
        signal_code1 = read_signal_from_file('/home/sakar2/lisa-test_branch_python/test/demo_baumer/dashboard/backend/decision.txt')

        if signal_code1 is not None:
            signal_code1_bool = str_to_bool(signal_code1)
            print(f"Signal from Code: {signal_code1} (as bool: {signal_code1_bool})")
            write_boolean_to_plc(db_number, byte_address, 0, signal_code1_bool)  # DBX0.0
        else:
            print("No signal from Code 1")

        time.sleep(1)

        root.mainloop()
except Exception as e:
    print(f"Error: {e}")
