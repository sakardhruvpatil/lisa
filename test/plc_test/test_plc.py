import snap7
from snap7.util import set_bool
import time

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

    def write_boolean_to_plc(db_number, start_byte, bool_value):
        """
        Writes a single boolean value to the specified DB on the Siemens S7 PLC.
        """
        # Define buffer size (1 boolean = 1 bit in a byte)
        data = bytearray(1)
        
        # Set the boolean value into the buffer
        set_bool(data, start_byte, 0, bool_value)
        
        # Write the buffer to the PLC
        try:
            plc.db_write(db_number, start_byte, data)
            print(f"PLC Write {'Accepted' if bool_value else 'Rejected'}: {bool_value}")
        except Exception as e:
            print(f"Failed to write to the PLC: {e}")
            raise

    # Example DB settings
    db_number = 1  # The DB number to write to (ensure this DB exists in the PLC)
    start_byte = 0  # Start at byte 0 (DBX0.0)

    # Alternate between sending Accepted and Rejected for 5 seconds each
    for _ in range(5):  # Loop 5 times (alternates commands)
        # Send "Accepted" command
        print("Sending 'Accepted' command...")
        write_boolean_to_plc(db_number, start_byte, True)
        print("Accepted")
        time.sleep(5)  # Wait for 5 seconds

        # Send "Rejected" command
        print("Sending 'Rejected' command...")
        write_boolean_to_plc(db_number, start_byte, False)
        print("Rejected")
        time.sleep(5)  # Wait for 5 seconds

except Exception as e:
    print(f"Error: {e}")
finally:
    # Disconnect from the PLC
    if plc.get_connected():
        plc.disconnect()
    print("Disconnected from PLC")