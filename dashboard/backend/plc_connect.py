import time
import snap7
from snap7.util import set_bool, set_int
from time import sleep

# PLC connection parameters
PLC_IP = '192.168.1.200'  # Replace with your PLC's IP address
RACK = 0
SLOT = 1

# Create a client to connect to the PLC
plc = snap7.client.Client()

def read_current_speed():
    try:
        # Open the log.txt file and read its contents
        with open("/home/sakar04/Documents/Ronak/lisa/dashboard/server/log.txt", "r") as log_file:
            lines = log_file.readlines()
            
            # Find the last line that contains the speed information
            last_line = lines[-1] if lines else None
            
            # Extract current speed from the last line (expected format: "Current Speed: <value>")
            if last_line and "Current Speed:" in last_line:
                current_speed = int(last_line.split(":")[1].strip())  # Extract the integer value
                return current_speed
            else:
                return None

    except Exception as e:
        print(f"Error reading from log.txt: {e}")
        return None

def write_boolean_to_plc(db_number, byte_address, bit_number, bool_value):
    """
    Writes a single boolean value to the specified bit in a byte on the Siemens S7 PLC.
    db_number: Data Block number
    byte_address: The byte within the DB where the boolean is located
    bit_number: The specific bit (0-7) in the byte
    bool_value: The boolean value to write (True or False)
    """
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

def write_integer_to_plc(db_number2, start_byte, value):
    """
    Writes an integer value to the PLC.
    """
    # Read the DB area
    data2 = plc.db_read(db_number2, start_byte, 2)  # Read 2 bytes for an integer

    # Set the integer value in the data block
    set_int(data2, 0, value)

    # Write the modified data back to the PLC
    try:
        plc.db_write(db_number2, start_byte, data2)
        print(f"PLC Write: DB{db_number2}, Start Byte {start_byte}, Integer Value {value}")
    except Exception as e:
        print(f"Failed to write integer to PLC: {e}")
        raise

def continuously_monitor_log():
    last_checked = None
    while True:
        current_speed = read_current_speed()
        if current_speed is not None:
            if current_speed != last_checked:
                print(f"Current Speed is: {current_speed}")
                
                # Write the corresponding boolean values to PLC based on the current speed
                db_number = 1  # The DB number to write to (ensure this DB exists in the PLC)
                byte_address = 0  # Start at byte 0 (DBX0.x)
                db_number2 = 2  # The data block number for integer
                start_byte = 0  # Start byte for integer value

                # Example speed control logic (just an example, adjust to your needs)
                if current_speed > 50:
                    print("Sending 'Accepted' command...")
                    write_boolean_to_plc(db_number, byte_address, 0, True)  # DBX0.0
                    write_boolean_to_plc(db_number, byte_address, 2, True)  # DBX0.2
                    write_integer_to_plc(db_number2, start_byte, current_speed)
                    print("Accepted")
                else:
                    print("Sending 'Rejected' command...")
                    write_boolean_to_plc(db_number, byte_address, 0, False)  # DBX0.0
                    write_boolean_to_plc(db_number, byte_address, 2, False)  # DBX0.1
                    write_integer_to_plc(db_number2, start_byte, current_speed)
                    print("Rejected")
                
                last_checked = current_speed
        else:
            print("No current speed found in log file.")
        
        # Wait for a while before checking the file again
        time.sleep(0.1)  # Adjust the interval as needed

if __name__ == "__main__":
    try:
        # Connect to the PLC
        print(f"Connecting to PLC @{PLC_IP}...")
        plc.connect(PLC_IP, RACK, SLOT)
        
        if not plc.get_connected():
            raise RuntimeError("Failed to connect to the PLC. Check IP address and network configuration.")
        
        # Start monitoring the log file and interact with the PLC
        continuously_monitor_log()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Disconnect from the PLC
        if plc.get_connected():
            plc.disconnect()
        print("Disconnected from PLC")
