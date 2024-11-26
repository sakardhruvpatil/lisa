import time
from pymodbus.client import ModbusSerialClient
import logging
import struct

# logging.basicConfig(level=logging.DEBUG)


def calculate_crc(data):
    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        for _ in range(8):
            if (crc & 1) != 0:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    return crc


def build_command(slave_id, function_code, address, data):
    command = [
        slave_id,
        function_code,
        address >> 8,
        address & 0xFF,
        data >> 8,
        data & 0xFF,
    ]
    crc = calculate_crc(command)
    command.append(crc & 0xFF)
    command.append((crc >> 8) & 0xFF)
    return bytes(command)


def int_to_bytes(value, length):
    """Convert an integer to bytes"""
    return value.to_bytes(length, byteorder="big", signed=True)


class MotorController:
    def __init__(self, port, slave_id, baudrate=38400, timeout=1):
        self.client = ModbusSerialClient(
            # method='rtu',
            port=port,
            baudrate=baudrate,
            timeout=timeout,
        )
        self.slave_id = slave_id
        self.initial_ticks = 0
        if not self.client.connect():
            raise Exception(
                f"Failed to connect to the Modbus server on port {port} for slave ID {slave_id}"
            )

    def send_command(self, command):
        logging.debug(f"Sending command to slave ID {self.slave_id}: {command}")
        response = self.client.send(command)
        logging.debug(f"Received response from slave ID {self.slave_id}: {response}")
        return response

    def set_mode_velocity_control(self):
        command = build_command(self.slave_id, 6, 0x6200, 0x0002)
        self.send_command(command)
        time.sleep(0.1)

    def set_mode_position_control(self):
        command = build_command(self.slave_id, 6, 0x6200, 0x0001)
        self.send_command(command)
        time.sleep(0.1)

    def set_velocity(self, rpm):
        rpm_bytes = int_to_bytes(rpm, 2)
        data = struct.unpack(">H", rpm_bytes)[0]
        command = build_command(self.slave_id, 6, 0x6203, data)
        self.send_command(command)
        time.sleep(0.1)

    def set_position(self, position):

        if not (-2147483648 <= position <= 2147483647):
            raise ValueError("Position out of range")

        # # Split the position into high and low bits
        # high_bit = (position >> 16) & 0xFFFF
        # low_bit = position & 0xFFFF

        # Convert position to bytes
        position_bytes = struct.pack(">i", position)
        high_bit = struct.unpack(">H", position_bytes[:2])[0]
        low_bit = struct.unpack(">H", position_bytes[2:])[0]

        # Send the high bit command
        high_bit_command = build_command(self.slave_id, 6, 0x6201, high_bit)
        self.send_command(high_bit_command)
        time.sleep(0.1)

        # Send the low bit command
        low_bit_command = build_command(self.slave_id, 6, 0x6202, low_bit)
        self.send_command(low_bit_command)
        time.sleep(0.1)

        self.start_motion()
        time.sleep(0.1)

    def get_position(self):
        """#! NOT WORKING, HAVE TO FIX THIS"""
        # Read the high bit from the register
        high_bit = self.read_register(0x6201)
        # Read the low bit from the register
        low_bit = self.read_register(0x6202)

        # Log the values for debugging
        logging.debug(f"High bit: {high_bit}, Low bit: {low_bit}")

        # Ensure the values are within the valid range
        if not (0 <= high_bit <= 65535):
            raise ValueError(f"High bit out of range: {high_bit}")
        if not (0 <= low_bit <= 65535):
            raise ValueError(f"Low bit out of range: {low_bit}")

        # Combine high and low bits to get the full position
        position_bytes = struct.pack(">HH", high_bit, low_bit)
        position = struct.unpack(">i", position_bytes)[0]

        return position

    def set_acceleration(self, acceleration):
        command = build_command(self.slave_id, 16, 0x6204, acceleration)
        self.send_command(command)
        time.sleep(0.1)

    def set_deceleration(self, deceleration):
        command = build_command(self.slave_id, 16, 0x6205, deceleration)
        self.send_command(command)
        time.sleep(0.1)

    def set_direction_and_velocity(self, direction, rpm):
        if direction == "reverse":
            rpm = -rpm  # Set negative RPM for reverse
        self.set_velocity(rpm)

    def start_motion(self):
        command = build_command(self.slave_id, 6, 0x6002, 0x0010)
        self.send_command(command)
        time.sleep(0.1)

    def stop_motion(self):
        command = build_command(self.slave_id, 6, 0x6002, 0x0040)
        self.send_command(command)
        time.sleep(0.1)

    def read_register(self, address):
        response = self.client.read_holding_registers(address, 2, self.slave_id)
        # print('Response:', response)
        if response.isError():
            raise Exception("Error reading 32-bit register data")
        if response.registers:
            h, l = response.registers
            value = (h << 16) + l
            if value >= 0x80000000:
                value -= 0x100000000
            return value
        else:
            raise Exception(f"No data received from address {address}")

    def reset_encoder(self):
        self.initial_ticks = self.read_register(0x0B1C)

    def get_encoder_ticks(self):
        return self.read_register(0x0B1C) - self.initial_ticks

    def close(self):
        self.client.close()


if __name__ == "__main__":
    motor1 = MotorController(port="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_B002FLVV-if00-port0", slave_id=1)

    try:
        # Set motor to velocity control mode
        logging.info("Setting motor 1 to velocity control mode")
        motor1.set_mode_position_control()

        # Define positions
        pos1 = -16000
        pos2 = 3000
        
        while True:  # Infinite loop
            # Move to pos1
            logging.info(f"Setting position to {pos1}")
            motor1.set_position(pos1)
            time.sleep(0.5)  # Wait for motor to reach position
            
            encoder_ticks_1 = motor1.read_register(0x0B1C)
            print(f"Encoder ticks after pos1: {encoder_ticks_1}")

            # Move to pos2 # Hi Prasad - Nishad
            logging.info(f"Setting position to {pos2}")
            motor1.set_position(pos2)
            time.sleep(1.0)  # Wait for motor to reach position

            encoder_ticks_2 = motor1.read_register(0x0B1C)
            print(f"Encoder ticks after pos2: {encoder_ticks_2}")

    except KeyboardInterrupt:
        print("Exiting program...")

    finally:
        motor1.stop_motion()
        motor1.close()
