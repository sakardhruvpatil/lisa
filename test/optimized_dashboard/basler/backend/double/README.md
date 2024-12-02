
# Motor Control README

This document provides instructions on setting up and using the `motor_control.py` script for controlling motors via Modbus. It also explains how to tune the PID parameters and includes an example script (`example_call.py`) to demonstrate how to call the `MotorController` class.

---

## Table of Contents

1. [Setup](#setup)
2. [Configuration](#configuration)
3. [Tuning the PID Controller](#tuning-the-pid-controller)
4. [Example Script](#example-script)

---

## Setup

### Requirements

Ensure you have the following installed:

- `pymodbus` library for Modbus communication

### Installation

Install `pymodbus` via pip:

```bash
pip install pymodbus
```

---

## Configuration

The `MotorConfig` class in `motor_control.py` contains default parameters for the motor and PID controller. Update these values as needed based on your hardware setup.

### Key Parameters

- **Port**: Specify the serial port connected to the motor driver (e.g., `/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_B002FFVD-if00-port0`).
- **Baudrate**: Communication speed, typically `38400`.
- **Driver ID**: Unique ID for the motor driver (default is `1`).
- **Encoder Ticks per Revolution**: Set the ticks per motor revolution (e.g., `10000`).
- **PID Gains**: Adjust `kp`, `ki`, and `kd` for proportional, integral, and derivative gains.
- **PID Output Limit**: Max allowable PID output ( Max and Min RPM)
- **PID Integral Limit**: Max allowable integral value.
- **Gearbox Ratio**: Ratio of motor rotations to shaft rotations.
- **Feedback Interval**: Time interval for PID feedback (in seconds).

Update the `MotorConfig` section in the `motor_control.py` file:

```python
class MotorConfig:
    def __init__(self):
        self.port = '/dev/...'  
        self.baudrate = 38400
        self.timeout = 0.1
        self.driver_id = 1
        self.encoder_ticks_per_revolution = 10000
        self.kp = 0.1
        self.ki = 0.0
        self.kd = 0.0
        self.gear_box_ratio = 20
        self.feedback_interval = 0.1
        self.pid_output_limit = 1000
        self.pid_integral_limit = 1000
```

---

## Tuning the PID Controller

Fine-tune the PID gains (`kp`, `ki`, `kd`) to achieve the desired motor behavior. Adjust these values in the `MotorConfig` class:

- **Proportional Gain (`kp`)**: Controls the response speed.
- **Integral Gain (`ki`)**: Corrects steady-state errors.
- **Derivative Gain (`kd`)**: Reduces overshoot and oscillations.

### Tuning Process

1. Start with a low `kp` and increase until the motor responds promptly without overshooting.
2. Gradually increase `ki` to eliminate steady-state errors.
3. Adjust `kd` to smooth out oscillations.

---

## Example Script

The `example_call.py` script demonstrates how to use the `MotorController` class in another script.

### `example_call.py`

```python
from motor_control import MotorSystem

motor_system = MotorSystem()
motor_system.start_motor_with_target_rotations(target_rotations=2.0, duration=15)
```

---

## Notes

- Make sure your motor driver and encoder are properly connected before running the scripts.
- Always test with low PID values and gradually increase them.
- Use the `emergency_stop()` method to safely stop the motor in case of an issue.
