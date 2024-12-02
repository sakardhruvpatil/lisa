#!/bin/bash

# Define directories
BACKEND_DIR=~/home/sr10/Documents/lisa/test/hadapsar/backend/double
DASHBOARD_DIR=~/home/sr10/Documents/lisa/test/hadapsar

# Function to open a new terminal and execute a command
open_terminal() {
    gnome-terminal -- bash -c "$1; exec bash"
}


# Navigate to the deploy folder and run the Python file for connecting to PLC in a new terminal
open_terminal "cd \"$BACKEND_DIR\" && python3 5secon_Off.py"

# Wait for a few seconds to ensure the PLC connection script starts properly
sleep 5

# Start the backend server in a new terminal
open_terminal "cd \"$BACKEND_DIR\" && python3 main.py"

sleep 10

# Start the dashboard server in a new terminal
open_terminal "cd \"$DASHBOARD_DIR\" && npm run electron"
