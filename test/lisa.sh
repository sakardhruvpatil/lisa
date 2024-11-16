#!/bin/bash

# Define directories
BACKEND_DIR=~/Documents/Sarthak/SakarRobotics/lisa/test/dashboard/backend
DASHBOARD_DIR=~/Documents/Sarthak/SakarRobotics/lisa/test/dashboard

# Function to open a new terminal and execute a command
open_terminal() {
    gnome-terminal -- bash -c "$1; exec bash"
}

# Start the backend server in a new terminal
open_terminal "cd \"$BACKEND_DIR\" && python3 rejection_database.py"

sleep 5

# Start the dashboard server in a new terminal
open_terminal "cd \"$DASHBOARD_DIR\" && serve -s build"

# Wait for a few seconds to ensure the dashboard server starts
sleep 5

# Open Chromium browser to the dashboard URL
google-chrome http://localhost:3000
