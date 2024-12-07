#!/bin/bash

# Exit the script immediately if any command fails
set -e


# Granting the user 'sakar03' permission to run sudo without a password
echo "Granting 'sakarws03' user passwordless sudo permissions..."
sudo bash -c 'echo "sakarws03 ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers'


# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install Python 3.10.12
echo "Installing Python 3.10.12..."
sudo apt install -y python3.10

# Verify if Python 3 is installed
echo "Verifying Python installation..."
python3.10 --version

# Install python3-pip if not already installed
echo "Installing python3-pip..."
sudo apt install -y python3-pip python3.10-distutils


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Reload systemd to apply the new service
sudo systemctl daemon-reload

sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.6.0-cuda-12.6_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.6.0-cuda-12.6/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

sudo apt-get install tensorrt

sudo apt install libgl1-mesa-dri libgl1-mesa-glx libxcb-xinerama0 libxcb-xinput0 libxcb-cursor0

sudo apt-get install ./pylon_*.deb ./codemeter*.deb

# Run the Pylon USB setup script and automatically answer 'yes' to prompts
echo "Running Pylon USB setup..."
yes | sudo bash /opt/pylon/share/pylon/setup-usb.sh

# Install required Python packages
echo "Installing Python packages..."
pip install Flask
pip install numpy
pip install opencv_contrib_python
pip install opencv_python
pip install psycopg2_binary
pip install pyserial
pip install filterpy
pip install ultralytics
pip install uvicorn # Install Gunicorn for production
pip install python-dotenv
pip install pymodbus
pip install pypylon
pip install python-snap7
pip install pymongo
pip install pytz
pip install fastapi
pip install asyncio


sudo apt-get install gnupg curl

curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc |    sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg    --dearmor

yes | echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

sudo apt-get update

sudo apt-get install -y mongodb-org

sudo systemctl start mongod

sudo systemctl enable mongod

curl -sL https://deb.nodesource.com/setup_20.x -o nodesource_setup.sh

sudo bash nodesource_setup.sh

sudo apt install nodejs

# Write the new configurations to sysctl.conf
echo "vm.max_map_count=128000" | sudo tee -a /etc/sysctl.conf
echo "vm.swappiness=0" | sudo tee -a /etc/sysctl.conf

# Apply the changes
sysctl -p


# Reload systemd to apply the new service
sudo systemctl daemon-reload

echo "Setup complete."

# Restart system to apply udev rules
echo "System restart is required to apply udev rules."
read -p "Do you want to reboot now? (y/n): " reboot_now
if [[ "$reboot_now" == "y" || "$reboot_now" == "Y" || "$reboot_now" == "yes" || "$reboot_now" == "Yes" ]]; then
    sudo reboot
else
    echo "Please reboot the system manually to apply udev rules."
fi
