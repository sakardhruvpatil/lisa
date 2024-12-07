#!/bin/bash

# Exit the script immediately if any command fails
set -e

# Folder URL and credentials
FOLDER_URL="https://sakarrobotics-my.sharepoint.com/:f:/p/sarthak/EljjtElqDUtDh0wQLB_gRu0B9qovbLSYnPix7LlPxaRpFw?e=zHVi3o"
DEST_DIR="$HOME/dependencies"

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


# Install dependencies for file downloading
echo "Installing dependencies for downloading files..."
sudo apt install -y wget unzip curl

# Create a directory to store the downloaded dependencies
echo "Creating directory for dependencies..."
mkdir -p "$DEST_DIR"

# Download dependencies
echo "Downloading dependencies from SharePoint..."
wget -O "$DEST_DIR/dependencies.zip" "$FOLDER_URL"

# Extract the downloaded files
echo "Extracting dependencies..."
unzip "$DEST_DIR/dependencies.zip" -d "$DEST_DIR"

# Navigate to the dependencies folder
cd "$DEST_DIR"

# Install TensorRT
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.6.0-cuda-12.6_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.6.0-cuda-12.6/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y tensorrt

# Install dependencies for Pylon and Codemeter
sudo apt install -y libgl1-mesa-dri libgl1-mesa-glx libxcb-xinerama0 libxcb-xinput0 libxcb-cursor0
sudo apt-get install ./pylon_*.deb ./codemeter*.deb

# Run the Pylon USB setup script and automatically answer 'yes' to prompts
echo "Running Pylon USB setup..."
yes | sudo bash /opt/pylon/share/pylon/setup-usb.sh


# Install NVIDIA Toolkit
echo "Installing NVIDIA Toolkit..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6


# Install required Python packages
echo "Installing Python packages..."
pip install Flask numpy opencv_contrib_python opencv_python psycopg2_binary pyserial \
    filterpy ultralytics uvicorn python-dotenv pymodbus pypylon python-snap7 pymongo pytz fastapi asyncio

# Install MongoDB
sudo apt-get install -y gnupg curl
curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod

# Install Node.js
curl -sL https://deb.nodesource.com/setup_20.x -o nodesource_setup.sh
sudo bash nodesource_setup.sh
sudo apt install -y nodejs

# Update system parameters
echo "vm.max_map_count=128000" | sudo tee -a /etc/sysctl.conf
echo "vm.swappiness=0" | sudo tee -a /etc/sysctl.conf
sysctl -p

# Reload systemd
sudo systemctl daemon-reload

# Create Polkit-enforced shell scripts
echo "Creating Polkit-enforced shell scripts..."
sudo bash -c 'cat > /usr/local/bin/exit_fullscreen.sh <<EOF
#!/bin/bash
# Script to exit fullscreen for LISA
echo "Exiting fullscreen for LISA application..."
wmctrl -r "LISA" -b remove,fullscreen
EOF'

sudo bash -c 'cat > /usr/local/bin/close_app.sh <<EOF
#!/bin/bash
# Script to close LISA application
echo "Closing LISA application..."
pkill -f "LISA"
EOF'

sudo bash -c 'cat > /usr/local/bin/switch_apps.sh <<EOF
#!/bin/bash
# Script to switch applications while LISA is running
echo "Switching applications is not permitted without authentication."
pkexec --disable-internal-agent wmctrl -s 1
EOF'

# Set proper permissions for the scripts
echo "Setting permissions for the scripts..."
sudo chmod 700 /usr/local/bin/exit_fullscreen.sh
sudo chmod 700 /usr/local/bin/close_app.sh
sudo chmod 700 /usr/local/bin/switch_apps.sh
sudo chown root:root /usr/local/bin/exit_fullscreen.sh
sudo chown root:root /usr/local/bin/close_app.sh
sudo chown root:root /usr/local/bin/switch_apps.sh

sudo systemctl restart polkit


# Create Polkit policy file
echo "Creating Polkit policy file..."
sudo mkdir -p /usr/share/polkit-1/actions/
sudo bash -c 'cat > /usr/share/polkit-1/actions/lisa.policy <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<policyconfig>

  <action id="lisa.exitfullscreen">
    <description>Exit Fullscreen Mode</description>
    <message>Authentication is required to exit fullscreen mode.</message>
    <defaults>
      <allow_any>no</allow_any>
      <allow_inactive>no</allow_inactive>
      <allow_active>auth_admin</allow_active>
    </defaults>
  </action>

  <action id="lisa.closeapp">
    <description>Close LISA</description>
    <message>Authentication is required to close the application.</message>
    <defaults>
      <allow_any>no</allow_any>
      <allow_inactive>no</allow_inactive>
      <allow_active>auth_admin</allow_active>
    </defaults>
  </action>

  <action id="lisa.switchapps">
    <description>Switch Applications while LISA is running</description>
    <message>Authentication is required to switch applications.</message>
    <defaults>
      <allow_any>no</allow_any>
      <allow_inactive>no</allow_inactive>
      <allow_active>auth_admin</allow_active>
    </defaults>
  </action>

</policyconfig>
EOF'

# Create Polkit rules file
echo "Creating Polkit rules file..."
sudo mkdir -p /etc/polkit-1/rules.d
sudo bash -c 'cat > /etc/polkit-1/rules.d/50-lisa.rules <<EOF
polkit.addRule(function(action, subject) {
  if (action.id == "lisa.exitfullscreen") {
    return polkit.Result.AUTH_ADMIN;
  }
  if (action.id == "lisa.closeapp") {
    return polkit.Result.AUTH_ADMIN;
  }
  if (action.id == "lisa.switchapps") {
    return polkit.Result.AUTH_ADMIN;
  }
});
EOF'

# Set correct permissions for the rules file
sudo chmod 644 /etc/polkit-1/rules.d/50-lisa.rules

echo "Setup complete. Please reboot the system for all changes to take effect."

# Restart system to apply changes
read -p "Do you want to reboot now? (y/n): " reboot_now
if [[ "$reboot_now" =~ ^[Yy]$ ]]; then
    sudo reboot
else
    echo "Please reboot the system manually to apply changes."
fi
