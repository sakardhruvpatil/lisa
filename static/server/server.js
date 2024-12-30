// Server.js

const express = require('express');
const cors = require('cors');
const fs = require('fs'); // Import fs module to write to log file
const path = require('path'); // Import path module for cross-platform compatibility
const os = require('os'); // Import os module to get HOME directory
const app = express();
const PORT = 5007;
const { writeFile } = require('fs/promises'); // Use promises for cleaner code

app.use(cors());
app.use(express.json());

// Define the LISA_LOGS directory path outside the AppImage
const LOGS_DIR = path.join(os.homedir(), 'LISA_LOGS');
const logFilePath = path.join(LOGS_DIR, 'log.txt');

// Ensure that the LISA_LOGS directory exists
if (!fs.existsSync(LOGS_DIR)) {
    fs.mkdirSync(LOGS_DIR, { recursive: true }); // Create the directory if it doesn't exist
    console.log(`Created directory: ${LOGS_DIR}`);
}

// Check if the log.txt file exists, if not, create it
if (!fs.existsSync(logFilePath)) {
    fs.writeFileSync(logFilePath, ''); // Create an empty log.txt if it doesn't exist
    console.log(`Created log file: ${logFilePath}`);
}

// Read the last logged speed from the file
let currentSpeed = 0; // Initial speed
fs.readFile(logFilePath, 'utf8', (err, data) => {
    if (err) {
        console.error('Error reading log file:', err);
    } else {
        // Get the last line of the log file to fetch the last logged speed
        const lines = data.trim().split('\n');
        const lastLine = lines[lines.length - 1];
        if (lastLine && lastLine.startsWith('Current Speed:')) {
            const speed = parseInt(lastLine.split(':')[1].trim(), 10);
            if (!isNaN(speed)) {
                currentSpeed = speed; // Set the current speed from the log
                console.log(`Loaded last speed: ${currentSpeed}`);
            }
        }
    }
});

app.get('/get-speed', (req, res) => {
    fs.readFile(logFilePath, 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading log file:', err);
            return res.status(500).json({ error: 'Failed to read log file' });
        } else {
            // Get the last line of the log file to fetch the last logged speed
            const lines = data.trim().split('\n');
            const lastLine = lines[lines.length - 1];
            let lastSpeed = 0;
            if (lastLine && lastLine.startsWith('Current Speed:')) {
                const speed = parseInt(lastLine.split(':')[1].trim(), 10);
                if (!isNaN(speed)) {
                    lastSpeed = speed;
                }
            }
            res.json({ speed: lastSpeed }); // Send the last speed to the frontend
        }
    });
});


app.post('/change-speed', (req, res) => {
    const { action } = req.body;

    // Update the current speed based on action
    if (action === 'increase') {
        currentSpeed = Math.min(100, currentSpeed + 1); // Increase speed
    } else if (action === 'decrease') {
        currentSpeed = Math.max(0, currentSpeed - 1); // Decrease speed, but not below 0
    }

    // Log the current speed to the terminal
    console.log(`Current Speed: ${currentSpeed}`);

    // Log the current speed to the log.txt file
    const logMessage = `Current Speed: ${currentSpeed}\n`;
    fs.appendFile(logFilePath, logMessage, (err) => {
        if (err) {
            console.error('Failed to write to log.txt', err);
        } else {
            console.log('Speed logged to log.txt');
        }
    });

    // Return the new speed as a JSON response
    res.json({ new_speed: currentSpeed });
});

// Endpoint to open the left shutter
app.post('/open_left', async (req, res) => {
    const { signal } = req.body;
    const decisionFilePath = path.join(LOGS_DIR, 'decision_left.txt');

    console.log(`Received signal for left shutter: ${signal}`); // Log the received signal

    try {
        await writeFile(decisionFilePath, signal.toString()); // Write the signal to the file
        console.log(`Left shutter decision written: ${signal}`);
        res.json({ message: 'Left shutter decision recorded' });
    } catch (error) {
        console.error('Error writing to decision_left.txt:', error);
        res.status(500).json({ error: 'Failed to write to decision_left.txt' });
    }
});

// Endpoint to open the right shutter
app.post('/open_right', async (req, res) => {
    const { signal } = req.body;
    const decisionFilePath = path.join(LOGS_DIR, 'decision_right.txt');

    console.log(`Received signal for right shutter: ${signal}`); // Log the received signal

    try {
        await writeFile(decisionFilePath, signal.toString()); // Write the signal to the file
        console.log(`Right shutter decision written: ${signal}`);
        res.json({ message: 'Right shutter decision recorded' });
    } catch (error) {
        console.error('Error writing to decision_right.txt:', error);
        res.status(500).json({ error: 'Failed to write to decision_right.txt' });
    }
});

// Endpoint to open both shutters
app.post('/open_both', async (req, res) => {
    const { signal } = req.body;
    const decisionLeftFilePath = path.join(LOGS_DIR, 'decision_left.txt');
    const decisionRightFilePath = path.join(LOGS_DIR, 'decision_right.txt');

    console.log(`Received signal for both shutters: ${signal}`);

    try {
        // Write the same signal to both files
        await writeFile(decisionLeftFilePath, signal.toString());
        await writeFile(decisionRightFilePath, signal.toString());

        console.log(`Both shutters decisions written: ${signal}`);
        res.json({ message: 'Both shutters decisions recorded' });
    } catch (error) {
        console.error('Error writing to decision files:', error);
        res.status(500).json({ error: 'Failed to write to decision files for both shutters' });
    }
});


app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
