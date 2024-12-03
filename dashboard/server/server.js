const express = require('express');
const cors = require('cors');
const fs = require('fs');
const app = express();
const PORT = 5007;

app.use(cors());
app.use(express.json());

let currentSpeed = 0; // Initial speed

// Function to read the current speed from log file
const readCurrentSpeed = () => {
    fs.readFile('log.txt', 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading log file:', err);
        } else {
            // Read the last line of the log file
            const lines = data.trim().split('\n');
            const lastLine = lines[lines.length - 1];
            const speedMatch = lastLine.match(/Current Speed: (\d+)/);
            if (speedMatch) {
                currentSpeed = parseInt(speedMatch[1], 10);
            }
        }
    });
};

// Read the current speed when the server starts
readCurrentSpeed();

app.post('/change-speed', (req, res) => {
    const { action } = req.body;

    // Update the current speed based on action
    if (action === 'increase') {
        // Ensure speed does not exceed 100
        currentSpeed = Math.min(100, currentSpeed + 10);
    } else if (action === 'decrease') {
        // Decrease speed, but not below 0
        currentSpeed = Math.max(0, currentSpeed - 10);
    }

    // Log the current speed to the terminal
    console.log(`Current Speed: ${currentSpeed}`);

    // Log the current speed to the log.txt file
    const logMessage = `Current Speed: ${currentSpeed}\n`;
    fs.appendFile('log.txt', logMessage, (err) => {
        if (err) {
            console.error('Failed to write to log.txt', err);
        } else {
            console.log('Speed logged to log.txt');
        }
    });

    // Return the new speed as a JSON response
    res.json({ new_speed: currentSpeed });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
