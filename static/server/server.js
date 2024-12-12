const express = require('express');
const cors = require('cors');
const fs = require('fs'); // Import fs module to write to log file
const app = express();
const PORT = 5007;

app.use(cors());
app.use(express.json());

let currentSpeed = 0; // Initial speed

// Check if the log.txt file exists, if not, create it
const logFilePath = '../../app/plc_integration/log.txt';
if (!fs.existsSync(logFilePath)) {
    fs.writeFileSync(logFilePath, ''); // Create an empty log.txt if it doesn't exist
}

app.post('/change-speed', (req, res) => {
    const { action } = req.body;

    // Update the current speed based on action
    if (action === 'increase') {
        currentSpeed += 10; // Increase speed
    } else if (action === 'decrease') {
        currentSpeed = Math.max(0, currentSpeed - 10); // Decrease speed, but not below 0
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

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});