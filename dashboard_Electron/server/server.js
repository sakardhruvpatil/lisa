const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

let currentSpeed = 0; // Initial speed

app.post('/change-speed', (req, res) => {
    const { action } = req.body;

    // Update the current speed based on action
    if (action === 'increase') {
        currentSpeed += 10; // Increase speed
    } else if (action === 'decrease') {
        currentSpeed = Math.max(0, currentSpeed - 10); // Decrease speed, but not below 0
    }

    // Output the new speed to the terminal
    console.log(`Current Speed: ${currentSpeed}`);

    // Return the new speed as a JSON response
    res.json({ new_speed: currentSpeed });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});

