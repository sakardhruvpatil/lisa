const { app, BrowserWindow, globalShortcut } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let win;
let backendProcess;
let plcProcess; 
let serverProcess;

// Function to start the backend (main.py)
function startBackend() {
    console.log('Starting backend...');
    const backendPath = path.join(app.getAppPath(), 'main.py'); // Path to main.py
    backendProcess = spawn('python3', [backendPath]);

    backendProcess.stdout.on('data', (data) => {
        console.log(`Backend output: ${data}`);
    });

    backendProcess.stderr.on('data', (data) => {
        console.error(`Backend error: ${data}`);
    });

    backendProcess.on('error', (error) => {
        console.error(`Failed to start backend: ${error.message}`);
    });

    backendProcess.on('close', (code) => {
        console.log(`Backend process exited with code ${code}`);
    });
}

// Function to start the PLC integration (messung_plc_com.py)
function startPlcIntegration() {
    console.log('Starting PLC Integration...');
    const plcPath = path.join(app.getAppPath(), 'plc_integration/messung_plc_com.py'); // Path to messung_plc_com.py
    plcProcess = spawn('python3', [plcPath]);

    plcProcess.stdout.on('data', (data) => {
        console.log(`PLC Integration output: ${data}`);
    });

    plcProcess.stderr.on('data', (data) => {
        console.error(`PLC Integration error: ${data}`);
    });

    plcProcess.on('error', (error) => {
        console.error(`Failed to start PLC integration: ${error.message}`);
    });

    plcProcess.on('close', (code) => {
        console.log(`PLC Integration process exited with code ${code}`);
    });
}

// Function to start the server (server.js)
function startServer() {
    console.log('Starting server...');
    const serverPath = path.join(app.getAppPath(), 'server/server.js'); // Path to server.js
    serverProcess = spawn('node', [serverPath]);

    serverProcess.stdout.on('data', (data) => {
        console.log(`Server output: ${data}`);
    });

    serverProcess.stderr.on('data', (data) => {
        console.error(`Server error: ${data}`);
    });

    serverProcess.on('error', (error) => {
        console.error(`Failed to start server: ${error.message}`);
    });

    serverProcess.on('close', (code) => {
        console.log(`Server process exited with code ${code}`);
    });
}

// Function to stop all processes gracefully
function stopAllProcesses() {
    return new Promise(async (resolve, reject) => {
        console.log('Stopping all processes...');
        try {
            // Stop Backend process
            if (backendProcess) {
                console.log('Stopping backend...');
                backendProcess.kill('SIGINT'); // Gracefully stop the backend
                await new Promise((resolve) => backendProcess.on('close', resolve));
            }

            // Stop PLC Integration process
            if (plcProcess) {
                console.log('Stopping PLC Integration...');
                plcProcess.kill('SIGINT'); // Gracefully stop PLC integration
                await new Promise((resolve) => plcProcess.on('close', resolve));
            }

            // Stop Server process
            if (serverProcess) {
                console.log('Stopping server...');
                serverProcess.kill('SIGINT'); // Gracefully stop the server
                await new Promise((resolve) => serverProcess.on('close', resolve));
            }

            console.log('All processes stopped.');
            // Close the frontend window after stopping all processes
            if (win) {
                console.log('Closing the frontend window...');
                win.close(); // Close the window gracefully
            }
            resolve();
        } catch (error) {
            console.error('Error stopping processes:', error);
            reject(error);
        }
    });
}

// Function to forcefully stop processes if they do not terminate within a timeout
function forceStopAllProcesses() {
    return new Promise(async (resolve, reject) => {
        console.log('Force stopping all processes...');

        try {
            const timeout = 5000; // Timeout in milliseconds before forcing a kill

            // Force stop Backend if it's still running
            if (backendProcess && !backendProcess.killed) {
                console.log('Force stopping backend...');
                backendProcess.kill('SIGTERM');
                setTimeout(() => {
                    if (!backendProcess.killed) backendProcess.kill('SIGKILL'); // Force kill if not stopped
                }, timeout);
            }

            // Force stop PLC Integration if it's still running
            if (plcProcess && !plcProcess.killed) {
                console.log('Force stopping PLC Integration...');
                plcProcess.kill('SIGTERM');
                setTimeout(() => {
                    if (!plcProcess.killed) plcProcess.kill('SIGKILL'); // Force kill if not stopped
                }, timeout);
            }

            // Force stop Server if it's still running
            if (serverProcess && !serverProcess.killed) {
                console.log('Force stopping server...');
                serverProcess.kill('SIGTERM');
                setTimeout(() => {
                    if (!serverProcess.killed) serverProcess.kill('SIGKILL'); // Force kill if not stopped
                }, timeout);
            }

            // Close the window after force stopping
            if (win) {
                console.log('Closing the frontend window...');
                win.close(); // Close the window gracefully
            }

            console.log('All processes attempted to stop.');
            resolve();
        } catch (error) {
            console.error('Error force stopping processes:', error);
            reject(error);
        }
    });
}

// Function to create the Electron window
function createWindow() {
    win = new BrowserWindow({
        width: 1280,
        height: 720,
        fullscreen: true, // Ensure the window opens in fullscreen
        kiosk: true,      // Enable kiosk mode
        webPreferences: {
            nodeIntegration: true, // Allows Node.js features in the renderer process
            contextIsolation: false, // Disable context isolation for compatibility
        },
    });

    // Correct path for the build folder
    const filePath = path.join(__dirname, 'build/index.html');
    console.log('Loading file:', filePath);

    win.loadFile(filePath).catch((err) => {
        console.error('Error loading file:', err);
    });

    // Optional: Remove the menu bar
    win.setMenu(null);

    // Set zoom factor to adjust app scaling
    win.webContents.on('did-finish-load', () => {
        win.webContents.setZoomFactor(0.8); // Adjust zoom level (0.8 = 80%)
    });

    // Handle window close event
    win.on('close', async (event) => {
        event.preventDefault(); // Prevent the app from being closed
        await stopAllProcesses();
        win.destroy();
    });

    win.on('closed', () => {
        win = null;
    });
}

app.whenReady()
    .then(() => {
        startBackend();  // Start the main backend
        console.log('Backend started. Waiting 15 seconds before launching frontend...');
        return new Promise((resolve) => setTimeout(resolve, 15000)); // Wait for 15 seconds
    })
    .then(() => {
        startPlcIntegration();  // Start PLC Integration after backend
        console.log('PLC integration started.');
        return new Promise((resolve) => setTimeout(resolve, 5000)); // Wait for 15 seconds
    })
    .then(() => {
        startServer();  // Start the server after PLC integration
        console.log('Server started. Launching frontend...');
        return new Promise((resolve) => setTimeout(resolve, 2000)); // Wait for 15 seconds
    })
    .then(() => {
        createWindow();  // Create the Electron window
        // Register global shortcuts to disable specific keys (e.g., Alt+Tab)
        globalShortcut.register('Alt+Tab', () => {
            return false; // Prevents Alt+Tab
        });

        globalShortcut.register('Ctrl+Shift+I', () => {
            return false; // Prevents opening developer tools
        });

        app.on('activate', () => {
            if (BrowserWindow.getAllWindows().length === 0) {
                createWindow();
            }
        });
    })
    .catch((error) => {
        console.error('Error during startup:', error);
        app.quit();
    });

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('will-quit', async (event) => {
    event.preventDefault(); // Prevent the app from being closed
    // Unregister all shortcuts when the app quits
    globalShortcut.unregisterAll();
    await forceStopAllProcesses();
    app.exit();
});

console.log('Electron main file loaded successfully.');
