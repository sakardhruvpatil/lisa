const { app, BrowserWindow, globalShortcut } = require('electron');
const path = require('path');

let win;

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

    // Load your React app
    // win.loadURL('http://localhost:3000'); // For development (if using React's development server)
    win.loadFile(path.join(__dirname, 'build', 'index.html')); // For production builds

    // Optional: Remove the menu bar
    win.setMenu(null);

    // Set zoom factor to adjust app scaling
    win.webContents.on('did-finish-load', () => {
        win.webContents.setZoomFactor(0.8); // Adjust zoom level (0.8 = 80%)
    });

    // Prevent closing the app
    win.on('close', (event) => {
        event.preventDefault(); // Prevent the app from being closed
    });

    win.on('closed', () => {
        win = null;
    });
}

app.whenReady().then(() => {
    createWindow();

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
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('will-quit', () => {
    // Unregister all shortcuts when the app quits
    globalShortcut.unregisterAll();
});
