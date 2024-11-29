// main.js
const { app, BrowserWindow } = require('electron');
const path = require('path');

let win;

function createWindow() {
	win = new BrowserWindow({
		width: 1280,
		height: 720,
		webPreferences: {
			nodeIntegration: true, // Allows Node.js features in the renderer process
		},
		fullscreen: true, // Ensure the window opens in fullscreen
	});

	// Load your React app from the build directory
	// win.loadURL('http://localhost:3000'); // For development (if using React's development server)

	// Uncomment the next line to load a packaged build later
	win.loadFile(path.join(__dirname, 'build', 'index.electron.html'));

	// Set zoom factor to make the app zoomed out
	win.webContents.on('did-finish-load', () => {
		win.webContents.setZoomFactor(1.0); // Adjust zoom level (0.8 = 80%)
	});

	win.on('closed', () => {
		win = null;
	});
}

app.whenReady().then(() => {
	createWindow();

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