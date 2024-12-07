// main.js
const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const { exec } = require('child_process');

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
		win.webContents.setZoomFactor(0.8); // Adjust zoom level (0.8 = 80%)
	});

	win.on('close', (e) => {
		e.preventDefault(); // Prevent default close
		exec('pkexec /usr/local/bin/close_app.sh', (error, stdout, stderr) => {
			if (error) {
				dialog.showErrorBox('Authentication Failed', 'You must authenticate to close the app.');
				console.error(`Error: ${error.message}`);
				return;
			}
			if (stderr) {
				console.error(`Stderr: ${stderr}`);
			}
			console.log(`Stdout: ${stdout}`);
			win.destroy(); // Close the window only after authentication
		});
	});

	win.on('leave-full-screen', (e) => {
		e.preventDefault(); // Prevent default fullscreen exit
		exec('pkexec /usr/local/bin/exit_fullscreen.sh', (error, stdout, stderr) => {
			if (error) {
				dialog.showErrorBox('Authentication Failed', 'You must authenticate to exit fullscreen.');
				console.error(`Error: ${error.message}`);
				return;
			}
			if (stderr) {
				console.error(`Stderr: ${stderr}`);
			}
			console.log(`Stdout: ${stdout}`);
			win.setFullScreen(false); // Exit fullscreen only after authentication
		});
	});

	win.on('blur', (e) => {
		// Trigger authentication when switching apps
		exec('pkexec /usr/local/bin/switch_apps.sh', (error, stdout, stderr) => {
			if (error) {
				dialog.showErrorBox('Authentication Failed', 'You must authenticate to switch applications.');
				console.error(`Error: ${error.message}`);
				win.focus(); // Return focus to the app if authentication fails
				return;
			}
			if (stderr) {
				console.error(`Stderr: ${stderr}`);
			}
			console.log(`Stdout: ${stdout}`);
		});
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