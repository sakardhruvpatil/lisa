// App.js

import React, { useState, useEffect } from 'react';
import { Route, Routes } from 'react-router-dom';
import './App.css';
import Navbar from './Navbar';
import About from './Pages/About';
import Contact from './Pages/Contact';
import AnalyticsData from './Pages/AnalyticsData';
import WebcamCapture from './WebcamCapture';
import Settings from './Pages/Settings';
import logo from './sakar.png'; // Import the logo image

// Main App component
const App = () => {
	const [cameraLayout, setCameraLayout] = useState('vertical');
	const [mode, setMode] = useState('production'); // Default to demo mode
	const [acceptanceRate, setAcceptanceRate] = useState(95); // Default acceptance rate
	const [showLayoutModal, setShowLayoutModal] = useState(false);
	const [screenResolution, setScreenResolution] = useState({
		width: window.innerWidth,
		height: window.innerHeight,
	});

	// Handle screen resolution changes
	useEffect(() => {
		const handleResize = () => {
			setScreenResolution({
				width: window.innerWidth,
				height: window.innerHeight,
			});
		};

		window.addEventListener('resize', handleResize);
		return () => window.removeEventListener('resize', handleResize);
	}, []);

	// Adjust camera layout based on screen width
	useEffect(() => {
		if (screenResolution.width < 600) {
			setCameraLayout('vertical');
		} else {
			setCameraLayout('horizontal');
		}
	}, [screenResolution]);

	// Handle mode change (demo or production)
	const handleModeChange = (newMode) => {
		setMode(newMode);
		if (newMode === 'production') {
			setShowLayoutModal(true);
		}
	};

	// Handle camera layout selection
	const handleLayoutChange = (layout) => {
		setCameraLayout(layout);
		setShowLayoutModal(false);
	};

	return (
		<div className="app-layout">
			<Navbar />
			<div className="content">
				<div className="dashboard">
					{/* Logo Section */}
					<div className="logo-container">
						<img src={logo} alt="Logo" className="logo" />
						<h1 className="Linen "> Linen Inspection and Sorting Assistant </h1> {/* Added name below the logo */}
					</div>
					<Routes>
						<Route
							path="/"
							element={
								<HomeWithWebcam
									mode={mode}
									acceptanceRate={acceptanceRate}
									cameraLayout={cameraLayout}
								/>
							}
						/>
						<Route path="/about" element={<About />} />
						<Route
							path="/AnalyticsData"
							element={
								<AnalyticsData
									acceptanceRate={acceptanceRate}
									cameraLayout={cameraLayout}
								/>
							}
						/>
						<Route path="/contact" element={<Contact />} />
						<Route
							path="/settings"
							element={
								<Settings
									setMode={handleModeChange}
									acceptanceRate={acceptanceRate}
									setAcceptanceRate={setAcceptanceRate}
								/>
							}
						/>
					</Routes>
				</div>
				<div className={`webcam-section ${cameraLayout}`}>
					<WebcamCapture mode={mode} cameraLayout={cameraLayout} />
				</div>
	
				{showLayoutModal && (
					<div className="layout-modal">
						<div className="modal-content">
							<h2>Select Camera Layout</h2>
							<button onClick={() => handleLayoutChange('vertical')}>Vertical</button>
							<button onClick={() => handleLayoutChange('horizontal')}>Horizontal</button>
						</div>
					</div>
				)}
			</div>
		</div>
	);
};
// Component for the Home page with webcam controls
const HomeWithWebcam = ({ mode, acceptanceRate, cameraLayout }) => {
	const [speed, setSpeed] = useState(0);
	const [currentTime, setCurrentTime] = useState(new Date());

	// State to hold counts for left and right cameras
	const [countDataLeft, setCountDataLeft] = useState({
		total_bedsheets: 0,
		total_accepted: 0,
		total_rejected: 0,
	});

	const [countDataRight, setCountDataRight] = useState({
		total_bedsheets: 0,
		total_accepted: 0,
		total_rejected: 0,
	});

	// WebSocket connections for left and right cameras
	useEffect(() => {
		let wsLeft;
		let wsRight;

		const connectWebSocketLeft = () => {
			wsLeft = new WebSocket('ws://localhost:8000/ws/todays_counts/left');

			wsLeft.onmessage = (event) => {
				const data = JSON.parse(event.data);
				setCountDataLeft(data);
			};
        
			wsLeft.onerror = (error) => {
				console.error('WebSocket error (left):', error);
			};

			wsLeft.onclose = () => {
				console.log('WebSocket connection closed (left), attempting to reconnect...');
				setTimeout(connectWebSocketLeft, 5000); // Reconnect after 5 seconds
			};
		};

		const connectWebSocketRight = () => {
			wsRight = new WebSocket('ws://localhost:8000/ws/todays_counts/right');

			wsRight.onmessage = (event) => {
				const data = JSON.parse(event.data);
				setCountDataRight(data);
			};

			wsRight.onerror = (error) => {
				console.error('WebSocket error (right):', error);
			};

			wsRight.onclose = () => {
				console.log('WebSocket connection closed (right), attempting to reconnect...');
				setTimeout(connectWebSocketRight, 5000); // Reconnect after 5 seconds
			};
		};

		if (mode === 'production') {
			connectWebSocketLeft();
			connectWebSocketRight();
		} else {
			// In demo mode, only connect left WebSocket
			connectWebSocketLeft();
		}

		// Clean up WebSocket connections on unmount or mode change
		return () => {
			if (wsLeft) wsLeft.close();
			if (wsRight) wsRight.close();
		};
	}, [mode]);

	// Update current time every second
	useEffect(() => {
		const timer = setInterval(() => {
			setCurrentTime(new Date());
		}, 1000); // Update every second
		return () => clearInterval(timer);
	}, []);

	// Function to change conveyor speed
	const changeSpeed = async (action) => {
		try {
			const response = await fetch('http://localhost:5007/change-speed', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ action }),
			});
			if (!response.ok) throw new Error('Failed to change speed');
			const data = await response.json();
			setSpeed(data.new_speed); // Assuming response has 'new_speed' field
		} catch (error) {
			console.error('Error:', error);
			alert('Failed to change speed: ' + error.message);
		}
	};

	return (
		<div className="dashboard">
			 <div className="acceptance-rate-display">
				<p>Acceptance Rate: {acceptanceRate}%</p>
				<div className="line"></div>
			</div>
           
        

			<div className="current-time">
				<p>
					{currentTime.toLocaleString('en-US', {
						weekday: 'long',
						year: 'numeric',
						month: 'long',
						day: 'numeric',
						hour: '2-digit',
						minute: '2-digit',
						hour12: true,
					})}
				</p>
			</div>

			{/* Display counts in separate tables for left and right cameras */}
			<div className={`tables-container ${cameraLayout}`}>
				{mode === 'demo' || mode === 'production' ? (
					<div className="table-wrapper">
						<h3>Left Camera Counts</h3>
						<div className="table-container">
							<table className="summary-table">
								<thead>
									<tr>
										<th>Accepted</th>
										<th>Rejected</th>
										<th>Total Bedsheets</th>
									</tr>
								</thead>
								<tbody>
									
									<tr>
										<td>{countDataLeft.total_accepted}</td>
										<td>{countDataLeft.total_rejected}</td>
										<td>{countDataLeft.total_bedsheets}</td>
									</tr>
								</tbody>
							</table>
						</div>
					</div>
				) : null}

				{mode === 'production' && (
					<div className="table-wrapper">
						<h3>Right Camera Counts</h3>
						<div className="table-container">
							<table className="summary-table">
								<thead>
									<tr>
										<th>Accepted</th>
										<th>Rejected</th>
										<th>Total Bedsheets</th>
									</tr>
								</thead>
								<tbody>
									<tr>
										<td>{countDataRight.total_accepted}</td>
										<td>{countDataRight.total_rejected}</td>
										<td>{countDataRight.total_bedsheets}</td>
									</tr>
								</tbody>
							</table>
						</div>
					</div>
				)}
			</div>

			{/* Centered Controls */}
			<div className="controls-container">
				<div className="controls">
					<button onClick={() => changeSpeed('decrease')}>-</button>
					<p className="speed-display">{speed}</p>
					<button onClick={() => changeSpeed('increase')}>+</button>
				</div>
				<div className="speed-button-label">
					<p>Conveyor Speed</p>
				</div>
			</div>
		</div>
	);
};

export default App;
