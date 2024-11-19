// App.js

import React, { useState, useEffect } from 'react';
import { Route, Routes, useLocation } from 'react-router-dom';
import './App.css';
import Navbar from './Navbar';
import About from './Pages/About';
import Contact from './Pages/Contact';
import AnalyticsData from './Pages/AnalyticsData';
import WebcamCapture from './WebcamCapture';
import Settings from './Pages/Settings';
import logo from './sakar.png'; // Import the logo image

// Remove unnecessary import of HomePage
// import HomePage from './Pages/HomePage';

// Main App component
const App = () => {
	const location = useLocation();
	const [cameraLayout, setCameraLayout] = useState('vertical');
	const [mode, setMode] = useState('demo'); // Default to demo mode
	const [acceptanceRate, setAcceptanceRate] = useState(95); // Default acceptance rate
	const [showLayoutModal, setShowLayoutModal] = useState(false);
	const [screenResolution, setScreenResolution] = useState({
		width: window.innerWidth,
		height: window.innerHeight,
	});

	// Handle screen resolution changes
	const handleResize = () => {
		setScreenResolution({
			width: window.innerWidth,
			height: window.innerHeight,
		});
	};

	// Fetch the current threshold from the backend when the app starts
	useEffect(() => {
		const fetchThreshold = async () => {
			try {
				const response = await fetch('http://localhost:8000/get_current_threshold');
				const result = await response.json();
				if (result.threshold !== undefined) {
					setAcceptanceRate(result.threshold);
				}
			} catch (error) {
				console.error('Error fetching current threshold:', error);
			}
		};

		fetchThreshold();
	}, []);

	useEffect(() => {
		window.addEventListener('resize', handleResize);
		return () => window.removeEventListener('resize', handleResize);
	}, []);

	useEffect(() => {
		if (screenResolution.width < 600) {
			setCameraLayout('vertical');
		} else {
			setCameraLayout('horizontal');
		}
	}, [screenResolution]);

	const handleModeChange = (newMode) => {
		setMode(newMode);
		if (newMode === 'production') {
			setShowLayoutModal(true);
		}
	};

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
			</div>

			{showLayoutModal && (
				<div className="layout-modal">
					<div className="modal-content">
						<h2>Select Camera Layout</h2>
						<button onClick={() => handleLayoutChange('vertical')}>Vertical</button>
						<button onClick={() => handleLayoutChange('horizontal')}>
							Horizontal
						</button>
					</div>
				</div>
			)}
		</div>
	);
};

// Component for the Home page with webcam controls
const HomeWithWebcam = ({ mode, acceptanceRate, cameraLayout }) => {
	const [speed, setSpeed] = useState(0);
	const [currentTime, setCurrentTime] = useState(new Date());

	// State to hold counts
	const [countData, setCountData] = useState({
		total_bedsheets: 0,
		total_accepted: 0,
		total_rejected: 0,
	});

	// WebSocket connection to receive real-time counts
	useEffect(() => {
		let ws;

		const connectWebSocket = () => {
			ws = new WebSocket('ws://localhost:8000/ws/todays_counts');

			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				setCountData(data);
			};

			ws.onerror = (error) => {
				console.error('WebSocket error:', error);
			};

			ws.onclose = () => {
				console.log('WebSocket connection closed, attempting to reconnect...');
				setTimeout(connectWebSocket, 5000); // Try to reconnect after 5 seconds
			};
		};

		connectWebSocket();

		// Clean up the WebSocket connection when the component unmounts
		return () => {
			if (ws) ws.close();
		};
	}, []);

	useEffect(() => {
		const timer = setInterval(() => {
			setCurrentTime(new Date());
		}, 500);
		return () => clearInterval(timer);
	}, []);

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
				<p>{currentTime.toLocaleString('en-US', {
					weekday: 'long',
					year: 'numeric',
					month: 'long',
					day: 'numeric',
					hour: '2-digit',
					minute: '2-digit',
					hour12: true
				})}</p>
			</div>


			{/* Display today's data counts in a table format */}
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
							<td>{countData.total_accepted}</td>
							<td>{countData.total_rejected}</td>
							<td>{countData.total_bedsheets}</td>
						</tr>
					</tbody>
				</table>
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
