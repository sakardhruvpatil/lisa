// App.js

import React, { useState, useEffect, useCallback } from 'react';
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
	const [cameraLayout, setCameraLayout] = useState(localStorage.getItem('cameraMode') || 'vertical'); // Initialize with mode from localStorage
	const [mode, setMode] = useState(localStorage.getItem('cameraMode') || 'vertical'); // Default to 'vertical'  
	const [acceptanceRate, setAcceptanceRate] = useState(null); // Default acceptance rate
	const [speed, setSpeed] = useState(0); // State to store the current speed
	const [loadingSpeed, setLoadingSpeed] = useState(true); // Loading state for speed
	const [screenResolution, setScreenResolution] = useState({
		width: window.innerWidth,
		height: window.innerHeight,
	});

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

	// Fetch the current threshold from the backend
	const fetchCurrentThreshold = useCallback(async () => {
		try {
			const response = await fetch(`http://localhost:8000/get_current_threshold/${mode}`);
			if (!response.ok) {
				throw new Error('Failed to fetch current threshold');
			}
			const data = await response.json();
			console.log('Fetched threshold data:', data); // Log the fetched data
			setAcceptanceRate(data.threshold !== undefined ? data.threshold : 95); // Default to 95 if undefined
		} catch (error) {
			console.error('Error fetching current threshold:', error);
			setAcceptanceRate(mode === 'horizontal' ? 95 : null); // Set default for horizontal mode
		}
	}, [mode]);

	useEffect(() => {
		const savedMode = localStorage.getItem('cameraMode') || 'vertical';
		setMode(savedMode);
		setCameraLayout(savedMode); // Adjust layout based on saved mode
		fetchCurrentThreshold(); // Fetch threshold on initial load
	}, [fetchCurrentThreshold]);

	useEffect(() => {
		if (mode === 'horizontal') {
			setCameraLayout('horizontal');
		} else if (mode === 'vertical') {
			setCameraLayout('vertical');
		}
		fetchCurrentThreshold(); // Fetch threshold whenever mode changes
	}, [mode, fetchCurrentThreshold]);

	// Handle mode change and update layout accordingly
	const handleModeChange = (newMode) => {
		setMode(newMode);
		localStorage.setItem('cameraMode', newMode); // Save the mode to localStorage
		setCameraLayout(newMode === 'vertical' ? 'horizontal' : 'vertical'); // Ensure layout reflects the mode
	};


	return (
		<div className="app-layout">
			<Navbar />
			<div className="content">
				<div className="dashboard">
					{/* Logo Section */}
					<div className="logo-container">
						<img src={logo} alt="Logo" className="logo" />
						<h1 className="Linen">
							<span className="highlight-first-letter">L</span>inen
							<span className="word-gap"> </span>
							<span className="highlight-first-letter">I</span>nspection
							<span className="word-gap"> </span>
							&
							<span className="word-gap"> </span>
							<span className="highlight-first-letter">S</span>orting
							<span className="word-gap"> </span>
							<span className="highlight-first-letter">A</span>ssistant
						</h1>
					</div>



					<Routes>
						<Route
							path="/"
							element={
								<HomeWithWebcam
									mode={mode}
									acceptanceRate={acceptanceRate}
									cameraLayout={cameraLayout}
									setLoadingSpeed={setLoadingSpeed} // Pass the function
									loadingSpeed={loadingSpeed} // Pass the loading state				 
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
		</div>
	);
};


// Component for the Home page with webcam controls
const HomeWithWebcam = ({ mode, acceptanceRate, cameraLayout, setLoadingSpeed, loadingSpeed  }) => {
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

	const [countDataHorizontal, setCountDataHorizontal] = useState({
		total_bedsheets: 0,
		total_accepted: 0,
		total_rejected: 0,
	});

	// WebSocket connections for left, right, and horizontal cameras
	useEffect(() => {
		let wsLeft;
		let wsRight;
		let wsHorizontal;

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
				setTimeout(connectWebSocketLeft, 5000);
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
				setTimeout(connectWebSocketRight, 5000);
			};
		};

		const connectWebSocketHorizontal = () => {
			wsHorizontal = new WebSocket('ws://localhost:8000/ws/todays_counts/horizontal');
			wsHorizontal.onmessage = (event) => {
				const data = JSON.parse(event.data);
				setCountDataHorizontal(data);
			};
			wsHorizontal.onerror = (error) => {
				console.error('WebSocket error (horizontal):', error);
			};
			wsHorizontal.onclose = () => {
				console.log('WebSocket connection closed (horizontal), attempting to reconnect...');
				setTimeout(connectWebSocketHorizontal, 5000);
			};
		};

		// Connect to the appropriate WebSocket based on the mode
		if (mode === 'vertical') {
			connectWebSocketLeft();
			connectWebSocketRight();
		} else if (mode === 'horizontal') {
			connectWebSocketHorizontal();
		}

		// Cleanup function to close WebSocket connections
		return () => {
			if (wsLeft) wsLeft.close();
			if (wsRight) wsRight.close();
			if (wsHorizontal) wsHorizontal.close();
		};
	}, [mode]);

	// Update current time every second
	useEffect(() => {
		const timer = setInterval(() => {
			setCurrentTime(new Date());
		}, 1000); // Update every second
		return () => clearInterval(timer);
	}, []);

	// Fetch the last logged speed from the server on startup
	useEffect(() => {
		const fetchSpeed = async () => {
			try {
				const response = await fetch('http://localhost:5007/get-speed');
				if (!response.ok) {
					throw new Error('Failed to fetch speed');
				}
				const data = await response.json();
				setSpeed(data.speed); // Set the speed to the fetched value
				setLoadingSpeed(false); // Set loading to false once speed is fetched
			} catch (error) {
				console.error('Error fetching speed:', error);
				setLoadingSpeed(false); // Set loading to false even in case of error
			}
		};
		fetchSpeed(); // Fetch speed when the component mounts
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
			setSpeed(data.new_speed); // Update the speed from the server response
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

			{/* Display counts in separate tables for left and right cameras or horizontal camera */}
			<div className={`tables-container ${cameraLayout}`}>
				{mode === 'horizontal' ? (
					<div className="table-wrapper">
						<h3>Horizontal Camera Counts</h3>
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
										<td>{countDataHorizontal.total_accepted}</td>
										<td>{countDataHorizontal.total_rejected}</td>
										<td>{countDataHorizontal.total_bedsheets}</td>
									</tr>
								</tbody>
							</table>
						</div>
					</div>
				) : (
					<>
						{mode === 'vertical' && (
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
						)}

						{mode === 'vertical' && (
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
					</>
				)}
			</div>

			{/* Centered Controls */}
			<div className="controls-container" style={{ marginTop: '50px' }}>
				<div className="controls">
					<speed-button onClick={() => changeSpeed('decrease')}>-</speed-button>
					{/* Display the current speed */}
					<p className="speed-display">{speed}</p>
					<speed-button onClick={() => changeSpeed('increase')}>+</speed-button>
				</div>
				<div className="speed-button-label">
					<p>Conveyor Speed</p>
				</div>
			</div>
		</div>
	);
};

export default App;