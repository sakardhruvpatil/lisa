// Settings.js

import React, { useState, useEffect } from 'react';

const Settings = ({ setMode, acceptanceRate, setAcceptanceRate }) => {
	const [localAcceptanceRate, setLocalAcceptanceRate] = useState(acceptanceRate);

	// Update localAcceptanceRate when acceptanceRate prop changes
	useEffect(() => {
		setLocalAcceptanceRate(acceptanceRate);
	}, [acceptanceRate]);

	// Function to handle slider changes
	const handleSliderChange = (e) => {
		const newValue = parseInt(e.target.value, 10);
		setLocalAcceptanceRate(newValue);
		// Do not update acceptanceRate or send to backend yet
	};

	// Function to save the threshold change
	const handleSaveThreshold = async () => {
		setAcceptanceRate(localAcceptanceRate); // Update acceptance rate in App component

		// Send threshold update to backend
		try {
			const response = await fetch('http://localhost:8000/update_threshold', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ threshold: localAcceptanceRate }),
			});

			const result = await response.json();
			console.log('Threshold update response:', result);
		} catch (error) {
			console.error('Error updating threshold:', error);
		}
	};

	// Function to change mode and notify backend
	const handleModeChange = async (mode) => {
		console.log(`Changing mode to ${mode}`);
		setMode(mode);
		localStorage.setItem('cameraLayout', mode); // Persist the mode in localStorage
		try {
			const response = await fetch(
				'http://localhost:8000/set_process_mode/' + mode,
				{ method: 'GET' }
			);
			const result = await response.json();
			console.log('Mode update response:', result);
		} catch (error) {
			console.error('Error updating mode:', error);
		}
	};	

	return (
		<div className="settings-container" style={{ marginBottom: '60px', fontSize: '48px', fontWeight: 'bold' }}>
			<label>Settings</label>

			<div className="mode-selection" style={{ marginTop: '40px'}}>
				<button onClick={() => handleModeChange('horizontal')}>Horizontal</button>
				<button onClick={() => handleModeChange('vertical')}>Vertical</button>
			</div>

			<div className="acceptance-rate" style={{ marginTop: '50px' }}>
				<label>Set Acceptance Rate</label>
				<div className="slider-container" style={{ marginTop: '30px' }}>
					<input
						type="range"
						min="0"
						max="100"
						value={localAcceptanceRate}
						onChange={handleSliderChange}
						style={{
							width: '33vw', // Width of the slider
							height: '25px', // Height of the slider (this affects the thumb size)
							background: '#ddd', // Background color of the track
							borderRadius: '5px', // Rounded corners
							outline: 'none', // Remove outline
							cursor: 'pointer' // Pointer cursor on hover
						}}
					/>
					<div className="acceptance-rate-value" style={{ marginTop: '20px' }}>{localAcceptanceRate}%</div>
				</div>
				<button className="save-button" style={{ marginTop: '20px' }} onClick={handleSaveThreshold}>
					Save
				</button>
			</div>
		</div>
	);
};

export default Settings;
