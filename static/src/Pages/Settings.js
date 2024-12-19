import React, { useState, useEffect } from 'react';

const Settings = ({ setMode, acceptanceRate, setAcceptanceRate }) => {
	const [localAcceptanceRate, setLocalAcceptanceRate] = useState(acceptanceRate);

	// Update localAcceptanceRate when acceptanceRate prop changes
	useEffect(() => {
		setLocalAcceptanceRate(acceptanceRate);
	}, [acceptanceRate]);

	const handleSliderChange = (e) => {
		const newValue = parseInt(e.target.value, 10);
		setLocalAcceptanceRate(newValue);
	};

	const handleSaveThreshold = async () => {
		setAcceptanceRate(localAcceptanceRate);

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

	const handleModeChange = async (mode) => {
		setMode(mode);
		localStorage.setItem('cameraLayout', mode);
		try {
			const response = await fetch(
				`http://localhost:8000/set_process_mode/${mode}`,
				{ method: 'GET' }
			);
			const result = await response.json();
			console.log('Mode update response:', result);
		} catch (error) {
			console.error('Error updating mode:', error);
		}
	};

	// Function to send a signal to open the left shut
	const handleOpenLeft = async () => {
		console.log('Open Left button pressed');
		try {
			// Send "1" signal to backend
			await fetch('http://localhost:8000/open_left', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ signal: 1 }),
			});

			console.log('Left shut activated');

			// Wait for 11 seconds
			setTimeout(async () => {
				// Send "0" signal to backend
				await fetch('http://localhost:8000/open_left', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ signal: 0 }),
				});
				console.log('Left shut deactivated');
			}, 11000);
		} catch (error) {
			console.error('Error sending Open Left signal:', error);
		}
	};

	// Function to send a signal to open the right shut
	const handleOpenRight = async () => {
		console.log('Open Right button pressed');
		try {
			// Send "1" signal to backend
			await fetch('http://localhost:8000/open_right', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ signal: 1 }),
			});

			console.log('Right shut activated');

			// Wait for 11 seconds
			setTimeout(async () => {
				// Send "0" signal to backend
				await fetch('http://localhost:8000/open_right', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ signal: 0 }),
				});
				console.log('Right shut deactivated');
			}, 11000);
		} catch (error) {
			console.error('Error sending Open Right signal:', error);
		}
	};

	return (
		<div className="settings-container" style={{ marginBottom: '60px', fontSize: '48px', fontWeight: 'bold' }}>
			<label>Settings</label>

			<div className="mode-selection" style={{ marginTop: '40px' }}>
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
							width: '33vw',
							height: '25px',
							background: '#ddd',
							borderRadius: '5px',
							outline: 'none',
							cursor: 'pointer',
						}}
					/>
					<div className="acceptance-rate-value" style={{ marginTop: '20px' }}>{localAcceptanceRate}%</div>
				</div>
				<button className="save-button" style={{ marginTop: '20px' }} onClick={handleSaveThreshold}>
					Save
				</button>
			</div>

			<div className="additional-controls" style={{ marginTop: '50px' }}>
				<button onClick={handleOpenLeft}>Open Left</button>
				<button style={{ marginLeft: '20px' }} onClick={handleOpenRight}>Open Right</button>
			</div>
		</div>
	);
};

export default Settings;
