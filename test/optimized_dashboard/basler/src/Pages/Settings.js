// Settings.js

import React, { useState, useEffect, useCallback } from 'react';

const Settings = ({ setMode, acceptanceRate, setAcceptanceRate }) => {
	const [localAcceptanceRate, setLocalAcceptanceRate] = useState(acceptanceRate);
	const [isModalOpen, setIsModalOpen] = useState(false);
	const [selectedDate, setSelectedDate] = useState('');
	const [data, setData] = useState([]);

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

	// Function to toggle modal visibility
	const toggleModal = () => {
		setIsModalOpen(!isModalOpen);
	};


	// Function to fetch data from backend, memoized with useCallback
	const fetchData = useCallback(async () => {
		try {
			const response = await fetch(
				`http://localhost:8000/analytics${selectedDate ? `?date=${selectedDate}` : ''}`
			);
			const result = await response.json();
			setData(result);
			console.log('Analytics data:', result);
		} catch (error) {
			console.error('Error fetching analytics data:', error);
		}
	}, [selectedDate]); // Depend on selectedDate

	// Function to fetch data when modal opens or date changes
	useEffect(() => {
		if (isModalOpen) {
			fetchData();
		}
	}, [isModalOpen, fetchData]); // Removed the eslint-disable comment

	// Function to handle date change and update state
	const handleDateChange = (e) => {
		const newSelectedDate = e.target.value; // Get the new date value
		setSelectedDate(newSelectedDate); // Update the state
	};

	// useEffect to fetch data when selectedDate changes
	useEffect(() => {
		if (selectedDate || isModalOpen) {
			fetchData();
		}
	}, [selectedDate, fetchData, isModalOpen]);


	return (
		<div className="settings-container" style={{ marginTop: '10px' }}>
			<h1>Settings</h1>

			<div className="mode-selection">
				<button onClick={() => setMode('demo')}>Demo Mode</button>
				<button onClick={() => setMode('production')}>Production Mode</button>
			</div>

			<div className="acceptance-rate" style={{ marginTop: '50px' }}>
				<label>Set Acceptance Rate:</label>
				<div className="slider-container" style={{ marginTop: '30px' }}>
					<input
						type="range"
						min="0"
						max="100"
						value={localAcceptanceRate}
						onChange={handleSliderChange}
					/>
					<div className="acceptance-rate-value" style={{ marginTop: '20px' }}>{localAcceptanceRate}%</div>
				</div>
				<button className="save-button"  style={{ marginTop: '20px' }} onClick={handleSaveThreshold}>
					Save
				</button>
			</div>
		</div>
	);
};

export default Settings;
