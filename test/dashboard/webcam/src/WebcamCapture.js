import React, { useState, useEffect } from 'react';

const WebcamCapture = ({ mode, cameraLayout }) => {
	const isHorizontalMode = mode === 'horizontal'; // Check if in horizontal mode
	const [activeFeed, setActiveFeed] = useState('left'); // Default feed

	// Fetch the current active feed from the backend
	useEffect(() => {
		const fetchActiveFeed = async () => {
			try {
				const response = await fetch('http://localhost:8000/current_feed');
				const data = await response.json();
				setActiveFeed(data.activeFeed);
			} catch (error) {
				console.error('Failed to fetch active feed:', error);
			}
		};

		fetchActiveFeed();
	}, []); // Only run once on component mount

	const videoSrcLeft = 'http://localhost:8000/video_feed/left';
	const videoSrcRight = 'http://localhost:8000/video_feed/right';

	const [leftFeedError, setLeftFeedError] = useState(false);
	const [rightFeedError, setRightFeedError] = useState(false);

	const handleError = (feed) => {
		if (feed === 'left') setLeftFeedError(true);
		if (feed === 'right') setRightFeedError(true);
	};

	const showLeftFeed = !leftFeedError && activeFeed === 'left';
	const showRightFeed = !rightFeedError && activeFeed === 'right';

	return (
		<div
			style={{
				display: cameraLayout === 'vertical' ? 'flex' : 'block',
				justifyContent: 'center',
				alignItems: 'center',
				flexDirection: cameraLayout === 'vertical' ? 'column' : 'row', // Adjust for vertical or horizontal layout
			}}
		>
			{isHorizontalMode ? (
				<>
					{/* Show left feed if no error, otherwise show right feed */}
					{showLeftFeed ? (
						<div style={{ width: '100%', margin: '5px' }}>
							<img
								src={videoSrcLeft}
								alt="Left Video Stream"
								style={{
									width: '100%',
									height: '400px',
									objectFit: 'cover',
									borderRadius: '15px',
									boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
								}}
								onError={() => {
									handleError('left');
									console.error('Failed to load left video feed from backend.');
								}}
							/>
						</div>
					) : showRightFeed ? (
						<div style={{ width: '100%', margin: '5px' }}>
							<img
								src={videoSrcRight}
								alt="Right Video Stream"
								style={{
									width: '100%',
									height: '400px',
									objectFit: 'cover',
									borderRadius: '15px',
									boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
								}}
								onError={() => {
									handleError('right');
									console.error('Failed to load right video feed from backend.');
								}}
							/>
						</div>
					) : null}
				</>
			) : (
				<>
					{/* Left Camera Feed */}
					<div
						style={{
							width: cameraLayout === 'vertical' ? '100%' : '78%',
							margin: '5px',
							borderRadius: '15px',
							overflow: 'hidden',
							boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
						}}
					>
						<img
							src={videoSrcLeft}
							alt="Left Video Stream"
							style={{
								width: '100%',
								height: '500px',
								objectFit: 'cover',
								borderRadius: '15px',
							}}
							onError={() => {
								handleError('left');
								console.error('Failed to load left video feed from backend.');
							}}
						/>
					</div>

					{/* Right Camera Feed */}
					<div
						style={{
							width: cameraLayout === 'vertical' ? '100%' : '78%',
							margin: '5px',
							borderRadius: '15px',
							overflow: 'hidden',
							boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
						}}
					>
						<img
							src={videoSrcRight}
							alt="Right Video Stream"
							style={{
								width: '100%',
								height: '500px',
								objectFit: 'cover',
								borderRadius: '15px',
							}}
							onError={() => {
								handleError('right');
								console.error('Failed to load right video feed from backend.');
							}}
						/>
					</div>
				</>
			)}
		</div>
	);
};

export default WebcamCapture;
