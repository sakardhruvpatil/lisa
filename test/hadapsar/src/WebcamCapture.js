// WebcamCapture.js

import React, { useEffect, useState } from 'react';

const WebcamCapture = ({ mode, cameraLayout }) => {
	const isDemoMode = mode === 'demo'; // Check if in demo mode (only one camera)

	// Define the video feed URLs
	const videoSrcLeft = 'http://localhost:8000/video_feed/left';
	const videoSrcRight = 'http://localhost:8000/video_feed/right';

	return (
		<div
			style={{
				display: cameraLayout === 'vertical' ? 'block' : 'flex',
				justifyContent: 'center',
				alignItems: 'center',
			}}
		>
			{isDemoMode ? (
				// In demo mode, show video feed from backend for left camera
				<div style={{ width: '100%', margin: '5px' }}>
					<img
						src={videoSrcLeft}
						alt="Left Video Stream"
						style={{ width: '100%', height: '400px', objectFit: 'cover' }}
						onError={(e) => {
							e.target.onerror = null;
							e.target.src = '';
							console.error('Failed to load left video feed from backend.');
						}}
					/>
				</div>
			) : (
				<>
					{/* Left Camera Feed */}
					<div
						style={{
							width: cameraLayout === 'vertical' ? '100%' : '48%',
							margin: '5px',
						}}
					>
						<img
							src={videoSrcLeft}
							alt="Left Video Stream"
							style={{ width: '100%', height: '400px', objectFit: 'cover' }}
							onError={(e) => {
								e.target.onerror = null;
								e.target.src = '';
								console.error('Failed to load left video feed from backend.');
							}}
						/>
					</div>

					{/* Right Camera Feed */}
					<div
						style={{
							width: cameraLayout === 'vertical' ? '100%' : '48%',
							margin: '5px',
						}}
					>
						<img
							src={videoSrcRight}
							alt="Right Video Stream"
							style={{ width: '100%', height: '400px', objectFit: 'cover' }}
							onError={(e) => {
								e.target.onerror = null;
								e.target.src = '';
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
