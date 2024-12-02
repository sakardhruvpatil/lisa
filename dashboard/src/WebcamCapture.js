import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';

const WebcamCapture = ({ mode, cameraLayout }) => {
	const [videoDeviceIds, setVideoDeviceIds] = useState([]); // Store video device IDs
	const [permissionsGranted, setPermissionsGranted] = useState(false); // Track permissions
	const webcamRef1 = useRef(null);
	const webcamRef2 = useRef(null);
	const videoStreamRef = useRef(null); // Ref to hold the video stream

	const isDemoMode = mode === 'demo'; // Check if in demo mode (only one camera)

	useEffect(() => {
		// Check if we have stored device IDs in sessionStorage
		const storedDeviceIds = sessionStorage.getItem('videoDeviceIds');
		const storedPermissions = sessionStorage.getItem('permissionsGranted');

		if (storedDeviceIds && storedPermissions) {
			// If device IDs and permissions are stored, use them
			setVideoDeviceIds(JSON.parse(storedDeviceIds));
			setPermissionsGranted(storedPermissions === 'true');
		} else {
			// Otherwise, request device permissions and store the data
			if (!isDemoMode) {
				const getVideoDevices = async () => {
					const devices = await navigator.mediaDevices.enumerateDevices();
					const videoInputs = devices.filter(device => device.kind === 'videoinput');

					if (videoInputs.length > 0) {
						const constraints = videoInputs.map(device => ({
							deviceId: { exact: device.deviceId },
						}));

						// Request access to both cameras
						try {
							await Promise.all(constraints.map(constraint => navigator.mediaDevices.getUserMedia({ video: constraint })));
							setVideoDeviceIds(videoInputs.map(device => device.deviceId)); // Set IDs for both cameras
							setPermissionsGranted(true); // Set permissions granted
							// Store device IDs and permissions in sessionStorage
							sessionStorage.setItem('videoDeviceIds', JSON.stringify(videoInputs.map(device => device.deviceId)));
							sessionStorage.setItem('permissionsGranted', 'true');
						} catch (error) {
							console.error('Error accessing cameras:', error);
						}
					}
				};

				getVideoDevices();
			}
		}

		return () => {
			// Clean up the video stream on unmount
			if (videoStreamRef.current) {
				videoStreamRef.current.getTracks().forEach(track => track.stop());
			}
		};
	}, [isDemoMode]); // Re-run effect when mode changes

	const videoSrc = 'http://localhost:8000/video_feed'; // Video feed from FastAPI

	return (
		<div style={{ display: cameraLayout === 'vertical' ? 'block' : 'flex', justifyContent: 'center' }}>
		  {isDemoMode ? (
			// In demo mode, show video feed from backend
			<div style={{ width: '100%', margin: '5px' }}>
			  <img
				src={videoSrc}
				alt="Video Stream"
				style={{
				  width: '100%',
				  height: '400px',
				  objectFit: 'cover',
				  borderRadius: '15px', // Rounded corners for demo video
				  boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)', // Add subtle shadow
				}}
				onError={(e) => {
				  e.target.onerror = null;
				  e.target.src = '';
				  console.error('Failed to load video feed from backend.');
				}}
			  />
			</div>
		  ) : (
			permissionsGranted && videoDeviceIds.length > 0 ? (
			  <>
				{/* First camera (always visible) */}
				<div
				  style={{
					width: cameraLayout === 'vertical' ? '100%' : '78%', // Adjust based on layout
					margin: '5px',
					borderRadius: '15px',
					overflow: 'hidden',
					boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)', // Shadow for depth
				  }}
				>
				  <h3 style={{ textAlign: 'center', color: '#333', marginBottom: '10px' }}>Left Camera</h3>
				  <Webcam
					audio={false}
					ref={webcamRef1}
					screenshotFormat="image/jpeg"
					videoConstraints={{ deviceId: { exact: videoDeviceIds[0] } }}
					style={{
					  width: '100%',
					  height: '500px',
					  objectFit: 'cover', // Ensure video fits within container
					  borderRadius: '15px', // Curved corners on the webcam
					}}
					onUserMedia={(stream) => {
					  // Store the video stream in the ref to persist it
					  if (!videoStreamRef.current) {
						videoStreamRef.current = stream;
					  }
					}}
				  />
				</div>
	
				{/* Second camera */}
				{videoDeviceIds[1] && (
				  <div
					style={{
					  width: cameraLayout === 'vertical' ? '100%' : '78%', // Adjust based on layout
					  margin: '5px',
					  borderRadius: '15px',
					  overflow: 'hidden',
					  boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)', // Shadow for depth
					}}
				  >
					<h3 style={{ textAlign: 'center', color: '#333', marginBottom: '10px' }}>Right Camera</h3>
					<Webcam
					  audio={false}
					  ref={webcamRef2}
					  screenshotFormat="image/jpeg"
					  videoConstraints={{ deviceId: { exact: videoDeviceIds[1] } }}
					  style={{
						width: '100%',
						height: '500px',
						objectFit: 'cover', // Ensure video fits within container
						borderRadius: '15px', // Curved corners on the webcam
					  }}
					  onUserMedia={(stream) => {
						// Store the video stream in the ref to persist it
						if (!videoStreamRef.current) {
						  videoStreamRef.current = stream;
						}
					  }}
					/>
				  </div>
				)}
			  </>
			) : (
			  <p>Loading cameras...</p>
			)
		  )}
		</div>
	  );
	};
	
	export default WebcamCapture;