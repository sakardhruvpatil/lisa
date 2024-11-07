import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';

const WebcamCapture = ({ mode, cameraLayout }) => {
  const [videoDeviceIds, setVideoDeviceIds] = useState([]); // Store video device IDs
  const [permissionsGranted, setPermissionsGranted] = useState(false); // Track permissions
  const webcamRef1 = useRef(null);
  const webcamRef2 = useRef(null);

  const isDemoMode = mode === 'demo'; // Check if in demo mode (only one camera)

  useEffect(() => {
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
        } catch (error) {
          console.error('Error accessing cameras:', error);
        }
      }
    };

    getVideoDevices();
  }, []);

  return (
    <div style={{ display: cameraLayout === 'vertical' ? 'block' : 'flex' }}>
      {permissionsGranted && videoDeviceIds.length > 0 && (
        <>
          {/* First camera (always visible) */}
          <div style={{ width: cameraLayout === 'vertical' ? '100%' : '48%', margin: '5px' }}>
            <Webcam
              audio={false}
              ref={webcamRef1}
              screenshotFormat="image/jpeg"
              videoConstraints={{ deviceId: { exact: videoDeviceIds[0] } }}
              style={{ width: '100%', height: '400px' }} // Adjusted to moderate height
            />
          </div>

          {/* Second camera, only visible if not in demo mode */}
          {!isDemoMode && videoDeviceIds[1] && (
            <div style={{ width: cameraLayout === 'vertical' ? '100%' : '48%', margin: '5px' }}>
              <Webcam
                audio={false}
                ref={webcamRef2}
                screenshotFormat="image/jpeg"
                videoConstraints={{ deviceId: { exact: videoDeviceIds[1] } }}
                style={{ width: '100%', height: '400px' }} // Adjusted to moderate height
              />
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default WebcamCapture;
