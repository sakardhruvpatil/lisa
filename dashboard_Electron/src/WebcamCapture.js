import React, { useRef } from 'react';
import Webcam from 'react-webcam';

const WebcamCapture = ({ mode, cameraLayout }) => {
  const webcamRef1 = useRef(null);
  const webcamRef2 = useRef(null);

  // Check for demo mode (only one camera in demo mode)
  const isDemoMode = mode === 'demo';

  return (
    <div className={`webcam-container ${cameraLayout}`}>
      {/* First camera (always visible) */}
      <div className="webcam">
        <Webcam
          audio={false}
          ref={webcamRef1}
          screenshotFormat="image/jpeg"
          width="100%"
        />
      </div>

      {/* Second camera, only visible if not in demo mode */}
      {!isDemoMode && (
        <div className="webcam">
          <Webcam
            audio={false}
            ref={webcamRef2}
            screenshotFormat="image/jpeg"
            width="100%"
          />
        </div>
      )}
    </div>
  );
};

export default WebcamCapture;
