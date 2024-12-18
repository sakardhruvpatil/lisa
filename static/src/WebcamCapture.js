import React, { useState, useEffect, useRef } from 'react';

const WebcamCapture = ({ mode, cameraLayout }) => {
    const isHorizontalMode = mode === 'horizontal'; // Check if in horizontal mode
    const [activeFeed, setActiveFeed] = useState('left'); // Start with null to indicate no active feed yet

    // Ref to store the interval id
    const intervalRef = useRef('left');

    // Function to fetch the active feed from the backend
    const fetchActiveFeed = async () => {
        try {
            const response = await fetch('http://localhost:8000/current_feed');
            const data = await response.json();
            if (data.activeFeed) {
                setActiveFeed(data.activeFeed); // Set the active feed once it's retrieved
                clearInterval(intervalRef.current); // Stop polling once feed is retrieved
            }
        } catch (error) {
            console.error('Failed to fetch active feed:', error);
        }
    };

    useEffect(() => {
        // Start polling for the active feed until it's available
        if (!activeFeed) {
            intervalRef.current = setInterval(fetchActiveFeed, 500); // Fetch every 2 seconds
        }

        // Cleanup the interval when the component unmounts or the active feed is found
        return () => {
            clearInterval(intervalRef.current);
        };
    }, [activeFeed]); // This will only run when activeFeed changes

    // Define video sources
    const videoSrcLeft = 'http://localhost:8000/video_feed/left';
    const videoSrcRight = 'http://localhost:8000/video_feed/right';
    const videoSrcHorizontal = 'http://localhost:8000/video_feed/horizontal';

    // Render the correct feed based on the active feed state
    return (
        <div
            style={{
                display: cameraLayout === 'vertical' ? 'flex' : 'block',
                justifyContent: 'center',
                alignItems: 'center',
                flexDirection: cameraLayout === 'vertical' ? 'column' : 'row',
            }}
        >
            {activeFeed ? (
                isHorizontalMode ? (
                    <div style={{ width: '100%', margin: '5px' }}>
                        <img
                            src={videoSrcHorizontal}
                            alt="Horizontal Video Stream"
                            style={{
                                width: '100%',
                                height: '400px',
                                objectFit: 'cover',
                                borderRadius: '15px',
                                boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
                            }}
                            onError={() => {
                                console.error('Failed to load horizontal video feed from backend.');
                            }}
                        />
                    </div>
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
                                    console.error('Failed to load right video feed from backend.');
                                }}
                            />
                        </div>
                    </>
                )
            ) : (
                <div style={{ textAlign: 'center', marginTop: '20px' }}>
                    <p>Loading feed...</p>
                </div>
            )}
        </div>
    );
};

export default WebcamCapture;
