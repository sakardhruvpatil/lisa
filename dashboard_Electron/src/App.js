import React, { useState, useEffect } from 'react';
import { Route, Routes, useLocation } from 'react-router-dom';
import './App.css';
import Navbar from './Navbar';
import About from './Pages/About';
import Contact from './Pages/Contact';
import AnalyticsData from './Pages/AnalyticsData'; // Updated import
import WebcamCapture from './WebcamCapture';
import Settings from './Pages/Settings';

// Main App component
const App = () => {
  const location = useLocation();
  const [cameraLayout, setCameraLayout] = useState('vertical');
  const [mode, setMode] = useState('demo'); // Default to demo mode
  const [acceptanceRate, setAcceptanceRate] = useState(50); // Default acceptance rate
  const [showLayoutModal, setShowLayoutModal] = useState(false);

  const handleModeChange = (newMode) => {
    setMode(newMode);
    if (newMode === 'production') {
      setShowLayoutModal(true); // Show layout modal for production mode
    }
  };

  const handleLayoutChange = (layout) => {
    setCameraLayout(layout);
    setShowLayoutModal(false); // Close the layout modal after layout change
  };

  return (
    <div className="app-layout">
      <Navbar />
      <div className="content">
        <div className="dashboard">
          <Routes>
            <Route
              path="/"
              element={<HomeWithWebcam mode={mode} acceptanceRate={acceptanceRate} cameraLayout={cameraLayout} />}
            />
            <Route path="/about" element={<About />} />
            <Route 
              path="/AnalyticsData" 
              element={<AnalyticsData acceptanceRate={acceptanceRate} cameraLayout={cameraLayout} />} // Pass props to AnalyticsData
            />
            <Route path="/contact" element={<Contact />} />
            <Route
              path="/settings"
              element={<Settings setMode={handleModeChange} setAcceptanceRate={setAcceptanceRate} />}
            />
          </Routes>
        </div>
        <div className={`webcam-section ${cameraLayout}`}>
          <WebcamCapture mode={mode} cameraLayout={cameraLayout} />
        </div>
      </div>

      {/* Layout Selection Modal (for production mode) */}
      {showLayoutModal && (
        <div className="layout-modal">
          <div className="modal-content">
            <h2>Select Camera Layout</h2>
            <button onClick={() => handleLayoutChange('vertical')}>Vertical</button>
            <button onClick={() => handleLayoutChange('horizontal')}>Horizontal</button>
          </div>
        </div>
      )}
    </div>
  );
};

// Component for the Home page with webcam controls
const HomeWithWebcam = ({ mode, acceptanceRate, cameraLayout }) => {
  const [speed, setSpeed] = useState(0);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const changeSpeed = async (action) => {
    try {
      const response = await fetch('http://localhost:5000/change-speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      if (!response.ok) throw new Error('Failed to change speed');
      const data = await response.json();
      setSpeed(data.new_speed);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to change speed: ' + error.message);
    }
  };

  const tableData = [
    { accept: 10, reject: 5 },
    { accept: 15, reject: 10 },
  ];

  // Determine the number of rows based on mode and layout
  let rowsToDisplay = 1; // Default 1 row for demo mode
  if (mode === 'production') {
    if (cameraLayout === 'vertical') {
      rowsToDisplay = 2; // 2 rows in vertical layout
    } else if (cameraLayout === 'horizontal') {
      rowsToDisplay = 1; // 1 row, 2 columns in horizontal layout
    }
  }

  // Create an array with the number of rows to display based on `rowsToDisplay`
  const displayedTableData = tableData.slice(0, rowsToDisplay);

  return (
    <div className="dashboard">
      <h1 className="main-heading">Welcome to the Dashboard</h1>

      {/* Acceptance Rate Display (Moved above the table) */}
      <div className="acceptance-rate-display">
        <p>Acceptance Rate: {acceptanceRate}%</p>
        <div className="line"></div>
      </div>

      {/* Current Time Display */}
      <div className="current-time">
        <p>{currentTime.toLocaleString()}</p>
      </div>

      {/* Table Display */}
      <div className="table-container">
        {displayedTableData.map((row, index) => (
          <table className="summary-table" key={index}>
            <thead>
              <tr>
                <th>Accept</th>
                <th>Reject</th>
                {mode === 'production' && <th>Total</th>}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{row.accept}</td>
                <td>{row.reject}</td>
                {mode === 'production' && <td>{row.accept + row.reject}</td>}
              </tr>
            </tbody>
          </table>
        ))}
      </div>

      {/* Speed Controls */}
      <div className="controls" style={{ marginTop: '20px' }}>
        <button onClick={() => changeSpeed('decrease')}>-</button>
        <p className="speed-display">{speed}</p>
        <button onClick={() => changeSpeed('increase')}>+</button>
      </div>

      {/* Speed Button Label */}
      <div className="speed-button-label">
        <p>Speed Button</p>
      </div>

      {/* Production Mode Popup */}
      {mode === 'production' && (
        <div className="popup">Production Mode Activated</div>
      )}
    </div>
  );
};

export default App;
