import React, { useState } from 'react';
import { Route, Routes, useLocation } from 'react-router-dom';
import './App.css';
import Navbar from './Navbar';
import About from './Pages/About';
import Contact from './Pages/Contact';
import Services from './Pages/Services';
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
              element={<HomeWithWebcam mode={mode} acceptanceRate={acceptanceRate} />}
            />
            <Route path="/about" element={<About />} />
            <Route path="/services" element={<Services />} />
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
const HomeWithWebcam = ({ mode, acceptanceRate }) => {
  const [speed, setSpeed] = useState(0);
  const [currentTime, setCurrentTime] = useState(new Date());

  React.useEffect(() => {
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

  return (
    <div className="dashboard">
      <h1 className="main-heading">Welcome to the Dashboard</h1>

      {/* Half Circle Display for Acceptance Rate */}
      <div className="half-circle-container">
        <div className="half-circle">
          <p className="value-text">{acceptanceRate}%</p>
        </div>
      </div>

      {/* Speed Controls */}
      <div className="controls">
        <button onClick={() => changeSpeed('decrease')}>-</button>
        <p className="speed-display">{speed}</p>
        <button onClick={() => changeSpeed('increase')}>+</button>
      </div>

      {/* Current Time Display */}
      <div className="current-time">
        <p>{currentTime.toLocaleString()}</p>
      </div>

      {/* Table Display */}
      <table className="summary-table">
        <thead>
          <tr>
            <th>Accept</th>
            <th>Reject</th>
            {mode === 'production' && <th>Total</th>}
          </tr>
        </thead>
        <tbody>
          {tableData.map((row, index) => (
            <tr key={index}>
              <td>{row.accept}</td>
              <td>{row.reject}</td>
              {mode === 'production' && <td>{row.accept + row.reject}</td>}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Production Mode Popup */}
      {mode === 'production' && (
        <div className="popup">Production Mode Activated</div>
      )}
    </div>
  );
};

export default App;
