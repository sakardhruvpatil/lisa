import React, { useState } from 'react';

const Settings = ({ setMode, setAcceptanceRate }) => {
  const [localAcceptanceRate, setLocalAcceptanceRate] = useState(50); // Local state for slider value

  const handleSliderChange = (e) => {
    const newValue = e.target.value;
    setLocalAcceptanceRate(newValue); // Update local state
    setAcceptanceRate(newValue); // Update global state in App.js
  };

  return (
    <div className="settings-container">
      <h2>Settings</h2>

      <div className="mode-selection">
        <button onClick={() => setMode('demo')}>Demo Mode</button>
        <button onClick={() => setMode('production')}>Production Mode</button>
      </div>

      <div className="acceptance-rate">
        <label>Set Acceptance Rate:</label>
        <input
          type="range"
          min="0"
          max="100"
          value={localAcceptanceRate}
          onChange={handleSliderChange}
        />
        <div className="acceptance-rate-value">{localAcceptanceRate}%</div>
      </div>
    </div>
  );
};

export default Settings;
