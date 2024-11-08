import React, { useState } from 'react';

const Settings = ({ setMode, setAcceptanceRate }) => {
  const [localAcceptanceRate, setLocalAcceptanceRate] = useState(50); // Local state for slider value
  const [isModalOpen, setIsModalOpen] = useState(false); // State to manage modal visibility
  const [selectedDate, setSelectedDate] = useState('');
  const [data, setData] = useState([
    { date: '2024-11-07', bedsheetNo: 1, status: 'Accepted' },
    { date: '2024-11-06', bedsheetNo: 2, status: 'Rejected' },
    { date: '2024-11-05', bedsheetNo: 3, status: 'Accepted' },
  ]); // Example static data

  const handleSliderChange = (e) => {
    const newValue = e.target.value;
    setLocalAcceptanceRate(newValue); // Update local state
    setAcceptanceRate(newValue); // Update global state in App.js
  };

  const toggleModal = () => {
    setIsModalOpen(!isModalOpen); // Toggle modal visibility
  };

  const handleDateChange = (e) => {
    setSelectedDate(e.target.value);
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

      <div className="view-data-button">
        <button onClick={toggleModal}>View Data</button> {/* Button to trigger modal */}
      </div>

      {/* Modal Integration */}
      {isModalOpen && (
        <div className="layout-modal" onClick={toggleModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Data for {selectedDate || 'Select a Date'}</h2>
            <div>
              <input
                type="date"
                value={selectedDate}
                onChange={handleDateChange}
                placeholder="Select Date"
              />
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>BedSheet No.</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((row, index) => (
                    <tr key={index}>
                      <td>{row.date}</td>
                      <td>{row.bedsheetNo}</td>
                      <td>{row.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <button className="close-btn" onClick={toggleModal}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Settings;
