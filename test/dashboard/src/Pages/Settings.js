import React, { useState, useEffect } from 'react';

const Settings = ({ setMode }) => {
  const [localAcceptanceRate, setLocalAcceptanceRate] = useState(50); // Local state for slider value
  const [isModalOpen, setIsModalOpen] = useState(false); // State to manage modal visibility
  const [selectedDate, setSelectedDate] = useState('');
  const [data, setData] = useState([]); // Data fetched from backend

  // Function to handle slider changes and send updated threshold to backend
  const handleSliderChange = async (e) => {
    const newValue = parseInt(e.target.value, 10);
    setLocalAcceptanceRate(newValue); // Update local state

    // Send threshold update to backend
    try {
      const response = await fetch('http://localhost:8000/update_threshold', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold: newValue }),
      });

      const result = await response.json();
      console.log('Threshold update response:', result);
    } catch (error) {
      console.error('Error updating threshold:', error);
    }
  };

  // Function to toggle modal visibility
  const toggleModal = () => {
    setIsModalOpen(!isModalOpen);
  };

  // Function to fetch data when modal opens or date changes
  useEffect(() => {
    if (isModalOpen) {
      fetchData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isModalOpen, selectedDate]);

  // Function to fetch data from backend
  const fetchData = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/analytics${selectedDate ? `?date=${selectedDate}` : ''}`
      );
      const result = await response.json();
      setData(result);
      console.log('Analytics data:', result); // Add this line for debugging
    } catch (error) {
      console.error('Error fetching analytics data:', error);
    }
  };

  const handleDateChange = (e) => {
    const selectedDate = e.target.value; // This is in 'YYYY-MM-DD' format
    setSelectedDate(selectedDate);
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
        <button onClick={toggleModal}>View Data</button>
      </div>

      {/* Modal Integration */}
      {isModalOpen && (
        <div className="layout-modal" onClick={toggleModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Data for {selectedDate || 'All Dates'}</h2>
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
                    <th>Bedsheet No.</th>
                    <th>Detected Threshold (%)</th>
                    <th>Set Threshold (%)</th>
                    <th>Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {data.length > 0 ? (
                    data.map((row, index) => (
                      <tr key={index}>
                        <td>{row.date}</td>
                        <td>{row.bedsheet_number}</td>
                        <td>{row.detected_threshold.toFixed(2)}%</td>
                        <td>{row.set_threshold}%</td>
                        <td>{row.decision}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="5">No data available for the selected date.</td>
                    </tr>
                  )}
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
