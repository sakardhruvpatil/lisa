import React, { useEffect, useState } from 'react';

const HomePage = () => {
  const [countData, setCountData] = useState({
    total_bedsheets: 0,
    total_accepted: 0,
    total_rejected: 0,
  });

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/todays_counts');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setCountData(data);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    // Clean up the WebSocket connection when the component unmounts
    return () => {
      ws.close();
    };
  }, []);

  return (
    <div className="dashboard">
      <h1 className="main-heading">Welcome to the Dashboard</h1>

      <div className="current-time">
        <p>{new Date().toLocaleString()}</p>
      </div>

      <div className="data-counts">
        <h2>Today's Data Counts</h2>
        <p>Accepted: {countData.total_accepted}</p>
        <p>Rejected: {countData.total_rejected}</p>
        <p>Total Bedsheets: {countData.total_bedsheets}</p>
      </div>
    </div>
  );
};

export default HomePage;
