import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const AnalyticsData = () => {
  const [analyticsData, setAnalyticsData] = useState([]);
  
  // Connect to WebSocket server
  useEffect(() => {
    const socket = new WebSocket('ws://localhost:5000'); // WebSocket server URL

    socket.onopen = () => {
      console.log('WebSocket connected');
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log(data);  // You can log or use this data directly

      // Update state with the new data
      setAnalyticsData(prevData => [...prevData, data]);
    };

    socket.onclose = () => {
      console.log('WebSocket closed');
    };

    return () => {
      socket.close(); // Clean up the connection when the component is unmounted
    };
  }, []);

  // Chart Data
  const chartData = {
    labels: analyticsData.map(item => item.date),  // Dates (timestamps)
    datasets: [
      {
        label: 'Accept',
        data: analyticsData.map(item => item.accept),  // Accept data
        borderColor: 'green',
        backgroundColor: 'rgba(0, 255, 0, 0.2)',
        fill: true,
      },
      {
        label: 'Reject',
        data: analyticsData.map(item => item.reject),  // Reject data
        borderColor: 'red',
        backgroundColor: 'rgba(255, 0, 0, 0.2)',
        fill: true,
      },
    ],
  };

  return (
    <div className="analytics-data">
      <h1>Analytics Data</h1>

      {/* Display the analytics data in a table */}
      <div className="table-container">
        <table className="summary-table">
          <thead>
            <tr>
              <th>Day</th>
              <th>Accept</th>
              <th>Reject</th>
              <th>Date</th>
            </tr>
          </thead>
          <tbody>
            {analyticsData.map((item, index) => (
              <tr key={index}>
                <td>{item.day}</td>
                <td>{item.accept}</td>
                <td>{item.reject}</td>
                <td>{item.date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Line Chart */}
      <div className="chart-container">
        <h2>Accept and Reject Data</h2>
        <Line data={chartData} options={{ responsive: true, plugins: { legend: { position: 'top' } } }} />
      </div>
    </div>
  );
};

export default AnalyticsData;
