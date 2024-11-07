import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const AnalyticsData = () => {
  // Sample data: Days of the week with corresponding accept/reject data
  const [analyticsData] = useState([
    { day: 'Monday', accept: 10, reject: 5 },
    { day: 'Tuesday', accept: 12, reject: 8 },
    { day: 'Wednesday', accept: 15, reject: 10 },
    { day: 'Thursday', accept: 20, reject: 5 },
    { day: 'Friday', accept: 18, reject: 7 },
    { day: 'Saturday', accept: 25, reject: 10 },
    { day: 'Sunday', accept: 30, reject: 5 },
  ]);

  // Chart Data
  const chartData = {
    labels: analyticsData.map(item => item.day), // Days
    datasets: [
      {
        label: 'Accept',
        data: analyticsData.map(item => item.accept), // Accept data
        borderColor: 'green',
        backgroundColor: 'rgba(0, 255, 0, 0.2)',
        fill: true,
      },
      {
        label: 'Reject',
        data: analyticsData.map(item => item.reject), // Reject data
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
            </tr>
          </thead>
          <tbody>
            {analyticsData.map((item, index) => (
              <tr key={index}>
                <td>{item.day}</td>
                <td>{item.accept}</td>
                <td>{item.reject}</td>
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
