import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const AnalyticsData = () => {
  const [analyticsData, setAnalyticsData] = useState([]);
  const [filteredData, setFilteredData] = useState([]); // State to hold filtered data based on selected date
  const [selectedDate, setSelectedDate] = useState(null); // State for selected date
  const [showMonthlyData, setShowMonthlyData] = useState(false); // State to show/hide the monthly data pop-up
  const [monthlyData, setMonthlyData] = useState([]); // To hold the aggregated data for the month

  // Connect to WebSocket server
  useEffect(() => {
    const socket = new WebSocket('ws://localhost:5000'); // WebSocket server URL

    socket.onopen = () => {
      console.log('WebSocket connected');
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log(data);  // You can log or use this data directly

      // Send data to backend via POST request
      fetch('http://localhost:8000/log-analytics-data/', { // FastAPI endpoint
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      })
      .then(response => response.json())
      .then(responseData => {
        console.log('Data logged to DB:', responseData);
      })
      .catch(error => {
        console.error('Error logging data to DB:', error);
      });

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

  // Filter data based on selected date
  useEffect(() => {
    if (selectedDate) {
      const selectedDateString = selectedDate.toLocaleDateString('en-CA'); // Get YYYY-MM-DD format to match date format in data
      const filtered = analyticsData.filter(item => item.date.split('T')[0] === selectedDateString);
      setFilteredData(filtered); // Update filtered data state
    } else {
      setFilteredData(analyticsData); // If no date is selected, show all data
    }
  }, [selectedDate, analyticsData]);

  // Aggregate monthly data based on the available analytics data
  useEffect(() => {
    if (analyticsData.length > 0) {
      const aggregatedData = analyticsData.reduce((acc, item) => {
        const month = new Date(item.date).toLocaleString('default', { month: 'long', year: 'numeric' }); // Format month-year
        if (!acc[month]) {
          acc[month] = { accept: 0, reject: 0 };
        }
        acc[month].accept += item.accept;
        acc[month].reject += item.reject;
        return acc;
      }, {});

      setMonthlyData(Object.keys(aggregatedData).map(month => ({
        month,
        accept: aggregatedData[month].accept,
        reject: aggregatedData[month].reject,
      })));
    }
  }, [analyticsData]);

  // Chart Data
  const chartData = {
    labels: filteredData.map(item => item.date),  // Dates (timestamps)
    datasets: [
      {
        label: 'Accept',
        data: filteredData.map(item => item.accept),  // Accept data
        borderColor: 'green',
        backgroundColor: 'rgba(0, 255, 0, 0.2)',
        fill: true,
      },
      {
        label: 'Reject',
        data: filteredData.map(item => item.reject),  // Reject data
        borderColor: 'red',
        backgroundColor: 'rgba(255, 0, 0, 0.2)',
        fill: true,
      },
    ],
  };

  // Handle date selection
  const handleDateChange = (date) => {
    setSelectedDate(date); // Update the selected date
  };

  // Show or hide the monthly data pop-up
  const toggleMonthlyDataPopup = () => {
    setShowMonthlyData(!showMonthlyData);
  };

  return (
    <div className="analytics-data">
      <h1>Analytics Data</h1>

      {/* Button to trigger monthly data pop-up */}
      <div style={{ textAlign: 'center', marginBottom: '20px' }}>
        <button onClick={toggleMonthlyDataPopup}>Show Monthly Data</button>
      </div>

      {/* Monthly Data Pop-up */}
      {showMonthlyData && (
        <div className="monthly-data-popup">
          <h2>Monthly Data</h2>
          <div className="monthly-data-content">
            {/* Display aggregated monthly data */}
            {monthlyData.length > 0 ? (
              <table className="summary-table">
                <thead>
                  <tr>
                    <th>Month</th>
                    <th>Accept</th>
                    <th>Reject</th>
                  </tr>
                </thead>
                <tbody>
                  {monthlyData.map((item, index) => (
                    <tr key={index}>
                      <td>{item.month}</td>
                      <td>{item.accept}</td>
                      <td>{item.reject}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p>No data available for the selected month.</p>
            )}
          </div>
          <button onClick={toggleMonthlyDataPopup}>Close</button>
        </div>
      )}

      {/* Date Picker for selecting date */}
      <div className="date-picker-container" style={{ marginBottom: '20px', textAlign: 'center' }}>
        <label htmlFor="date-picker">Select Date: </label>
        <DatePicker
          id="date-picker"
          selected={selectedDate}
          onChange={handleDateChange}
          dateFormat="yyyy/MM/dd" // Customize the format as needed
          placeholderText="Click to select a date"
          maxDate={new Date()} // Optionally, prevent future date selection
        />
      </div>

      {/* Display the filtered analytics data in a table */}
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
            {filteredData.map((item, index) => (
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
