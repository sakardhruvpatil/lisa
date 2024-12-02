import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const AnalyticsData = () => {
    const [dailyData, setDailyData] = useState([]);
    const [filteredData, setFilteredData] = useState([]);
    const [selectedDate, setSelectedDate] = useState(null);
    const [showMonthlyData, setShowMonthlyData] = useState(false);
    const [monthlyData, setMonthlyData] = useState([]);

    // Fetch daily analytics data from backend
    useEffect(() => {
        const fetchDailyData = async () => {
            try {
                const response = await fetch('http://localhost:8000/daily_analytics');
                if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
                const data = await response.json();
                setDailyData(Array.isArray(data) ? data : []);
            } catch (error) {
                console.error('Error fetching daily analytics data:', error);
                setDailyData([]);
            }
        };
        fetchDailyData();
    }, []);

    // Fetch monthly data from backend
    useEffect(() => {
        const fetchMonthlyData = async () => {
            try {
                const response = await fetch('http://localhost:8000/monthly_analytics');
                const data = await response.json();
                setMonthlyData(data);
            } catch (error) {
                console.error('Error fetching monthly analytics data:', error);
            }
        };

        fetchMonthlyData();
    }, []);

    // Filter data based on selected date
    useEffect(() => {
        if (selectedDate) {
            // Convert selectedDate to UTC date string in 'YYYY-MM-DD' format
            const selectedDateUTC = new Date(
                selectedDate.getTime() - selectedDate.getTimezoneOffset() * 60000
            );
            const selectedDateString = selectedDateUTC.toISOString().split('T')[0];

            // For debugging: log the selected date and the dates in dailyData
            console.log('Selected Date String:', selectedDateString);
            dailyData.forEach((item) => {
                console.log('Item Date:', item.date);
            });

            const filtered = dailyData.filter(
                (item) => item.date === selectedDateString
            );
            setFilteredData(filtered);
        } else {
            setFilteredData(dailyData);
        }
    }, [selectedDate, dailyData]);

    // Handle date selection
    const handleDateChange = (date) => {
        setSelectedDate(date);
    };

    // Toggle monthly data pop-up
    const toggleMonthlyDataPopup = () => {
        setShowMonthlyData(!showMonthlyData);
    };

    // Prepare chart data for daily analytics
    const sortedDailyData = [...dailyData].sort(
        (a, b) => new Date(a.date) - new Date(b.date)
    );

    const dailyChartData = {
        labels: sortedDailyData.map((item) => item.date),
        datasets: [
            {
                label: 'Total Accepted',
                data: sortedDailyData.map((item) => item.total_accepted),
                borderColor: 'green',
                backgroundColor: 'rgba(0, 255, 0, 0.2)',
                fill: true,
            },
            {
                label: 'Total Rejected',
                data: sortedDailyData.map((item) => item.total_rejected),
                borderColor: 'red',
                backgroundColor: 'rgba(255, 0, 0, 0.2)',
                fill: true,
            },
        ],
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
                                        <th>Total Bedsheets</th> {/* Added Header */}
                                        <th>Accepted</th>
                                        <th>Rejected</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {monthlyData.map((item, index) => (
                                        <tr key={index}>
                                            <td>{item.month}</td>
                                            <td>{item.total_bedsheets}</td> {/* Display Total Bedsheets */}
                                            <td>{item.accepted}</td>
                                            <td>{item.rejected}</td>
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

            {/* Date Picker */}
            <div
                className="date-picker-container"
                style={{ marginBottom: '20px', textAlign: 'center' }}
            >
                <label htmlFor="date-picker">Select Date: </label>
                <DatePicker
                    id="date-picker"
                    selected={selectedDate}
                    onChange={handleDateChange}
                    dateFormat="yyyy-MM-dd"
                    placeholderText="Click to select a date"
                    maxDate={new Date()}
                />
                <button onClick={() => setSelectedDate(null)}>Clear Date</button>
            </div>

            {/* Data Table */}
            <div className="table-container">
                <table className="summary-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Total Bedsheets</th>
                            <th>Total Accepted</th>
                            <th>Total Rejected</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filteredData.length > 0 ? (
                            filteredData.map((item, index) => (
                                <tr key={index}>
                                    <td>{item.date}</td>
                                    <td>{item.total_bedsheets}</td>
                                    <td>{item.total_accepted}</td>
                                    <td>{item.total_rejected}</td>
                                </tr>
                            ))
                        ) : (
                            <tr>
                                <td colSpan="4">No data available for the selected date.</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>

            {/* Line Chart for Daily Analytics */}
            <div className="chart-container">
                <h2>Daily Analytics</h2>
                <Line
                    data={dailyChartData}
                    options={{
                        responsive: true,
                        plugins: { legend: { position: 'top' } },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Date',
                                },
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Count',
                                },
                                beginAtZero: true,
                            },
                        },
                    }}
                />
            </div>
        </div>
    );
};

export default AnalyticsData;
