import React, { useEffect, useState } from 'react';

const HomePage = () => {
  const [countData, setCountData] = useState({ accept: 0, reject: 0, total: 0 });
  const [type, setType] = useState("accept");  // This can be randomly selected or cycled through

  // Fetch the initial data once
  useEffect(() => {
    fetch("http://localhost:8000/update_counts/")
      .then(response => response.json())
      .then(data => setCountData(data))
      .catch(error => console.error("Error fetching data:", error));
  }, []);

  // Function to send an entry to the backend
  const addEntryAutomatically = async () => {
    const value = 1; // Value to increment by

    try {
      const response = await fetch("http://localhost:8000/log-analytics-data/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ type, value }),
      });
      const result = await response.json();
      if (result.message === "Entry added successfully") {
        setCountData(result.data); // Update the data displayed
      } else {
        console.error(result.message);
      }
    } catch (error) {
      console.error("Error adding entry:", error);
    }
  };

  // Set up automatic data addition at intervals
  useEffect(() => {
    const intervalId = setInterval(() => {
      addEntryAutomatically();
    }, 5000); // Adjust interval time as needed (e.g., every 5 seconds)

    // Clean up the interval on component unmount
    return () => clearInterval(intervalId);
  }, [type]); // Optional: You can cycle `type` automatically too

  return (
    <div>
      <h2>Data Counts</h2>
      <p>Accepted: {countData.accept}</p>
      <p>Rejected: {countData.reject}</p>
      <p>Total: {countData.total}</p>
    </div>
  );
};

export default HomePage;
