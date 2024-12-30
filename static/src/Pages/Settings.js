import React, { useState, useEffect } from "react";
import PasswordModal from "../PasswordModals";

const Settings = ({ setMode, acceptanceRate, setAcceptanceRate }) => {
    const [localAcceptanceRate, setLocalAcceptanceRate] = useState(acceptanceRate);
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    // Track any error message to display instead of using alert
    const [errorMessage, setErrorMessage] = useState("");

    useEffect(() => {
        setLocalAcceptanceRate(acceptanceRate);
    }, [acceptanceRate]);

    // Instead of alert, update errorMessage state for immediate feedback
    const handlePasswordSubmit = (enteredPassword) => {
        const storedPassword = "1234";
        if (enteredPassword === storedPassword) {
            setIsAuthenticated(true);
            setErrorMessage("");
        } else {
            setErrorMessage("Incorrect password. Please try again.");
        }
    };

    // If user is not authenticated, always show the PasswordModal
    if (!isAuthenticated) {
        // Pass the errorMessage down to PasswordModal so it can display it
        return <PasswordModal onPasswordSubmit={handlePasswordSubmit} errorMessage={errorMessage} />;
    }

    const handleSliderChange = (e) => {
        const newValue = parseInt(e.target.value, 10);
        setLocalAcceptanceRate(newValue);
    };

    const handleSaveThreshold = async () => {
        setAcceptanceRate(localAcceptanceRate);

        try {
            const response = await fetch("http://localhost:8000/update_threshold", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ threshold: localAcceptanceRate }),
            });

            const result = await response.json();
            console.log("Threshold update response:", result);
        } catch (error) {
            console.error("Error updating threshold:", error);
        }
    };

    const handleModeChange = async (mode) => {
        setMode(mode);
        localStorage.setItem("cameraLayout", mode);
        try {
            const response = await fetch(
                `http://localhost:8000/set_process_mode/${mode}`,
                { method: "GET" },
            );
            const result = await response.json();
            console.log("Mode update response:", result);
        } catch (error) {
            console.error("Error updating mode:", error);
        }
    };

    const handleOpenLeft = async () => {
        console.log("Open Left button pressed");
        try {
            const response = await fetch("http://localhost:5007/open_left", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ signal: 1 }),
            });

            if (!response.ok) {
                throw new Error("Failed to activate left chute");
            }

            console.log("Left chute activated");
        } catch (error) {
            console.error("Error sending Open Left signal:", error);
        }
    };

    const handleOpenRight = async () => {
        console.log("Open Right button pressed");
        try {
            const response = await fetch("http://localhost:5007/open_right", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ signal: 1 }),
            });

            if (!response.ok) {
                throw new Error("Failed to activate right chute");
            }

            console.log("Right chute activated");
        } catch (error) {
            console.error("Error sending Open Right signal:", error);
        }
    };

    const handleOpenBoth = async () => {
        console.log("Open Both button pressed");
        try {
            const response = await fetch("http://localhost:5007/open_both", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ signal: 1 }),
            });

            if (!response.ok) {
                throw new Error("Failed to activate both chute");
            }

            console.log("Both chute activated");
        } catch (error) {
            console.error("Error sending Open both signal:", error);
        }
    };

    return (
        <div
            className="settings-container"
            style={{
                marginBottom: "60px",
                fontSize: "48px",
                fontWeight: "bold",
            }}
        >
            <label>Settings</label>

            <div className="mode-selection" style={{ marginTop: "40px" }}>
                <button onClick={() => handleModeChange("horizontal")}>
                    Horizontal
                </button>
                <button onClick={() => handleModeChange("vertical")}>Vertical</button>
            </div>

            <div className="acceptance-rate" style={{ marginTop: "50px" }}>
                <label>Set Acceptance Rate</label>
                <div className="slider-container" style={{ marginTop: "30px" }}>
                    <input
                        type="range"
                        min="0"
                        max="100"
                        value={localAcceptanceRate}
                        onChange={handleSliderChange}
                        style={{
                            width: "33vw",
                            height: "25px",
                            background: "#ddd",
                            borderRadius: "5px",
                            outline: "none",
                            cursor: "pointer",
                        }}
                    />
                    <div
                        className="acceptance-rate-value"
                        style={{ marginTop: "20px" }}
                    >
                        {localAcceptanceRate}%
                    </div>
                </div>
                <button
                    className="save-button"
                    style={{ marginTop: "20px" }}
                    onClick={handleSaveThreshold}
                >
                    Save
                </button>
            </div>

            <div
                className="additional-controls"
                style={{
                    marginTop: "60px",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                }}
            >
                <div style={{ display: "flex", justifyContent: "center" }}>
                    <button onClick={handleOpenLeft}>Open Left</button>
                    <button style={{ marginLeft: "40px" }} onClick={handleOpenRight}>
                        Open Right
                    </button>
                </div>
                <button style={{ marginTop: "40px" }} onClick={handleOpenBoth}>
                    Open Both
                </button>
            </div>
        </div>
    );
};

export default Settings;