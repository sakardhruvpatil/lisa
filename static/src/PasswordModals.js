import React, { useState, useRef } from "react"; // Import useRef

const PasswordModal = ({ onPasswordSubmit }) => {
    const [password, setPassword] = useState("");
    const inputRef = useRef(null); // Create a ref

    const handleSubmit = (e) => {
        e.preventDefault();
        onPasswordSubmit(password);
        setPassword("");
    };

    return (
        <div className="modal">
            <div
                className="modal-content"
                style={{ width: "400px", padding: "40px" }}
            >
                <h2 style={{ fontSize: "24px" }}>Enter Password</h2>
                <form onSubmit={handleSubmit}>
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        style={{
                            padding: "12px",
                            fontSize: "16px",
                            marginBottom: "20px",
                        }}
                        ref={inputRef} // Attach the ref to the input
                    />
                    <button
                        type="submit"
                        style={{ padding: "12px 20px", fontSize: "16px" }}
                        onClick={() => inputRef.current.focus()} // Focus on button click
                    >
                        Submit
                    </button>
                </form>
            </div>
        </div>
    );
};

export default PasswordModal;