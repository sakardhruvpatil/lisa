import React from 'react';

const Contact = () => {
  const containerStyle = {
    maxWidth: '800px',
    margin: '50px auto',
    padding: '20px',
    border: '1px solid #ddd',
    borderRadius: '8px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    fontFamily: 'Arial, sans-serif',
    backgroundColor: '#f9f9f9',
  };

  const headerStyle = {
    textAlign: 'center',
    color: 'orange', // Changed color to orange
    marginBottom: '20px',
  };

  const textStyle = {
    fontSize: '16px',
    color: '#555',
    lineHeight: '1.6',
  };

  const listStyle = {
    listStyleType: 'none',
    padding: 0,
    marginTop: '20px',
    fontSize: '16px',
    color: '#555',
  };

  const listItemStyle = {
    marginBottom: '10px',
  };

  return (
    <div style={containerStyle}>
      <h1 style={headerStyle}>Contact Us</h1>
      <p style={textStyle}>
        We value your feedback and inquiries. Our team is here to assist you with any questions or concerns you may have. Please feel free to reach out to us using the information below:
      </p>
      <ul style={listStyle}>
        <li style={listItemStyle}>
          <strong>Email:</strong> contact@sakarrobotics.com
        </li>
        <li style={listItemStyle}>
          <strong>Phone:</strong> +91 79722 51272
        </li>
        <li style={listItemStyle}>
          <strong>Address:</strong> Sakar Robotics, 2nd Floor, ANSEC House, Tank Road, Shanti Nagar, Yerwada, Pune, MH 411006
        </li>
      </ul>
      <p style={textStyle}>
        Whether you have a question about our services, pricing, or anything else, we are here to help you.
      </p>
    </div>
  );
};

export default Contact;
