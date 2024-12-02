import React from 'react';


const About = () => {
  return (
    <div className="about-container">
    
      <div className="company-info">
        <h2 className="section-heading">About Sakar Robotics</h2>
        <p className="company-text">
          Sakar Robotics is India’s leading Intelligent Automation company leveraging Machine Vision & Mobile Robotics technologies to serve the requirements of the industry. Located in Pune, Sakar Robotics started in May'2023, with a focus on construction, railways, bio-technology, and manufacturing industries. We have satisfied multiple major clients with 3 core products with high-volume repeat orders.
        </p>
        <p className="company-text">
          One of our standout products is the **LISA** (Linen Inspection and Sorting Assistant), a state-of-the-art stain and damage detection machine trusted by **Indian Railways**. LISA automates the detection of stains and damages on cotton bedsheets, enhancing the quality control process and ensuring high standards of hygiene for passenger satisfaction. With the ability to process bedsheets at a rate of 15-20 per minute (customizable), LISA provides real-time data collection and visualization for efficient monitoring.
        </p>
      </div>

      <div className="lisa-features">
        <h2 className="section-heading">Features of LISA</h2>
        <ul className="features-list">
          <li>Robust stain and damage detection algorithm.</li>
          <li>Smart AI-based decision making.</li>
          <li>Quick, responsive, and intuitive dashboard for real-time data visualization and monitoring.</li>
          <li>User-configurable system speed ranging from 1 - 50 meters per minute.</li>
          <li>Automated decision-making and rejection mechanism (optional) for acceptance or rejection.</li>
          <li>Backlight and High-Res camera for two-pass scan.</li>
        </ul>
      </div>

   
    </div>
  );
};

export default About;
