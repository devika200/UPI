import React from "react";
import "./About.css"; // Create About.css for styling

const About = () => {
  return (
    <div className="about-container">
      <section className="about-section">
        <h2>About UPI Fraud Detection</h2>
        <p>
          Welcome to UPI Fraud Detection, a platform committed to safeguarding your digital transactions. In an increasingly digital world, the threat of online fraud is a serious concern. Our mission is to provide you with a robust, reliable, and intelligent system that detects and prevents fraudulent UPI transactions in real-time.
        </p>
        <p>
          We leverage cutting-edge AI and machine learning technologies to analyze transaction patterns, identify suspicious activities, and provide instant verification. Our system is designed to work seamlessly across all UPI apps and banking platforms, ensuring comprehensive protection for all users.
        </p>

        <h3>Our Core Values</h3>
        <ul>
          <li><strong>Security:</strong> We prioritize the safety and integrity of your transactions.</li>
          <li><strong>Innovation:</strong> We continuously innovate to stay ahead of evolving fraud techniques.</li>
          <li><strong>Reliability:</strong> We provide accurate and dependable fraud detection services.</li>
          <li><strong>Accessibility:</strong> We aim to make our platform accessible to all users, regardless of their technical expertise.</li>
        </ul>

        <p>
          Our team comprises experienced professionals in cybersecurity, data science, and software development. We are dedicated to providing you with the peace of mind that comes from knowing your digital transactions are secure.
        </p>
      </section>

      
     
    </div>
  );
};

export default About;