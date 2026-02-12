import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import "./Home.css";

const Home = () => {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  return (
    <div className={`home-page ${isLoaded ? 'loaded' : ''}`}>
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">
            Secure Your Transactions with <span className="accent">UPI Fraud Detection</span>
          </h1>
          <p className="hero-subtitle">
            Advanced AI-powered system that detects and prevents fraudulent UPI transactions in real-time.
            Protect your digital payments with industry-leading security technology.
          </p>
          <div className="hero-buttons">
            <Link to="/login" className="btn btn-primary">Get Started</Link>
            <Link to="/transaction" className="btn btn-secondary">Check Transaction</Link>
          </div>
          <div className="trust-indicators">
            <div className="trust-item">
              <div className="trust-icon">🛡️</div>
              <span>99.8% Accuracy</span>
            </div>
            <div className="trust-item">
              <div className="trust-icon">⚡</div>
              <span>Real-time Analysis</span>
            </div>
            <div className="trust-item">
              <div className="trust-icon">🔒</div>
              <span>Bank-grade Security</span>
            </div>
          </div>
        </div>
        <div className="hero-image">
          <div className="security-graphic">
            <div className="graphic-circle circle-1"></div>
            <div className="graphic-circle circle-2"></div>
            <div className="graphic-circle circle-3"></div>
            <div className="graphic-shield"></div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <h2 className="section-title">How It Works</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Advanced Analytics</h3>
            <p>Our system analyzes transaction patterns and behaviors to identify suspicious activities.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🤖</div>
            <h3>Machine Learning</h3>
            <p>AI algorithms continuously learn and adapt to new fraud techniques and patterns.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⏱️</div>
            <h3>Instant Verification</h3>
            <p>Get immediate results with our high-performance fraud detection engine.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📱</div>
            <h3>Cross-Platform</h3>
            <p>Works across all UPI apps and banking platforms for comprehensive protection.</p>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats-section">
        <div className="stat-container">
          <div className="stat-item">
            <h3 className="stat-number">10M+</h3>
            <p className="stat-label">Transactions Monitored</p>
          </div>
          <div className="stat-item">
            <h3 className="stat-number">99.8%</h3>
            <p className="stat-label">Detection Accuracy</p>
          </div>
          <div className="stat-item">
            <h3 className="stat-number">50K+</h3>
            <p className="stat-label">Frauds Prevented</p>
          </div>
          <div className="stat-item">
            <h3 className="stat-number">₹100M+</h3>
            <p className="stat-label">Savings for Users</p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h2>Ready to Secure Your Transactions?</h2>
          <p>Join thousands of users who trust our platform for their payment security.</p>
          <div className="cta-buttons">
            <Link to="/signup" className="btn btn-primary">Create Free Account</Link>
           
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;

