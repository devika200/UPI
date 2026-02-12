import React, { useState } from "react";
import axios from "axios";
import "./TransactionForm.css";

const TransactionForm = () => {
  const initialFormState = {
    amount: "",
    sender: "",
    receiver: "",
    timestamp: "",
    description: "",
    category: "personal"
  };

  const [formData, setFormData] = useState(initialFormState);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [formErrors, setFormErrors] = useState({});

  const validateForm = () => {
    const errors = {};
    
    if (!formData.amount || isNaN(Number(formData.amount)) || Number(formData.amount) <= 0) {
      errors.amount = "Please enter a valid amount";
    }
    
    if (!formData.sender || formData.sender.trim() === "") {
      errors.sender = "Sender UPI ID is required";
    } else if (!formData.sender.includes("@")) {
      errors.sender = "Enter a valid UPI ID (e.g., name@bank)";
    }
    
    if (!formData.receiver || formData.receiver.trim() === "") {
      errors.receiver = "Receiver UPI ID is required";
    } else if (!formData.receiver.includes("@")) {
      errors.receiver = "Enter a valid UPI ID (e.g., name@bank)";
    }
    
    if (!formData.timestamp) {
      errors.timestamp = "Transaction time is required";
    }
    
    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    
    // Clear specific error when field is being edited
    if (formErrors[name]) {
      setFormErrors({ ...formErrors, [name]: null });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    
    if (!validateForm()) {
      return;
    }
    
    setLoading(true);
    
    try {
      // Format data before sending
      const dataToSend = {
        ...formData,
        amount: Number(formData.amount),
        // Ensure timestamp is in ISO format
        timestamp: new Date(formData.timestamp).toISOString()
      };
      
      const response = await axios.post(
        "http://localhost:5000/api/check_fraud", 
        dataToSend,
        { timeout: 10000 } // 10 second timeout
      );
      
      setResult(response.data);
    } catch (error) {
      console.error("Error:", error);
      if (error.response) {
        // Server responded with error status
        setError(`Server error: ${error.response.data.message || error.response.statusText}`);
      } else if (error.request) {
        // Request made but no response received
        setError("No response from server. Please check your connection and try again.");
      } else {
        // Error setting up request
        setError(`Request error: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData(initialFormState);
    setResult(null);
    setError(null);
    setFormErrors({});
  };

  const getRiskLevel = (score) => {
    if (score >= 0.8) return { level: "High", color: "#ef4444" };
    if (score >= 0.5) return { level: "Medium", color: "#f59e0b" };
    return { level: "Low", color: "#10b981" };
  };

  return (
    <div className="transaction-form-container">
      <h2>Transaction Risk Assessment</h2>
      
      {error && (
        <div className="error-message">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>{error}</span>
        </div>
      )}
      
      {!result ? (
        <form onSubmit={handleSubmit} className="form-grid">
          <div className="form-group">
            <label htmlFor="amount">Amount (₹)</label>
            <input 
              type="text" 
              id="amount"
              name="amount" 
              value={formData.amount}
              placeholder="Enter amount" 
              onChange={handleChange}
            />
            {formErrors.amount && <p className="field-error">{formErrors.amount}</p>}
          </div>
          
          <div className="form-group">
            <label htmlFor="sender">Sender UPI ID</label>
            <input 
              type="text" 
              id="sender"
              name="sender" 
              value={formData.sender}
              placeholder="e.g., yourname@bank" 
              onChange={handleChange}
            />
            {formErrors.sender && <p className="field-error">{formErrors.sender}</p>}
          </div>
          
          <div className="form-group">
            <label htmlFor="receiver">Receiver UPI ID</label>
            <input 
              type="text" 
              id="receiver"
              name="receiver" 
              value={formData.receiver}
              placeholder="e.g., merchant@bank" 
              onChange={handleChange}
            />
            {formErrors.receiver && <p className="field-error">{formErrors.receiver}</p>}
          </div>
          
          <div className="form-group">
            <label htmlFor="timestamp">Transaction Time</label>
            <input 
              type="datetime-local" 
              id="timestamp"
              name="timestamp" 
              value={formData.timestamp}
              onChange={handleChange}
            />
            {formErrors.timestamp && <p className="field-error">{formErrors.timestamp}</p>}
          </div>
          
          <div className="form-group">
            <label htmlFor="category">Category</label>
            <select 
              id="category"
              name="category" 
              value={formData.category}
              onChange={handleChange}
            >
              <option value="personal">Personal</option>
              <option value="shopping">Shopping</option>
              <option value="bills">Bills & Utilities</option>
              <option value="food">Food & Dining</option>
              <option value="other">Other</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="description">Description (Optional)</label>
            <textarea 
              id="description"
              name="description" 
              value={formData.description}
              placeholder="Add transaction notes" 
              onChange={handleChange}
              rows="3"
            ></textarea>
          </div>
          
          <div className="form-actions">
            <button type="button" onClick={resetForm} className="reset-button">Reset</button>
            <button type="submit" className="submit-button" disabled={loading}>
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Processing...
                </>
              ) : (
                "Check Transaction"
              )}
            </button>
          </div>
        </form>
      ) : (
        <div className="result-container">
          <div className={`result-header ${result.is_fraud ? 'fraudulent' : 'legitimate'}`}>
            <h3>
              {result.is_fraud ? (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                  </svg>
                  Potentially Fraudulent Transaction
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                    <polyline points="22 4 12 14.01 9 11.01"></polyline>
                  </svg>
                  Transaction Appears Legitimate
                </>
              )}
            </h3>
          </div>
          
          <div className="result-details">
            <div className="result-summary">
              <div className="summary-item">
                <span>Risk Score:</span>
                <div 
                  className="risk-meter" 
                  style={{ 
                    "--risk-percent": `${result.fraud_score * 100}%`,
                    "--risk-color": getRiskLevel(result.fraud_score).color
                  }}
                >
                  <div className="risk-level">{getRiskLevel(result.fraud_score).level}</div>
                </div>
              </div>
              
              {result.risk_factors && (
                <div className="summary-item">
                  <span>Risk Factors:</span>
                  <ul className="risk-factors">
                    {result.risk_factors.map((factor, index) => (
                      <li key={index}>{factor}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {result.recommendation && (
                <div className="summary-item">
                  <span>Recommendation:</span>
                  <p>{result.recommendation}</p>
                </div>
              )}
            </div>
            
            <div className="transaction-summary">
              <h4>Transaction Details</h4>
              <div className="details-grid">
                <div>
                  <strong>Amount:</strong> ₹{formData.amount}
                </div>
                <div>
                  <strong>Sender:</strong> {formData.sender}
                </div>
                <div>
                  <strong>Receiver:</strong> {formData.receiver}
                </div>
                <div>
                  <strong>Time:</strong> {new Date(formData.timestamp).toLocaleString()}
                </div>
                {formData.category && (
                  <div>
                    <strong>Category:</strong> {formData.category}
                  </div>
                )}
                {formData.description && (
                  <div className="full-width">
                    <strong>Description:</strong> {formData.description}
                  </div>
                )}
              </div>
            </div>
          </div>
          
          <div className="result-actions">
            {result.is_fraud ? (
              <>
                <button className="danger-button">Cancel Transaction</button>
                <button className="caution-button">Proceed Anyway (Risky)</button>
              </>
            ) : (
              <button className="success-button">Proceed with Payment</button>
            )}
            <button className="secondary-button" onClick={resetForm}>New Check</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default TransactionForm;
