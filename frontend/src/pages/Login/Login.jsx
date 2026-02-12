
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Login.css";

const Login = () => {
  const [credentials, setCredentials] = useState({ email: "", password: "" });
  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [loginAttempts, setLoginAttempts] = useState(0);
  const navigate = useNavigate();

  const validateForm = () => {
    const newErrors = {};
    
    // Email validation
    if (!credentials.email) {
      newErrors.email = "Email is required";
    } else if (!/\S+@\S+\.\S+/.test(credentials.email)) {
      newErrors.email = "Email is invalid";
    }
    
    // Password validation
    if (!credentials.password) {
      newErrors.password = "Password is required";
    } else if (credentials.password.length < 8) {
      newErrors.password = "Password must be at least 8 characters";
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setCredentials({ ...credentials, [name]: value });
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors({ ...errors, [name]: "" });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    setIsLoading(true);
    
    try {
      // Simulating API call with timeout
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // TODO: Replace with actual API call for authentication
      if (credentials.email === "admin@example.com" && credentials.password === "password") {
        // Store auth token instead of user credentials
        localStorage.setItem("authToken", "example-token-would-come-from-server");
        navigate("/transaction");
      } else {
        setLoginAttempts(prev => prev + 1);
        
        if (loginAttempts >= 2) {
          setErrors({ general: "Too many failed attempts. Please try again later or reset your password." });
        } else {
          setErrors({ general: "Invalid credentials. Please try again." });
        }
      }
    } catch (error) {
      setErrors({ general: "Login failed. Please try again later." });
      console.error("Login error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleForgotPassword = () => {
    navigate("/reset-password");
  };

  return (
    <div className="login-container">
      <h2>Welcome Back</h2>
      <p className="login-subtitle">Please enter your credentials to continue</p>
      
      {errors.general && (
        <div className="error-message">{errors.general}</div>
      )}
      
      <form onSubmit={handleSubmit} noValidate>
        <div className="form-group">
          <label htmlFor="email">Email</label>
          <input 
            type="email" 
            id="email"
            name="email" 
            value={credentials.email}
            onChange={handleChange} 
            className={errors.email ? "input-error" : ""}
            disabled={isLoading}
            autoComplete="username"
          />
          {errors.email && <span className="error-text">{errors.email}</span>}
        </div>
        
        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input 
            type="password" 
            id="password"
            name="password" 
            value={credentials.password}
            onChange={handleChange}
            className={errors.password ? "input-error" : ""}
            disabled={isLoading}
            autoComplete="current-password"
          />
          {errors.password && <span className="error-text">{errors.password}</span>}
        </div>
        
        <div className="form-actions">
          <button 
            type="button" 
            className="text-button" 
            onClick={handleForgotPassword}
            disabled={isLoading}
          >
            Forgot Password?
          </button>
        </div>
        
        <button 
          type="submit" 
          className={`submit-button ${isLoading ? "loading" : ""}`}
          disabled={isLoading}
        >
          {isLoading ? "Logging in..." : "Log In"}
        </button>
      </form>
      
      <div className="register-link">
        <p>Don't have an account? <a href="/register">Sign up</a></p>
      </div>
    </div>
  );
};

export default Login;

