import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { Shield, Menu, X, Bell, User } from "lucide-react";
import "./Navbar.css";

const Navbar = ({ isLoggedIn }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const location = useLocation();

  // Handle scroll effect for navbar
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 20) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Close mobile menu when navigating to a new page
  useEffect(() => {
    setIsMenuOpen(false);
  }, [location]);

  // Toggle mobile menu
  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  // Check if a link is active
  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <nav className={`navbar ${isScrolled ? "scrolled" : ""}`}>
      <div className="navbar-container">
        <Link to="/" className="logo">
          <Shield size={22} />
          <span>UPI<span className="logo-accent">UPI SECURE</span></span>
        </Link>

        {/* Mobile menu button */}
        <button className="mobile-menu-button" onClick={toggleMenu}>
          {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>

        {/* Nav links - will be hidden on mobile and shown when menu is toggled */}
        <ul className={`nav-links ${isMenuOpen ? "active" : ""}`}>
          <li>
            <Link to="/" className={isActive("/") ? "active" : ""}>
              Home
            </Link>
          </li>
          
          <li>
            <Link to="/transaction" className={isActive("/transaction") ? "active" : ""}>
              Check Transaction
            </Link>
          </li>
          <li>
                <Link to="/history" className={isActive("/history") ? "active" : ""}>
                  History
                </Link>
              </li>
          

          
          {isLoggedIn && (
            <>
              <li>
                <Link to="/dashboard" className={isActive("/dashboard") ? "active" : ""}>
                  Dashboard
                </Link>
              </li>
              <li>
                <Link to="/history" className={isActive("/history") ? "active" : ""}>
                  History
                </Link>
              </li>
            </>
          )}
          
          <li>
            <Link to="/about" className={isActive("/about") ? "active" : ""}>
              About
            </Link>
          </li>
        </ul>

        {/* Auth section */}
        <div className="auth-section">
          {isLoggedIn ? (
            <>
              <Link to="/notifications" className="icon-button">
                <Bell size={20} />
                <span className="notification-badge">3</span>
              </Link>
              <div className="user-dropdown">
                <button className="user-button">
                  <div className="user-avatar">
                    <User size={18} />
                  </div>
                  <span className="user-name">User</span>
                </button>
                <div className="dropdown-menu">
                  <Link to="/profile">Profile</Link>
                  <Link to="/settings">Settings</Link>
                  <Link to="/logout">Logout</Link>
                </div>
              </div>
            </>
          ) : (
            <div className="auth-buttons">
              <Link to="/login" className="login-button">
                Login
              </Link>
              <Link to="/register" className="register-button">
                Register
              </Link>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
