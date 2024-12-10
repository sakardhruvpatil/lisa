import React, { useState, useEffect } from "react";
import "./Navbar.css";
import { Link, NavLink, useLocation } from "react-router-dom";

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    // Close the menu when the location changes (i.e., when navigating to a different page)
    setMenuOpen(false);
  }, [location]);

  useEffect(() => {
    // Close the menu when clicking outside of the sidebar
    const handleClickOutside = (event) => {
      if (event.target.closest('.sidebar') || event.target.closest('.menu')) return;
      setMenuOpen(false);
    };

    document.addEventListener("click", handleClickOutside);
    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, []);

  return (
    <nav className="navbar">
      <Link to="/" className="title" style={{ marginBottom: '30px' }}>LISA</Link>
      <div className="menu" onClick={() => setMenuOpen(!menuOpen)}>
        <div className={menuOpen ? "open" : ""}></div>
        <div className={menuOpen ? "open" : ""}></div>
        <div className={menuOpen ? "open" : ""}></div>
      </div>

      {/* Sidebar */}
      <div className={`sidebar ${menuOpen ? "open" : ""}`}>
        {menuOpen && <div className="sidebar-title" style={{ marginBottom: '30px' }}>LISA</div>} {/* Add title in the sidebar */}
        <ul>
          <li><NavLink to="/" activeClassName="active">Home</NavLink></li>
          <li><NavLink to="/about" activeClassName="active">About</NavLink></li>
          <li><NavLink to="/AnalyticsData" activeClassName="active">Analytics</NavLink></li>
          <li><NavLink to="/contact" activeClassName="active">Contact</NavLink></li>
          <li><Link to="/settings">Settings</Link></li>
        </ul>
      </div>

      {/* Overlay for closing the sidebar */}
      {menuOpen && <div className="overlay" onClick={() => setMenuOpen(false)}></div>}
    </nav>
  );
};

export default Navbar;