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

  return (
    <nav className="navbar">
      <Link to="/" className="title">LISA</Link>
      <div className="menu" onClick={() => setMenuOpen(!menuOpen)}>
        <div className={menuOpen ? "open" : ""}></div>
        <div className={menuOpen ? "open" : ""}></div>
        <div className={menuOpen ? "open" : ""}></div>
      </div>

      {/* Sidebar */}
      <div className={`sidebar ${menuOpen ? "open" : ""}`}>
        <ul>
          <li><NavLink to="/" activeClassName="active">Home</NavLink></li>
          <li><NavLink to="/about" activeClassName="active">About</NavLink></li>
          <li className="analytics"><NavLink to="/AnalyticsData" activeClassName="active">AnalyticsData</NavLink></li>
          <li><NavLink to="/contact" activeClassName="active">Contact</NavLink></li>
          <li><Link to="/settings">Settings</Link></li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
