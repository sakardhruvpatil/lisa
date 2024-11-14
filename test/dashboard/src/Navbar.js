import React, { useState, useEffect } from "react";
import "./Navbar.css";
import { Link, NavLink, useLocation } from "react-router-dom";

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    // Close the menu when the location changes (i.e., when navigating to a different page)
    setMenuOpen(false);
  }, [location]); // Effect runs every time the location changes

  return (
    <nav className="navbar">
      <Link to="/" className="title">Website</Link>
      <div className="menu" onClick={() => setMenuOpen(!menuOpen)}>
        <span></span>
        <span></span>
        <span></span>
      </div>
      <ul className={menuOpen ? "open" : "closed"}>
        <li><NavLink to="/">Home</NavLink></li>
        <li><NavLink to="/about">About</NavLink></li>
        <li><NavLink to="/AnalyticsData">AnalyticsData</NavLink></li>
        <li><NavLink to="/contact">Contact</NavLink></li>
        <li><Link to="/settings">Settings</Link></li>
      </ul>
    </nav>
  );
};

export default Navbar;



