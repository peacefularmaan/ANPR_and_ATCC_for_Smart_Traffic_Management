import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Navbar.css';

const Navbar = ({ onLogout }) => {
  const navigate = useNavigate();

  const handleLogout = () => {
    onLogout();
    navigate('/login');
  };

  return (
    <div className="nav">
      <div className="nav-logo">SmartTraffix</div>
      <ul className="nav-menu">
        <li onClick={() => navigate('/')}>Home</li>
        <li onClick={() => navigate('/live-monitoring')}>Live Monitoring</li>
        <li onClick={() => navigate('/features')}>Features</li>
        <li onClick={handleLogout}>Logout</li>
      </ul>
    </div>
  );
};

export default Navbar;

