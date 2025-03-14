
// import React, { useState } from 'react';
// import './Login.css';

// const Login = ({ setIsAuthenticated }) => {
//   const [username, setUsername] = useState('');
//   const [password, setPassword] = useState('');

//   const handleLogin = (e) => {
//     e.preventDefault();
//     // Replace with your actual authentication logic
//     if (username === 'user' && password === 'pass') {
//       setIsAuthenticated(true);
//     } else {
//       alert('Invalid credentials');
//     }
//   };

//   return (
//     <div className="login-wrapper">
//       <div className="login-box">
//         <h2>Login</h2>
//         <form onSubmit={handleLogin}>
//           <div className="input-group">
//             <label htmlFor="username">
//               <i className="icon-user"></i> Username
//             </label>
//             <input
//               id="username"
//               type="text"
//               value={username}
//               onChange={(e) => setUsername(e.target.value)}
//               required
//             />
//           </div>

//           <div className="input-group">
//             <label htmlFor="password">
//               <i className="icon-lock"></i> Password
//             </label>
//             <input
//               id="password"
//               type="password"
//               value={password}
//               onChange={(e) => setPassword(e.target.value)}
//               required
//             />
//           </div>

//           <div className="forgot-password">
//             <a href="#forgot">Forgot Password?</a>
//           </div>

//           <button type="submit">Login</button>
//         </form>
//       </div>
//       <footer>
//         <p>
//           © Magnificent Login Form. All Rights Reserved | Design By <span>W3layouts</span>
//         </p>
//       </footer>
//     </div>
//   );
// };

// export default Login;



import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';

const Login = ({ setIsAuthenticated }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();
    if (username === 'user' && password === 'pass') {
      setIsAuthenticated(true);
      navigate('/'); // Redirect to home after login
    } else {
      alert('Invalid credentials');
    }
  };

  return (
    <div className="login-wrapper">
      <div className="login-box">
        <h2>Login</h2>
        <form onSubmit={handleLogin}>
          <div className="input-group">
            <label htmlFor="username">
              <i className="icon-user"></i> Username
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div className="input-group">
            <label htmlFor="password">
              <i className="icon-lock"></i> Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <div className="forgot-password">
            <a href="#forgot">Forgot Password?</a>
          </div>
          <button type="submit">Login</button>
        </form>
      </div>
      <footer>
        <p>
          © Stylish Login Form. All Rights Reserved | Design By <span>PeacefulArmaan</span>
        </p>
      </footer>
    </div>
  );
};

export default Login;
