// import React, { useState, useEffect } from 'react';
// import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// import Background from './Components/Background/Background';
// import Navbar from './Components/Navbar/Navbar';
// import Hero from './Components/Hero/Hero';
// import LiveMonitoring from './Components/LiveMonitoring/LiveMonitoring';
// import Login from './Components/Login/Login';

// const Home = ({ playStatus, heroCount, setHeroCount, heroData, setPlayStatus }) => (
//   <>
//     <Background playStatus={playStatus} heroCount={heroCount} />
//     <Hero
//       setPlayStatus={setPlayStatus}
//       heroData={heroData[heroCount]}
//       heroCount={heroCount}
//       setHeroCount={setHeroCount}
//       playStatus={playStatus}
//     />
//   </>
// );

// const App = () => {
//   const [isAuthenticated, setIsAuthenticated] = useState(false);
//   const heroData = [
//     { text1: "Let's make", text2: "our roads safer." },
//     { text1: "Making", text2: "travelling easier." },
//     { text1: "Saving", text2: "more lives." },
//   ];
//   const [heroCount, setHeroCount] = useState(0);
//   const [playStatus, setPlayStatus] = useState(false);

//   useEffect(() => {
//     const interval = setInterval(() => {
//       setHeroCount((count) => (count + 1) % heroData.length);
//     }, 7000);
//     return () => clearInterval(interval);
//   }, [heroData.length]);

//   const handleLogout = () => {
//     setIsAuthenticated(false);
//   };

//   return (
//     <Router>
//       {!isAuthenticated ? (
//         <Routes>
//           <Route path="*" element={<Login setIsAuthenticated={setIsAuthenticated} />} />
//         </Routes>
//       ) : (
//         <>
//           {/* Navbar is placed here so it's always visible */}
//           <Navbar onLogout={handleLogout} />
//           <Routes>
//             <Route
//               path="/"
//               element={
//                 <Home
//                   playStatus={playStatus}
//                   heroCount={heroCount}
//                   setHeroCount={setHeroCount}
//                   heroData={heroData}
//                   setPlayStatus={setPlayStatus}
//                 />
//               }
//             />
//             <Route path="/live-monitoring" element={<LiveMonitoring />} />
//             <Route path="/login" element={<Login setIsAuthenticated={setIsAuthenticated} />} />
//           </Routes>
//         </>
//       )}
//     </Router>
//   );
// };

// export default App;


import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Background from './Components/Background/Background';
import Navbar from './Components/Navbar/Navbar';
import Hero from './Components/Hero/Hero';
import LiveMonitoring from './Components/LiveMonitoring/LiveMonitoring';
import Features from './Components/Features/Features'; // Adjust the import path as needed
import Login from './Components/Login/Login';

const Home = ({ playStatus, heroCount, setHeroCount, heroData, setPlayStatus }) => (
  <>
    <Background playStatus={playStatus} heroCount={heroCount} />
    <Hero
      setPlayStatus={setPlayStatus}
      heroData={heroData[heroCount]}
      heroCount={heroCount}
      setHeroCount={setHeroCount}
      playStatus={playStatus}
    />
  </>
);

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const heroData = [
    { text1: "Let's make", text2: "our roads safer." },
    { text1: "Make our", text2: "travelling easier." },
    { text1: "And save", text2: "more lives." },
  ];
  const [heroCount, setHeroCount] = useState(0);
  const [playStatus, setPlayStatus] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setHeroCount((count) => (count + 1) % heroData.length);
    }, 7000);
    return () => clearInterval(interval);
  }, [heroData.length]);

  const handleLogout = () => {
    setIsAuthenticated(false);
  };

  return (
    <Router>
      {!isAuthenticated ? (
        <Routes>
          <Route path="*" element={<Login setIsAuthenticated={setIsAuthenticated} />} />
        </Routes>
      ) : (
        <>
          <Navbar onLogout={handleLogout} />
          <Routes>
            <Route path="/" element={<Home playStatus={playStatus} heroCount={heroCount} setHeroCount={setHeroCount} heroData={heroData} setPlayStatus={setPlayStatus} />} />
            <Route path="/live-monitoring" element={<LiveMonitoring />} />
            <Route path="/features" element={<Features />} />
            <Route path="/login" element={<Login setIsAuthenticated={setIsAuthenticated} />} />
          </Routes>
        </>
      )}
    </Router>
  );
};

export default App;



