// import React from 'react'
// import './Hero.css'
// import arrow_btn from '../../assets/arrow_btn.png'
// import play_icon from '../../assets/play_icon.png'
// import pause_icon from '../../assets/pause_icon.png'

// const Hero = ({heroData, setHeroCount, heroCount, setPlayStatus, playStatus}) => {
//   return (
//     <div className='hero'>
//         <div className="hero-text">
//             <p>{heroData.text1}</p>
//             <p>{heroData.text2}</p>
//         </div>
//         <div className="hero-explore">
//             {/* <p>Explore the features</p> */}
//             <p>About</p>
//             <img src={arrow_btn} alt="" />
//         </div>
//         <div className="hero-dot-play">
//             <ul className="hero-dots">
//                 <li onClick={()=>setHeroCount(0)} className={heroCount === 0? "hero-dot orange": "hero-dot"}></li>
//                 <li onClick={()=>setHeroCount(1)} className={heroCount === 1? "hero-dot orange": "hero-dot"}></li>
//                 <li onClick={()=>setHeroCount(2)} className={heroCount === 2? "hero-dot orange": "hero-dot"}></li>
//             </ul>
//             <div className="hero-play">
//                 <img onClick={()=>setPlayStatus(!playStatus)} src={playStatus?pause_icon:play_icon} alt="" />
//                 <p>See the video</p>
//             </div>
//         </div>
//     </div>
//   )
// }

// export default Hero


import React, { useState } from 'react';
import './Hero.css';
import arrow_btn from '../../assets/arrow_btn.png';
import play_icon from '../../assets/play_icon.png';
import pause_icon from '../../assets/pause_icon.png';

const Hero = ({ heroData, setHeroCount, heroCount, setPlayStatus, playStatus }) => {
  // State to handle dialog visibility
  const [showAboutDialog, setShowAboutDialog] = useState(false);

  const toggleAboutDialog = () => {
    setShowAboutDialog(!showAboutDialog);
  };

  return (
    <div className='hero'>
      <div className="hero-text-container">
        <div className="hero-text">
          <p>{heroData.text1}</p>
          <p>{heroData.text2}</p>
        </div>
      </div>
      <div className="hero-explore" onClick={toggleAboutDialog}>
        <p>About</p>
        <img src={arrow_btn} alt="Explore" />
      </div>
      <div className="hero-dot-play">
        <ul className="hero-dots">
          <li onClick={()=>setHeroCount(0)} className={heroCount === 0 ? "hero-dot orange" : "hero-dot"}></li>
          <li onClick={()=>setHeroCount(1)} className={heroCount === 1 ? "hero-dot orange" : "hero-dot"}></li>
          <li onClick={()=>setHeroCount(2)} className={heroCount === 2 ? "hero-dot orange" : "hero-dot"}></li>
        </ul>
        <div className="hero-play">
          <img onClick={()=>setPlayStatus(!playStatus)} src={playStatus ? pause_icon : play_icon} alt="Play/Pause" />
          <p>See the video</p>
        </div>
      </div>
      {showAboutDialog && (
        <div className="about-dialog-backdrop" onClick={toggleAboutDialog}>
          <div className="about-dialog" onClick={(e) => e.stopPropagation()}>
            <h2>About SmartTraffix</h2>
            <p>
              SmartTraffix is a cutting-edge solution designed to enhance road safety and traffic management.
              Explore features like live monitoring, advanced video processing, and automated detection systems.
              Our platform helps make roads safer, streamline travel, and save lives.
            </p>
            <button onClick={toggleAboutDialog}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Hero;
