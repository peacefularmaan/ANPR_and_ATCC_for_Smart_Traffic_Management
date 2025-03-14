import React from 'react'
import './Background.css'
import video2 from '../../assets/video2.mp4'
import image4 from '../../assets/image4.jpg'
import image5 from '../../assets/image5.jpg'
import image10 from '../../assets/image10.jpg'

const Background = ({playStatus, heroCount}) => {
  if (playStatus) {
    return(
        <video className='background fade-in' autoPlay loop muted>
            <source src={video2} type='video/mp4'/>
        </video>
    )
  }
  else if (heroCount === 0) {
    return <img src={image4} className='background fade-in' alt=''/>
  }
  else if (heroCount === 1) {
    return <img src={image5} className='background fade-in' alt="" />
  }
  else if (heroCount === 2) {
    return <img src={image10} className='background fade-in' alt="" />
  }
}

export default Background
