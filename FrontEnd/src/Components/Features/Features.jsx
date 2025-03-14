// import React, { useState } from 'react';
// import './Features.css';

// const Features = () => {
//   const featureOptions = [
//     'Helmet Detection',
//     'Triple Riding Detection',
//     'Accident Detection',
//     'Wrong-Way Driving Detection',
//     'Wrong-Parking Detection',
//     'Red Signal Violation Detection'
//   ];
  
//   const [selectedFeature, setSelectedFeature] = useState(featureOptions[0]);
//   const [numCameras, setNumCameras] = useState(0);
//   const [files, setFiles] = useState({});

//   // Handler for file changes
//   const handleFileChange = (e, cameraIndex) => {
//     setFiles({
//       ...files,
//       [cameraIndex]: e.target.files[0],
//     });
//   };

//   const handleStartProcessing = () => {
//     // Backend processing logic goes here.
//     console.log(`Processing ${selectedFeature} for ${numCameras} cameras`);
//   };

//   return (
//     <div className="features-page">
//       <div className="left-pane">
//         <h2>Features</h2>
//         <ul>
//           {featureOptions.map((feature, index) => (
//             <li 
//               key={index}
//               className={selectedFeature === feature ? 'active' : ''}
//               onClick={() => setSelectedFeature(feature)}
//             >
//               {feature}
//             </li>
//           ))}
//         </ul>
//       </div>
//       <div className="center-pane">
//         <div className="features-box">
//           <h1>{selectedFeature}</h1>
//           <p>
//             Configure settings for {selectedFeature}. Please specify the number of cameras, upload videos, and start processing.
//           </p>
//           <div className="config-section">
//             <label>No. of Cameras</label>
//             <input
//               type="number"
//               value={numCameras}
//               onChange={(e) => setNumCameras(Number(e.target.value))}
//               min="0"
//             />
//           </div>
//           {Array.from({ length: numCameras }, (_, index) => (
//             <div key={index} className="camera-config">
//               <label>{`Camera ${index + 1}`}</label>
//               <input
//                 type="file"
//                 onChange={(e) => handleFileChange(e, index + 1)}
//               />
//             </div>
//           ))}
//           <div className="config-section">
//             <button onClick={handleStartProcessing}>Start Processing</button>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Features;


import React, { useState } from 'react';
import './Features.css';

const Features = () => {
  const featureOptions = [
    'Helmet Detection',
    'Triple Riding Detection',
    'Accident Detection',
    'Overspeeding Detection',
    'Heatmap Visualization',
    'Emergency Vehicle Detection',
    'Crosswalk Detection',
  ];
  
  const [selectedFeature, setSelectedFeature] = useState(featureOptions[0]);
  const [file, setFile] = useState(null);

  // Handle file selection
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleStartProcessing = async () => {
    if (!file) {
      alert("Please upload a video file.");
      return;
    }

    const formData = new FormData();
    formData.append("feature", selectedFeature);
    formData.append("video", file);

    try {
      const response = await fetch("http://localhost:5000/process", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        // alert(`Processing started for ${selectedFeature}! OpenCV window will appear.`);
      } else {
        alert("Error: " + data.error);
      }
    } catch (error) {
      console.error("Error processing video:", error);
      alert("Failed to connect to the server.");
    }
  };

  return (
    <div className="features-page">
      <div className="left-pane">
        <h2>Features</h2>
        <ul>
          {featureOptions.map((feature, index) => (
            <li 
              key={index}
              className={selectedFeature === feature ? 'active' : ''}
              onClick={() => setSelectedFeature(feature)}
            >
              {feature}
            </li>
          ))}
        </ul>
      </div>
      <div className="center-pane">
        <div className="features-box">
          <h1>{selectedFeature}</h1>
          <p>Upload a video and start processing for {selectedFeature}.</p>
          
          <div className="config-section">
            <label>Upload Video</label>
            <input type="file" accept="video/*" onChange={handleFileChange} />
          </div>

          <div className="config-section">
            <button onClick={handleStartProcessing}>Start Processing</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Features;
