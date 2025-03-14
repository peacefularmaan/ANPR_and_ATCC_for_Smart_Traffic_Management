import React, { useState } from 'react';
import './LiveMonitoring.css';

const LiveMonitoring = () => {
  const [numCameras, setNumCameras] = useState(1); // Default to 1 camera
  const [files, setFiles] = useState([]); // Store files as an array
  const [process, setProcess] = useState('ANPR'); // Default selection

  // Handler for file selection per camera
  const handleFileChange = (e, cameraIndex) => {
    const selectedFiles = e.target.files;
    
    if (process === 'ANPR') {
      setFiles([selectedFiles[0]]); // Only 1 file allowed for ANPR
    } else {
      const newFiles = [...files];
      newFiles[cameraIndex] = selectedFiles[0];
      setFiles(newFiles);
    }
  };

  const handleStartProcessing = async () => {
    if (process === 'ANPR' && files.length !== 1) {
      alert('Please upload exactly one video for ANPR.');
      return;
    }
    if (process === 'ATCC' && (files.length === 0 || files.includes(undefined))) {
      alert('Please upload at least one video for ATCC.');
      return;
    }

    const formData = new FormData();
    formData.append("process", process);

    files.forEach((file) => {
      formData.append("videos", file);
    });

    try {
      const response = await fetch("http://localhost:5000/process", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        console.log(data);
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      alert("Failed to start processing. Check console for details.");
      console.error("Error processing video:", error);
    }
  };

  return (
    <div className="live-monitoring-page">
      <div className="live-monitoring-box">
        <h1>Live Monitoring Configuration</h1>
        <p>
          Configure your live monitoring settings here. Please specify the number
          of cameras and upload the relevant files for each camera.
        </p>
        
        {/* Select Process */}
        <div className="config-section">
          <label>Select Process</label>
          <select 
            value={process} 
            onChange={(e) => {
              setProcess(e.target.value);
              setFiles([]); // Reset file uploads
              setNumCameras(1); // Reset camera count
            }}
          >
            <option value="ANPR">ANPR</option>
            <option value="ATCC">ATCC</option>
          </select>
        </div>

        {/* Set Number of Cameras (Only for ATCC) */}
        {process === 'ATCC' && (
          <div className="config-section">
            <label>No. of Cameras</label>
            <input
              type="number"
              value={numCameras}
              onChange={(e) => {
                setNumCameras(Number(e.target.value));
                setFiles([]); // Reset files on number change
              }}
              min="1"
            />
          </div>
        )}

        {/* File Upload for ANPR */}
        {process === 'ANPR' && (
          <div className="config-section">
            <label>Upload Video</label>
            <input 
              type="file" 
              accept="video/*" 
              onChange={(e) => handleFileChange(e, 0)} 
            />
          </div>
        )}

        {/* File Upload for ATCC */}
        {process === 'ATCC' && Array.from({ length: numCameras }, (_, index) => (
          <div key={index} className="camera-config">
            <label>{`Camera ${index + 1}`}</label>
            <input 
              type="file" 
              accept="video/*" 
              onChange={(e) => handleFileChange(e, index)} 
            />
          </div>
        ))}

        <div className="config-section">
          <button onClick={handleStartProcessing}>Start Processing</button>
        </div>
      </div>
    </div>
  );
};

export default LiveMonitoring;
