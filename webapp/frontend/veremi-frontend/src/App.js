import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [singleInput, setSingleInput] = useState('');
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [activeTab, setActiveTab] = useState('csv'); // 'csv' or 'single'

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid CSV file');
    }
  };

  const handleFileDrop = (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === 'text/csv') {
      setFile(droppedFile);
      setError(null);
    } else {
      setError('Please drop a valid CSV file');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const removeFile = () => {
    setFile(null);
    setPredictionData(null);
    setError(null);
  };

  const handleCSVUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPredictionData(data);
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
    } finally {
      setLoading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  const handleSinglePrediction = async () => {
    if (!singleInput.trim()) {
      setError('Please enter the data values');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('data', singleInput.trim());

    try {
      const response = await fetch('http://localhost:8000/predict_single', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPredictionData(data);
    } catch (err) {
      setError(`Prediction failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = () => {
    if (!predictionData) return;

    const csvContent = [
      ['Row', 'Predicted Class', 'Confidence', 'Probabilities'],
      ...predictionData.predictions.map((pred, index) => [
        index + 1,
        `Class ${pred}`,
        `${(Math.max(...predictionData.probabilities[index]) * 100).toFixed(2)}%`,
        predictionData.probabilities[index].map(p => p.toFixed(4)).join(';')
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'prediction_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>🤖 ML Model Prediction Service</h1>
          <p className="subtitle">Upload CSV files or test single predictions with your trained model</p>
          <div className="api-status">
            <span className="status-label">API Status:</span>
            <div className="status-indicator status-ok">
              <div className="status-dot"></div>
              <span>Online</span>
            </div>
          </div>
        </div>
      </header>

      <main className="main-content">
        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button 
            className={`tab-button ${activeTab === 'csv' ? 'active' : ''}`}
            onClick={() => setActiveTab('csv')}
          >
            📊 CSV Upload
          </button>
          <button 
            className={`tab-button ${activeTab === 'single' ? 'active' : ''}`}
            onClick={() => setActiveTab('single')}
          >
            🎯 Single Prediction
          </button>
        </div>

        {/* CSV Upload Tab */}
        {activeTab === 'csv' && (
          <section className="upload-section">
            <div className="section-header">
              <h2>📊 Upload CSV File</h2>
              <p>Upload your dataset for batch prediction analysis</p>
            </div>

            <div 
              className={`file-drop-zone ${file ? 'file-selected' : ''}`}
              onDrop={handleFileDrop}
              onDragOver={handleDragOver}
            >
              <input
                type="file"
                accept=".csv"
                onChange={handleFileSelect}
                className="file-input-hidden"
              />
              
              {!file ? (
                <div className="drop-zone-content">
                  <div className="upload-icon">📁</div>
                  <div className="drop-zone-placeholder">
                    <h3>Drop your CSV file here</h3>
                    <p>or click to browse files</p>
                    <div className="file-requirements">
                      <small>Accepts .csv files with 29 columns</small>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="file-selected-content">
                  <div className="file-icon">📄</div>
                  <div className="file-details">
                    <h3>{file.name}</h3>
                    <p>{(file.size / 1024).toFixed(2)} KB</p>
                  </div>
                  <button className="remove-file-btn" onClick={removeFile}>
                    ✕
                  </button>
                </div>
              )}
            </div>

            {uploadProgress > 0 && (
              <div className="progress-section">
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="progress-text">Processing... {uploadProgress}%</p>
              </div>
            )}

            <div className="button-group">
              <button 
                className="primary-btn" 
                onClick={handleCSVUpload}
                disabled={!file || loading}
              >
                {loading ? (
                  <>
                    <div className="spinner"></div>
                    Processing...
                  </>
                ) : (
                  <>
                    🚀 Start Prediction
                  </>
                )}
              </button>
            </div>
          </section>
        )}

        {/* Single Prediction Tab */}
        {activeTab === 'single' && (
          <section className="upload-section">
            <div className="section-header">
              <h2>🎯 Single Line Prediction</h2>
              <p>Enter a single row of data for instant prediction</p>
            </div>

            <div className="single-input-section">
              <label htmlFor="single-input" className="input-label">
                Enter 29 comma-separated values:
              </label>
              <textarea
                id="single-input"
                className="single-input-field"
                value={singleInput}
                onChange={(e) => setSingleInput(e.target.value)}
                placeholder="4,72002.30294185807,130137,101301377,422013806,266.98240149315575,32.33695481096056,0.0,3.480882465673816,3.473183912873589,0.0,-0.12466057308754501,1.2063426886358601,0.0,-0.0,-0.0,0.0,-0.21909979067814803,2.120327763513944,0.0,0.000862295952139,0.000862295952139,0.0,-0.102790238633832,0.9947030546055441,0.0,20.038218064395213,17.5410005137198,0.0"
                rows="4"
              />
              <small className="input-help">
                Enter exactly 29 numerical values separated by commas
              </small>
            </div>

            <div className="button-group">
              <button 
                className="primary-btn" 
                onClick={handleSinglePrediction}
                disabled={!singleInput.trim() || loading}
              >
                {loading ? (
                  <>
                    <div className="spinner"></div>
                    Predicting...
                  </>
                ) : (
                  <>
                    🎯 Predict
                  </>
                )}
              </button>
            </div>
          </section>
        )}

        {/* Error Display */}
        {error && (
          <div className="error-message">
            <div className="error-icon">⚠️</div>
            <div className="error-content">
              <h3>Error</h3>
              <p>{error}</p>
            </div>
          </div>
        )}

        {/* Results Section */}
        {predictionData && (
          <section className="results-section">
            <div className="results-header">
              <h2>📈 Prediction Results</h2>
              <button className="download-btn" onClick={downloadResults}>
                📥 Download Results
              </button>
            </div>

            <div className="results-summary">
              <div className="summary-card">
                <div className="summary-icon">📊</div>
                <div className="summary-content">
                  <h3>{predictionData.total_samples}</h3>
                  <p>Samples Processed</p>
                </div>
              </div>
              <div className="summary-card">
                <div className="summary-icon">✅</div>
                <div className="summary-content">
                  <h3>100%</h3>
                  <p>Success Rate</p>
                </div>
              </div>
              <div className="summary-card">
                <div className="summary-icon">🎯</div>
                <div className="summary-content">
                  <h3>{predictionData.predictions.length > 0 ? 'Complete' : 'N/A'}</h3>
                  <p>Status</p>
                </div>
              </div>
            </div>

            <div className="results-table-container">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Row #</th>
                    <th>Predicted Class</th>
                    <th>Confidence</th>
                    <th>All Probabilities</th>
                  </tr>
                </thead>
                <tbody>
                  {predictionData.predictions.slice(0, 100).map((pred, index) => (
                    <tr key={index} className="table-row">
                      <td className="row-number">{index + 1}</td>
                      <td className="predicted-class">
                        <span className={`class-badge class-${pred}`}>
                          Class {pred}
                        </span>
                      </td>
                      <td className="confidence">
                        {(Math.max(...predictionData.probabilities[index]) * 100).toFixed(2)}%
                      </td>
                      <td className="probabilities">
                        <div className="prob-bars">
                          {predictionData.probabilities[index].map((prob, i) => (
                            <div key={i} className="prob-item">
                              <span className="prob-label">C{i}</span>
                              <div className="prob-bar">
                                <div 
                                  className="prob-fill" 
                                  style={{ width: `${prob * 100}%` }}
                                ></div>
                              </div>
                              <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {predictionData.predictions.length > 100 && (
                <div className="table-footer">
                  Showing first 100 results. Download full results using the button above.
                </div>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
