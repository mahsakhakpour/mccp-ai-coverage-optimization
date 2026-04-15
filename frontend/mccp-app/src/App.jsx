import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [points, setPoints] = useState('');
  const [eps, setEps] = useState('2.0');
  const [minSamples, setMinSamples] = useState('3');
  const [radius, setRadius] = useState('1.5');
  const [loading, setLoading] = useState(false);
  const [loadingAI, setLoadingAI] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [aiSuggestions, setAiSuggestions] = useState(null);
  const [hotspots, setHotspots] = useState(null);
  const [insights, setInsights] = useState(null);
  const [activeTab, setActiveTab] = useState('results');
  const canvasRef = useRef(null);

  const parsePoints = (pointsText) => {
    const lines = pointsText.trim().split('\n');
    return lines.map(line => {
      const coords = line.split(',').map(Number);
      return [coords[0], coords[1]];
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let pointsArray;
      if (points.includes('\n')) {
        pointsArray = parsePoints(points);
      } else {
        pointsArray = JSON.parse(points);
      }

      const requestData = {
        points: pointsArray,
        eps: parseFloat(eps),
        min_samples: parseInt(minSamples),
        radius: parseFloat(radius)
      };

      const response = await axios.post('http://localhost:8000/optimize', requestData);
      setResult(response.data);
      setActiveTab('results');
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const getAISuggestions = async () => {
    setLoadingAI(true);
    setError(null);
    try {
      let pointsArray;
      if (points.includes('\n')) {
        pointsArray = parsePoints(points);
      } else {
        pointsArray = JSON.parse(points);
      }
      
      const response = await axios.post('http://localhost:8000/ai/suggest-parameters', {
        points: pointsArray,
        eps: parseFloat(eps),
        min_samples: parseInt(minSamples),
        radius: parseFloat(radius)
      });
      setAiSuggestions(response.data);
      
      // Auto-fill the suggested parameters
      setEps(response.data.suggested_eps.toString());
      setMinSamples(response.data.suggested_min_samples.toString());
      setRadius(response.data.suggested_radius.toString());
      
      setActiveTab('ai');
    } catch (err) {
      setError("AI suggestion failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoadingAI(false);
    }
  };

  const getHotspots = async () => {
    setLoadingAI(true);
    setError(null);
    try {
      let pointsArray;
      if (points.includes('\n')) {
        pointsArray = parsePoints(points);
      } else {
        pointsArray = JSON.parse(points);
      }
      
      const response = await axios.post('http://localhost:8000/ai/predict-hotspots', {
        points: pointsArray,
        eps: parseFloat(eps),
        min_samples: parseInt(minSamples),
        radius: parseFloat(radius)
      });
      setHotspots(response.data);
      setActiveTab('hotspots');
    } catch (err) {
      setError("Hotspot prediction failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoadingAI(false);
    }
  };

  const getInsights = async () => {
    setLoadingAI(true);
    setError(null);
    try {
      let pointsArray;
      if (points.includes('\n')) {
        pointsArray = parsePoints(points);
      } else {
        pointsArray = JSON.parse(points);
      }
      
      const response = await axios.post('http://localhost:8000/ai/insights', {
        points: pointsArray,
        eps: parseFloat(eps),
        min_samples: parseInt(minSamples),
        radius: parseFloat(radius)
      });
      setInsights(response.data);
      setActiveTab('insights');
    } catch (err) {
      setError("Insights failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoadingAI(false);
    }
  };

  const loadSampleData = () => {
    setPoints(`0,0
1,1
2,2
10,10
11,11
12,12
0,10
1,11
2,12
5,5
5.5,5.5
6,6`);
    setEps('2.0');
    setMinSamples('3');
    setRadius('1.5');
  };

  const loadRandomData = () => {
    const randomPoints = [];
    for (let i = 0; i < 50; i++) {
      const x = (Math.random() * 20).toFixed(2);
      const y = (Math.random() * 20).toFixed(2);
      randomPoints.push(`${x},${y}`);
    }
    setPoints(randomPoints.join('\n'));
  };

  // Draw visualization
  useEffect(() => {
    if (!result || !result.best_center || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    let pointsArray;
    try {
      if (points.includes('\n')) {
        pointsArray = parsePoints(points);
      } else {
        pointsArray = JSON.parse(points);
      }
    } catch(e) {
      return;
    }

    if (pointsArray.length === 0) return;

    const slidingCenter = result.best_center;
    const bruteCenter = result.brute_force_center;
    const circleRadius = parseFloat(radius);

    let minX = Math.min(...pointsArray.map(p => p[0]), slidingCenter[0] - circleRadius);
    let maxX = Math.max(...pointsArray.map(p => p[0]), slidingCenter[0] + circleRadius);
    let minY = Math.min(...pointsArray.map(p => p[1]), slidingCenter[1] - circleRadius);
    let maxY = Math.max(...pointsArray.map(p => p[1]), slidingCenter[1] + circleRadius);
    
    if (bruteCenter) {
      minX = Math.min(minX, bruteCenter[0] - circleRadius);
      maxX = Math.max(maxX, bruteCenter[0] + circleRadius);
      minY = Math.min(minY, bruteCenter[1] - circleRadius);
      maxY = Math.max(maxY, bruteCenter[1] + circleRadius);
    }
    
    const padding = 60;
    const scaleX = (width - 2 * padding) / (maxX - minX);
    const scaleY = (height - 2 * padding) / (maxY - minY);
    
    const transformX = (x) => padding + (x - minX) * scaleX;
    const transformY = (y) => height - padding - (y - minY) * scaleY;

    // Draw grid
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = minX + (i / 10) * (maxX - minX);
      const y = minY + (i / 10) * (maxY - minY);
      ctx.beginPath();
      ctx.moveTo(transformX(x), transformY(minY));
      ctx.lineTo(transformX(x), transformY(maxY));
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(transformX(minX), transformY(y));
      ctx.lineTo(transformX(maxX), transformY(y));
      ctx.stroke();
    }

    // Draw Brute Force Circle
    if (bruteCenter) {
      const bruteCenterX = transformX(bruteCenter[0]);
      const bruteCenterY = transformY(bruteCenter[1]);
      const bruteRadiusPx = circleRadius * scaleX;
      
      ctx.beginPath();
      ctx.arc(bruteCenterX, bruteCenterY, bruteRadiusPx, 0, 2 * Math.PI);
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 3;
      ctx.setLineDash([10, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
      
      ctx.fillStyle = '#22c55e';
      ctx.font = '12px Arial';
      ctx.fillText(`Brute: ${result.brute_force_count} pts`, bruteCenterX - 45, bruteCenterY - 10);
    }

    // Draw Sliding Circle
    const slidingCenterX = transformX(slidingCenter[0]);
    const slidingCenterY = transformY(slidingCenter[1]);
    const slidingRadiusPx = circleRadius * scaleX;
    
    ctx.beginPath();
    ctx.arc(slidingCenterX, slidingCenterY, slidingRadiusPx, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(170, 59, 255, 0.1)';
    ctx.fill();
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.stroke();

    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 20px Arial';
    ctx.fillText('×', slidingCenterX - 7, slidingCenterY + 8);
    
    ctx.fillStyle = '#ef4444';
    ctx.font = '12px Arial';
    ctx.fillText(`Sliding: ${result.max_count} pts`, slidingCenterX - 40, slidingCenterY - 20);

    // Draw points with cluster colors
    const clusterLabels = result.cluster_labels;
    const clusterColors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4'];
    
    pointsArray.forEach((point, idx) => {
      const x = transformX(point[0]);
      const y = transformY(point[1]);
      const isInside = Math.sqrt(Math.pow(point[0] - slidingCenter[0], 2) + Math.pow(point[1] - slidingCenter[1], 2)) <= circleRadius;
      
      let pointColor = '#9ca3af';
      if (clusterLabels && clusterLabels[idx] !== -1) {
        pointColor = clusterColors[clusterLabels[idx] % clusterColors.length];
      }
      if (isInside) pointColor = '#aa3bff';
      
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = pointColor;
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    ctx.fillStyle = '#333';
    ctx.font = 'bold 14px Arial';
    ctx.fillText('🔴 Red = Sliding Circle | 🟢 Green Dashed = Brute Force', 20, 30);
  }, [result, points, radius]);

  return (
    <div className="app-container">
      <h1>Maximum Circular Coverage Problem</h1>
      <p className="subtitle">AI-Powered Spatial Coverage Optimization</p>
      
      <div className="input-section">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Data Points:</label>
            <textarea
              rows="5"
              value={points}
              onChange={(e) => setPoints(e.target.value)}
              placeholder="0,0&#10;1,1&#10;2,2"
              required
            />
            <div className="button-group">
              <button type="button" onClick={loadSampleData} className="btn-secondary">Sample</button>
              <button type="button" onClick={loadRandomData} className="btn-secondary">Random</button>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>eps (cluster distance):</label>
              <input type="number" step="0.1" value={eps} onChange={(e) => setEps(e.target.value)} required />
            </div>
            <div className="form-group">
              <label>min_samples:</label>
              <input type="number" step="1" value={minSamples} onChange={(e) => setMinSamples(e.target.value)} required />
            </div>
            <div className="form-group">
              <label>radius:</label>
              <input type="number" step="0.1" value={radius} onChange={(e) => setRadius(e.target.value)} required />
            </div>
          </div>

          <div className="button-group-full">
            <button type="submit" disabled={loading} className="btn-primary">
              {loading ? 'Running...' : 'Run MCCP'}
            </button>
            <button type="button" onClick={getAISuggestions} disabled={loadingAI} className="btn-ai">
              AI Suggest Parameters
            </button>
            <button type="button" onClick={getHotspots} disabled={loadingAI} className="btn-ai">
              Find Hotspots
            </button>
            <button type="button" onClick={getInsights} disabled={loadingAI} className="btn-ai">
              Get Insights
            </button>
          </div>
        </form>

        {error && <div className="error">❌ {error}</div>}
      </div>

      {/* Tab Navigation */}
      {(result || aiSuggestions || hotspots || insights) && (
        <div className="tabs">
          <button className={`tab ${activeTab === 'results' ? 'active' : ''}`} onClick={() => setActiveTab('results')}>
            Results
          </button>
          <button className={`tab ${activeTab === 'ai' ? 'active' : ''}`} onClick={() => setActiveTab('ai')}>
            AI Suggestions
          </button>
          <button className={`tab ${activeTab === 'hotspots' ? 'active' : ''}`} onClick={() => setActiveTab('hotspots')}>
            Hotspots
          </button>
          <button className={`tab ${activeTab === 'insights' ? 'active' : ''}`} onClick={() => setActiveTab('insights')}>
            Insights
          </button>
        </div>
      )}

      {/* Results Tab */}
      {activeTab === 'results' && result && (
        <div className="results-section">
          <h2>Algorithm Comparison</h2>
          <div className="results-grid">
            <div className="result-card">
              <h3>🔴 Sliding Circle</h3>
              <div className="result-value">{result.max_count}</div>
              <div className="result-detail">points covered</div>
              <div className="result-detail">{result.sliding_time.toFixed(4)}s</div>
            </div>
            <div className="result-card">
              <h3>Brute Force</h3>
              <div className="result-value">{result.brute_force_count}</div>
              <div className="result-detail">points covered</div>
              <div className="result-detail">{result.brute_force_time.toFixed(4)}s</div>
            </div>
            <div className="result-card">
              <h3>Speedup</h3>
              <div className="result-value">{result.speedup_percentage.toFixed(1)}%</div>
              <div className="result-detail">faster than brute force</div>
            </div>
            <div className="result-card">
              <h3>Accuracy</h3>
              <div className="result-value">{result.accuracy_percentage.toFixed(1)}%</div>
              <div className="result-detail">of optimal solution</div>
            </div>
          </div>

          <div className="visualization">
            <canvas ref={canvasRef} width={750} height={500} style={{ border: '1px solid #ccc', borderRadius: '8px', width: '100%', height: 'auto', background: 'white' }} />
          </div>

          <div className="info-box">
            <p>✅ <strong>Sliding Circle:</strong> ({result.best_center[0].toFixed(2)}, {result.best_center[1].toFixed(2)})</p>
            <p>✅ <strong>Brute Force:</strong> ({result.brute_force_center[0].toFixed(2)}, {result.brute_force_center[1].toFixed(2)})</p>
          </div>
        </div>
      )}

      {/* AI Suggestions Tab */}
      {activeTab === 'ai' && aiSuggestions && (
        <div className="ai-section">
          <h2>AI Parameter Suggestions</h2>
          <div className="results-grid">
            <div className="result-card ai-card">
              <h3>Suggested eps</h3>
              <div className="result-value">{aiSuggestions.suggested_eps}</div>
              <button className="btn-use" onClick={() => setEps(aiSuggestions.suggested_eps.toString())}>Use</button>
            </div>
            <div className="result-card ai-card">
              <h3>Suggested min_samples</h3>
              <div className="result-value">{aiSuggestions.suggested_min_samples}</div>
              <button className="btn-use" onClick={() => setMinSamples(aiSuggestions.suggested_min_samples.toString())}>Use</button>
            </div>
            <div className="result-card ai-card">
              <h3>Suggested radius</h3>
              <div className="result-value">{aiSuggestions.suggested_radius}</div>
              <button className="btn-use" onClick={() => setRadius(aiSuggestions.suggested_radius.toString())}>Use</button>
            </div>
          </div>
          <div className="info-box">
            <p><strong>AI Reasoning:</strong> {aiSuggestions.reasoning || "Parameters optimized for maximum coverage"}</p>
            <p><strong>Data Density:</strong> {aiSuggestions.density?.toFixed(3) || "N/A"}</p>
            <p><strong>Confidence:</strong> {(aiSuggestions.confidence * 100).toFixed(0)}%</p>
            <p>{aiSuggestions.message}</p>
          </div>
        </div>
      )}

      {/* Hotspots Tab */}
      {activeTab === 'hotspots' && hotspots && (
        <div className="ai-section">
          <h2>AI-Powered Hotspots</h2>
          <p className="subtitle">{hotspots.message}</p>
          <div className="hotspots-list">
            {hotspots.hotspots.map((spot, idx) => (
              <div key={idx} className="hotspot-card">
                <div className={`hotspot-priority ${spot.priority}`}>
                  {spot.priority === 'high' ? '' : ''} Priority {idx + 1}
                </div>
                <div className="hotspot-location">
                  Location: ({spot.center[0].toFixed(2)}, {spot.center[1].toFixed(2)})
                </div>
                <div className="hotspot-coverage">
                  Potential Coverage: {spot.potential_coverage} points
                </div>
                <div className="hotspot-radius">
                  Suggested Radius: {spot.suggested_radius}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Insights Tab */}
      {activeTab === 'insights' && insights && (
        <div className="ai-section">
          <h2>AI Data Insights</h2>
          <p className="subtitle">{insights.summary}</p>
          <div className="insights-list">
            {insights.insights.map((insight, idx) => (
              <div key={idx} className="insight-card">
                {insight}
              </div>
            ))}
          </div>
          {insights.statistics && (
            <div className="stats-box">
              <h3>Statistics</h3>
              <p>X Range: {insights.statistics.x_range}</p>
              <p>Y Range: {insights.statistics.y_range}</p>
              <p>Density: {insights.statistics.density}</p>
              <p>Total Points: {insights.statistics.total_points}</p>
            </div>
          )}
        </div>
      )}

      <footer>
        <p>© 2025 by Mahsa Khakpour | AI-Powered MCCP | Sliding Circle vs Brute Force</p>
      </footer>
    </div>
  );
}

export default App;