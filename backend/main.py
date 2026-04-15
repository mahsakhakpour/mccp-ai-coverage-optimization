from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import time

app = FastAPI(title="MCCP API", description="Maximum Circular Coverage Problem Solver with AI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Schema
class RequestData(BaseModel):
    points: List[List[float]]
    eps: float
    min_samples: int
    radius: float

# AI Optimizer Class
class MCCPAIOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.history = []
        
    def extract_features(self, points):
        """Extract features from points for AI analysis"""
        points_array = np.array(points)
        
        x_range = np.max(points_array[:, 0]) - np.min(points_array[:, 0])
        y_range = np.max(points_array[:, 1]) - np.min(points_array[:, 1])
        
        features = {
            'num_points': len(points_array),
            'x_range': x_range,
            'y_range': y_range,
            'x_std': np.std(points_array[:, 0]),
            'y_std': np.std(points_array[:, 1]),
            'density': len(points_array) / ((x_range + 1) * (y_range + 1)) if (x_range + 1) * (y_range + 1) > 0 else 0,
        }
        
        # Add KMeans cluster info
        if len(points_array) >= 3:
            n_clusters = min(5, max(2, len(points_array) // 10))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(points_array)
            features['num_clusters'] = len(np.unique(labels))
        else:
            features['num_clusters'] = 1
            
        return features
    
    def suggest_parameters(self, points, target_coverage=None):
        """AI-powered parameter suggestion"""
        features = self.extract_features(points)
        
        # Rule-based suggestions based on data characteristics
        if features['num_points'] < 10:
            suggestions = {
                'eps': round(max(1.0, features['x_range'] / 5, features['y_range'] / 5), 1),
                'min_samples': max(2, int(features['num_points'] / 5)),
                'radius': round(max(1.0, min(features['x_range'], features['y_range']) / 6), 1)
            }
        elif features['num_points'] < 50:
            suggestions = {
                'eps': round(max(1.0, features['x_range'] / 8, features['y_range'] / 8), 1),
                'min_samples': max(3, int(features['num_points'] / 10)),
                'radius': round(max(1.0, min(features['x_range'], features['y_range']) / 8), 1)
            }
        else:
            suggestions = {
                'eps': round(max(1.0, features['x_range'] / 10, features['y_range'] / 10), 1),
                'min_samples': max(5, int(features['num_points'] / 15)),
                'radius': round(max(1.0, min(features['x_range'], features['y_range']) / 10), 1)
            }
        
        # Adjust based on density
        if features['density'] > 1.0:
            suggestions['eps'] = round(suggestions['eps'] * 0.7, 1)
            suggestions['radius'] = round(suggestions['radius'] * 0.8, 1)
        elif features['density'] < 0.1:
            suggestions['eps'] = round(suggestions['eps'] * 1.5, 1)
            suggestions['radius'] = round(suggestions['radius'] * 1.3, 1)
            
        # Adjust for target coverage
        if target_coverage and target_coverage < len(points):
            coverage_ratio = target_coverage / len(points)
            suggestions['radius'] = round(suggestions['radius'] * (0.7 + 0.6 * coverage_ratio), 1)
            
        # Ensure min_samples is at least 2
        suggestions['min_samples'] = max(2, suggestions['min_samples'])
        
        # Add confidence score
        confidence = 0.85
        if features['num_points'] < 10:
            confidence = 0.70
        elif features['num_points'] > 100:
            confidence = 0.90
            
        suggestions['confidence'] = confidence
        suggestions['density'] = features['density']
        
        return suggestions
    
    def learn_from_results(self, features, parameters, coverage_count, total_points):
        """Learn from optimization results"""
        self.history.append({
            'features': features,
            'parameters': parameters,
            'coverage_ratio': coverage_count / total_points if total_points > 0 else 0
        })
        
        # Keep only last 100 results
        if len(self.history) > 100:
            self.history.pop(0)

# Create global AI optimizer instance
ai_optimizer = MCCPAIOptimizer()

# Sliding Circle Algorithm
def sliding_circle_algorithm(points: np.ndarray, radius: float, cluster_points: Dict[int, np.ndarray]):
    if len(cluster_points) == 0:
        return None, 0, []

    best_center = None
    max_count = 0
    all_circles = []
    
    sorted_clusters = sorted(cluster_points.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cid, cpoints in sorted_clusters:
        current_points = np.array(cpoints)
        
        if len(current_points) < max_count:
            continue
            
        sample_indices = range(0, len(current_points), max(1, len(current_points)//10))
        sample_points = current_points[list(sample_indices)]
        
        cluster_best_center = None
        cluster_best_count = 0
        
        for start_point in sample_points:
            current_center = start_point.copy()
            current_count = np.sum(np.linalg.norm(current_points - current_center, axis=1) <= radius)
            
            for _ in range(75):
                in_circle = np.linalg.norm(current_points - current_center, axis=1) <= radius
                points_in_circle = current_points[in_circle]
                
                if len(points_in_circle) == 0:
                    break
                    
                new_center = points_in_circle.mean(axis=0)
                direction = new_center - current_center
                step_size = min(radius/2, np.linalg.norm(direction))
                
                if step_size < 1e-6:
                    break
                    
                current_center += direction * (step_size / np.linalg.norm(direction))
                new_count = np.sum(np.linalg.norm(current_points - current_center, axis=1) <= radius)
                
                if new_count > current_count:
                    current_count = new_count
                else:
                    step_size *= 0.7
                    if step_size < 1e-6:
                        break
            
            if current_count > cluster_best_count:
                cluster_best_count = current_count
                cluster_best_center = current_center
        
        if cluster_best_center is not None:
            if cluster_best_count > max_count:
                max_count = cluster_best_count
                best_center = cluster_best_center
                all_circles.append(('optimal', best_center, max_count))
            elif cluster_best_count == max_count:
                all_circles.append(('optimal', cluster_best_center, cluster_best_count))
            else:
                all_circles.append(('secondary', cluster_best_center, cluster_best_count))

    return best_center, max_count, all_circles

# Brute Force Algorithm
def brute_force_algorithm(points: np.ndarray, radius: float, resolution: float = 0.05):
    if len(points) == 0:
        return None, 0
    
    padding = radius * 1.5
    x_min, y_min = np.min(points, axis=0) - padding
    x_max, y_max = np.max(points, axis=0) + padding
    step = radius * resolution

    best_center = None
    max_count = 0
    
    x_grid = np.arange(x_min, x_max + step, step)
    y_grid = np.arange(y_min, y_max + step, step)
    
    for x in x_grid:
        for y in y_grid:
            center = np.array([x, y])
            distances = np.linalg.norm(points - center, axis=1)
            count = np.sum(distances <= radius)
            if count > max_count:
                max_count = count
                best_center = center

    return best_center, max_count

# DBSCAN Clustering for visualization
def get_clusters(points: np.ndarray, eps: float, min_samples: int):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    return labels.tolist()

# Main Optimization Function
def run_mccp(points: List[List[float]], eps: float, min_samples: int, radius: float) -> Dict[str, Any]:
    points_array = np.array(points)
    
    # Get clusters for visualization
    cluster_labels = get_clusters(points_array, eps, min_samples)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)
    labels = clustering.labels_
    clusters = set(labels) - {-1}
    cluster_points = {cid: points_array[labels == cid] for cid in clusters}
    
    if not clusters:
        cluster_points = {-1: points_array}
    
    # Run Sliding Circle Algorithm
    start_time = time.time()
    sliding_center, sliding_count, sliding_circles = sliding_circle_algorithm(points_array, radius, cluster_points)
    sliding_time = time.time() - start_time
    
    # Run Brute Force Algorithm
    start_time = time.time()
    brute_center, brute_count = brute_force_algorithm(points_array, radius)
    brute_time = time.time() - start_time
    
    result = {
        "best_center": sliding_center.tolist() if sliding_center is not None else None,
        "max_count": int(sliding_count),
        "sliding_time": sliding_time,
        "brute_force_center": brute_center.tolist() if brute_center is not None else None,
        "brute_force_count": int(brute_count),
        "brute_force_time": brute_time,
        "accuracy_percentage": (sliding_count / brute_count * 100) if brute_count > 0 else 0,
        "speedup_percentage": ((brute_time - sliding_time) / brute_time * 100) if brute_time > 0 else 0,
        "cluster_labels": cluster_labels
    }
    
    return result

# ============ AI ENDPOINTS ============

# AI Parameter Suggestions
@app.post("/ai/suggest-parameters")
async def suggest_parameters(data: RequestData):
    try:
        if len(data.points) < 2:
            raise HTTPException(status_code=400, detail="At least 2 points are required")
        
        suggestions = ai_optimizer.suggest_parameters(data.points)
        
        return {
            "suggested_eps": suggestions['eps'],
            "suggested_min_samples": suggestions['min_samples'],
            "suggested_radius": suggestions['radius'],
            "confidence": suggestions['confidence'],
            "density": suggestions['density'],
            "message": f"AI analyzed {len(data.points)} points with density {suggestions['density']:.3f}",
            "reasoning": "Parameters optimized for maximum coverage based on spatial distribution"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI Hotspot Prediction
@app.post("/ai/predict-hotspots")
async def predict_hotspots(data: RequestData):
    try:
        points_array = np.array(data.points)
        
        if len(points_array) < 3:
            return {
                "hotspots": [],
                "message": "Need at least 3 points for hotspot detection"
            }
        
        # Use KMeans to identify potential hotspot locations
        n_clusters = min(5, max(2, len(points_array) // 8))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(points_array)
        
        hotspots = []
        for i, center in enumerate(kmeans.cluster_centers_):
            # Count points near this hotspot
            distances = np.linalg.norm(points_array - center, axis=1)
            nearby_count = np.sum(distances <= data.radius * 2)
            
            priority = "high" if nearby_count > len(points_array) / n_clusters else "medium"
            
            hotspots.append({
                "id": i,
                "center": center.tolist(),
                "potential_coverage": int(nearby_count),
                "priority": priority,
                "suggested_radius": round(data.radius * 1.2, 2)
            })
        
        # Sort by potential coverage
        hotspots.sort(key=lambda x: x['potential_coverage'], reverse=True)
        
        return {
            "hotspots": hotspots,
            "total_hotspots": len(hotspots),
            "message": f"AI identified {len(hotspots)} potential service locations",
            "analysis": f"Based on {len(points_array)} points distributed across {n_clusters} clusters"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI Coverage Prediction
@app.post("/ai/predict-coverage")
async def predict_coverage(data: RequestData):
    try:
        points_array = np.array(data.points)
        
        if len(points_array) < 2:
            return {
                "predicted_coverage": 0,
                "confidence": 0,
                "message": "Need at least 2 points for prediction"
            }
        
        # Extract features
        x_range = np.max(points_array[:, 0]) - np.min(points_array[:, 0])
        y_range = np.max(points_array[:, 1]) - np.min(points_array[:, 1])
        area = (x_range + 1) * (y_range + 1)
        density = len(points_array) / area if area > 0 else 0
        
        # Simple prediction model based on density and radius
        coverage_ratio = min(1.0, (data.radius * data.radius * np.pi * density) * 0.5)
        predicted_coverage = int(len(points_array) * coverage_ratio)
        
        # Ensure prediction is within bounds
        predicted_coverage = max(1, min(len(points_array), predicted_coverage))
        
        # Calculate confidence
        confidence = 0.7 + (0.2 * min(1.0, density))
        if len(points_array) < 20:
            confidence *= 0.8
        
        return {
            "predicted_coverage": predicted_coverage,
            "confidence": round(confidence, 2),
            "total_points": len(points_array),
            "factors": {
                "point_density": round(density, 3),
                "area": round(area, 2),
                "radius_used": data.radius,
                "recommended_radius": round(max(1.0, min(x_range, y_range) / 6), 2)
            },
            "message": f"AI predicts {predicted_coverage} out of {len(points_array)} points can be covered"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI Data Insights
@app.post("/ai/insights")
async def get_insights(data: RequestData):
    try:
        points_array = np.array(data.points)
        
        if len(points_array) < 2:
            return {
                "insights": ["Need more data for insights"],
                "summary": "Insufficient data"
            }
        
        # Calculate statistics
        x_range = np.max(points_array[:, 0]) - np.min(points_array[:, 0])
        y_range = np.max(points_array[:, 1]) - np.min(points_array[:, 1])
        area = (x_range + 1) * (y_range + 1)
        density = len(points_array) / area if area > 0 else 0
        
        insights = []
        
        # Generate insights
        if density > 0.5:
            insights.append("High density area detected - multiple points close together")
        elif density < 0.1:
            insights.append("Sparse distribution - consider larger coverage radius")
        
        if x_range > y_range * 2:
            insights.append("Data is spread horizontally - consider elliptical coverage")
        elif y_range > x_range * 2:
            insights.append("Data is spread vertically - consider vertical optimization")
        
        if len(points_array) < 10:
            insights.append("Small dataset - brute force will be fast and accurate")
        elif len(points_array) > 100:
            insights.append("Large dataset - sliding circle algorithm recommended")
        
        if data.radius < min(x_range, y_range) / 10:
            insights.append("Small radius - will cover local clusters only")
        elif data.radius > max(x_range, y_range) / 2:
            insights.append("Large radius - may cover most points from optimal center")
        
        insights.append(f"Found {len(points_array)} points across {area:.0f} unit area")
        
        return {
            "insights": insights,
            "summary": f"AI analyzed {len(points_array)} points with {density:.3f} density",
            "statistics": {
                "x_range": round(x_range, 2),
                "y_range": round(y_range, 2),
                "density": round(density, 3),
                "total_points": len(points_array)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ MAIN ENDPOINTS ============

# Optimization Endpoint
@app.post("/optimize")
async def optimize(data: RequestData):
    try:
        if len(data.points) < 2:
            raise HTTPException(status_code=400, detail="At least 2 points are required")
        
        if data.eps <= 0 or data.min_samples <= 0 or data.radius <= 0:
            raise HTTPException(status_code=400, detail="All parameters must be positive numbers")
        
        result = run_mccp(data.points, data.eps, data.min_samples, data.radius)
        
        # Learn from this result
        features = ai_optimizer.extract_features(data.points)
        ai_optimizer.learn_from_results(features, {
            'eps': data.eps,
            'min_samples': data.min_samples,
            'radius': data.radius
        }, result['max_count'], len(data.points))
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health Check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "MCCP API with AI is running"}

# Root Endpoint
@app.get("/")
async def root():
    return {
        "name": "MCCP API with AI",
        "version": "3.0.0",
        "description": "Maximum Circular Coverage Problem Solver with Sliding Circle Algorithm and AI Features",
        "ai_features": [
            "Parameter suggestions",
            "Hotspot prediction",
            "Coverage prediction",
            "Data insights"
        ]
    }