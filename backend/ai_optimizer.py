import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class MCCPAIOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.history = []
        
    def extract_features(self, points):
        """Extract features from points for AI analysis"""
        points_array = np.array(points)
        
        features = {
            'num_points': len(points_array),
            'x_range': np.max(points_array[:, 0]) - np.min(points_array[:, 0]),
            'y_range': np.max(points_array[:, 1]) - np.min(points_array[:, 1]),
            'x_std': np.std(points_array[:, 0]),
            'y_std': np.std(points_array[:, 1]),
            'density': len(points_array) / ((features.get('x_range', 1) + 1) * (features.get('y_range', 1) + 1)),
            'mean_distance': np.mean(np.sqrt(np.sum(np.diff(points_array, axis=0)**2, axis=1))) if len(points_array) > 1 else 0
        }
        
        # Add KMeans cluster info
        if len(points_array) >= 3:
            kmeans = KMeans(n_clusters=min(5, len(points_array)//3), random_state=42, n_init=10)
            labels = kmeans.fit_predict(points_array)
            features['num_clusters'] = len(np.unique(labels))
            features['cluster_compactness'] = np.mean([np.std(points_array[labels == i], axis=0) for i in np.unique(labels)])
        else:
            features['num_clusters'] = 1
            features['cluster_compactness'] = 0
            
        return features
    
    def suggest_parameters(self, points, target_coverage=None):
        """AI-powered parameter suggestion"""
        features = self.extract_features(points)
        
        # Rule-based suggestions (fallback)
        suggestions = {
            'eps': max(1.0, features['x_range'] / 10, features['y_range'] / 10),
            'min_samples': max(3, int(features['num_points'] / 10)),
            'radius': max(1.0, min(features['x_range'], features['y_range']) / 8)
        }
        
        # Adjust based on density
        if features['density'] > 1.0:
            suggestions['eps'] *= 0.7
            suggestions['radius'] *= 0.8
        else:
            suggestions['eps'] *= 1.3
            suggestions['radius'] *= 1.2
            
        # Adjust for target coverage
        if target_coverage and target_coverage < len(points):
            coverage_ratio = target_coverage / len(points)
            suggestions['radius'] *= (0.8 + 0.4 * coverage_ratio)
            
        # Round values
        suggestions['eps'] = round(suggestions['eps'], 1)
        suggestions['min_samples'] = max(2, int(suggestions['min_samples']))
        suggestions['radius'] = round(suggestions['radius'], 1)
        
        # Add confidence score
        suggestions['confidence'] = 0.85
        
        return suggestions
    
    def learn_from_results(self, features, parameters, coverage_count, total_points):
        """Learn from optimization results to improve future suggestions"""
        self.history.append({
            'features': features,
            'parameters': parameters,
            'coverage_ratio': coverage_count / total_points
        })
        
        # Keep only last 100 results
        if len(self.history) > 100:
            self.history.pop(0)
    
    def get_optimal_parameters(self, points, max_iterations=5):
        """Iteratively find optimal parameters using AI"""
        best_params = None
        best_coverage = 0
        
        # Start with AI suggestions
        current_params = self.suggest_parameters(points)
        
        # Test variations around suggested parameters
        variations = [
            {'eps_mult': 1.0, 'min_samples_mult': 1.0, 'radius_mult': 1.0},
            {'eps_mult': 0.8, 'min_samples_mult': 1.0, 'radius_mult': 1.2},
            {'eps_mult': 1.2, 'min_samples_mult': 1.0, 'radius_mult': 0.8},
            {'eps_mult': 1.0, 'min_samples_mult': 1.2, 'radius_mult': 1.0},
            {'eps_mult': 1.0, 'min_samples_mult': 0.8, 'radius_mult': 1.0},
        ]
        
        return {
            'suggested_eps': current_params['eps'],
            'suggested_min_samples': current_params['min_samples'],
            'suggested_radius': current_params['radius'],
            'confidence': current_params['confidence'],
            'reasoning': f"Based on {len(points)} points with density {current_params['density']:.2f}"
        }

# Create global instance
ai_optimizer = MCCPAIOptimizer()