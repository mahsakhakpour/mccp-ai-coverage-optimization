# MCCP AI Coverage Optimization Application

## Overview
The **Maximum Circular Coverage Problem (MCCP)** project implements a hybrid algorithm to find the optimal placement of a fixed-radius circle that maximizes coverage of 2D points.

The core approach combines:
- **DBSCAN clustering** to identify dense regions
- **Sliding Circle optimization** to refine the best coverage position

This hybrid method significantly reduces computation time while maintaining high accuracy compared to brute-force search.

---

## Key Features
- Hybrid MCCP algorithm (DBSCAN + Sliding Circle)
- Fast approximation with high accuracy (96–99%)
- Brute-force comparison for validation
- Full-stack implementation with AI integration
- Real-time parameter tuning and visualization
- AI-driven insights and parameter recommendations

---

## Tech Stack

### Frontend
- Next.js
- TypeScript

### Backend
- Python
- FastAPI

### AI Layer
- Parameter recommendation engine
- Dense region detection
- Coverage prediction & insights

---

## Algorithm

### 1. DBSCAN Clustering
Groups nearby points into dense clusters based on:
- `eps` (distance threshold)
- `minPts` (minimum points per cluster)

### 2. Sliding Circle Optimization
For each cluster:
- A fixed-radius circle is slid across candidate positions
- The position maximizing point coverage is selected

### 3. Validation
Results are compared against brute-force search:
- Coverage accuracy: 96–99%
- Speed improvement: up to ~99%

---

## Performance
- Up to ~99% faster than brute-force methods
- Near-optimal coverage results
- Scales efficiently with larger datasets

---

## How to Run

### Frontend
```bash
cd frontend
npm install
npm run dev
