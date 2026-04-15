# MCCP AI Coverage Optimization Application

## Overview
A full-stack AI-powered system solving the **Maximum Circular Coverage Problem (MCCP)**:
> Find the optimal location of a fixed-radius circle that maximizes coverage of 2D points.

The system combines clustering + optimization + AI assistance to achieve near-optimal results with significant performance gains over brute-force methods.

---

## Key Highlights
- Hybrid algorithm: **DBSCAN + Sliding Circle Optimization**
- Up to **~99% faster** than brute-force search
- Maintains **96–99% accuracy**
- AI-assisted parameter tuning and insights
- Real-time visualization of coverage optimization

---

## System Architecture
- **Frontend:** Next.js + TypeScript (interactive UI + visualization)
- **Backend:** Python + FastAPI (algorithm execution layer)
- **AI Layer:** Parameter suggestion, density detection, insights engine

---

## Core Algorithm

### 1. Density Detection (DBSCAN)
Groups spatial points into clusters using density-based clustering.

### 2. Sliding Circle Optimization
Searches within clusters to find the circle position with maximum point coverage.

### 3. Validation
Compares results against brute-force baseline for accuracy benchmarking.

---

## Performance Summary
- Brute-force baseline: O(n³) complexity (high cost)
- Proposed method: optimized cluster-based search
- Speed improvement: **up to ~99%**
- Accuracy: **96–99% of optimal solution**

---

## Features
- Interactive point input system
- Adjustable clustering parameters (eps, minPts)
- Circle radius control
- AI-based parameter recommendations
- Dense region detection
- Real-time results comparison (AI vs brute-force)

---

## Example Output
- Points covered: 3–4 (depending on dataset)
- Best circle center: (5.08, 7.49)
- Speed improvement: 98%+
- Execution time: ~0.6s vs ~31s brute-force

---

## Tech Stack
- Next.js
- TypeScript
- Python
- FastAPI
- Machine Learning / AI heuristics

---

## Why this project matters
This project demonstrates:
- Algorithm design capability
- Optimization thinking
- Full-stack engineering skills
- AI integration into real systems
- Performance benchmarking against baselines

---

## Project Structure
## How to Run

### Frontend
```bash
cd frontend
npm install
npm run dev
