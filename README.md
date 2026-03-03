# OrbitalCompute-AI: On-Board Satellite Graph Neural Networks for Real-Time Disaster Infrastructure Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/Status-Research_Prototype-success)

**OrbitalCompute-AI** (also known as Sat-XKG) is a decentralized, edge-native framework designed to bypass the terrestrial communication bottlenecks typical of large-scale natural disasters. By shifting Graph Neural Network (GNN) inference directly to "Edge" hardware (such as LEO satellites or high-altitude UAVs), this system enables real-time resource routing and infrastructure assessment even when ground networks are completely destroyed.



## 🛰️ The Core Problem: The Terrestrial Bottleneck
During severe disaster events (e.g., the Wayanad landslides or Odisha cyclones), traditional disaster-response AI models fail because they rely on transmitting massive amounts of high-resolution satellite imagery (GeoTIFFs) to centralized cloud servers. This "Terrestrial Bottleneck" creates critical delays—often hours or days—in identifying survival routes and prioritizing cut-off hospitals.

## 🧠 Our Solution: "AI for Satellite" Edge Fusion
OrbitalCompute-AI solves this latency crisis through a three-stage on-board intelligence pipeline:

1. **Multi-Modal Data Ingestion:** The framework dynamically ingests real-world road networks and hospital locations using `OSMnx`, structuring them into a Spatial Knowledge Graph.
2. **Raster-to-Vector Fusion:** Using a custom spatial intersection engine (`rasterio`), the AI maps real-time Sentinel-1 SAR (Synthetic Aperture Radar) flood pixels directly onto road "edges," dynamically severing blocked routes in the mathematical graph.
3. **On-Board GNN Inference:** A Graph Convolutional Network (GCN), built with PyTorch Geometric, runs directly on the edge hardware. It predicts "criticality" scores for infrastructure nodes, identifying high-priority assets that are physically intact but operationally constrained by surrounding flood damage.

## ⚡ Performance Metrics & Hardware Efficiency
Simulations against traditional cloud-based processing architectures yield the following telemetry optimizations:
* **Latency:** End-to-end processing reduced from **~53 hours** (cloud-upload estimate for massive payloads) to **~3.1 milliseconds** (local edge inference).
* **Bandwidth:** **99.9% reduction** in data transmission by sending a lightweight GNN deployment manifest rather than raw satellite imagery.
* **Energy Optimization:** **~98% battery saving** for edge hardware SoC by minimizing high-power, long-range radio telemetry.

## 🛠️ Technology Stack
* **AI & Machine Learning:** PyTorch, PyTorch Geometric (GNN)
* **Geospatial Processing:** NetworkX, OSMnx, Rasterio, Shapely
* **Simulation & Optimization:** Custom Dijkstra-based deployment manifest generator
* **Frontend:** Streamlit, Matplotlib (Dynamic Graph Visualization)

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment.

### Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Sat-XKG.git](https://github.com/YOUR_USERNAME/Sat-XKG.git)
   cd Sat-XKG
