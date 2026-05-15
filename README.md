# Swarm AI Maze Solver: Evolutionary Robotics with ROS2 & Gazebo

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble/Foxy-orange.svg)](https://docs.ros.org/en/humble/index.html)
[![Gazebo](https://img.shields.io/badge/Simulator-Gazebo-orange.svg)](https://gazebosim.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced autonomous robotics framework that trains a swarm of robots to solve procedurally generated 5x5 mazes using **Neuroevolution (Genetic Algorithms + RNN)** within a high-fidelity **Gazebo** simulation.

![Current Maze Preview](current_maze_preview.png)
*(Note: Please ensure this image is uploaded to your repo)*

## 🚀 Overview

This project implements a complete **Curriculum Learning** pipeline:
1. **Specialist Training:** Individual robots are trained on specific maze layouts until they "graduate."
2. **Generalist Evolution:** Successful "Genius" brains are used as seeds for a multi-maze environment where the AI evolves to solve *any* unseen maze layout through generalized spatial logic.

## 🧠 Technical Architecture

The system is structured into 4 distinct layers:
- **Layer 4: Evolution:** Manages populations (30-96 agents), fitness evaluation, crossover, and Gaussian mutation.
- **Layer 3: Intelligence (RNN):** A custom Recurrent Neural Network (~1450 parameters) processing 26 inputs (Lidar + Compass) to drive wheel velocities.
- **Layer 2: Game Logic:** Handles sensor fusion (48 to 24 Lidar rays), BFS-based navigation waypoints, and death/goal detection.
- **Layer 1: Physics:** Gazebo simulator integrated with ROS2 for high-fidelity physics and sensor simulation.

## 🛠 Tech Stack

- **Framework:** ROS2 (Robot Operating System)
- **Simulation:** Gazebo (Ignition/Harmonic)
- **AI/ML:** Genetic Algorithms, RNN (NumPy-based for maximum performance)
- **Algorithms:** DFS (Maze Generation), BFS (Pathfinding & Fitness Scoring)
- **Language:** Python 3.10+

## 📈 Key Features

- **Procedural Content Generation:** Automatically generates "perfect" 5x5 mazes using DFS, ensuring a single path to the goal.
- **BFS-Driven Fitness:** A robust fitness function that prevents "cheating" by rewarding progress based on actual path distance to the goal.
- **Auto-Recovery & Curriculum:** The system automatically saves "champions," handles simulator crashes, and progresses to more difficult maps autonomously.
- **Parallel Evaluation:** Evaluates up to 16 robots simultaneously in Gazebo to accelerate training.

## 📂 Project Structure

- `src/core/ga_model.py`: Neural network architecture (RNN).
- `src/agent/`: Core robot logic and sensor processing.
- `train_ga.py`: Main training script for single-map specialization.
- `train_multi_ga.py`: Training script for general intelligence.
- `auto_curriculum.py`: Orchestrator for the learning pipeline.

## 🚦 Getting Started

### Prerequisites
- ROS2 Humble/Foxy
- Gazebo Sim
- Python dependencies: `numpy`, `matplotlib`

### Running the Project
1. **Launch Training:**
   ```bash
   python3 auto_curriculum.py
   ```
2. **Visualize Results:**
   ```bash
   python3 plot_fitness.py
   ```

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
**Author:** Nguyễn Gia Bảo  
**Portfolio:** [Your Portfolio Link Here]
