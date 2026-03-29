# SG-GAN-Q-Learning: EV Routing & Charging Optimization

This repository contains the simulation code for optimizing Electric Vehicle (EV) routing and charging using a PI-DDQN (Proportional-Integral Double Deep Q-Network) approach combined with an SG-GAN (Spatial Graph Generative Adversarial Network) road network.

## Getting Started

Follow the instructions below to set up your environment and run the code. You can choose to run the project using an isolated Virtual Environment (Recommended), or install dependencies globally on your system.

---

## Method 1: Using a Virtual Environment (Recommended)

Using a virtual environment (`venv`) ensures that this project's dependencies do not clash with other Python projects on your computer.

### 1. Open your terminal or command prompt
Navigate to the root directory of this project:
```bash
cd path/to/SG-GAN-Q-Learning
```

### 2. Create the Virtual Environment
Run the following command to create a virtual environment named `.venv`:
```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment
* **On Windows:**
  ```cmd
  .venv\Scripts\activate
  ```
* **On macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```
*(When activated, you should see `(.venv)` at the beginning of your terminal prompt.)*

### 4. Install Dependencies
Install the required packages. *(If a `requirements.txt` is missing, you may generally need packages like `numpy`, `matplotlib`, `torch` or `tensorflow`, `gym` depending on the models used.)*
```bash
pip install -r requirements.txt
```
*(Alternatively, manually install the specific packages imported in the scripts: `pip install numpy matplotlib torch pyMuPDF`)*

### 5. Run the Files
Execute the main evaluation script:
```bash
python evaluate_pi_dqn.py
```

### 6. Deactivate
When you are done, simply run `deactivate` in your terminal to close the virtual environment.

---

## Method 2: Normal Setup (Without `venv`)

If you prefer to install the packages directly onto your system's global Python environment:

### 1. Open your terminal or command prompt
Navigate to the root directory of this project:
```bash
cd path/to/SG-GAN-Q-Learning
```

### 2. Install Dependencies Globally
```bash
pip install -r requirements.txt
```
*(Or install manually: `pip install numpy matplotlib torch pyMuPDF`)*

### 3. Run the Files
Execute the main evaluation script directly:
```bash
python evaluate_pi_dqn.py
```

---

## Project Structure Overview

* `evaluate_pi_dqn.py`: The main script to evaluate the PI-DDQN agent's performance and generate comparison graphs.
* `network_env.py`: Defines the SG-GAN generated road network environment, nodes, charging stations, and grid mechanics.
* `pi_dqn_routing.py`: Contains the deep reinforcement learning model (PI-DDQN architecture) handling the routing decisions.
* `results/`: Directory where output figures (e.g., convergence graphs, KLD values, energy consumption) are saved.
* `BasePaper/`: Contains older versions and baseline benchmarking code for comparison.
