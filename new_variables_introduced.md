# Novel Variables Introduced in PI-DDQN (vs Base Paper)

This document formally details the new theoretical and programmatic variables introduced into the architecture by upgrading the legacy Tabular Q-Learning system to a novel **Physics-Informed Double Deep Q-Network (PI-DDQN)**.

Because both algorithms evaluate the identical SG-GAN map and utilize the identical physical properties of the EV (Mass, Drag, Area, etc.), **all variables listed below are strictly architectural upgrades** that prove the intelligence of the new PI-DDQN system.

---

### 1. Physics-Informed Variables (The Novelty)

*   **`λ` (Lambda) / `PIDDQN_PHYS_LAMBDA` (0.5):** 
    *   **Description:** The Physics Regularization Coefficient.
    *   **Purpose:** This is the most crucial new variable. It determines how strictly the Neural Network is penalized for predicting Q-Values that violate the laws of physics. It balances standard Temporal Difference (TD) learning against physical reality.
*   **`L_phys` (Physics Loss):** 
    *   **Description:** The calculated Mean Squared Error (MSE) penalty applied during backpropagation.
    *   **Purpose:** Calculated strictly when the predicted Q-value falls outside the theoretically possible minimum or maximum energy bounds.
*   **`E_min` & `E_max` (Physical Energy Bounds):** 
    *   **Description:** The absolute minimum and maximum theoretical energy bounds for any given edge traversal.
    *   **Purpose:** The Q-learning algorithm simply accepted whatever reward it got. The PI-DDQN dynamically calculates these physical bounds and uses them to anchor the neural network's predictions to reality.

### 2. Deep Reinforcement Learning Variables (The DNN Upgrade)

*   **`Target Network` (θ-):** 
    *   **Description:** A secondary, cloned Neural Network with delayed weight updates.
    *   **Purpose:** Replaces the standard max-Q update. The base paper suffered from "overestimation bias" (thinking a route is better than it actually is). The Target Network variable eliminates this bias, providing stable, long-term learning.
*   **`PIDDQN_BUFFER_CAP` (50,000):** 
    *   **Description:** The Experience Replay Buffer capacity.
    *   **Purpose:** The old tabular method threw away experience immediately after updating. The new algorithm stores up to 50,000 past driving transitions (State, Action, Reward, Next State) to randomly sample from later, vastly improving learning efficiency.
*   **`PIDDQN_BATCH_SIZE` (64):** 
    *   **Description:** The mini-batch sampling size.
    *   **Purpose:** Dictates how many historical driving experiences are pulled from the buffer simultaneously to compute the loss gradient for the neural network updates.
*   **`PIDDQN_LR` (0.001):** 
    *   **Description:** The Adam Optimizer Learning Rate.
    *   **Purpose:** Replaces the tabular learning rate `α`. It dynamically controls the step size of the neural network's gradient descent, automatically decaying over time to ensure convergence.

### 3. State & Constraint Formulation

*   **`Action Masking Filter`:** 
    *   **Description:** A dynamic binary array applied over the output layer.
    *   **Purpose:** The old Q-learning allowed the car to pick a road, drive, and "die" if the battery ran out (receiving a negative reward). The new PI-DDQN logically masks out any neighboring paths where `Battery Required > Current SOC`, forcing the network to natively navigate around impossible constraints.
