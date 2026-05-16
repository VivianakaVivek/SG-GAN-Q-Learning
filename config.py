"""
Central Configuration File for SG-GAN EV Routing
Modify variables here to apply changes across BOTH base_paper and my_research.
"""

# ==========================================
# 1. MAP & ENVIRONMENT SETTINGS
# ==========================================
MAP_LAT = 28.6193       # Dwarka Mod Latitude
MAP_LON = 77.0334       # Dwarka Mod Longitude
MAP_DIST = 800          # Radius in meters
DEFAULT_NODES = 29      # Number of junctions in default map
DEFAULT_CS = 7          # Number of charging stations in default map
CONNECTION_THRESH_A = 0.25 # Sparsity for Graph A (Wider)
CONNECTION_THRESH_B = 0.35 # Sparsity for Graph B (Dense)

# ==========================================
# 2. EV PHYSICAL PARAMETERS (Table I)
# ==========================================
EV_PARAMS = {
    'eta': 0.85,        # Drivetrain efficiency
    'rho': 1.225,       # Air density (kg/m^3)
    'Cd': 0.28,         # Drag coefficient
    'A': 2.3,           # Frontal area (m^2)
    'Cr': 0.01,         # Rolling resistance
    'm': 1500,          # Vehicle mass (kg)
    'g': 9.81,          # Gravity (m/s^2)
    'battery': 10.0,    # Total battery capacity (kWh)
    'soc_thresh': 0.25, # Critical SOC threshold
    'buffer': 2.5       # Emergency buffer (kWh)
}

# ==========================================
# 3. SG-GAN HYPERPARAMETERS
# ==========================================
GAN_EPOCHS = 401
GAN_NOISE_DIM = 10
GAN_LR = 0.0002
GAN_BETA1 = 0.5
GAN_BETA2 = 0.999
GAN_BATCH_SIZE = 32
GAN_GRAD_CLIP = 5.0

# ==========================================
# 4. ROUTING HYPERPARAMETERS (Q-Learning & PI-DDQN)
# ==========================================
NUM_EVS_DEFAULT = 50
BETA_WEIGHT = 0.8       # Weight for Energy vs Time in Reward (Eq. 12)
GAMMA = 0.95            # Discount factor

# Tabular Q-Learning Specific
QL_EPOCHS = 10000
QL_ALPHA = 0.1
QL_EPS_START = 1.0
QL_EPS_MIN = 0.05
QL_EPS_DECAY = 0.995

# PI-DDQN Specific
PIDDQN_EPOCHS = 800
PIDDQN_LR = 0.0005
PIDDQN_BUFFER_CAP = 50000
PIDDQN_BATCH_SIZE = 64
PIDDQN_PHYS_LAMBDA = 0.25 # Weight of physics regularization loss

# ==========================================
# 5. SCALABILITY TESTING (Fig 5)
# ==========================================
SCALE_EVS = [50, 100, 150]
SCALE_NODES = [29, 50, 100]
SCALE_TRAIN_EPOCHS_QL = 300
SCALE_TRAIN_EPOCHS_PIDDQN = 100
