"""
qlearning/config.py
===================
All simulation parameters from the paper (Tables I and II).
No hardcoding anywhere else — import from here.
"""

# ── Table I: Physical / energy parameters ────────────────────────────────────
PHYSICS = {
    "eta":   0.90,        # mechanical-to-electrical efficiency η
    "rho":   1.225,       # air density  ρ  (kg/m³)
    "Cd":    0.30,        # drag coefficient
    "A":     2.20,        # frontal area (m²)
    "Cr":    0.015,       # rolling resistance coefficient
    "m":     1500.0,      # vehicle mass (kg)
    "g":     9.81,        # gravity (m/s²)
    "theta": 0.0,         # road incline angle (rad); 0 = flat
}

# ── EV fleet parameters ───────────────────────────────────────────────────────
EV = {
    "n_evs":         5,       # EVs used for Table III/IV/V
    "n_evs_total":   50,      # total EVs in network (paper mentions 50)
    "battery_kWh":   10.0,    # uniform 10 kWh battery (paper)
    "soc_min_pct":   0.30,    # SOC low: 30%
    "soc_max_pct":   1.00,    # SOC high: 100%
    "soc_mean_pct":  0.65,    # mean 65%
    "soc_std_pct":   0.202,   # std 20.2%
    # Critical SOC thresholds (proportional to 2.5 kWh buffer / battery_kWh)
    # 10kWh→25%, 15kWh→16.7%, 20kWh→12.5%, 30kWh→8.3%
    "energy_buffer_kWh": 2.5,
    "v_kmh":         40.0,    # cruising speed (km/h)  — from OSM default
}

# ── Charging station parameters ───────────────────────────────────────────────
CS = {
    "n_cs":          7,       # 7 CS nodes in network
    "charger_cap":   3,       # C_j: max simultaneous chargers per CS
    "eta_s":         11.0,    # charging rate η_s (kW)  — Level-2 AC
    "lambda_arr":    5.0,     # mean EV arrival rate λ (EVs/hour) — Poisson
}

# ── Table II: Q-learning hyperparameters ─────────────────────────────────────
QLEARN = {
    "alpha":         0.15,    # learning rate α
    "gamma":         0.90,    # discount factor γ
    "epsilon":       1.0,     # initial exploration rate ε
    "epsilon_min":   0.01,
    "epsilon_decay": 0.995,   # multiplicative decay per episode
    "n_epochs":      10000,   # training episodes (paper: ~10k epochs)
    "beta":          0.8,     # weighting factor β (energy vs time); paper uses 0.8
    # Reward shaping constants (Eq. 12)
    "R_base":        1000.0,
    "K_obj":         500.0,
    "K_congestion":  200.0,
    "K_bonus":       300.0,
}

# ── Congestion levels tested in Tables III / IV / V ──────────────────────────
CONGESTION_LEVELS = [0.25, 0.50, 1.00]   # 25%, 50%, 100%

# ── Network topology ──────────────────────────────────────────────────────────
NETWORK = {
    "n_junctions": 22,
    "n_cs":         7,
    "n_nodes":     29,
}

# ── Output directories ────────────────────────────────────────────────────────
OUTPUT = {
    "root":   "qlearning/outputs",
    "plots":  "qlearning/outputs/plots",
    "tables": "qlearning/outputs/tables",
    "data":   "qlearning/outputs/data",
}
