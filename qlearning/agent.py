"""
qlearning/agent.py
==================
Single EV Q-learning agent (Algorithm 2 from paper).

Each agent:
  - Maintains its own Q-table Q_i(s, a)
  - Uses ε-greedy policy (Eq. 15-16)
  - Updates via Bellman equation (Eq. 13)
  - Tracks SOC, visited nodes, route taken
"""

import numpy as np
import networkx as nx


class EVAgent:
    """
    Decentralised Q-learning agent representing one electric vehicle.

    Parameters
    ----------
    agent_id   : int
    n_nodes    : int   — state space size
    start_node : int
    dest_node  : int
    soc_kWh    : float — initial state of charge
    params     : dict  — merged config params
    seed       : int
    """

    def __init__(self, agent_id, n_nodes, start_node, dest_node, soc_kWh, params, seed):
        self.id        = agent_id
        self.n         = n_nodes
        self.start     = start_node
        self.dest      = dest_node
        self.soc_kWh   = soc_kWh
        self.p         = params
        self.rng       = np.random.default_rng(seed + agent_id * 100)

        # Q-table Q_i(s, a)  — initialised to zero (Algorithm 2 line 3)
        self.Q = np.zeros((n_nodes, n_nodes), dtype=np.float64)

        # Exploration
        self.epsilon = params["epsilon"]

        # Current state
        self.state      = start_node
        self.done       = False
        self.at_cs      = False

        # Trajectory tracking
        self.route          = [start_node]
        self.total_energy   = 0.0       # kWh
        self.total_time     = 0.0       # seconds
        self.visited_nodes  = {start_node}

    # ── ε-greedy action selection (Eq. 15-16) ─────────────────────────────────

    def select_action(self, neighbours: list) -> int:
        """
        From current state s, choose next node a ∈ neighbours.
          π_t(a) = 1 - ε + ε/|A|   if Q_t(a) = max
                   ε/|A|            otherwise
        """
        if not neighbours:
            return self.state   # stuck — stay

        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(neighbours))     # explore
        else:
            q_vals = self.Q[self.state, neighbours]
            best   = int(neighbours[np.argmax(q_vals)]) # exploit
            return best

    # ── Q-update (Eq. 13) ─────────────────────────────────────────────────────

    def update_q(self, s: int, a: int, reward: float, s_next: int):
        """
        Q_i(s,a) ← Q_i(s,a) + α[R_i + γ max_a' Q_i(s',a') - Q_i(s,a)]
        """
        alpha = self.p["alpha"]
        gamma = self.p["gamma"]
        td_target = reward + gamma * np.max(self.Q[s_next])
        self.Q[s, a] += alpha * (td_target - self.Q[s, a])

    # ── Epsilon decay ─────────────────────────────────────────────────────────

    def decay_epsilon(self):
        self.epsilon = max(self.p["epsilon_min"],
                           self.epsilon * self.p["epsilon_decay"])

    # ── Step (one move in the environment) ────────────────────────────────────

    def step(self, action: int, env, use_bipartite: bool = True):
        """
        Execute action (move to node `action`), collect reward, update Q.
        Returns: reward (float)
        """
        s      = self.state
        a      = action
        s_next = action

        # ── Physics ──────────────────────────────────────────────────────────
        dist_m  = env.dist_mat[s, a]
        ec_kWh  = env.energy_kWh(dist_m)
        delta_ij = env.local_congestion(s, a)

        # Determine if visiting a CS
        cs_visit = None
        critical = env.soc_critical_kWh()
        need_cs  = (self.soc_kWh - ec_kWh) < critical

        if need_cs and use_bipartite:
            # Bipartite mapping to nearest available CS
            mapping = env.assign_cs_bipartite([s_next])
            cs_visit = mapping.get(s_next)
        elif need_cs and not use_bipartite:
            # Random CS assignment (no bipartite — Table III scenario)
            if env.cs_nodes:
                cs_visit = int(env.rng.choice(env.cs_nodes))

        # Update SOC — drain en route, refill at CS
        self.soc_kWh = max(0.0, self.soc_kWh - ec_kWh)
        if cs_visit is not None:
            env.cs_queue[cs_visit] = env.cs_queue.get(cs_visit, 0) + 1
            tc_s = env.total_travel_time_s(dist_m, self.soc_kWh, cs_visit)
            self.soc_kWh = env.p["battery_kWh"]           # fully charged
            env.cs_queue[cs_visit] = max(0, env.cs_queue[cs_visit] - 1)
        else:
            tc_s = env.total_travel_time_s(dist_m, self.soc_kWh, None)

        # Normalisation denominators
        E_norm = env.p["battery_kWh"]
        T_norm = max(env.time_mat[env.time_mat > 0].max() * 20, 1.0)

        # Reached destination?
        reached = (s_next == self.dest)

        # Reward (Eq. 12)
        r = env.reward(ec_kWh, tc_s, delta_ij, reached, E_norm, T_norm)

        # Q-update (Eq. 13)
        self.update_q(s, a, r, s_next)

        # State transition
        self.state = s_next
        self.route.append(s_next)
        self.visited_nodes.add(s_next)
        self.total_energy += ec_kWh
        self.total_time   += tc_s

        if reached:
            self.done = True

        return r

    def reset(self, soc_kWh: float = None):
        """Reset agent to start for a new episode (keeps Q-table)."""
        self.state       = self.start
        self.done        = False
        self.route       = [self.start]
        self.visited_nodes = {self.start}
        self.total_energy  = 0.0
        self.total_time    = 0.0
        self.soc_kWh = soc_kWh if soc_kWh is not None else self.p["battery_kWh"] * 0.65
