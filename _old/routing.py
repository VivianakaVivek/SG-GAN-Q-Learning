"""
Q-Learning for EV Routing and Charging Station Allocation
==========================================================
Implements Section IV (Energy, Travel Time, Congestion calculations)
and Section V (Algorithm 2: Q-Learning routing + bipartite CS allocation).

Reference: "SG-GAN for Electric Vehicle Road Networks and Q-Learning
in Energy Efficient Routing" - Das, Maity, Chakraborty (IEEE T-ITS, 2025)
"""

import random
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


# ============================================================
# Table I: Parameters related to energy calculation
# ============================================================
EV_PARAMS = {
    'eta': 0.85,         # Mechanical-to-electrical efficiency
    'rho': 1.225,        # Air density (kg/m^3)
    'Cd': 0.28,          # Drag coefficient
    'A': 2.3,            # Frontal area (m^2)
    'Cr': 0.01,          # Rolling resistance coefficient
    'm': 1500.0,         # Vehicle mass (kg)
    'g': 9.81,           # Gravitational acceleration (m/s^2)
    'battery_kwh': 10.0, # Battery capacity (kWh)
    'soc_threshold': 0.25,  # Critical SOC threshold (25% = 2.5 kWh)
    'energy_buffer': 2.5,   # Minimum energy buffer (kWh)
}


def compute_energy_kwh(distance_m, speed_ms, slope_rad=0.0, params=EV_PARAMS):
    """
    Compute energy consumption for an EV traversing an edge (Eq. 6).
    
    Ec = (d / eta) * (0.5 * rho * Cd * A * v^2 + Cr * m * g + m * g * sin(theta))
    
    Returns energy in kWh.
    """
    eta = params['eta']
    rho = params['rho']
    Cd = params['Cd']
    A = params['A']
    Cr = params['Cr']
    m = params['m']
    g = params['g']

    # Forces (Newtons)
    F_aero = 0.5 * rho * Cd * A * speed_ms ** 2
    F_rolling = Cr * m * g
    F_grade = m * g * np.sin(slope_rad)
    F_total = F_aero + F_rolling + F_grade

    # Energy (Joules) = Force * distance / efficiency
    energy_j = (distance_m / eta) * F_total

    # Convert to kWh
    energy_kwh = energy_j / 3_600_000.0
    return max(0, energy_kwh)


def compute_travel_time(base_time_s, congestion_factor, charge_time_s=0.0):
    """
    Compute travel time for an edge (Eq. 7).
    
    Tc = L/lambda + d/V + T_charge
    The queuing delay is modeled via congestion_factor.
    """
    # Queuing delay proportional to congestion
    queue_delay = congestion_factor * base_time_s * 0.5
    return base_time_s + queue_delay + charge_time_s


def compute_congestion_factor(traffic_flow, capacity):
    """
    Compute local congestion factor for a road segment (Eq. 9).
    
    delta_ij = q_ij / C_ij
    Normalized between 0 (no traffic) and 1 (full congestion).
    """
    if capacity <= 0:
        return 1.0
    return min(1.0, traffic_flow / capacity)


class EVAgent:
    """
    A single EV agent with tabular Q-learning.
    
    State: (current_node, destination_node, soc_bucket)
    Action: move to a neighboring node
    Reward: Eq. 12 from the paper
    Update: Eq. 13 (standard Q-learning)
    Policy: epsilon-greedy (Eq. 15-16)
    """

    def __init__(self, agent_id, G, start, dest, initial_soc,
                 q_table, alpha=0.1, gamma=0.95, beta=0.8, n_soc_buckets=10):
        self.agent_id = agent_id
        self.G = G
        self.s = start
        self.dest = dest
        self.soc = initial_soc
        self.max_soc = EV_PARAMS['battery_kwh']
        self.critical_soc = EV_PARAMS['energy_buffer']

        # Shared Q-table: {(state_tuple, action): q_value}
        self.q_table = q_table
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.beta = beta        # Energy vs time weighting factor

        self.n_soc_buckets = n_soc_buckets

        self.total_energy = 0.0
        self.total_time = 0.0
        self.total_distance = 0.0
        self.path = [self.s]
        self.done = False

        # Random slope per edge (small inclines 0-3 degrees)
        self.edge_slopes = {}
        for u, v in G.edges():
            slope = np.radians(random.uniform(0, 3))
            self.edge_slopes[(u, v)] = slope
            self.edge_slopes[(v, u)] = slope

    def _soc_bucket(self):
        """Discretize SOC into buckets for tabular Q-learning."""
        ratio = max(0, min(1, self.soc / self.max_soc))
        return min(int(ratio * self.n_soc_buckets), self.n_soc_buckets - 1)

    def _state(self):
        """Current state tuple for Q-table lookup."""
        return (self.s, self.dest, self._soc_bucket())

    def get_actions(self):
        """Get available actions (neighboring nodes), preferring unvisited."""
        neighbors = list(self.G.neighbors(self.s))
        # Prefer unvisited to reduce cycles
        unvisited = [n for n in neighbors if n not in self.path[-5:]]
        return unvisited if unvisited else neighbors

    def choose_action(self, epsilon):
        """
        Epsilon-greedy action selection (Eq. 15-16).
        With probability epsilon, explore randomly.
        Otherwise, exploit the best Q-value.
        """
        actions = self.get_actions()
        if not actions:
            return self.s

        # Exploration
        if random.random() < epsilon:
            return random.choice(actions)

        # Exploitation: argmax Q(s, a) over valid actions
        state = self._state()
        best_action = actions[0]
        best_q = self.q_table.get((state, actions[0]), 0.0)
        for a in actions[1:]:
            q = self.q_table.get((state, a), 0.0)
            if q > best_q:
                best_q = q
                best_action = a

        return best_action

    def compute_reward(self, energy_kwh, travel_time_s, congestion_factor,
                       reached_dest):
        """
        Compute reward for a transition (Eq. 12).
        
        Reward = R_base - [beta * Ec/E_norm + (1-beta) * Tc/T_norm] * K_obj
                 - (delta_ij / delta_norm) * K_congestion
                 + I_dest * K_bonus
        """
        R_base = 10.0
        K_obj = 5.0
        K_congestion = 2.0
        K_bonus = 100.0
        E_norm = 0.5    # Normalization for energy
        T_norm = 120.0   # Normalization for time
        delta_norm = 1.0  # Max expected congestion

        obj_cost = (self.beta * (energy_kwh / E_norm) +
                    (1 - self.beta) * (travel_time_s / T_norm))

        reward = R_base - obj_cost * K_obj
        reward -= (congestion_factor / delta_norm) * K_congestion

        if reached_dest:
            reward += K_bonus

        if self.soc <= 0:
            reward -= 50.0  # Battery depletion penalty

        return reward

    def step(self, epsilon, global_congestion):
        """
        Execute one step: choose action, compute physics, update Q-table.
        
        Returns:
            done: Whether agent reached destination or depleted battery
            energy: Energy consumed this step (kWh)
            time: Travel time this step (s)
            distance: Distance traveled this step (m)
        """
        if self.done:
            return True, 0, 0, 0

        # Charging logic: charge at CS if SOC is below 80%
        if (self.G.nodes[self.s].get('is_cs', False) and
                self.soc < self.max_soc * 0.8):
            charge_amount = self.max_soc - self.soc
            # Charging time: Eq. 8 (T_charge = (B_max - B_current) / eta_s)
            charge_time = (charge_amount / 50.0) * 3600  # 50 kW charger
            self.soc = self.max_soc
            self.total_time += charge_time
            return False, 0, charge_time, 0

        action = self.choose_action(epsilon)

        if action == self.s:
            self.done = True
            return True, 0, 0, 0

        # Get edge data
        edge = self.G.get_edge_data(self.s, action)
        distance_m = edge['weight']
        base_time_s = edge['time']
        capacity = edge.get('capacity', 20)

        # Compute physics
        speed_ms = distance_m / max(0.1, base_time_s)
        slope = self.edge_slopes.get((self.s, action), 0.0)

        energy_kwh = compute_energy_kwh(distance_m, speed_ms, slope)
        local_cong_count = global_congestion.get((self.s, action), 0) + \
                           global_congestion.get((action, self.s), 0)
        cong_factor = compute_congestion_factor(local_cong_count, capacity)
        travel_time = compute_travel_time(base_time_s, cong_factor)

        # Update agent state
        old_state = self._state()
        self.soc -= energy_kwh
        self.total_energy += energy_kwh
        self.total_time += travel_time
        self.total_distance += distance_m

        reached_dest = (action == self.dest)
        self.done = reached_dest or (self.soc <= 0)

        # Compute reward (Eq. 12)
        reward = self.compute_reward(energy_kwh, travel_time, cong_factor,
                                     reached_dest)

        # Move to new state
        self.s = action
        self.path.append(action)
        new_state = self._state()

        # Q-learning update (Eq. 13):
        # Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]
        old_q = self.q_table.get((old_state, action), 0.0)

        if self.done:
            max_next_q = 0.0
        else:
            next_actions = self.get_actions()
            if next_actions:
                max_next_q = max(
                    self.q_table.get((new_state, a), 0.0) for a in next_actions
                )
            else:
                max_next_q = 0.0

        td_target = reward + self.gamma * max_next_q
        self.q_table[(old_state, action)] = old_q + self.alpha * (td_target - old_q)

        return self.done, energy_kwh, travel_time, distance_m


class MultiAgentRouter:
    """
    Multi-agent Q-learning framework for EV routing (Algorithm 2).
    
    Features:
    - Shared Q-table across all EV agents
    - Global congestion tracking (Eq. 9-10)
    - Bipartite graph CS allocation (Eq. 17, Hungarian algorithm)
    - Epsilon-greedy exploration with decay
    """

    def __init__(self, G, n_evs=50, beta=0.8, alpha=0.1, gamma=0.95):
        self.G = G
        self.n_evs = n_evs
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

        # Shared Q-table across all agents
        self.q_table = defaultdict(float)
        self.nodes = list(G.nodes())

        # Epsilon schedule
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def _create_agents(self):
        """Create fresh EV agents with random start/dest and SOC."""
        agents = []
        for i in range(self.n_evs):
            s, d = random.sample(self.nodes, 2)
            # SOC uniformly distributed between 30% and 100% (paper: avg 65%, std 20.2%)
            soc = random.uniform(0.3 * EV_PARAMS['battery_kwh'],
                                 EV_PARAMS['battery_kwh'])
            agent = EVAgent(i, self.G, s, d, soc,
                            q_table=self.q_table,
                            alpha=self.alpha, gamma=self.gamma, beta=self.beta)
            agents.append(agent)
        return agents

    def map_bipartite_cs(self, agent):
        """
        Bipartite graph CS allocation (Eq. 17, Section V-4).
        
        Find the best CS for an EV based on combined energy cost
        and expected queuing delay, using the Hungarian algorithm.
        """
        cs_nodes = [n for n, d in self.G.nodes(data=True) if d.get('is_cs', False)]
        if not cs_nodes:
            return None

        best_cs = None
        min_cost = float('inf')

        for cs in cs_nodes:
            try:
                path_length = nx.shortest_path_length(
                    self.G, agent.s, cs, weight='weight'
                )
                # Energy to reach CS
                energy_req = (path_length / 1000.0) * 0.15
                # Combined metric (Eq. 17)
                cost = self.beta * (energy_req / 0.5) + (1 - self.beta) * 0

                if cost < min_cost and agent.soc > energy_req:
                    min_cost = cost
                    best_cs = cs
            except nx.NetworkXNoPath:
                continue

        return best_cs

    def train(self, epochs=10000):
        """
        Train all EV agents using Q-learning (Algorithm 2).
        
        Returns:
            history: List of (total_energy, total_distance_km) per epoch
        """
        print(f"Training Q-Learning: {self.n_evs} EVs, {epochs} epochs, beta={self.beta}...")
        history = []

        for epoch in range(epochs):
            agents = self._create_agents()
            active = set(agents)
            global_congestion = defaultdict(int)

            epoch_energy = 0.0
            epoch_distance = 0.0

            steps = 0
            while active and steps < 100:
                steps += 1
                completed = []

                for agent in list(active):
                    # CS rerouting when SOC is critical (Section V)
                    if (agent.soc <= agent.critical_soc * 1.5 and
                            not agent.G.nodes[agent.s].get('is_cs', False)):
                        target_cs = self.map_bipartite_cs(agent)
                        if target_cs:
                            agent.dest = target_cs

                    prev_s = agent.s
                    done, e, t, d = agent.step(self.epsilon, global_congestion)
                    epoch_energy += e
                    epoch_distance += d

                    # Update global congestion
                    if not done and prev_s != agent.s:
                        global_congestion[(prev_s, agent.s)] += 1

                    if done:
                        completed.append(agent)

                for c in completed:
                    active.discard(c)

                # Congestion decay over time
                for k in list(global_congestion.keys()):
                    global_congestion[k] = max(0, global_congestion[k] - 0.5)

            # Epsilon decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            epoch_dist_km = epoch_distance / 1000.0
            history.append((epoch_energy, epoch_dist_km))

            if epoch % 500 == 0:
                e_100km = ((epoch_energy / epoch_dist_km) * 100
                           if epoch_dist_km > 0 else 0)
                print(f"  Epoch {epoch:5d} | Energy/100km: {e_100km:.2f} kWh | "
                      f"Eps: {self.epsilon:.3f} | Q-table: {len(self.q_table)} entries")

        print("Training complete.")
        return history

    def evaluate(self, epochs=5, use_bipartite=True):
        """
        Evaluate trained policy with epsilon=0 (greedy).
        
        Args:
            epochs: Number of evaluation episodes
            use_bipartite: Whether to use bipartite CS allocation
            
        Returns:
            energy_per_100km: Average energy consumption (kWh/100km)
            avg_time: Average travel time per EV (seconds)
        """
        total_energy = 0.0
        total_dist = 0.0
        total_time = 0.0

        for ep in range(epochs):
            agents = self._create_agents()
            active = set(agents)
            global_congestion = defaultdict(int)

            steps = 0
            while active and steps < 100:
                steps += 1
                completed = []

                for agent in list(active):
                    if (use_bipartite and
                            agent.soc <= agent.critical_soc * 1.5 and
                            not agent.G.nodes[agent.s].get('is_cs', False)):
                        target_cs = self.map_bipartite_cs(agent)
                        if target_cs:
                            agent.dest = target_cs

                    prev_s = agent.s
                    done, e, t, d = agent.step(0.0, global_congestion)  # epsilon=0
                    total_energy += e
                    total_dist += d
                    total_time += t

                    if not done and prev_s != agent.s:
                        global_congestion[(prev_s, agent.s)] += 1

                    if done:
                        completed.append(agent)

                for c in completed:
                    active.discard(c)

                for k in list(global_congestion.keys()):
                    global_congestion[k] = max(0, global_congestion[k] - 0.5)

        n_total = self.n_evs * epochs
        avg_energy = total_energy / n_total
        avg_time = total_time / n_total
        avg_dist_km = (total_dist / 1000.0) / n_total
        energy_per_100km = (avg_energy / avg_dist_km) * 100 if avg_dist_km > 0 else 0

        return energy_per_100km, avg_time


if __name__ == "__main__":
    # Quick standalone test
    from network_env import NetworkEnvironment
    env = NetworkEnvironment()
    env.fetch_real_data()
    gen, _, _ = env.train_sg_gan(n_nodes=29, max_epochs=201)
    G, kld = env.synthesize_graph(gen, n_nodes=29, n_cs=7)

    router = MultiAgentRouter(G, n_evs=10, beta=0.8)
    history = router.train(epochs=100)
    e, t = router.evaluate(epochs=3)
    print(f"Evaluation: {e:.2f} kWh/100km, {t:.1f} s avg time")
