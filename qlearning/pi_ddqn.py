import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import networkx as nx

from qlearning.simulate import _merged_params

class DDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PIDDQNAgent:
    def __init__(self, n_nodes, device="cpu", extra_params=None):
        self.p = _merged_params(extra_params)
        self.n_nodes = n_nodes
        self.device = device
        
        self.input_dim = n_nodes * 2 + 1  # current node one-hot, dest node one-hot, SOC
        self.output_dim = n_nodes
        
        self.main_net = DDQNNetwork(self.input_dim, self.output_dim).to(device)
        self.target_net = DDQNNetwork(self.input_dim, self.output_dim).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.gamma = self.p["gamma"]
        self.epsilon = self.p["epsilon"]
        self.epsilon_min = self.p["epsilon_min"]
        self.epsilon_decay = self.p["epsilon_decay"]
        
        # PI Target Update Controller
        self.Kp = 0.1
        self.Ki = 0.01
        self.integral_error = {name: torch.zeros_like(param) for name, param in self.main_net.named_parameters()}
        
    def get_state(self, current_node, dest_node, soc):
        s = np.zeros(self.input_dim, dtype=np.float32)
        s[current_node] = 1.0
        s[self.n_nodes + dest_node] = 1.0
        s[-1] = soc / self.p["battery_kWh"]
        return s
        
    def select_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_net(state_t).squeeze(0).cpu().numpy()
        
        valid_q = {a: q_values[a] for a in valid_actions}
        return max(valid_q, key=valid_q.get)
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.main_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        current_q = self.main_net(states).gather(1, actions).squeeze(1)
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network_pi(self):
        """ PI controller for target network weights. """
        with torch.no_grad():
            for name, main_param in self.main_net.named_parameters():
                target_param = dict(self.target_net.named_parameters())[name]
                
                error = main_param.data - target_param.data
                self.integral_error[name] += error
                
                # PI update rule
                update = self.Kp * error + self.Ki * self.integral_error[name]
                target_param.data.copy_(target_param.data + update)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def run_piddqn_scenario(G, dist_mat, time_mat, cs_nodes, congestion_level, use_bipartite=True, n_evs=None, seed=42):
    params = _merged_params()
    if n_evs is None:
        n_evs = params["n_evs"]
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    n_nodes = G.number_of_nodes()
    
    agent = PIDDQNAgent(n_nodes, device)
    
    # Adjacency
    adj_mat = nx.to_numpy_array(G).astype(bool)
    cs_arr = np.array(cs_nodes, dtype=np.int32)
    
    v_ms = params["v_kmh"] / 3.6
    B_max = params["battery_kWh"]
    crit_soc = params["energy_buffer_kWh"] / 10.0 * B_max
    E_norm = B_max
    T_norm = float(time_mat[time_mat > 0].max() * 20) if time_mat.max() > 0 else 1.0
    beta = params["beta"]
    
    t_mat = np.where(dist_mat > 0, dist_mat / (v_ms + 1e-9), 0.0)
    aero = 0.5 * params["rho"] * params["Cd"] * params["A"] * v_ms**2
    roll = params["Cr"] * params["m"] * params["g"] * v_ms
    incl_f = params["m"] * params["g"] * dist_mat * np.sin(params["theta"])
    E_mat = (1.0 / params["eta"]) * ((aero + roll) * t_mat + incl_f) / 3_600_000.0
    
    cs_set = set(cs_nodes)
    non_cs = [n for n in range(n_nodes) if n not in cs_set]
    
    starts = np.random.choice(non_cs, size=n_evs, replace=True)
    dests = np.array([np.random.choice([n for n in non_cs if n != s]) for s in starts])
    
    # We train for 500 epochs (enough for NN with small graph)
    n_epochs = 500
    
    last_routes = []
    last_energy = np.zeros(n_evs)
    last_time = np.zeros(n_evs)
    
    for ep in range(n_epochs):
        socs = np.clip(
            np.random.normal(params["soc_mean_pct"], params["soc_std_pct"], n_evs),
            params["soc_min_pct"], params["soc_max_pct"]
        ) * B_max
        
        q_ij = float(min(np.random.poisson(congestion_level * 10.0), 10.0))
        delta_ij = q_ij / 10.0
        
        ep_energy = np.zeros(n_evs)
        ep_time = np.zeros(n_evs)
        routes = [[int(s)] for s in starts]
        
        for ev in range(n_evs):
            curr_node = int(starts[ev])
            dest_node = int(dests[ev])
            done = False
            
            for step in range(n_nodes * 3):
                if done: break
                
                state_vec = agent.get_state(curr_node, dest_node, socs[ev])
                valid_nbrs = np.where(adj_mat[curr_node])[0]
                
                if len(valid_nbrs) == 0:
                    break
                    
                action = agent.select_action(state_vec, valid_nbrs)
                
                ec = float(E_mat[curr_node, action])
                tc = float(t_mat[curr_node, action])
                
                needs_cs = (socs[ev] - ec) < crit_soc
                if needs_cs and len(cs_arr) > 0:
                    if use_bipartite:
                        cs_node = int(cs_arr[np.argmin(dist_mat[action, cs_arr])])
                    else:
                        cs_node = int(np.random.choice(cs_arr))
                    socs[ev] = max(0.0, socs[ev] - ec)
                    deficit = max(0.0, B_max - socs[ev])
                    tc += (deficit / params["eta_s"]) * 3600.0
                    socs[ev] = B_max
                else:
                    socs[ev] = max(0.0, socs[ev] - ec)
                    
                reached = (action == dest_node)
                cost = beta * (ec / E_norm) + (1.0 - beta) * (tc / T_norm)
                r = params["R_base"] - cost * params["K_obj"] - delta_ij * params["K_congestion"]
                if reached:
                    r += params["K_bonus"]
                
                next_state_vec = agent.get_state(action, dest_node, socs[ev])
                
                agent.store_transition(state_vec, action, r, next_state_vec, reached)
                
                ep_energy[ev] += ec
                ep_time[ev] += tc
                routes[ev].append(action)
                curr_node = action
                
                if reached:
                    done = True
            # Replay once per EV per episode instead of every step to speed up significantly
            agent.replay()
            
        # PI Control for beta: adjust beta to favor energy if average energy > target (14.0 kWh/100km)
        target_energy_kWh_100km = 14.0
        avg_energy = ep_energy.mean()
        avg_dist = ep_time.mean() * v_ms
        current_energy_kWh_100km = (avg_energy / (avg_dist / 1000.0) * 100.0) if avg_dist > 0 else 20.0
        
        # error: positive if energy is too high -> increase beta
        error_e = current_energy_kWh_100km - target_energy_kWh_100km
        beta = np.clip(beta + 0.05 * error_e, 0.1, 0.99)
                    
        agent.update_target_network_pi()
        agent.decay_epsilon()
        
        if ep == n_epochs - 1:
            last_routes = routes
            last_energy = ep_energy.copy()
            last_time = ep_time.copy()
            
    return {
        "final_routes": last_routes,
        "energy_per_ev": last_energy,
        "time_per_ev": last_time,
        "global_cong": congestion_level,
    }
