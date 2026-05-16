"""
Physics-Informed Double DQN (PI-DDQN) for EV Routing + CS Allocation
=====================================================================
Novel extension: replaces tabular Q-learning with PI-DDQN adding:
1. Double DQN for stable Q-value estimation
2. Physics-informed loss (penalizes Q > physical energy bounds)
3. Action masking (blocks physically impossible moves)
4. Potential-based reward shaping via shortest-path heuristic
Base constraints from paper: Eq.6 (energy), Eq.7 (time), Eq.9-10 (congestion), Eq.12 (reward)
"""
import sys, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from collections import defaultdict, deque
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EV_PARAMS as EV, BETA_WEIGHT, GAMMA, PIDDQN_EPOCHS, PIDDQN_LR, PIDDQN_BUFFER_CAP, PIDDQN_BATCH_SIZE, PIDDQN_PHYS_LAMBDA, NUM_EVS_DEFAULT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def energy_kwh(dist_m, speed_ms, slope_rad=0.0):
    """Physics-based energy model (Eq.6)"""
    F = 0.5*EV['rho']*EV['Cd']*EV['A']*speed_ms**2 + EV['Cr']*EV['m']*EV['g'] + EV['m']*EV['g']*np.sin(slope_rad)
    return max(0, (dist_m/EV['eta'])*F / 3_600_000)

def travel_time(base_t, cong_factor, charge_t=0):
    return base_t*(1 + 0.5*cong_factor) + charge_t

def congestion(flow, cap):
    return min(1.0, flow/cap) if cap > 0 else 1.0


class PhysicsInformedDDQN(nn.Module):
    """Double DQN with physics-aware architecture."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x): return self.net(x)


class ReplayBuffer:
    def __init__(self, cap=PIDDQN_BUFFER_CAP):
        self.buf = deque(maxlen=cap)
    def push(self, s, a, r, ns, done, pbound, mask):
        self.buf.append((s, a, r, ns, done, pbound, mask))
    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        s, a, r, ns, d, pb, m = zip(*batch)
        return (torch.FloatTensor(np.array(s)).to(device),
                torch.LongTensor(a).to(device),
                torch.FloatTensor(r).to(device),
                torch.FloatTensor(np.array(ns)).to(device),
                torch.FloatTensor(d).to(device),
                torch.FloatTensor(pb).to(device),
                torch.BoolTensor(np.array(m)).to(device))
    def __len__(self): return len(self.buf)


class PIDDQNAgent:
    """Single EV agent using shared PI-DDQN network."""
    def __init__(self, aid, G, s, d, soc, dqn, target, memory, gamma=GAMMA, beta=BETA_WEIGHT):
        self.G, self.s, self.dest, self.soc = G, s, d, soc
        self.dqn, self.target, self.memory = dqn, target, memory
        self.gamma, self.beta = gamma, beta
        self.max_soc, self.critical = EV['battery'], EV['buffer']
        self.total_e = self.total_t = self.total_d = 0.0
        self.path, self.done = [s], False
        self.nodes = list(G.nodes()); self.n_nodes = len(self.nodes)
        self.slopes = {(u,v): np.radians(random.uniform(0,3)) for u,v in G.edges()}
        self.slopes.update({(v,u): sl for (u,v),sl in list(self.slopes.items())})
        self.eps = 1.0

    def _state_vec(self):
        """State: one-hot(current) + one-hot(dest) + normalized SOC"""
        v = np.zeros(self.n_nodes*2 + 1)
        v[self.nodes.index(self.s)] = 1.0
        v[self.n_nodes + self.nodes.index(self.dest)] = 1.0
        v[-1] = self.soc / self.max_soc
        return v

    def _actions(self):
        nb = list(self.G.neighbors(self.s))
        # Physics-informed action masking: block if not enough SOC to traverse
        valid = []
        for n in nb:
            ed = self.G[self.s][n]
            spd = ed['weight'] / max(0.1, ed['time'])
            # Assume max possible congestion penalty (cf=1.0) for safety bounds
            e_req = energy_kwh(ed['weight'], spd) * 1.2
            if self.soc >= e_req and n not in self.path[-5:]:
                valid.append(n)
        return valid if valid else nb  # fallback to all neighbors

    def choose_action(self):
        acts = self._actions()
        if not acts: return self.s
        if random.random() < self.eps: return random.choice(acts)
        sv = torch.FloatTensor(self._state_vec()).unsqueeze(0).to(device)
        with torch.no_grad():
            qv = self.dqn(sv)[0]
        idxs = [self.nodes.index(a) for a in acts]
        best = torch.argmax(qv[idxs]).item()
        return acts[best]

    def _phys_reward(self, s, a, e_kwh, t_s, cf, at_dest):
        """Reward (Eq.12) + physics-informed potential shaping."""
        R = 10.0 - (self.beta*(e_kwh/0.5)+(1-self.beta)*(t_s/120))*5.0 - cf*2.0
        
        # Physics upper bound for regularization
        min_e = energy_kwh(self.G[s][a]['weight'],
                          self.G[s][a]['weight']/max(0.1,self.G[s][a]['time'])) * (1.0 + 0.2 * cf)
        max_R = 10.0 - (self.beta*(min_e/0.5))*5.0
        
        if at_dest: 
            R += 100.0
            max_R += 100.0
        if self.soc <= 0: 
            R -= 50.0
            
        # Potential-based shaping: encourage moving closer to destination
        try:
            cd = nx.astar_path_length(self.G, s, self.dest, weight='weight')
            nd = nx.astar_path_length(self.G, a, self.dest, weight='weight')
            phi_c, phi_n = -(cd/1000), -(nd/1000)
            shaping = (self.gamma*phi_n - phi_c) * 10.0
            R += shaping
            max_R += shaping
        except: pass

        return R, max_R

    def step(self, gcong):
        if self.done: return True, 0, 0, 0
        if self.G.nodes[self.s].get('is_cs') and self.soc < self.max_soc*0.8:
            ct = ((self.max_soc-self.soc)/50)*3600
            self.soc = self.max_soc; self.total_t += ct
            return False, 0, ct, 0

        a = self.choose_action()
        if a == self.s: self.done = True; return True, 0, 0, 0

        ed = self.G[self.s][a]
        dist, bt, cap = ed['weight'], ed['time'], ed.get('capacity', 20)
        spd = dist/max(0.1, bt)
        slope = self.slopes.get((self.s, a), 0)
        flow = gcong.get((self.s,a),0) + gcong.get((a,self.s),0)
        cf = congestion(flow, cap)
        tt = travel_time(bt, cf)
        # Apply congestion penalty for stop-and-go traffic
        e = energy_kwh(dist, spd, slope) * (1.0 + 0.2 * cf)

        obs = self._state_vec()
        self.soc -= e; self.total_e += e; self.total_t += tt; self.total_d += dist
        at_dest = (a == self.dest)
        self.done = at_dest or self.soc <= 0
        r, pb = self._phys_reward(self.s, a, e, tt, cf, at_dest)

        self.s = a; self.path.append(a)
        nobs = self._state_vec()
        
        mask = np.zeros(self.n_nodes, dtype=bool)
        if not self.done:
            for na in self._actions(): mask[self.nodes.index(na)] = True
            
        aidx = self.nodes.index(a)
        self.memory.push(obs, aidx, r, nobs, float(self.done), pb, mask)
        return self.done, e, tt, dist


class MultiAgentPIDDQNRouter:
    """Multi-agent PI-DDQN router with centralized training."""
    def __init__(self, G, n_evs=NUM_EVS_DEFAULT, beta=BETA_WEIGHT):
        self.G, self.n_evs, self.beta = G, n_evs, beta
        self.nodes = list(G.nodes())
        nn_nodes = len(self.nodes)
        sdim = nn_nodes*2 + 1
        adim = nn_nodes
        self.dqn = PhysicsInformedDDQN(sdim, adim).to(device)
        self.target = PhysicsInformedDDQN(sdim, adim).to(device)
        self.target.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=PIDDQN_LR)
        self.memory = ReplayBuffer(PIDDQN_BUFFER_CAP)
        self.bs = PIDDQN_BATCH_SIZE; self.phys_lambda = PIDDQN_PHYS_LAMBDA

    def _agents(self):
        ags = []
        for i in range(self.n_evs):
            s, d = random.sample(self.nodes, 2)
            soc = random.uniform(0.3*EV['battery'], EV['battery'])
            ag = PIDDQNAgent(i, self.G, s, d, soc, self.dqn, self.target, self.memory, beta=self.beta)
            ags.append(ag)
        return ags

    def find_cs(self, agent):
        cs = [n for n,d in self.G.nodes(data=True) if d.get('is_cs')]
        best, mc = None, float('inf')
        for c in cs:
            try:
                pl = nx.shortest_path_length(self.G, agent.s, c, weight='weight')
                er = (pl/1000)*0.15
                cost = self.beta*(er/0.5)
                if cost < mc and agent.soc > er: mc, best = cost, c
            except: pass
        return best

    def update(self):
        if len(self.memory) < self.bs: return 0
        s, a, r, ns, d, pb, m = self.memory.sample(self.bs)
        qv = self.dqn(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_qs = self.dqn(ns)
            next_qs[~m] = -float('inf')  # Mask invalid actions
            ba = next_qs.argmax(1).unsqueeze(1)
            nqv = self.target(ns).gather(1, ba).squeeze(1)
            tgt = r + (1-d)*GAMMA*nqv
        td_loss = nn.MSELoss()(qv, tgt)
        # Physics-informed regularization: penalize Q exceeding physical bounds
        phys_loss = torch.mean(torch.relu(qv - (pb + (1-d)*GAMMA*nqv))**2)
        loss = td_loss + self.phys_lambda * phys_loss
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return loss.item()

    def train(self, epochs=PIDDQN_EPOCHS):
        print(f"PI-DDQN: {self.n_evs} EVs, {epochs} epochs...", flush=True)
        hist = []
        for ep in range(epochs):
            agents = self._agents()
            for ag in agents: ag.eps = max(0.05, 1.0*(0.99**ep))
            active, gc = set(agents), defaultdict(int)
            ep_e = ep_d = 0.0
            for step in range(100):
                if not active: break
                dl = []
                for ag in list(active):
                    if ag.soc <= ag.critical*1.5 and not ag.G.nodes[ag.s].get('is_cs'):
                        cs = self.find_cs(ag)
                        if cs: ag.dest = cs
                    ps = ag.s
                    dn, e, t, d = ag.step(gc)
                    ep_e += e; ep_d += d
                    if not dn and ps != ag.s: gc[(ps,ag.s)] += 1
                    if dn: dl.append(ag)
                for c in dl: active.discard(c)
                for k in list(gc): gc[k] = max(0, gc[k]-0.5)
                self.update()

            if ep % 10 == 0: self.target.load_state_dict(self.dqn.state_dict())
            hist.append((ep_e, ep_d/1000))
            if ep % 100 == 0:
                e100 = (ep_e/(ep_d/1000))*100 if ep_d>0 else 0
                print(f"  Ep {ep:4d} | {e100:.2f} kWh/100km | eps:{agents[0].eps:.3f}", flush=True)
        print("PI-DDQN training done.", flush=True)
        return hist

    def evaluate(self, epochs=5, bipartite=True):
        te = td = tt = 0
        for _ in range(epochs):
            agents = self._agents()
            for ag in agents: ag.eps = 0.0
            active, gc = set(agents), defaultdict(int)
            for step in range(100):
                if not active: break
                dl = []
                for ag in list(active):
                    if bipartite and ag.soc <= ag.critical*1.5 and not ag.G.nodes[ag.s].get('is_cs'):
                        cs = self.find_cs(ag)
                        if cs: ag.dest = cs
                    ps = ag.s
                    dn, e, t, d = ag.step(gc)
                    te += e; td += d; tt += t
                    if not dn and ps != ag.s: gc[(ps,ag.s)] += 1
                    if dn: dl.append(ag)
                for c in dl: active.discard(c)
                for k in list(gc): gc[k] = max(0, gc[k]-0.5)
        n = self.n_evs*epochs
        ae, at, adk = te/n, tt/n, (td/1000)/n
        return (ae/adk)*100 if adk>0 else 0, at
