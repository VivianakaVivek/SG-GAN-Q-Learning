"""
Tabular Q-Learning for EV Routing + CS Allocation (Section V, Algorithm 2)
Energy model (Eq.6), Travel time (Eq.7), Congestion (Eq.9-10), Reward (Eq.12)
"""
import sys, os, random
import numpy as np
import networkx as nx
from collections import defaultdict
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EV_PARAMS as EV, BETA_WEIGHT, GAMMA, QL_EPOCHS, QL_ALPHA, QL_EPS_START, QL_EPS_MIN, QL_EPS_DECAY, NUM_EVS_DEFAULT

def energy_kwh(dist_m, speed_ms, slope_rad=0.0):
    F = 0.5*EV['rho']*EV['Cd']*EV['A']*speed_ms**2 + EV['Cr']*EV['m']*EV['g'] + EV['m']*EV['g']*np.sin(slope_rad)
    return max(0, (dist_m/EV['eta'])*F / 3_600_000)

def travel_time(base_t, cong_factor, charge_t=0):
    """Travel time (Eq.7): Tc = queueing + travel + charge"""
    return base_t * (1 + 0.5*cong_factor) + charge_t

def congestion(flow, cap):
    """Local congestion (Eq.9): δij = qij/Cij"""
    return min(1.0, flow/cap) if cap > 0 else 1.0

class EVAgent:
    def __init__(self, aid, G, s, d, soc, qtable, alpha=QL_ALPHA, gamma=GAMMA, beta=BETA_WEIGHT):
        self.G, self.s, self.dest, self.soc = G, s, d, soc
        self.qtable, self.alpha, self.gamma, self.beta = qtable, alpha, gamma, beta
        self.max_soc, self.critical = EV['battery'], EV['buffer']
        self.total_e = self.total_t = self.total_d = 0.0
        self.path, self.done = [s], False
        self.slopes = {(u,v): np.radians(random.uniform(0,3)) for u,v in G.edges()}
        self.slopes.update({(v,u): s for (u,v),s in list(self.slopes.items())})

    def _state(self):
        bucket = min(int(self.soc/self.max_soc*10), 9)
        return (self.s, self.dest, bucket)

    def _actions(self):
        nb = list(self.G.neighbors(self.s))
        unv = [n for n in nb if n not in self.path[-5:]]
        return unv if unv else nb

    def choose_action(self, eps):
        acts = self._actions()
        if not acts: return self.s
        if random.random() < eps: return random.choice(acts)
        st = self._state()
        return max(acts, key=lambda a: self.qtable.get((st, a), 0.0))

    def reward(self, e_kwh, t_s, cf, at_dest):
        """Reward function (Eq.12)"""
        R = 10.0 - (self.beta*(e_kwh/0.5) + (1-self.beta)*(t_s/120))*5.0 - cf*2.0
        if at_dest: R += 100.0
        if self.soc <= 0: R -= 50.0
        return R

    def step(self, eps, gcong):
        if self.done: return True, 0, 0, 0
        # Charge at CS if SOC < 80%
        if self.G.nodes[self.s].get('is_cs') and self.soc < self.max_soc*0.8:
            ct = ((self.max_soc - self.soc)/50)*3600
            self.soc = self.max_soc; self.total_t += ct
            return False, 0, ct, 0

        a = self.choose_action(eps)
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

        old_st = self._state()
        self.soc -= e; self.total_e += e; self.total_t += tt; self.total_d += dist
        at_dest = (a == self.dest)
        self.done = at_dest or self.soc <= 0

        r = self.reward(e, tt, cf, at_dest)
        self.s = a; self.path.append(a)
        new_st = self._state()

        # Q-update (Eq.13)
        old_q = self.qtable.get((old_st, a), 0.0)
        if self.done:
            nq = 0
        else:
            na = self._actions()
            nq = max((self.qtable.get((new_st, x), 0.0) for x in na), default=0)
        self.qtable[(old_st, a)] = old_q + self.alpha*(r + self.gamma*nq - old_q)
        return self.done, e, tt, dist


class MultiAgentRouter:
    def __init__(self, G, n_evs=NUM_EVS_DEFAULT, beta=BETA_WEIGHT):
        self.G, self.n_evs, self.beta = G, n_evs, beta
        self.qtable = defaultdict(float)
        self.nodes = list(G.nodes())
        self.eps, self.eps_min, self.eps_decay = QL_EPS_START, QL_EPS_MIN, QL_EPS_DECAY

    def _agents(self):
        ags = []
        for i in range(self.n_evs):
            s, d = random.sample(self.nodes, 2)
            soc = random.uniform(0.3*EV['battery'], EV['battery'])
            ags.append(EVAgent(i, self.G, s, d, soc, self.qtable, beta=self.beta))
        return ags

    def find_cs(self, agent):
        cs = [n for n,d in self.G.nodes(data=True) if d.get('is_cs')]
        best, mc = None, float('inf')
        for c in cs:
            try:
                pl = nx.shortest_path_length(self.G, agent.s, c, weight='weight')
                er = (pl/1000)*0.15
                cost = self.beta*(er/0.5) + (1-self.beta)*0
                if cost < mc and agent.soc > er: mc, best = cost, c
            except: pass
        return best

    def train(self, epochs=QL_EPOCHS):
        print(f"Q-Learning: {self.n_evs} EVs, {epochs} epochs...", flush=True)
        hist = []
        for ep in range(epochs):
            agents = self._agents()
            active, gcong = set(agents), defaultdict(int)
            ep_e = ep_d = 0.0
            for step in range(100):
                if not active: break
                done_list = []
                for ag in list(active):
                    if ag.soc <= ag.critical*1.5 and not ag.G.nodes[ag.s].get('is_cs'):
                        cs = self.find_cs(ag)
                        if cs: ag.dest = cs
                    ps = ag.s
                    dn, e, t, d = ag.step(self.eps, gcong)
                    ep_e += e; ep_d += d
                    if not dn and ps != ag.s: gcong[(ps, ag.s)] += 1
                    if dn: done_list.append(ag)
                for c in done_list: active.discard(c)
                for k in list(gcong): gcong[k] = max(0, gcong[k]-0.5)

            self.eps = max(self.eps_min, self.eps*self.eps_decay)
            hist.append((ep_e, ep_d/1000))
            if ep % 1000 == 0:
                e100 = (ep_e/(ep_d/1000))*100 if ep_d > 0 else 0
                print(f"  Ep {ep:5d} | {e100:.2f} kWh/100km | eps:{self.eps:.3f}", flush=True)
        print("Training done.", flush=True)
        return hist

    def evaluate(self, epochs=5, bipartite=True):
        te = td = tt = 0
        for _ in range(epochs):
            agents = self._agents()
            active, gc = set(agents), defaultdict(int)
            for step in range(100):
                if not active: break
                dl = []
                for ag in list(active):
                    if bipartite and ag.soc <= ag.critical*1.5 and not ag.G.nodes[ag.s].get('is_cs'):
                        cs = self.find_cs(ag)
                        if cs: ag.dest = cs
                    ps = ag.s
                    dn, e, t, d = ag.step(0.0, gc)
                    te += e; td += d; tt += t
                    if not dn and ps != ag.s: gc[(ps, ag.s)] += 1
                    if dn: dl.append(ag)
                for c in dl: active.discard(c)
                for k in list(gc): gc[k] = max(0, gc[k]-0.5)
        n = self.n_evs * epochs
        ae, at = te/n, tt/n
        adk = (td/1000)/n
        return (ae/adk)*100 if adk > 0 else 0, at
