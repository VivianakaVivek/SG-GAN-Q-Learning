"""
COMPARISON TABLE EXPERIMENT
============================
Mirrors the base paper's Table III vs Table IV experiment, but extended to compare:
  - Table A: Q-Learning (Base Paper) — the PROBLEM
  - Table B: PI-DDQN (Our Research) — the SOLUTION

For 5 fixed EV source→destination pairs, at 25%, 50%, 100% congestion:
Records actual path taken, energy (kWh), and travel time (s).
"""
import sys, os, random, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))
from base_paper.network_env import NetworkEnvironment, Generator
from base_paper.routing import MultiAgentRouter, EVAgent
from my_research.pi_ddqn_routing import MultiAgentPIDDQNRouter, PIDDQNAgent
from config import *

# ── Load shared map ───────────────────────────────────────────────
map_path = os.path.join(os.path.dirname(__file__), 'shared_map.pkl')
with open(map_path, 'rb') as f:
    M = pickle.load(f)

ga = M['graph_a']
nodes = list(ga.nodes())

print("=" * 70)
print("COMPARISON EXPERIMENT: Q-Learning (Problem) vs PI-DDQN (Solution)")
print("=" * 70)
print(f"Graph A: {ga.number_of_nodes()} nodes, {ga.number_of_edges()} edges")

# ── Fix 5 EV source→destination pairs (reproducible) ─────────────
random.seed(99)
np.random.seed(99)
EV_PAIRS = []
cs_nodes = [n for n, d in ga.nodes(data=True) if d.get('is_cs')]
non_cs   = [n for n in nodes if n not in cs_nodes]
for _ in range(5):
    s = random.choice(non_cs)
    d = random.choice([x for x in non_cs if x != s])
    EV_PAIRS.append((s, d))

print(f"\nFixed EV Source→Destination Pairs:")
for i, (s, d) in enumerate(EV_PAIRS):
    print(f"  EV {i+1}: Node {s} -> Node {d}")

# ── Train both models on full 50-EV graph ────────────────────────
print("\n[Training Q-Learning...]")
ql_router = MultiAgentRouter(ga, n_evs=50, beta=BETA_WEIGHT)
ql_router.train(epochs=QL_EPOCHS)

print("\n[Training PI-DDQN...]")
pi_router = MultiAgentPIDDQNRouter(ga, n_evs=50, beta=BETA_WEIGHT)
pi_router.train(epochs=PIDDQN_EPOCHS)

# ── Helper: evaluate a single EV and record path + metrics ───────
def eval_ql_single(s, d, soc_init, gcong, eps=0.0):
    ag = EVAgent(99, ga, s, d, soc_init, ql_router.qtable, alpha=0, beta=BETA_WEIGHT)
    for _ in range(80):
        if ag.done: break
        ps = ag.s
        dn, e, t, dist = ag.step(eps, gcong)
        if not dn and ps != ag.s:
            gcong[(ps, ag.s)] = gcong.get((ps, ag.s), 0) + 1
    path_str = "→".join(str(x) for x in ag.path[:8])
    if len(ag.path) > 8: path_str += "→..."
    # Add CS info
    cs_visited = [str(n) for n in ag.path if ga.nodes[n].get('is_cs')]
    if cs_visited:
        path_str += f" [CS:{','.join(cs_visited[:2])}]"
    dist_km = ag.total_d / 1000.0
    e100 = (ag.total_e / dist_km) * 100 if dist_km > 0 else 0
    return path_str, round(e100, 2), round(ag.total_t, 1), ag.soc > 0

def eval_pi_single(s, d, soc_init, gcong):
    ag = PIDDQNAgent(99, ga, s, d, soc_init,
                     pi_router.dqn, pi_router.target,
                     pi_router.memory, beta=BETA_WEIGHT)
    ag.eps = 0.0
    for _ in range(80):
        if ag.done: break
        ps = ag.s
        dn, e, t, dist = ag.step(gcong)
        if not dn and ps != ag.s:
            gcong[(ps, ag.s)] = gcong.get((ps, ag.s), 0) + 1
    path_str = "→".join(str(x) for x in ag.path[:8])
    if len(ag.path) > 8: path_str += "→..."
    cs_visited = [str(n) for n in ag.path if ga.nodes[n].get('is_cs')]
    if cs_visited:
        path_str += f" [CS:{','.join(cs_visited[:2])}]"
    dist_km = ag.total_d / 1000.0
    e100 = (ag.total_e / dist_km) * 100 if dist_km > 0 else 0
    return path_str, round(e100, 2), round(ag.total_t, 1), ag.soc > 0

# ── Run experiment across 3 congestion levels ─────────────────────
# Congestion simulated by pre-filling gcong counter to reflect traffic density
CONG_LEVELS = [25, 50, 100]

def make_gcong(level_pct):
    """Pre-fill congestion proportional to congestion level."""
    gcong = defaultdict(int)
    edge_list = list(ga.edges())
    n_congested = max(0, int(len(edge_list) * level_pct / 100 * 0.5))
    for u, v in random.sample(edge_list, min(n_congested, len(edge_list))):
        cap = ga[u][v].get('capacity', 20)
        gcong[(u, v)] = int(cap * level_pct / 100)
    return gcong

SOC_INIT = EV_PARAMS['battery'] * 0.9  # 90% charged at start

rows_ql = []
rows_pi = []

print("\n[Running per-EV path evaluation...]\n")

for ev_idx, (s, d) in enumerate(EV_PAIRS):
    ql_row = {'EV': ev_idx+1, 'Start→End': f"{s}→{d}"}
    pi_row = {'EV': ev_idx+1, 'Start→End': f"{s}→{d}"}

    for cong in CONG_LEVELS:
        random.seed(42 + cong + ev_idx)
        np.random.seed(42 + cong + ev_idx)
        gcong_ql = make_gcong(cong)
        gcong_pi = make_gcong(cong)

        path_ql, e_ql, t_ql, alive_ql = eval_ql_single(s, d, SOC_INIT, gcong_ql)
        path_pi, e_pi, t_pi, alive_pi = eval_pi_single(s, d, SOC_INIT, gcong_pi)

        ql_row[f'Path_{cong}'] = path_ql + ('' if alive_ql else ' [STRANDED]')
        ql_row[f'E_{cong}']    = e_ql
        ql_row[f'T_{cong}']    = t_ql

        pi_row[f'Path_{cong}'] = path_pi + ('' if alive_pi else ' [STRANDED]')
        pi_row[f'E_{cong}']    = e_pi
        pi_row[f'T_{cong}']    = t_pi

        print(f"  EV{ev_idx+1} | Cong={cong}% | QL: {e_ql:.2f}kWh {t_ql:.0f}s | PI: {e_pi:.2f}kWh {t_pi:.0f}s")

    rows_ql.append(ql_row)
    rows_pi.append(pi_row)

# ── Print Tables ──────────────────────────────────────────────────
print("\n" + "="*70)
print("TABLE A — Q-Learning Routes (BASE PAPER — The PROBLEM)")
print("="*70)
print(f"{'EV':<4} {'Route':<12} {'Path (25%)':<35} {'E25':>6} {'T25':>6} | {'Path (50%)':<35} {'E50':>6} {'T50':>6} | {'Path (100%)':<35} {'E100':>6} {'T100':>6}")
print("-"*170)
for r in rows_ql:
    print(f"{r['EV']:<4} {r['Start→End']:<12} {r['Path_25']:<35} {r['E_25']:>6} {r['T_25']:>6} | {r['Path_50']:<35} {r['E_50']:>6} {r['T_50']:>6} | {r['Path_100']:<35} {r['E_100']:>6} {r['T_100']:>6}")

print("\n" + "="*70)
print("TABLE B — PI-DDQN Routes (OUR RESEARCH — The SOLUTION)")
print("="*70)
print(f"{'EV':<4} {'Route':<12} {'Path (25%)':<35} {'E25':>6} {'T25':>6} | {'Path (50%)':<35} {'E50':>6} {'T50':>6} | {'Path (100%)':<35} {'E100':>6} {'T100':>6}")
print("-"*170)
for r in rows_pi:
    print(f"{r['EV']:<4} {r['Start→End']:<12} {r['Path_25']:<35} {r['E_25']:>6} {r['T_25']:>6} | {r['Path_50']:<35} {r['E_50']:>6} {r['T_50']:>6} | {r['Path_100']:<35} {r['E_100']:>6} {r['T_100']:>6}")

# ── Summary comparison ─────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY: Average Energy (kWh/100km) & Time (s) Comparison")
print("="*70)
for cong in CONG_LEVELS:
    avg_e_ql = np.mean([r[f'E_{cong}'] for r in rows_ql])
    avg_t_ql = np.mean([r[f'T_{cong}'] for r in rows_ql])
    avg_e_pi = np.mean([r[f'E_{cong}'] for r in rows_pi])
    avg_t_pi = np.mean([r[f'T_{cong}'] for r in rows_pi])
    savings_e = ((avg_e_ql - avg_e_pi) / avg_e_ql) * 100 if avg_e_ql > 0 else 0
    savings_t = ((avg_t_ql - avg_t_pi) / avg_t_ql) * 100 if avg_t_ql > 0 else 0
    print(f"  {cong:3d}% Congestion:")
    print(f"    Q-Learning  -> Energy: {avg_e_ql:.2f} kWh/100km | Time: {avg_t_ql:.1f}s")
    print(f"    PI-DDQN     -> Energy: {avg_e_pi:.2f} kWh/100km | Time: {avg_t_pi:.1f}s")
    print(f"    PI-DDQN Savings: {savings_e:+.2f}% energy | {savings_t:+.2f}% time")
    print()

print("Done. Use these results to fill in the markdown table.", flush=True)
