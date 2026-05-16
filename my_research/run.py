"""
MY RESEARCH: SG-GAN + PI-DDQN (Physics-Informed Double DQN)
Uses SAME shared map as base paper for fair comparison.
Generates ALL figures + tables + comparison against base paper Q-Learning.
"""
import sys, os, time, random, pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from network_env import NetworkEnvironment, Generator
from pi_ddqn_routing import MultiAgentPIDDQNRouter, PIDDQNAgent, EV
from config import *

os.makedirs("results", exist_ok=True)
print("="*60); print("MY RESEARCH: SG-GAN + PI-DDQN (Dwarka Mod)"); print("="*60, flush=True)

# ── Load SAME shared map ─────────────────────────────────────
map_path = os.path.join(os.path.dirname(__file__), '..', 'shared_map.pkl')
with open(map_path, 'rb') as f:
    M = pickle.load(f)

ga, ka = M['graph_a'], M['kld_a']
gb, kb = M['graph_b'], M['kld_b']
gh, dh = M['g_hist'], M['d_hist']

env = NetworkEnvironment()
env.real_edges = M['env_real_edges']
env.real_coords = M['env_real_coords']
env.x_min, env.x_max = M['env_x_min'], M['env_x_max']
env.y_min, env.y_max = M['env_y_min'], M['env_y_max']

gen = Generator(10, 29*2)
gen.load_state_dict(M['generator_state'])

print(f"Loaded SAME shared map: A({ga.number_of_nodes()}N,{ga.number_of_edges()}E,KLD={ka:.4f}), "
      f"B({gb.number_of_nodes()}N,{gb.number_of_edges()}E,KLD={kb:.4f})", flush=True)

# ═══════════════════════════════════════════════════════════════
# FIG 2: Road Networks (same map, same plots)
# ═══════════════════════════════════════════════════════════════
def plot_net(G, title, fn):
    plt.figure(figsize=(12,9))
    pos = nx.get_node_attributes(G, 'pos')
    cs = ['#2ecc71' if G.nodes[n].get('is_cs') else '#3498db' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=cs, node_size=500, edgecolors='k', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, edge_color='#7f8c8d', width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='white')
    el = {(u,v): f"{d['weight']/1000:.1f}km" for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, el, font_size=5, font_color='darkred',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.2))
    plt.legend(handles=[
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#3498db',markersize=10,label='Junction (22)'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#2ecc71',markersize=10,label='CS (7)')
    ], loc='upper right', fontsize=10)
    plt.title(title, fontsize=13); plt.tight_layout()
    plt.savefig(f"results/{fn}.png", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved {fn}.png", flush=True)

plot_net(ga, f"Fig 2(a): Wider Network (KLD={ka:.4f})", "fig_2a")
plot_net(gb, f"Fig 2(b): Dense Network (KLD={kb:.4f})", "fig_2b")

# ═══════════════════════════════════════════════════════════════
# TRAIN PI-DDQN
# ═══════════════════════════════════════════════════════════════
print("\n[Training PI-DDQN on Graph A]...", flush=True)
t0 = time.time()
router = MultiAgentPIDDQNRouter(ga, n_evs=NUM_EVS_DEFAULT, beta=BETA_WEIGHT)
hist = router.train(epochs=PIDDQN_EPOCHS)
train_t = time.time() - t0
print(f"Converged in {train_t:.0f}s ({train_t/60:.1f} min)", flush=True)

e_bip, t_bip = router.evaluate(5, bipartite=True)
e_no, t_no = router.evaluate(5, bipartite=False)
print(f"  PI-DDQN WITH bipartite:    {e_bip:.2f} kWh/100km, {t_bip:.1f}s", flush=True)
print(f"  PI-DDQN WITHOUT bipartite: {e_no:.2f} kWh/100km, {t_no:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════════════
# FIG 3(a): SG-GAN accuracy + FIG 3(b): PI-DDQN convergence
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
n = min(201, len(gh))
ax[0].plot(range(n), gh[:n], label='Generator', color='#e74c3c', linewidth=1.5)
ax[0].plot(range(n), dh[:n], label='Discriminator', color='#3498db', linewidth=1.5)
ax[0].set_title('Fig 3(a): SG-GAN Accuracy', fontsize=12)
ax[0].set_xlabel('Iterations'); ax[0].set_ylabel('Accuracy (%)')
ax[0].legend(); ax[0].grid(alpha=0.3)

epe = [(e/d)*100 if d>0 else 0 for e,d in hist]
ax[1].plot(epe, color='#9b59b6', alpha=0.2, linewidth=0.5)
if len(epe)>30:
    sm = np.convolve(epe, np.ones(30)/30, 'valid')
    ax[1].plot(range(len(sm)), sm, color='#9b59b6', linewidth=2, label='PI-DDQN (smoothed)')
ax[1].axhline(y=e_bip, color='green', linestyle='--', label=f'Final: {e_bip:.2f}')
ax[1].set_title('Fig 3(b): PI-DDQN Energy Convergence', fontsize=12)
ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('kWh/100km'); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/fig_3.png", dpi=200); plt.close()
print("  Saved fig_3.png", flush=True)

# ═══════════════════════════════════════════════════════════════
# FIG 4(a): 3D Congestion + FIG 4(b): Beta sensitivity
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 5))
a3 = fig.add_subplot(121, projection='3d')
cong_pcts = [25, 50, 100]; ec, tc = [], []
for c in cong_pcts:
    tr = MultiAgentPIDDQNRouter(ga, max(1,int(50*c/100)), 0.8)
    tr.dqn.load_state_dict(router.dqn.state_dict())
    e, t = tr.evaluate(3); ec.append(e); tc.append(t)
a3.scatter(cong_pcts, ec, tc, c='green', s=120, zorder=5)
a3.plot(cong_pcts, ec, tc, 'b-', alpha=0.5)
for i,c in enumerate(cong_pcts):
    a3.text(c, ec[i], tc[i], f'  C({c}%)\n  {ec[i]:.1f}kWh', fontsize=8)
a3.set_xlabel('Congestion(%)'); a3.set_ylabel('Energy(kWh/100km)'); a3.set_zlabel('Time(s)')
a3.set_title('Fig 4(a): PI-DDQN Congestion Analysis', fontsize=11)

ab = fig.add_subplot(122)
betas = np.linspace(0.1, 0.9, 9); eb, tb = [], []
for b in betas:
    tr = MultiAgentPIDDQNRouter(ga, 50, b)
    tr.dqn.load_state_dict(router.dqn.state_dict())
    e, t = tr.evaluate(2); eb.append(e); tb.append(t)
ab.plot(betas, eb, 'b-o', label='Energy', linewidth=1.5)
ab.set_ylabel('kWh/100km', color='b'); ab.tick_params(axis='y', labelcolor='b')
ab2 = ab.twinx()
ab2.plot(betas, tb, 'r-s', label='Time', linewidth=1.5)
ab2.set_ylabel('Time(s)', color='r'); ab2.tick_params(axis='y', labelcolor='r')
ab.set_xlabel('Beta (β)'); ab.set_title('Fig 4(b): PI-DDQN β Sensitivity', fontsize=11)
ab.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
ab.grid(alpha=0.3)
l1,lb1 = ab.get_legend_handles_labels(); l2,lb2 = ab2.get_legend_handles_labels()
ab.legend(l1+l2, lb1+lb2, fontsize=9)
plt.tight_layout(); plt.savefig("results/fig_4.png", dpi=200); plt.close()
print("  Saved fig_4.png", flush=True)

# ═══════════════════════════════════════════════════════════════
# FIG 5: Scalability
# ═══════════════════════════════════════════════════════════════
print("\n[Scalability (Fig 5)]...", flush=True)
evs = SCALE_EVS; nds = SCALE_NODES
em, tm = np.zeros((len(nds),len(evs))), np.zeros((len(nds),len(evs)))
gs = {}
for nd in nds:
    g_gen = gen if nd==DEFAULT_NODES else Generator(GAN_NOISE_DIM, nd*2)
    gs[nd], _ = env.synthesize_graph(g_gen, nd, max(1,int(nd*0.24)), connection_threshold=0.25)
for i, nd in enumerate(nds):
    for j, ne in enumerate(evs):
        tr = MultiAgentPIDDQNRouter(gs[nd], ne, BETA_WEIGHT); tr.train(SCALE_TRAIN_EPOCHS_PIDDQN)
        em[i,j], tm[i,j] = tr.evaluate(2)
        print(f"  N={nd},EV={ne}: {em[i,j]:.1f} kWh/100km", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
mk = ['o','s','^','D']
for i,nd in enumerate(nds):
    ax[0].plot(evs, em[i], f'-{mk[i]}', label=f'{nd}N')
    ax[1].plot(evs, tm[i], f'-{mk[i]}', label=f'{nd}N')
ax[0].set_title('Fig 5(a): EVs vs Energy (PI-DDQN)', fontsize=12)
ax[0].set_xlabel('EVs'); ax[0].set_ylabel('kWh/100km')
ax[1].set_title('Fig 5(b): EVs vs Time (PI-DDQN)', fontsize=12)
ax[1].set_xlabel('EVs'); ax[1].set_ylabel('Time(s)')
for a in ax: a.legend(); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/fig_5.png", dpi=200); plt.close()
print("  Saved fig_5.png", flush=True)

# ═══════════════════════════════════════════════════════════════
# ALL TABLES
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  PI-DDQN RESULTS TABLES\n" + "="*60)

savings = (e_no-e_bip)/e_no*100 if e_no>0 else 0

# Table IV: Congestion analysis
print("\n--- Table IV: Energy at Various Congestion (PI-DDQN + Bipartite) ---")
print(pd.DataFrame({
    'Congestion': ['25%','50%','100%'],
    'Energy(kWh/100km)': [f"{ec[0]:.2f}", f"{ec[1]:.2f}", f"{ec[2]:.2f}"],
    'Time(s)': [f"{tc[0]:.1f}", f"{tc[1]:.1f}", f"{tc[2]:.1f}"]
}).to_string(index=False))

# Table VI: With vs Without bipartite
print("\n--- Table VI: Bipartite Mapping Impact ---")
print(pd.DataFrame({
    'Config': ['Without Bipartite','With Bipartite','Savings'],
    'Energy(kWh/100km)': [f"{e_no:.2f}", f"{e_bip:.2f}", f"{savings:.2f}%"],
    'Time(s)': [f"{t_no:.1f}", f"{t_bip:.1f}", "—"]
}).to_string(index=False))

# Table VII: Method comparison (EXTENDED with PI-DDQN)
print("\n--- Table VII: Comparison (Extended with PI-DDQN) ---")
print(pd.DataFrame({
    'Method': ['Clustering [28]','A* [29]','Dijkstra [30]','MILP [31]',
               'PSO [32]','SARSA [33]','DRL [28]',
               'Q-Learning (Paper)','PI-DDQN (Ours)'],
    'Energy(kWh/100km)': [19.56,18.6,18.6,18.6,21.6,36.8,15.7,15.64,round(e_bip,2)],
    'Time': ['N/A','N/A','N/A','~10m','~30m','~2h','~5h','~3.5h',f'{train_t/60:.1f}m']
}).to_string(index=False))

# Table VIII: KLD
print("\n--- Table VIII: KL Divergence ---")
print(pd.DataFrame({
    'Network': ['Graph A (Wider)','Graph B (Dense)'],
    'Nodes': [29,29], 'Edges': [ga.number_of_edges(), gb.number_of_edges()],
    'KLD': [f"{ka:.4f}", f"{kb:.4f}"]
}).to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
improvement = ((15.64-e_bip)/15.64)*100 if e_bip < 15.64 else -((e_bip-15.64)/15.64)*100
print(f"\n{'='*60}\n  FINAL SUMMARY — PI-DDQN vs Q-Learning\n{'='*60}")
print(f"  SG-GAN: KLD(A)={ka:.4f}, KLD(B)={kb:.4f}")
print(f"  PI-DDQN energy (bipartite): {e_bip:.2f} kWh/100km")
print(f"  Q-Learning energy (paper):  15.64 kWh/100km")
print(f"  Improvement: {improvement:.2f}%")
print(f"  PI-DDQN convergence: {train_t/60:.1f} min ({train_t:.0f}s)")
print(f"\n  Physics-Informed Constraints:")
print(f"    ✓ Energy: Ec=(d/η)(½ρCdAv²+Crmg+mg·sinθ)")
print(f"    ✓ SOC action masking (blocks infeasible moves)")
print(f"    ✓ Physics loss regularization (Q ≤ physical bound)")
print(f"    ✓ A* potential shaping (guides exploration)")
print(f"    ✓ Double DQN (stable Q-estimates)")
print(f"    ✓ m={EV['m']}kg, Cd={EV['Cd']}, A={EV['A']}m², η={EV['eta']}")
print(f"    ✓ Battery: {EV['battery']}kWh, buffer: {EV['buffer']}kWh")
print(f"\n  Results in: {os.path.abspath('results')}/", flush=True)
