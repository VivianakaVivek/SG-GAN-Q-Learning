"""
BASE PAPER REPLICATION: SG-GAN + Q-Learning (Dwarka Mod)
Loads shared map from generate_map.py, generates ALL paper figures and tables:
  Fig 2(a,b)  - Wide & Dense road networks
  Fig 3(a)    - SG-GAN Generator/Discriminator accuracy
  Fig 3(b)    - Cumulative reward for 5 EVs
  Fig 4(a)    - 3D Congestion vs Energy vs Time
  Fig 4(b)    - Beta sensitivity analysis
  Fig 5(a,b)  - Scalability (EVs vs Energy/Time)
  Table IV    - Optimal paths with bipartite at various congestion
  Table VI    - Energy comparison vs [22] at different congestion
  Table VII   - Comparison of routing methods
  Table VIII  - KLD for different graph sizes/iterations
  + Energy convergence plot, KLD summary, convergence time
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
from routing import MultiAgentRouter, EVAgent, EV
from config import *

os.makedirs("results", exist_ok=True)
print("="*60); print("BASE PAPER: SG-GAN + Q-Learning Replication"); print("="*60, flush=True)

# ── Load shared map ──────────────────────────────────────────
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

print(f"Loaded shared map: A({ga.number_of_nodes()}N,{ga.number_of_edges()}E,KLD={ka:.4f}), "
      f"B({gb.number_of_nodes()}N,{gb.number_of_edges()}E,KLD={kb:.4f})", flush=True)

# ═══════════════════════════════════════════════════════════════
# FIG 2: Road Networks
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

plot_net(ga, f"Fig 2(a): Wider Network — 22 junctions, 7 CS (KLD={ka:.4f})", "fig_2a")
plot_net(gb, f"Fig 2(b): Dense Network — 22 junctions, 7 CS (KLD={kb:.4f})", "fig_2b")

# ═══════════════════════════════════════════════════════════════
# TRAIN Q-LEARNING (Algorithm 2)
# ═══════════════════════════════════════════════════════════════
print("\n[Training Q-Learning on Graph A (wider network)]...", flush=True)
t0 = time.time()
router = MultiAgentRouter(ga, n_evs=NUM_EVS_DEFAULT, beta=BETA_WEIGHT)
hist = router.train(epochs=QL_EPOCHS)
train_t = time.time() - t0
print(f"Converged in {train_t:.0f}s ({train_t/60:.1f} min)", flush=True)

# Evaluate
e_bip, t_bip = router.evaluate(5, bipartite=True)
e_no, t_no = router.evaluate(5, bipartite=False)
print(f"  WITH bipartite:    {e_bip:.2f} kWh/100km, {t_bip:.1f}s avg", flush=True)
print(f"  WITHOUT bipartite: {e_no:.2f} kWh/100km, {t_no:.1f}s avg", flush=True)

# ═══════════════════════════════════════════════════════════════
# FIG 3(a): SG-GAN Learning Accuracy
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
n = min(201, len(gh))
ax[0].plot(range(n), gh[:n], label='Generator Accuracy', color='#e74c3c', linewidth=1.5)
ax[0].plot(range(n), dh[:n], label='Discriminator Accuracy', color='#3498db', linewidth=1.5)
ax[0].set_title('Fig 3(a): SG-GAN Learning Accuracy', fontsize=12)
ax[0].set_xlabel('Iterations'); ax[0].set_ylabel('Accuracy (%)')
ax[0].legend(fontsize=10); ax[0].grid(alpha=0.3); ax[0].set_xlim(0, 200)

# FIG 3(b): Cumulative Reward (5 EVs)
nodes = list(ga.nodes())
cr = np.zeros((5, 2500))
for ep in range(2500):
    for i in range(5):
        s, d = random.sample(nodes, 2)
        ag = EVAgent(i, ga, s, d, random.uniform(3,10), router.qtable, alpha=0, beta=0.8)
        rw = 0
        for _ in range(50):
            if ag.done: break
            dn, e, t, dd = ag.step(0.0, defaultdict(int))
            rw += ag.reward(e, t, 0, dn and ag.s == ag.dest)
        cr[i][ep] = (cr[i][ep-1] if ep>0 else 0) + rw
colors5 = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']
for i in range(5): ax[1].plot(range(2500), cr[i], label=f'EV {i+1}', color=colors5[i], alpha=0.8)
ax[1].set_title('Fig 3(b): Cumulative Reward (5 EVs)', fontsize=12)
ax[1].set_xlabel('Iterations'); ax[1].set_ylabel('Cumulative Reward')
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/fig_3.png", dpi=200); plt.close()
print("  Saved fig_3.png", flush=True)

# ═══════════════════════════════════════════════════════════════
# FIG 4(a): 3D Congestion-Energy-Time
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 5))
a3 = fig.add_subplot(121, projection='3d')
cong_pcts = [25, 50, 100]
ec, tc = [], []
for c in cong_pcts:
    tr = MultiAgentRouter(ga, max(1,int(50*c/100)), 0.8)
    tr.qtable = router.qtable; tr.eps = 0
    e, t = tr.evaluate(3); ec.append(e); tc.append(t)
a3.scatter(cong_pcts, ec, tc, c='green', s=120, zorder=5)
a3.plot(cong_pcts, ec, tc, 'b-', alpha=0.5)
labels3d = [f'C({c}%)\n{ec[i]:.1f}kWh\n{tc[i]:.0f}s' for i,c in enumerate(cong_pcts)]
for i in range(3): a3.text(cong_pcts[i], ec[i], tc[i], f'  {labels3d[i]}', fontsize=8)
a3.set_xlabel('Congestion (%)'); a3.set_ylabel('Energy (kWh/100km)'); a3.set_zlabel('Time (s)')
a3.set_title('Fig 4(a): Congestion vs Energy vs Time', fontsize=11)

# FIG 4(b): Beta Sensitivity
ab = fig.add_subplot(122)
betas = np.linspace(0.1, 0.9, 9)
eb, tb = [], []
for b in betas:
    tr = MultiAgentRouter(ga, 50, b); tr.qtable = router.qtable; tr.eps = 0
    e, t = tr.evaluate(2); eb.append(e); tb.append(t)
ab.plot(betas, eb, 'b-o', label='Energy (kWh/100km)', linewidth=1.5)
ab.set_ylabel('Energy (kWh/100km)', color='b'); ab.tick_params(axis='y', labelcolor='b')
ab2 = ab.twinx()
ab2.plot(betas, tb, 'r-s', label='Travel Time (s)', linewidth=1.5)
ab2.set_ylabel('Travel Time (s)', color='r'); ab2.tick_params(axis='y', labelcolor='r')
ab.set_xlabel('Beta (β)'); ab.set_title('Fig 4(b): Sensitivity Analysis on β', fontsize=11)
ab.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='β=0.8 (selected)')
ab.grid(alpha=0.3)
l1,lb1 = ab.get_legend_handles_labels(); l2,lb2 = ab2.get_legend_handles_labels()
ab.legend(l1+l2, lb1+lb2, fontsize=9)
plt.tight_layout(); plt.savefig("results/fig_4.png", dpi=200); plt.close()
print("  Saved fig_4.png", flush=True)

# ═══════════════════════════════════════════════════════════════
# FIG 5: Scalability
# ═══════════════════════════════════════════════════════════════
print("\n[Scalability Analysis (Fig 5)]...", flush=True)
evs = SCALE_EVS; nds = SCALE_NODES
em, tm = np.zeros((len(nds),len(evs))), np.zeros((len(nds),len(evs)))
gs = {}
for nd in nds:
    g_gen = gen if nd==DEFAULT_NODES else Generator(GAN_NOISE_DIM, nd*2)
    gs[nd], _ = env.synthesize_graph(g_gen, nd, max(1,int(nd*0.24)), connection_threshold=0.25)
for i, nd in enumerate(nds):
    for j, ne in enumerate(evs):
        tr = MultiAgentRouter(gs[nd], ne, BETA_WEIGHT); tr.train(SCALE_TRAIN_EPOCHS_QL)
        em[i,j], tm[i,j] = tr.evaluate(2)
        print(f"  N={nd},EV={ne}: {em[i,j]:.1f} kWh/100km, {tm[i,j]:.0f}s", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
markers = ['o','s','^','D']
for i, nd in enumerate(nds):
    ax[0].plot(evs, em[i], f'-{markers[i]}', label=f'{nd} nodes')
    ax[1].plot(evs, tm[i], f'-{markers[i]}', label=f'{nd} nodes')
ax[0].set_title('Fig 5(a): No. of EVs vs Energy', fontsize=12)
ax[0].set_xlabel('Number of EVs'); ax[0].set_ylabel('Energy (kWh/100km)')
ax[1].set_title('Fig 5(b): No. of EVs vs Travel Time', fontsize=12)
ax[1].set_xlabel('Number of EVs'); ax[1].set_ylabel('Travel Time (s)')
for a in ax: a.legend(); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/fig_5.png", dpi=200); plt.close()
print("  Saved fig_5.png", flush=True)

# ═══════════════════════════════════════════════════════════════
# ENERGY CONVERGENCE PLOT
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
epe = [(e/d)*100 if d>0 else 0 for e,d in hist]
ax.plot(epe, color='purple', alpha=0.15, linewidth=0.5)
if len(epe)>100:
    sm = np.convolve(epe, np.ones(100)/100, 'valid')
    ax.plot(range(len(sm)), sm, color='purple', linewidth=2, label='Smoothed (window=100)')
ax.axhline(y=e_bip, color='green', linestyle='--', label=f'Final: {e_bip:.2f} kWh/100km')
ax.set_title('Q-Learning Energy Convergence', fontsize=12)
ax.set_xlabel('Epoch'); ax.set_ylabel('Energy (kWh/100km)'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/energy_convergence.png", dpi=200); plt.close()
print("  Saved energy_convergence.png", flush=True)

# ═══════════════════════════════════════════════════════════════
# ALL TABLES
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  RESULTS TABLES\n" + "="*60)

# Table IV: Optimal paths with bipartite at various congestion
print("\n--- Table IV: Energy at Various Congestion (WITH bipartite) ---")
t4 = pd.DataFrame({
    'Congestion': ['25%', '50%', '100%'],
    'Energy(kWh/100km)': [f"{ec[0]:.2f}", f"{ec[1]:.2f}", f"{ec[2]:.2f}"],
    'Travel Time(s)': [f"{tc[0]:.1f}", f"{tc[1]:.1f}", f"{tc[2]:.1f}"]
})
print(t4.to_string(index=False))

# Table VI: Energy comparison
print("\n--- Table VI: With vs Without Bipartite Mapping ---")
savings = (e_no - e_bip)/e_no*100 if e_no>0 else 0
t6 = pd.DataFrame({
    'Method': ['Without Bipartite', 'With Bipartite', 'Savings'],
    'Energy(kWh/100km)': [f"{e_no:.2f}", f"{e_bip:.2f}", f"{savings:.2f}%"],
    'Avg Time(s)': [f"{t_no:.1f}", f"{t_bip:.1f}", "—"]
})
print(t6.to_string(index=False))

# Table VII: Comparison of routing methods
print("\n--- Table VII: Comparison of EV Routing Methods ---")
t7 = pd.DataFrame({
    'Method': ['Clustering [28]','A* [29]','Dijkstra [30]','MILP [31]',
               'PSO [32]','SARSA [33]','DRL [28]','Q-Learning (Ours)'],
    'Energy(kWh/100km)': [19.56,18.6,18.6,18.6,21.6,36.8,15.7,round(e_bip,2)],
    'Training Time': ['N/A','N/A','N/A','~10 min','~30 min','~2 hrs','~5 hrs',f'{train_t/60:.1f} min']
})
print(t7.to_string(index=False))

# Table VIII: KLD for different iterations
print("\n--- Table VIII: KL Divergence ---")
t8 = pd.DataFrame({
    'Network': ['Graph A (Wider)', 'Graph B (Dense)'],
    'Nodes': [29, 29], 'Edges': [ga.number_of_edges(), gb.number_of_edges()],
    'KL Divergence': [f"{ka:.4f}", f"{kb:.4f}"]
})
print(t8.to_string(index=False))

# Final Summary
print(f"\n{'='*60}\n  FINAL SUMMARY\n{'='*60}")
print(f"  SG-GAN: 401 epochs, G_acc={gh[-1]:.1f}%, D_acc={dh[-1]:.1f}%")
print(f"  KLD (Wider): {ka:.4f}")
print(f"  KLD (Dense): {kb:.4f}")
print(f"  Q-Learning: 10,000 epochs, {train_t/60:.1f} min convergence")
print(f"  Energy (with bipartite): {e_bip:.2f} kWh/100km")
print(f"  Energy (without bipartite): {e_no:.2f} kWh/100km")
print(f"  Bipartite savings: {savings:.2f}%")
print(f"  beta = 0.8, 50 EVs, 29 nodes, 7 CS")
print(f"\n  All figures saved to: {os.path.abspath('results')}/", flush=True)
