"""
FIG 6 EQUIVALENT: 3D Computational Time Surface Plots
======================================================
Mirrors the base paper's Fig. 6 — three 3D surface plots:
  (a) Training time for SG-GAN (shared)
  (b) Training time for Q-Learning  [BASE PAPER]  vs  PI-DDQN  [OURS]
  (c) Inference time for Q-Learning [BASE PAPER]  vs  PI-DDQN  [OURS]

Axes: Number of Nodes (x) × Number of EVs (y) → Time (z)
"""
import sys, os, time, random, pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))
from base_paper.network_env import NetworkEnvironment, Generator
from base_paper.routing import MultiAgentRouter, EVAgent
from my_research.pi_ddqn_routing import MultiAgentPIDDQNRouter, PIDDQNAgent
from config import *

os.makedirs("base_paper/results", exist_ok=True)
os.makedirs("my_research/results", exist_ok=True)

# ── Load shared map env ───────────────────────────────────────
with open('shared_map.pkl', 'rb') as f:
    M = pickle.load(f)
env = NetworkEnvironment()
env.real_edges = M['env_real_edges']; env.real_coords = M['env_real_coords']
env.x_min, env.x_max = M['env_x_min'], M['env_x_max']
env.y_min, env.y_max = M['env_y_min'], M['env_y_max']

# ── Sweep parameters (matching paper's Fig 6 axes) ───────────
NODE_COUNTS = [50, 100, 150]
EV_COUNTS   = [50, 100, 150, 200]

# Reduced epochs for sweep (same as paper's scaling experiment)
QL_SWEEP_EPOCHS    = 200
PI_SWEEP_EPOCHS    = 80
STEPS_PER_EP       = 50   # Reduced for speed

print("=" * 65)
print("FIG 6: 3D Computational Time Surface — Base Paper vs PI-DDQN")
print("=" * 65)
print(f"Node sweep: {NODE_COUNTS}")
print(f"EV sweep:   {EV_COUNTS}")
print(f"QL epochs (sweep): {QL_SWEEP_EPOCHS} | PI epochs (sweep): {PI_SWEEP_EPOCHS}\n")

# ── Pre-generate all graphs once ─────────────────────────────
print("[Pre-generating graphs for all node sizes...]")
graphs = {}
for nd in NODE_COUNTS:
    random.seed(nd * 7)
    np.random.seed(nd * 7)
    gen_s = Generator(GAN_NOISE_DIM, nd * 2)
    g, _ = env.synthesize_graph(gen_s, nd, max(1, int(nd * 0.24)), connection_threshold=0.25)
    graphs[nd] = g
    print(f"  N={nd}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

# ── Result matrices ───────────────────────────────────────────
ql_train  = np.zeros((len(NODE_COUNTS), len(EV_COUNTS)))   # Q-Learning train time
pi_train  = np.zeros((len(NODE_COUNTS), len(EV_COUNTS)))   # PI-DDQN train time
ql_inf    = np.zeros((len(NODE_COUNTS), len(EV_COUNTS)))   # Q-Learning inference ms
pi_inf    = np.zeros((len(NODE_COUNTS), len(EV_COUNTS)))   # PI-DDQN inference ms

print("\n[Running timing sweep...]")
print(f"{'N':>5} {'EVs':>5} | {'QL Train':>10} {'QL Inf(ms)':>12} | {'PI Train':>10} {'PI Inf(ms)':>12}")
print("-" * 65)

for i, nd in enumerate(NODE_COUNTS):
    G = graphs[nd]
    nodes = list(G.nodes())

    for j, ne in enumerate(EV_COUNTS):
        random.seed(i * 100 + j)
        np.random.seed(i * 100 + j)

        # ── Q-Learning ─────────────────────────────────────
        ql = MultiAgentRouter(G, n_evs=ne, beta=BETA_WEIGHT)
        # Monkey-patch steps per episode for speed
        t0 = time.time()
        ql.train(epochs=QL_SWEEP_EPOCHS)
        ql_train[i, j] = time.time() - t0

        # Q-Learning inference: 200 single steps
        s, d = random.sample(nodes, 2)
        ag = EVAgent(0, G, s, d, EV_PARAMS['battery'], ql.qtable, alpha=0, beta=BETA_WEIGHT)
        inf_t = []
        for _ in range(200):
            t0 = time.perf_counter()
            ag.choose_action(0.0)
            inf_t.append((time.perf_counter() - t0) * 1000)
        ql_inf[i, j] = np.mean(inf_t)

        # ── PI-DDQN ────────────────────────────────────────
        pi = MultiAgentPIDDQNRouter(G, n_evs=ne, beta=BETA_WEIGHT)
        t0 = time.time()
        pi.train(epochs=PI_SWEEP_EPOCHS)
        pi_train[i, j] = time.time() - t0

        # PI-DDQN inference: 200 single steps
        s2, d2 = random.sample(nodes, 2)
        ag2 = PIDDQNAgent(0, G, s2, d2, EV_PARAMS['battery'],
                          pi.dqn, pi.target, pi.memory, beta=BETA_WEIGHT)
        ag2.eps = 0.0
        inf_t2 = []
        for _ in range(200):
            t0 = time.perf_counter()
            ag2.choose_action()
            inf_t2.append((time.perf_counter() - t0) * 1000)
        pi_inf[i, j] = np.mean(inf_t2)

        print(f"  N={nd:3d} EV={ne:3d} | {ql_train[i,j]:8.1f}s  {ql_inf[i,j]:10.4f}ms | {pi_train[i,j]:8.1f}s  {pi_inf[i,j]:10.4f}ms")

# ── Plotting helper ───────────────────────────────────────────
def make_3d_surface(Z, xlabel, ylabel, zlabel, title, save_path, color='viridis', label_pts=True):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    X_raw = np.array(NODE_COUNTS)
    Y_raw = np.array(EV_COUNTS)
    X, Y = np.meshgrid(X_raw, Y_raw, indexing='ij')

    # Smooth surface with interpolation
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((X_raw, Y_raw), Z, method='linear')
    xi = np.linspace(X_raw.min(), X_raw.max(), 30)
    yi = np.linspace(Y_raw.min(), Y_raw.max(), 30)
    Xi, Yi = np.meshgrid(xi, yi, indexing='ij')
    Zi = interp((Xi, Yi))

    surf = ax.plot_surface(Xi, Yi, Zi, cmap=color, alpha=0.85,
                           linewidth=0, antialiased=True, edgecolor='none')

    # Scatter actual data points
    ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c='red', s=60, zorder=10)

    # Label specific corner/key points matching paper style (A,B,C,D)
    if label_pts:
        labels = ['A', 'B', 'C', 'D']
        key_pts = [(0, 0), (0, 1), (1, 2), (2, 3)]   # (node_idx, ev_idx)
        for lbl, (ni, ei) in zip(labels, key_pts):
            xv, yv, zv = NODE_COUNTS[ni], EV_COUNTS[ei], Z[ni, ei]
            ax.text(xv, yv, zv * 1.05, f'{lbl}({xv},{yv},{zv:.1f})',
                    fontsize=7.5, color='red', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    ax.set_zlabel(zlabel, fontsize=10, labelpad=8)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.view_init(elev=25, azim=-55)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

# ── Generate plots — BASE PAPER (Q-Learning) ─────────────────
print("\n[Generating BASE PAPER plots (Fig 6 equivalent)...]")

make_3d_surface(
    ql_train, 'Number of Nodes', 'Number of EVs', 'Training Time (s)',
    'Fig 6(b): Training Time — Q-Learning (Base Paper)',
    'base_paper/results/fig6b_ql_train_time.png', color='cool'
)
make_3d_surface(
    ql_inf * 1000, 'Number of Nodes', 'Number of EVs', 'Inference Time (µs)',
    'Fig 6(c): Inference Time — Q-Learning (Base Paper)',
    'base_paper/results/fig6c_ql_inference_time.png', color='plasma'
)

# ── Generate plots — OUR RESEARCH (PI-DDQN) ──────────────────
print("\n[Generating OUR RESEARCH plots (PI-DDQN equivalent)...]")

make_3d_surface(
    pi_train, 'Number of Nodes', 'Number of EVs', 'Training Time (s)',
    'Fig 6(b): Training Time — PI-DDQN (Our Research)',
    'my_research/results/fig6b_piddqn_train_time.png', color='viridis'
)
make_3d_surface(
    pi_inf, 'Number of Nodes', 'Number of EVs', 'Inference Time (ms)',
    'Fig 6(c): Inference Time — PI-DDQN (Our Research)',
    'my_research/results/fig6c_piddqn_inference_time.png', color='magma'
)

# ── Side-by-side comparison plot ─────────────────────────────
print("\n[Generating side-by-side comparison figure...]")

fig, axes = plt.subplots(1, 2, figsize=(18, 7), subplot_kw={'projection': '3d'})
X, Y = np.meshgrid(NODE_COUNTS, EV_COUNTS, indexing='ij')

from scipy.interpolate import RegularGridInterpolator
xi = np.linspace(50, 150, 30)
yi = np.linspace(50, 200, 30)
Xi, Yi = np.meshgrid(xi, yi, indexing='ij')

for ax, Z, title, cmap in zip(
    axes,
    [ql_train, pi_train],
    ['Training Time — Q-Learning (Base)', 'Training Time — PI-DDQN (Ours)'],
    ['Blues', 'Greens']
):
    interp = RegularGridInterpolator((np.array(NODE_COUNTS), np.array(EV_COUNTS)), Z)
    Zi = interp((Xi, Yi))
    surf = ax.plot_surface(Xi, Yi, Zi, cmap=cmap, alpha=0.85, linewidth=0)
    ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c='red', s=60, zorder=10)
    labels = ['A', 'B', 'C', 'D']
    key_pts = [(0, 0), (0, 1), (1, 2), (2, 3)]
    for lbl, (ni, ei) in zip(labels, key_pts):
        xv, yv, zv = NODE_COUNTS[ni], EV_COUNTS[ei], Z[ni, ei]
        ax.text(xv, yv, zv * 1.05, f'{lbl}({xv},{yv},{zv:.1f})',
                fontsize=7.5, color='darkred', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
    ax.set_xlabel('Number of Nodes', fontsize=9, labelpad=6)
    ax.set_ylabel('Number of EVs', fontsize=9, labelpad=6)
    ax.set_zlabel('Training Time (s)', fontsize=9, labelpad=6)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.view_init(elev=25, azim=-55)
    fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10, pad=0.1)

plt.suptitle('Fig 6: Training Time Comparison — Q-Learning vs PI-DDQN\n(Computational Overhead by Network Scale)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('my_research/results/fig6_comparison_train_time.png', dpi=200, bbox_inches='tight')
plt.savefig('base_paper/results/fig6_comparison_train_time.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: fig6_comparison_train_time.png (both folders)")

# ── Print final table ─────────────────────────────────────────
print("\n" + "="*65)
print("FINAL TIMING TABLE (seconds)")
print("="*65)
print(f"{'Nodes':>6} {'EVs':>5} | {'QL Train':>10} {'QL Inf(ms)':>12} | {'PI Train':>10} {'PI Inf(ms)':>12} | {'Speedup':>9}")
print("-"*75)
for i, nd in enumerate(NODE_COUNTS):
    for j, ne in enumerate(EV_COUNTS):
        ratio = ql_train[i,j] / pi_train[i,j] if pi_train[i,j] > 0 else 0
        print(f"  {nd:4d} {ne:5d} | {ql_train[i,j]:9.2f}s {ql_inf[i,j]*1000:10.2f}us | "
              f"{pi_train[i,j]:9.2f}s {pi_inf[i,j]:10.4f}ms | x{ratio:.2f} QL faster")
print("\nAll plots saved. Done.", flush=True)
