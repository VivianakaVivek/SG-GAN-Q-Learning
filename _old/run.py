"""
Main Evaluation Script - Replicating Results from Section VI
=============================================================
Generates all figures and tables from the paper:
- Fig 2(a,b): Wide and Dense SG-GAN road networks
- Fig 3(a): SG-GAN learning accuracy curves
- Fig 3(b): Cumulative reward for 5 EVs
- Fig 4(a): 3D Congestion vs Energy vs Travel Time
- Fig 4(b): Beta sensitivity analysis
- Fig 5(a,b): Scalability analysis (EVs vs Energy/Time)
- Tables III-VII: Comparative results

Reference: "SG-GAN for Electric Vehicle Road Networks and Q-Learning
in Energy Efficient Routing" - Das, Maity, Chakraborty (IEEE T-ITS, 2025)
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

from network_env import NetworkEnvironment, Generator
from routing import MultiAgentRouter, EVAgent, EV_PARAMS

# ============================================================
# Setup
# ============================================================
os.makedirs("results", exist_ok=True)
print("=" * 60)
print("SG-GAN + Q-Learning for EV Routing - Full Replication")
print("=" * 60)

# ============================================================
# Step 1: Train SG-GAN and generate road networks
# ============================================================
print("\n[Step 1] Training SG-GAN...")
env = NetworkEnvironment()
env.fetch_real_data()

# Train GAN (paper uses up to 400 iterations for KLD evaluation)
gen, g_hist, d_hist = env.train_sg_gan(n_nodes=29, max_epochs=401)

# Fig 2(a): Wider network (lower threshold = fewer edges = wider)
print("\n[Step 2] Synthesizing road networks...")
graph_a, kld_a = env.synthesize_graph(gen, n_nodes=29, n_cs=7,
                                       connection_threshold=0.25)
# Fig 2(b): Dense network (higher threshold = more edges = denser)
graph_b, kld_b = env.synthesize_graph(gen, n_nodes=29, n_cs=7,
                                       connection_threshold=0.35)

# ============================================================
# Plot Fig 2: Road Networks
# ============================================================
def plot_network(G, title, filename):
    """Plot a road network graph with CS nodes highlighted."""
    plt.figure(figsize=(14, 10))
    pos = nx.get_node_attributes(G, 'pos')

    # Color nodes: green=CS, blue=junction
    colors = ['#2ecc71' if G.nodes[n].get('is_cs', False) else '#3498db'
              for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=600,
                           edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, edge_color='#7f8c8d', width=1.5, alpha=0.8,
                           arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold',
                            font_color='white')

    # Edge labels
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        d_km = data.get('weight', 0) / 1000.0
        t_min = data.get('time', 0) / 60.0
        edge_labels[(u, v)] = f"{d_km:.1f}km"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=6, font_color='darkred',
                                 bbox=dict(facecolor='white', alpha=0.7,
                                           edgecolor='none', pad=0.3))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=12, label='Junction Nodes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=12, label='Charging Stations'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=11)
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f"results/{filename}.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved results/{filename}.png")


plot_network(graph_a, f"Fig 2(a): Wider Network (KLD={kld_a:.4f})", "fig_2a")
plot_network(graph_b, f"Fig 2(b): Dense Network (KLD={kld_b:.4f})", "fig_2b")

# ============================================================
# Step 3: Train Q-Learning Router
# ============================================================
print("\n[Step 3] Training Q-Learning router on wider network (Fig 2a)...")
start_time = time.time()

router = MultiAgentRouter(graph_a, n_evs=50, beta=0.8)
q_history = router.train(epochs=10000)

train_time = time.time() - start_time
print(f"Training completed in {train_time / 60:.1f} minutes.")

# Evaluate with and without bipartite mapping
e_with_bip, t_with_bip = router.evaluate(epochs=5, use_bipartite=True)
e_without_bip, t_without_bip = router.evaluate(epochs=5, use_bipartite=False)

print(f"\n[Results]")
print(f"  With bipartite CS mapping:    {e_with_bip:.2f} kWh/100km")
print(f"  Without bipartite CS mapping: {e_without_bip:.2f} kWh/100km")
print(f"  KLD (Wider):  {kld_a:.4f}")
print(f"  KLD (Dense):  {kld_b:.4f}")

# ============================================================
# Plot Fig 3(a): SG-GAN Learning Accuracy
# ============================================================
print("\n[Step 4] Generating figures...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fig 3(a) - GAN accuracy curves (first 200 iterations as in paper)
ax1 = axes[0]
n_plot = min(201, len(g_hist))
iterations = np.arange(n_plot)
ax1.plot(iterations, g_hist[:n_plot], label='Generator Accuracy', color='#e74c3c')
ax1.plot(iterations, d_hist[:n_plot], label='Discriminator Accuracy', color='#3498db')
ax1.set_title('Fig 3(a): SG-GAN Learning Accuracy', fontsize=12)
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Fig 3(b) - Cumulative reward for 5 sample EVs
ax2 = axes[1]
# Run a short simulation tracking individual EV rewards
sample_agents = []
nodes = list(graph_a.nodes())
for i in range(5):
    s, d = random.sample(nodes, 2)
    soc = random.uniform(3.0, 10.0)
    agent = EVAgent(i, graph_a, s, d, soc, q_table=router.q_table,
                    alpha=0.0, gamma=0.95, beta=0.8)  # alpha=0: no learning
    sample_agents.append(agent)

# Track cumulative rewards over episodes
n_reward_episodes = 2000
cumulative_rewards = np.zeros((5, n_reward_episodes))
for ep in range(n_reward_episodes):
    for idx, ag in enumerate(sample_agents):
        # Reset agent
        ag.s, ag.dest = random.sample(nodes, 2)
        ag.soc = random.uniform(3.0, 10.0)
        ag.path = [ag.s]
        ag.total_energy = 0
        ag.total_time = 0
        ag.total_distance = 0
        ag.done = False

        ep_reward = 0
        for step in range(50):
            if ag.done:
                break
            old_s = ag.s
            done, e, t, d = ag.step(0.0, defaultdict(int))
            # Approximate reward
            ep_reward += ag.compute_reward(e, t, 0.0, done and ag.s == ag.dest)

        prev = cumulative_rewards[idx][ep - 1] if ep > 0 else 0
        cumulative_rewards[idx][ep] = prev + ep_reward



for i in range(5):
    ax2.plot(range(n_reward_episodes), cumulative_rewards[i],
             label=f'EV {i + 1}', alpha=0.8)
ax2.set_title('Fig 3(b): Cumulative Reward (5 EVs)', fontsize=12)
ax2.set_xlabel('Episodes')
ax2.set_ylabel('Cumulative Reward')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/fig_3.png", dpi=200)
plt.close()
print("  Saved results/fig_3.png")

# ============================================================
# Plot Fig 4(a): 3D Congestion-Energy-Time plot
# ============================================================
fig = plt.figure(figsize=(14, 5))

ax3d = fig.add_subplot(121, projection='3d')
congestion_levels = [25, 50, 100]
energies_cong = []
times_cong = []

for c_pct in congestion_levels:
    n_evs_test = max(1, int(50 * (c_pct / 100.0)))
    test_router = MultiAgentRouter(graph_a, n_evs=n_evs_test, beta=0.8)
    test_router.q_table = router.q_table  # Reuse trained Q-table
    test_router.epsilon = 0.0
    e, t = test_router.evaluate(epochs=3)
    energies_cong.append(e)
    times_cong.append(t)

ax3d.plot(congestion_levels, energies_cong, times_cong, 'b-', alpha=0.5)
ax3d.scatter(congestion_levels, energies_cong, times_cong, c='green',
             marker='o', s=120, zorder=5)
labels_3d = [f'C({c}%)' for c in congestion_levels]
for i, txt in enumerate(labels_3d):
    ax3d.text(congestion_levels[i], energies_cong[i], times_cong[i], f'  {txt}')
ax3d.set_xlabel('Congestion (%)')
ax3d.set_ylabel('Energy (kWh/100km)')
ax3d.set_zlabel('Travel Time (s)')
ax3d.set_title('Fig 4(a): Congestion vs Energy vs Time', fontsize=11)

# ============================================================
# Plot Fig 4(b): Beta sensitivity analysis
# ============================================================
ax_beta = fig.add_subplot(122)
betas = np.linspace(0.1, 0.9, 9)
e_beta, t_beta = [], []

for b in betas:
    test_router = MultiAgentRouter(graph_a, n_evs=50, beta=b)
    test_router.q_table = router.q_table
    test_router.epsilon = 0.0
    e, t = test_router.evaluate(epochs=2)
    e_beta.append(e)
    t_beta.append(t)

ax_beta.plot(betas, e_beta, 'b-o', label='Energy (kWh/100km)')
ax_beta.set_ylabel('Energy (kWh/100km)', color='blue')
ax_beta.set_xlabel('Beta Value')
ax_beta.tick_params(axis='y', labelcolor='blue')

ax_beta2 = ax_beta.twinx()
ax_beta2.plot(betas, t_beta, 'r-s', label='Travel Time (s)')
ax_beta2.set_ylabel('Travel Time (s)', color='red')
ax_beta2.tick_params(axis='y', labelcolor='red')

ax_beta.set_title('Fig 4(b): Sensitivity Analysis on Beta', fontsize=11)
ax_beta.grid(True, alpha=0.3)

# Combined legend
lines1, labels1 = ax_beta.get_legend_handles_labels()
lines2, labels2 = ax_beta2.get_legend_handles_labels()
ax_beta.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.tight_layout()
plt.savefig("results/fig_4.png", dpi=200)
plt.close()
print("  Saved results/fig_4.png")

# ============================================================
# Plot Fig 5: Scalability Analysis
# ============================================================
print("\n[Step 5] Running scalability analysis (Fig 5)...")
ev_counts = [50, 100, 150, 200]
node_counts = [29, 50, 100, 150]
energy_matrix = np.zeros((len(node_counts), len(ev_counts)))
time_matrix = np.zeros((len(node_counts), len(ev_counts)))

# Pre-generate graphs for each node count
graphs_scaled = {}
for n in node_counts:
    if n == 29:
        scaled_gen = gen
    else:
        scaled_gen = Generator(10, n * 2)  # Untrained generator for different sizes
    g, _ = env.synthesize_graph(scaled_gen, n_nodes=n,
                                 n_cs=max(1, int(n * 0.24)),
                                 connection_threshold=0.25)
    graphs_scaled[n] = g

for i, n in enumerate(node_counts):
    g = graphs_scaled[n]
    for j, n_ev in enumerate(ev_counts):
        test_router = MultiAgentRouter(g, n_evs=n_ev, beta=0.8)
        # Quick training for scalability test
        test_router.train(epochs=200)
        e, t = test_router.evaluate(epochs=2)
        energy_matrix[i, j] = e
        time_matrix[i, j] = t
        print(f"  Nodes={n}, EVs={n_ev}: {e:.2f} kWh/100km, {t:.1f}s")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, n in enumerate(node_counts):
    axes[0].plot(ev_counts, energy_matrix[i, :], marker='o', label=f'{n} nodes')
axes[0].set_title('Fig 5(a): No. of EVs vs Energy Consumption', fontsize=12)
axes[0].set_xlabel('Number of EVs')
axes[0].set_ylabel('Energy (kWh/100km)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for i, n in enumerate(node_counts):
    axes[1].plot(ev_counts, time_matrix[i, :], marker='s', label=f'{n} nodes')
axes[1].set_title('Fig 5(b): No. of EVs vs Travel Time', fontsize=12)
axes[1].set_xlabel('Number of EVs')
axes[1].set_ylabel('Travel Time (s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/fig_5.png", dpi=200)
plt.close()
print("  Saved results/fig_5.png")

# ============================================================
# Print Results Tables
# ============================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print(f"\n[KL Divergence - Table VIII]")
print(f"  Wider Network (Graph A): KLD = {kld_a:.4f}")
print(f"  Dense Network (Graph B): KLD = {kld_b:.4f}")

print(f"\n[Energy Consumption Comparison]")
print(f"  With bipartite mapping:    {e_with_bip:.2f} kWh/100km")
print(f"  Without bipartite mapping: {e_without_bip:.2f} kWh/100km")
if e_without_bip > 0:
    savings = ((e_without_bip - e_with_bip) / e_without_bip) * 100
    print(f"  Energy savings:            {savings:.2f}%")

print(f"\n[Training Performance]")
print(f"  Q-Learning training time:  {train_time / 60:.1f} minutes")
print(f"  Q-table entries:           {len(router.q_table)}")

# Table VII: Comparison of routing methods
print(f"\n[Table VII: Comparison of EV Routing Methods]")
df = pd.DataFrame({
    'Method': ['Clustering [28]', 'A* [29]', 'Dijkstra [30]', 'MILP [31]',
               'PSO [32]', 'SARSA [33]', 'DRL [28]',
               'Proposed Q-Learning'],
    'Energy (kWh/100km)': [19.56, 18.6, 18.6, 18.6, 21.6, 36.8, 15.7,
                            round(e_with_bip, 2)],
    'Training Time': ['N/A', 'N/A', 'N/A', '~10 min', '~30 min',
                      '~2 hours', '~5 hours',
                      f'{train_time / 60:.1f} min']
})
print(df.to_string(index=False))

print(f"\n[Energy Convergence Plot]")
# Plot energy convergence over training
fig, ax = plt.subplots(figsize=(10, 5))
energies_per_epoch = []
for e_total, d_km in q_history:
    if d_km > 0:
        energies_per_epoch.append((e_total / d_km) * 100)
    else:
        energies_per_epoch.append(0)

# Smooth with moving average
window = 100
if len(energies_per_epoch) > window:
    smoothed = np.convolve(energies_per_epoch, np.ones(window) / window, mode='valid')
    ax.plot(range(len(smoothed)), smoothed, color='purple', alpha=0.9,
            label='Energy (kWh/100km) - smoothed')
ax.plot(range(len(energies_per_epoch)), energies_per_epoch,
        color='purple', alpha=0.2, linewidth=0.5)
ax.set_title('Energy Convergence Over Training', fontsize=12)
ax.set_xlabel('Epoch')
ax.set_ylabel('Energy (kWh/100km)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/energy_convergence.png", dpi=200)
plt.close()
print("  Saved results/energy_convergence.png")

print("\n" + "=" * 60)
print("All results saved to ./results/")
print("=" * 60)
