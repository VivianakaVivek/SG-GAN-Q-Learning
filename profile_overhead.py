"""
Computational Overhead Profiling
Measures training time, inference time, memory, and model size for both models.
"""
import sys, os, time, pickle, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from config import *
from base_paper.network_env import NetworkEnvironment, Generator
from base_paper.routing import MultiAgentRouter, EVAgent
from my_research.pi_ddqn_routing import MultiAgentPIDDQNRouter, PIDDQNAgent
from collections import defaultdict
import torch

# ── Load shared map ──────────────────────────────────────────────
with open('shared_map.pkl', 'rb') as f:
    M = pickle.load(f)
ga = M['graph_a']
nodes = list(ga.nodes())
print("="*60)
print("COMPUTATIONAL OVERHEAD PROFILING")
print("="*60)
print(f"Graph A: {ga.number_of_nodes()} nodes, {ga.number_of_edges()} edges\n")

# ══════════════════════════════════════════════════════════════
# 1. Q-LEARNING MODEL SIZE
# ══════════════════════════════════════════════════════════════
print("--- Q-Learning Model Analysis ---")
n = ga.number_of_nodes()
# State space: (current_node, dest_node, soc_bucket)
state_space = n * n * 10
action_space = n
qtable_entries = state_space * action_space
qtable_bytes = qtable_entries * 8  # float64
print(f"  State space size:       {state_space:,} states")
print(f"  Action space size:      {action_space} actions per state")
print(f"  Max Q-table entries:    {qtable_entries:,}")
print(f"  Q-table memory (worst): {qtable_bytes/1024:.1f} KB = {qtable_bytes/1024/1024:.2f} MB")
print(f"  Training epochs:        {QL_EPOCHS:,}")
print(f"  EVs per episode:        {NUM_EVS_DEFAULT}")
print(f"  Steps per episode:      100")
print(f"  Total EV-steps:         {QL_EPOCHS * NUM_EVS_DEFAULT * 100:,}")

# Train and time QL
print(f"\n  [Timing Q-Learning training — {QL_EPOCHS} epochs]...")
ql = MultiAgentRouter(ga, n_evs=NUM_EVS_DEFAULT, beta=BETA_WEIGHT)
t0 = time.time()
ql.train(epochs=QL_EPOCHS)
ql_train_t = time.time() - t0
# Actual Q-table size after training
actual_qtable = len(ql.qtable)
actual_bytes = actual_qtable * (8 + 8 + 8 + 8 + 8)  # key tuple + float
print(f"  Actual Q-table entries: {actual_qtable:,}")
print(f"  Actual Q-table memory:  {actual_bytes/1024:.1f} KB")
print(f"  Total training time:    {ql_train_t:.1f}s ({ql_train_t/60:.2f} min)")
print(f"  Time per epoch:         {ql_train_t/QL_EPOCHS*1000:.2f} ms/epoch")

# Inference time QL
random.seed(7); s, d = random.sample(nodes, 2)
ag = EVAgent(0, ga, s, d, EV_PARAMS['battery'], ql.qtable, alpha=0, beta=BETA_WEIGHT)
inf_times = []
for _ in range(200):
    t0 = time.perf_counter()
    ag.choose_action(0.0)
    inf_times.append((time.perf_counter()-t0)*1000)
print(f"  Avg inference time:     {np.mean(inf_times):.4f} ms/step")
print(f"  Max inference time:     {np.max(inf_times):.4f} ms/step")

# ══════════════════════════════════════════════════════════════
# 2. PI-DDQN MODEL SIZE
# ══════════════════════════════════════════════════════════════
print("\n--- PI-DDQN Model Analysis ---")
state_dim = n * 2 + 1
action_dim = n
# Count parameters
pi_r = MultiAgentPIDDQNRouter(ga, n_evs=NUM_EVS_DEFAULT, beta=BETA_WEIGHT)
online_params = sum(p.numel() for p in pi_r.dqn.parameters())
target_params = sum(p.numel() for p in pi_r.target.parameters())
total_nn_params = online_params + target_params
nn_memory_bytes = total_nn_params * 4  # float32
# Replay buffer memory
replay_entry_floats = state_dim + 1 + 1 + state_dim + 1 + 1 + action_dim
replay_memory_bytes = PIDDQN_BUFFER_CAP * replay_entry_floats * 4
total_pi_memory = nn_memory_bytes + replay_memory_bytes

print(f"  State vector dim:       {state_dim} ({n} one-hot curr + {n} one-hot dest + 1 SOC)")
print(f"  Action vector dim:      {action_dim}")
print(f"  Network architecture:   {state_dim} -> 128 -> 128 -> {action_dim}")
print(f"  Online DQN params:      {online_params:,}")
print(f"  Target DQN params:      {target_params:,}")
print(f"  Total NN parameters:    {total_nn_params:,}")
print(f"  NN memory (float32):    {nn_memory_bytes/1024:.2f} KB")
print(f"  Replay buffer capacity: {PIDDQN_BUFFER_CAP:,} transitions")
print(f"  Replay buffer memory:   {replay_memory_bytes/1024/1024:.2f} MB")
print(f"  Total PI-DDQN memory:   {total_pi_memory/1024/1024:.2f} MB")
print(f"  Training epochs:        {PIDDQN_EPOCHS}")
print(f"  Batch size:             {PIDDQN_BATCH_SIZE}")
print(f"  Target sync frequency:  every 10 epochs")
print(f"  Physics lambda:         {PIDDQN_PHYS_LAMBDA}")

# Train and time PI-DDQN
print(f"\n  [Timing PI-DDQN training — {PIDDQN_EPOCHS} epochs]...")
pi = MultiAgentPIDDQNRouter(ga, n_evs=NUM_EVS_DEFAULT, beta=BETA_WEIGHT)
t0 = time.time()
pi.train(epochs=PIDDQN_EPOCHS)
pi_train_t = time.time() - t0
print(f"  Total training time:    {pi_train_t:.1f}s ({pi_train_t/60:.2f} min)")
print(f"  Time per epoch:         {pi_train_t/PIDDQN_EPOCHS*1000:.1f} ms/epoch")

# Inference time PI-DDQN
random.seed(7); s2, d2 = random.sample(nodes, 2)
ag2 = PIDDQNAgent(0, ga, s2, d2, EV_PARAMS['battery'],
                  pi.dqn, pi.target, pi.memory, beta=BETA_WEIGHT)
ag2.eps = 0.0
inf_times2 = []
for _ in range(200):
    t0 = time.perf_counter()
    ag2.choose_action()
    inf_times2.append((time.perf_counter()-t0)*1000)
print(f"  Avg inference time:     {np.mean(inf_times2):.4f} ms/step")
print(f"  Max inference time:     {np.max(inf_times2):.4f} ms/step")

# ══════════════════════════════════════════════════════════════
# 3. SCALABILITY OVERHEAD
# ══════════════════════════════════════════════════════════════
print("\n--- Scalability: Training Time vs Network Size ---")
env = NetworkEnvironment()
env.real_edges = M['env_real_edges']; env.real_coords = M['env_real_coords']
env.x_min, env.x_max = M['env_x_min'], M['env_x_max']
env.y_min, env.y_max = M['env_y_min'], M['env_y_max']

scale_results = []
for nd in [29, 50]:
    gen_s = Generator(GAN_NOISE_DIM, nd*2)
    g_s, _ = env.synthesize_graph(gen_s, nd, max(1, int(nd*0.24)), connection_threshold=0.25)

    # QL
    ql_s = MultiAgentRouter(g_s, n_evs=10, beta=BETA_WEIGHT)
    t0 = time.time()
    ql_s.train(epochs=SCALE_TRAIN_EPOCHS_QL)
    ql_t = time.time() - t0
    ql_mem = g_s.number_of_nodes() ** 3 * 10 * 8 / 1024

    # PI
    pi_s = MultiAgentPIDDQNRouter(g_s, n_evs=10, beta=BETA_WEIGHT)
    pi_state = g_s.number_of_nodes() * 2 + 1
    pi_mem = (sum(p.numel() for p in pi_s.dqn.parameters()) * 2 * 4 / 1024)
    t0 = time.time()
    pi_s.train(epochs=SCALE_TRAIN_EPOCHS_PIDDQN)
    pi_t = time.time() - t0

    scale_results.append((nd, g_s.number_of_edges(), ql_t, ql_mem, pi_t, pi_mem))
    print(f"  N={nd:3d}: QL={ql_t:.1f}s ({ql_mem:.0f}KB Q-table) | PI-DDQN={pi_t:.1f}s ({pi_mem:.0f}KB NN)")

# ══════════════════════════════════════════════════════════════
# 4. FINAL SUMMARY PRINT (for copy-paste into markdown)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL NUMBERS FOR MARKDOWN")
print("="*60)
print(f"QL_TRAIN_TIME={ql_train_t:.1f}s, PI_TRAIN_TIME={pi_train_t:.1f}s")
print(f"QL_INF={np.mean(inf_times):.4f}ms, PI_INF={np.mean(inf_times2):.4f}ms")
print(f"QL_ACTUAL_QTABLE={actual_qtable}, PI_NN_PARAMS={total_nn_params}")
print(f"PI_TOTAL_MEM={total_pi_memory/1024/1024:.2f}MB")
for nd, ne, qt, qm, pt, pm in scale_results:
    print(f"N={nd}: QL={qt:.1f}s/{qm:.0f}KB | PI={pt:.1f}s/{pm:.0f}KB")
