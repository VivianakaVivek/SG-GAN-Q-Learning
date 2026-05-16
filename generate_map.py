"""
Generate SG-GAN road network map ONCE and save for both base_paper and my_research.
This ensures both use the EXACT same map for fair comparison.
Run this FIRST before running either folder's run.py.
"""
import sys, os, random, pickle
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from base_paper.network_env import NetworkEnvironment, Generator

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("="*60)
print("GENERATING SHARED SG-GAN MAP (Dwarka Mod)")
print("="*60, flush=True)

env = NetworkEnvironment(lat=MAP_LAT, lon=MAP_LON, dist=MAP_DIST)
env.fetch_real_data()

# Train SG-GAN (401 epochs as per paper)
gen, g_hist, d_hist = env.train_sg_gan(n_nodes=DEFAULT_NODES, max_epochs=GAN_EPOCHS, noise_dim=GAN_NOISE_DIM)

# Generate both network topologies
print("\nGenerating networks...", flush=True)
random.seed(SEED+1); np.random.seed(SEED+1); torch.manual_seed(SEED+1)
graph_a, kld_a = env.synthesize_graph(gen, DEFAULT_NODES, DEFAULT_CS, connection_threshold=0.25)

random.seed(SEED+2); np.random.seed(SEED+2); torch.manual_seed(SEED+2)
graph_b, kld_b = env.synthesize_graph(gen, DEFAULT_NODES, DEFAULT_CS, connection_threshold=0.35)

# Save everything
data = {
    'env_real_edges': env.real_edges,
    'env_real_coords': env.real_coords,
    'env_x_min': env.x_min, 'env_x_max': env.x_max,
    'env_y_min': env.y_min, 'env_y_max': env.y_max,
    'generator_state': gen.state_dict(),
    'graph_a': graph_a, 'kld_a': kld_a,
    'graph_b': graph_b, 'kld_b': kld_b,
    'g_hist': g_hist, 'd_hist': d_hist,
}

save_path = os.path.join(os.path.dirname(__file__), 'shared_map.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(data, f)

print(f"\n{'='*60}")
print(f"Map saved to: {save_path}")
print(f"  Graph A (Wider):  {graph_a.number_of_nodes()}N, {graph_a.number_of_edges()}E, KLD={kld_a:.4f}")
print(f"  Graph B (Dense):  {graph_b.number_of_nodes()}N, {graph_b.number_of_edges()}E, KLD={kld_b:.4f}")
print(f"  GAN epochs: 401, G_acc final: {g_hist[-1]:.1f}%, D_acc final: {d_hist[-1]:.1f}%")
print(f"{'='*60}")
print("Now run:  python -u base_paper\\run.py")
print("Then run: python -u my_research\\run.py", flush=True)
