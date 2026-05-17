"""
models/generate.py
Uses a trained SG-GAN generator to synthesize a new road network graph.
"""

import numpy as np
import torch
import networkx as nx


def generate_graph(
    generator,
    n_nodes: int,
    config: dict,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple:
    """
    Generate a new road network adjacency matrix using the trained Generator.

    Strategy:
      For each pair of nodes (i, j), generate an embedding for j
      conditioned on i's noise, then ask the Discriminator-like heuristic
      whether an edge should exist.
      Here we use a simpler generative strategy:
        - Generate n_nodes synthetic embeddings from noise.
        - Build edges by cosine similarity threshold.

    Args:
        generator  : trained GraphGenerator
        n_nodes    : number of nodes in the generated graph
        config     : config dict
        device     : torch device
        threshold  : cosine similarity threshold for connecting nodes

    Returns:
        gen_adj       : (n_nodes x n_nodes) binary adjacency matrix
        gen_embeddings: (n_nodes x output_dim) numpy array
    """
    sg_cfg = config["sggan"]
    noise_dim = sg_cfg["noise_dim"]

    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_nodes, noise_dim, device=device)
        embeddings = generator(z).cpu().numpy()   # (n_nodes, output_dim)

    # Compute pairwise cosine similarities
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms
    sim_matrix = normed @ normed.T   # (n_nodes x n_nodes)

    # Build adjacency: connect if similarity > threshold
    gen_adj = (sim_matrix > threshold).astype(np.float32)
    np.fill_diagonal(gen_adj, 0)  # no self-loops

    # Ensure the generated graph is connected (add edges to isolated nodes)
    G = nx.from_numpy_array(gen_adj)
    for node in list(nx.isolates(G)):
        # Connect isolated node to its most similar non-self node
        sim_row = sim_matrix[node].copy()
        sim_row[node] = -1
        best = int(np.argmax(sim_row))
        gen_adj[node, best] = gen_adj[best, node] = 1.0

    return gen_adj, embeddings


def build_generated_weight_matrices(gen_adj: np.ndarray, real_dist: np.ndarray, real_time: np.ndarray):
    """
    Assign plausible distance and travel-time weights to the generated graph
    by sampling from the real graph's edge weight distributions.

    Returns:
        gen_dist : (n x n) distance matrix
        gen_time : (n x n) time matrix
    """
    # Sample distances and times from the real graph's non-zero edges
    real_dists = real_dist[real_dist > 0]
    real_times = real_time[real_time > 0]

    if len(real_dists) == 0 or len(real_times) == 0:
        return gen_adj.copy(), gen_adj.copy()

    n = gen_adj.shape[0]
    gen_dist = np.zeros((n, n), dtype=np.float32)
    gen_time = np.zeros((n, n), dtype=np.float32)

    rows, cols = np.where(gen_adj > 0)
    rng = np.random.default_rng(seed=42)

    sampled_dists = rng.choice(real_dists, size=len(rows), replace=True)
    sampled_times = rng.choice(real_times, size=len(rows), replace=True)

    for idx, (i, j) in enumerate(zip(rows, cols)):
        gen_dist[i, j] = gen_dist[j, i] = sampled_dists[idx]
        gen_time[i, j] = gen_time[j, i] = sampled_times[idx]

    return gen_dist, gen_time
