"""
evaluation/metrics.py
KL Divergence and other graph similarity metrics.
All parameters are read from config.
"""

import numpy as np
from scipy.stats import entropy
import networkx as nx


def _edge_distance_time_ratios(adj: np.ndarray, dist_mat: np.ndarray, time_mat: np.ndarray) -> np.ndarray:
    """Compute distance-to-time ratio R_uv for every edge in the graph."""
    rows, cols = np.where((adj > 0) & (time_mat > 0))
    ratios = dist_mat[rows, cols] / time_mat[rows, cols]
    return ratios


def _normalize(ratios: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = ratios.min(), ratios.max()
    if mx - mn < 1e-8:
        return np.zeros_like(ratios)
    return (ratios - mn) / (mx - mn)


def compute_kld(
    real_adj: np.ndarray,
    real_dist: np.ndarray,
    real_time: np.ndarray,
    gen_adj: np.ndarray,
    gen_dist: np.ndarray,
    gen_time: np.ndarray,
    config: dict,
) -> float:
    """
    Compute KL Divergence between real and SG-GAN generated road networks
    using the distance-to-time ratio distribution (as in the paper).

    Returns:
        kld : KL divergence scalar (lower = more similar to real)
    """
    kld_cfg = config["kld"]
    n_bins   = kld_cfg["n_bins"]
    smooth   = kld_cfg["smoothing"]

    real_ratios = _edge_distance_time_ratios(real_adj, real_dist, real_time)
    gen_ratios  = _edge_distance_time_ratios(gen_adj, gen_dist, gen_time)

    if len(real_ratios) == 0 or len(gen_ratios) == 0:
        print("[KLD] Warning: empty ratio arrays — returning inf.")
        return float("inf")

    # Normalize using the combined range (as in paper Eq. 3)
    all_ratios = np.concatenate([real_ratios, gen_ratios])
    mn, mx = all_ratios.min(), all_ratios.max()

    def norm(r):
        if mx - mn < 1e-8:
            return np.zeros_like(r)
        return (r - mn) / (mx - mn)

    real_norm = norm(real_ratios)
    gen_norm  = norm(gen_ratios)

    # Build probability distributions via histogram
    bin_edges = np.linspace(0, 1, n_bins + 1)
    p, _ = np.histogram(real_norm, bins=bin_edges, density=False)
    q, _ = np.histogram(gen_norm,  bins=bin_edges, density=False)

    # Normalize to probability and smooth
    p = p.astype(float) + smooth
    q = q.astype(float) + smooth
    p /= p.sum()
    q /= q.sum()

    kld = float(entropy(p, q))
    return kld


def graph_stats(G: nx.Graph) -> dict:
    """Compute basic structural statistics of a graph."""
    # Convert to simple Graph in case G is a MultiGraph (OSM returns MultiGraphs)
    G_simple = nx.Graph(G) if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) else G
    degrees = [d for _, d in G_simple.degree()]
    return {
        "num_nodes"        : G_simple.number_of_nodes(),
        "num_edges"        : G_simple.number_of_edges(),
        "avg_degree"       : float(np.mean(degrees)),
        "density"          : nx.density(G_simple),
        "num_components"   : nx.number_connected_components(G_simple),
        "avg_clustering"   : nx.average_clustering(G_simple),
    }
