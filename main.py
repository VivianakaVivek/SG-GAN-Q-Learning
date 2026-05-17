"""
main.py
─────────────────────────────────────────────────────────────────────────────
SG-GAN Pipeline Entry Point — Dwarka, New Delhi (configurable via config.yaml)

Steps:
  1. Load config
  2. Download/load OSM road network
  3. Compute node embeddings
  4. Train SG-GAN
  5. Generate new road network
  6. Evaluate with KL Divergence
  7. Visualize and save results
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

# ── Local imports ─────────────────────────────────────────────────────────────
from utils.config import load_config, ensure_dirs
from data.extract_map import prepare_graph
from models.embeddings import get_embeddings
from training.train import train_sggan, save_models
from models.generate import generate_graph, build_generated_weight_matrices
from evaluation.metrics import compute_kld, graph_stats
from visualization.plots import (
    plot_real_graph,
    plot_generated_graph,
    plot_training_curves,
    plot_degree_distributions,
    plot_summary_dashboard,
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="SG-GAN Road Network Generation")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--force-reload", action="store_true",
        help="Re-download OSM data even if cached data exists"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training and load existing model weights"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Config ─────────────────────────────────────────────────────────────
    config = load_config(args.config)
    ensure_dirs(config)

    device = get_device()
    print(f"[Device] Using: {device}")
    print(f"[Config] Location: {config['map']['place_name']}")

    # ── 2. OSM graph ──────────────────────────────────────────────────────────
    G, nodes_list, node_idx, adj, dist_mat, time_mat = prepare_graph(
        config, force_reload=args.force_reload
    )
    n = len(nodes_list)
    print(f"[Graph] {n} nodes, {G.number_of_edges()} edges")

    stats = graph_stats(G)
    print(f"[Graph Stats] {json.dumps(stats, indent=2)}")

    # ── 3. Node embeddings ────────────────────────────────────────────────────
    emb_path = os.path.join(config["output"]["data_dir"], "embeddings.npy")
    if os.path.exists(emb_path) and not args.force_reload:
        print("[Embedding] Loading cached embeddings ...")
        embeddings = np.load(emb_path)
    else:
        embeddings = get_embeddings(G, config)
        np.save(emb_path, embeddings)
        print(f"[Embedding] Saved to {emb_path}")

    # ── 4. Train SG-GAN ───────────────────────────────────────────────────────
    if args.skip_train:
        from training.train import load_models
        generator, discriminator = load_models(config, device)
        model_dir = config["output"]["model_dir"]
        history_path = os.path.join(model_dir, "history.json")
        with open(history_path) as f:
            history = json.load(f)
    else:
        generator, discriminator, history = train_sggan(
            embeddings, adj, config, device
        )
        save_models(generator, discriminator, history, config)

    # ── 5. Generate road network ──────────────────────────────────────────────
    print(f"\n[Generate] Synthesizing new road network ({n} nodes) ...")
    gen_adj, gen_embeddings = generate_graph(generator, n, config, device)
    gen_dist, gen_time = build_generated_weight_matrices(gen_adj, dist_mat, time_mat)

    gen_edges = int(gen_adj.sum() // 2)
    print(f"[Generate] Generated graph: {n} nodes, {gen_edges} edges")

    # Save generated adjacency
    data_dir = config["output"]["data_dir"]
    np.save(os.path.join(data_dir, "gen_adj.npy"),  gen_adj)
    np.save(os.path.join(data_dir, "gen_dist.npy"), gen_dist)
    np.save(os.path.join(data_dir, "gen_time.npy"), gen_time)

    # ── 6. Evaluate — KL Divergence ───────────────────────────────────────────
    kld_final = compute_kld(adj, dist_mat, time_mat, gen_adj, gen_dist, gen_time, config)
    print(f"\n[KLD] KL Divergence (generated vs real): {kld_final:.4f}")
    print(f"      (Paper reports ~0.54 as a good result)")

    # Save metrics
    results = {
        "place"     : config["map"]["place_name"],
        "kld_final" : kld_final,
        "graph_stats": stats,
        "generated" : {
            "num_nodes": int(n),
            "num_edges": int(gen_edges),
        },
    }
    results_path = os.path.join(config["output"]["save_dir"], "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved to {results_path}")

    # ── 7. Visualize ──────────────────────────────────────────────────────────
    print("\n[Plots] Generating visualizations ...")
    plot_real_graph(G, config)
    plot_generated_graph(gen_adj, config)
    plot_training_curves(history, config)
    plot_degree_distributions(adj, gen_adj, config)
    plot_summary_dashboard(G, gen_adj, history, kld_final, config)

    print(f"\n{'='*60}")
    print(f"  ✅  Pipeline complete!")
    print(f"  📍  Location  : {config['map']['place_name']}")
    print(f"  📊  KLD Score : {kld_final:.4f}")
    print(f"  📁  Outputs   : {config['output']['save_dir']}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
