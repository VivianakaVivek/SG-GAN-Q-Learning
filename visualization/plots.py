"""
visualization/plots.py
All visualizations for the SG-GAN pipeline.
Saves plots to the output directory configured in config.yaml.
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _save(fig, filename: str, config: dict):
    path = os.path.join(config["output"]["plot_dir"], filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {path}")


def plot_real_graph(G: nx.Graph, config: dict, title: str = None):
    """Plot the OSM road network extracted for the configured location."""
    place = config["map"]["place_name"]
    title = title or f"Real Road Network — {place}"
    G = nx.Graph(G) if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) else G

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)
           if "x" in data and "y" in data}

    # Fall back to spring layout if OSM coords are unavailable
    if len(pos) < G.number_of_nodes():
        pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_size=15, node_color="#4FC3F7",
        edge_color="#90CAF9", width=0.8,
        with_labels=False, alpha=0.9,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    _save(fig, "real_graph.png", config)


def plot_generated_graph(gen_adj: np.ndarray, config: dict, title: str = None):
    """Plot a road network synthesized by SG-GAN from its adjacency matrix."""
    place = config["map"]["place_name"]
    title = title or f"SG-GAN Generated Road Network — {place}"

    G_gen = nx.from_numpy_array(gen_adj)
    pos = nx.spring_layout(G_gen, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx(
        G_gen, pos=pos, ax=ax,
        node_size=15, node_color="#A5D6A7",
        edge_color="#66BB6A", width=0.8,
        with_labels=False, alpha=0.9,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    _save(fig, "generated_graph.png", config)


def plot_training_curves(history: dict, config: dict):
    """
    Plot generator and discriminator loss AND accuracy over training epochs.
    Accuracy plot matches Fig. 3(a) of the paper:
      - Generator: 45-65% in first 50 iters → 91% at convergence
      - Discriminator: 40-75% in first 50 iters → 93% at convergence
    """
    epochs = range(1, len(history["loss_G"]) + 1)
    has_acc = "acc_G" in history and len(history["acc_G"]) > 0

    n_cols = 2 if not has_acc else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 2:
        axes = list(axes)

    # Loss plots
    for ax, key, color, label in zip(
        axes[:2],
        ["loss_G", "loss_D"],
        ["#EF5350", "#42A5F5"],
        ["Generator Loss", "Discriminator Loss"],
    ):
        ax.plot(epochs, history[key], color=color, linewidth=1.5)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

    # Accuracy plots (paper Fig. 3a)
    if has_acc:
        paper_targets = {"acc_G": 91, "acc_D": 93}
        for ax, key, color, label in zip(
            axes[2:],
            ["acc_G", "acc_D"],
            ["#FF7043", "#29B6F6"],
            ["Generator Accuracy", "Discriminator Accuracy"],
        ):
            acc_pct = [v * 100 for v in history[key]]
            ax.plot(epochs, acc_pct, color=color, linewidth=1.5)
            # Draw paper target line
            target = paper_targets[key]
            ax.axhline(y=target, color=color, linestyle="--", alpha=0.5,
                       label=f"Paper target: {target}%")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_title(label, fontsize=13, fontweight="bold")
            ax.set_ylim(0, 105)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

    fig.suptitle("SG-GAN Training Curves (Paper: Fig. 3a)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "training_curves.png", config)


def plot_kld_over_training(kld_values: list, config: dict):
    """Plot how KL Divergence decreases during training."""
    fig, ax = plt.subplots(figsize=(9, 5))
    checkpoints = [
        i * (config["sggan"]["epochs"] // len(kld_values))
        for i in range(1, len(kld_values) + 1)
    ]
    ax.plot(checkpoints, kld_values, marker="o", color="#AB47BC", linewidth=2)
    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("KL Divergence", fontsize=12)
    ax.set_title(
        f"KL Divergence (Generated vs Real) — {config['map']['place_name']}",
        fontsize=13, fontweight="bold",
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "kld_over_training.png", config)


def plot_degree_distributions(real_adj: np.ndarray, gen_adj: np.ndarray, config: dict):
    """Compare degree distributions of the real and generated graphs."""
    real_degrees = real_adj.sum(axis=1).astype(int)
    gen_degrees  = gen_adj.sum(axis=1).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, degrees, color, label in zip(
        axes,
        [real_degrees, gen_degrees],
        ["#4FC3F7", "#A5D6A7"],
        ["Real Graph", "Generated Graph"],
    ):
        unique, counts = np.unique(degrees, return_counts=True)
        ax.bar(unique, counts, color=color, edgecolor="white", width=0.7)
        ax.set_xlabel("Node Degree", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Degree Distribution — {label}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, "degree_distributions.png", config)


def plot_summary_dashboard(
    G_real: nx.Graph,
    gen_adj: np.ndarray,
    history: dict,
    kld_final: float,
    config: dict,
):
    """Single dashboard combining all key plots."""
    place = config["map"]["place_name"]
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Real graph
    ax1 = fig.add_subplot(gs[0, 0])
    G_real_simple = nx.Graph(G_real) if isinstance(G_real, (nx.MultiGraph, nx.MultiDiGraph)) else G_real
    pos_real = {node: (data["x"], data["y"]) for node, data in G_real_simple.nodes(data=True)
                if "x" in data and "y" in data}
    if len(pos_real) < G_real_simple.number_of_nodes():
        pos_real = nx.spring_layout(G_real_simple, seed=42)
    nx.draw_networkx(G_real_simple, pos=pos_real, ax=ax1, node_size=8,
                     node_color="#4FC3F7", edge_color="#90CAF9",
                     width=0.6, with_labels=False, alpha=0.9)
    ax1.set_title("Real OSM Network", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # 2. Generated graph
    ax2 = fig.add_subplot(gs[0, 1])
    G_gen = nx.from_numpy_array(gen_adj)
    pos_gen = nx.spring_layout(G_gen, seed=42)
    nx.draw_networkx(G_gen, pos=pos_gen, ax=ax2, node_size=8,
                     node_color="#A5D6A7", edge_color="#66BB6A",
                     width=0.6, with_labels=False, alpha=0.9)
    ax2.set_title("SG-GAN Generated Network", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # 3. KLD annotation
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.text(0.5, 0.6, f"KL Divergence\n(Generated vs Real)",
             ha="center", va="center", fontsize=13, color="#555")
    ax3.text(0.5, 0.35, f"{kld_final:.4f}",
             ha="center", va="center", fontsize=36, fontweight="bold", color="#AB47BC")
    ax3.text(0.5, 0.15, "Lower = More Similar to Real",
             ha="center", va="center", fontsize=10, color="#888")

    # 4. Generator loss
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(history["loss_G"], color="#EF5350", linewidth=1.5)
    ax4.set_title("Generator Loss", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("Loss"); ax4.grid(alpha=0.3)

    # 5. Discriminator loss
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(history["loss_D"], color="#42A5F5", linewidth=1.5)
    ax5.set_title("Discriminator Loss", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Epoch"); ax5.set_ylabel("Loss"); ax5.grid(alpha=0.3)

    # 6. Degree distributions
    ax6 = fig.add_subplot(gs[1, 2])
    gen_deg  = gen_adj.sum(axis=1).astype(int)
    real_adj_np = nx.to_numpy_array(nx.Graph(G_real) if isinstance(G_real, (nx.MultiGraph, nx.MultiDiGraph)) else G_real)
    real_deg = real_adj_np.sum(axis=1).astype(int)
    u_r, c_r = np.unique(real_deg, return_counts=True)
    u_g, c_g = np.unique(gen_deg,  return_counts=True)
    ax6.bar(u_r, c_r, color="#4FC3F7", alpha=0.7, label="Real",      width=0.4, align="edge")
    ax6.bar(u_g, c_g, color="#A5D6A7", alpha=0.7, label="Generated", width=-0.4, align="edge")
    ax6.set_title("Degree Distributions", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Degree"); ax6.set_ylabel("Count")
    ax6.legend(fontsize=9); ax6.grid(axis="y", alpha=0.3)

    fig.suptitle(f"SG-GAN Road Network Summary — {place}",
                 fontsize=15, fontweight="bold", y=1.01)
    _save(fig, "summary_dashboard.png", config)
