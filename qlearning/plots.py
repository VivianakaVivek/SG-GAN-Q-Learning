"""
qlearning/plots.py
==================
All figures from the paper's results section.

Figures produced:
  Fig. 3b — Cumulative Reward for 5 EVs vs Epochs
  Fig. 4a — 3D plot: Congestion × Energy × Travel Time
  Fig. 4b — β sensitivity analysis
  Fig. 5a — No. of EVs vs Energy consumption
  Fig. 5b — No. of EVs vs Travel time
  Fig. 6  — Network graph with optimal paths overlaid (wide + dense)
  Bonus   — Q-table heatmap, degree distribution, SOC evolution
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.linewidth":   1.2,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

from qlearning.config import OUTPUT, CONGESTION_LEVELS, QLEARN


def _save(fig, name):
    os.makedirs(OUTPUT["plots"], exist_ok=True)
    path = os.path.join(OUTPUT["plots"], name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved → {path}")


# ── Fig. 3b — Cumulative Reward ───────────────────────────────────────────────

def plot_cumulative_reward(rewards_matrix: np.ndarray, n_evs: int = 5):
    """
    Replicates paper Fig. 3(b):
      X-axis: No. of Epoch (0–10000)
      Y-axis: Cumulative Reward Gained
      EV5, EV4, EV2 higher (~Group II); EV1, EV3 lower (~Group I)
    """
    colours = ["#d62728", "#ff7f0e", "#9467bd", "#17becf", "#2ca02c"]
    labels  = [f"EV {i+1}" for i in range(n_evs)]
    n_ep    = rewards_matrix.shape[1]
    epochs  = np.arange(n_ep)

    # Running cumulative mean (window=50) — matches paper's smooth curves
    window = 50
    smooth = np.zeros_like(rewards_matrix)
    for ev in range(n_evs):
        for ep in range(n_ep):
            s = max(0, ep - window)
            smooth[ev, ep] = rewards_matrix[ev, s:ep+1].mean()

    fig, ax = plt.subplots(figsize=(6, 4.8))
    for ev in range(min(n_evs, 5)):
        ax.plot(epochs, smooth[ev], color=colours[ev], linewidth=1.6, label=labels[ev])

    ax.set_xlabel("No. of Epoch", fontsize=12)
    ax.set_ylabel("Cumulative Reward Gained", fontsize=12)
    ax.set_xlim(0, n_ep)
    ax.set_title("(b)", loc="left", fontsize=12, fontweight="bold")

    # Legend: high → low reward order
    order   = np.argsort(smooth[:n_evs, -1])[::-1]
    handles = [mpatches.Patch(color=colours[i], label=labels[i]) for i in order]
    ax.legend(handles=handles, loc="center right", fontsize=9,
              framealpha=0.9, handlelength=1.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    for sp in ax.spines.values():
        sp.set_linestyle("--")
    plt.tight_layout()
    _save(fig, "fig3b_cumulative_reward.png")


# ── Fig. 4a — 3D Congestion / Energy / Time ───────────────────────────────────

def plot_3d_congestion(cong_energy_time: list):
    """
    Paper Fig. 4(a):
      Points A(100, 16.66, 404.57), B(50, 15.64, 333.33), C(25, 13.83, 280.95)
      X = congestion %, Y = mean energy (kWh), Z = mean travel time (min)
    """
    congs   = [d["cong_pct"]    for d in cong_energy_time]
    energies = [d["energy_mean"] for d in cong_energy_time]
    times    = [d["time_mean"]   for d in cong_energy_time]   # hours

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(congs, energies, times,
                     c=congs, cmap="plasma", s=120, depthshade=True, zorder=5)
    ax.plot(congs, energies, times, "--", color="#888", linewidth=1.2, alpha=0.7)

    for d in cong_energy_time:
        ax.text(d["cong_pct"], d["energy_mean"], d["time_mean"],
                f" {d['label']}", fontsize=9)

    ax.set_xlabel("Congestion (%)",   fontsize=10)
    ax.set_ylabel("Mean Energy (kWh)", fontsize=10)
    ax.set_zlabel("Mean Time (h)",    fontsize=10)
    ax.set_title("Fig. 4(a) — 3D: Congestion × Energy × Travel Time",
                 fontsize=11, fontweight="bold")
    plt.colorbar(sc, ax=ax, shrink=0.5, label="Congestion %")
    plt.tight_layout()
    _save(fig, "fig4a_3d_congestion.png")


# ── Fig. 4b — β sensitivity ───────────────────────────────────────────────────

def plot_beta_sensitivity(beta_records: list):
    """
    Paper Fig. 4(b): β vs energy and travel time.
    Two y-axes: left = energy (kWh), right = time (h).
    """
    betas   = [r["beta"]        for r in beta_records]
    energies = [r["energy_mean"] for r in beta_records]
    times    = [r["time_mean"]   for r in beta_records]

    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(betas, energies, "o-", color="#2ca02c", linewidth=1.8,
             label="Energy (kWh)")
    ax2.plot(betas, times,    "s--", color="#1f77b4", linewidth=1.8,
             label="Travel Time (h)")

    ax1.axvline(0.8, color="#d62728", linestyle=":", linewidth=1.2,
                label="β = 0.8 (selected)")
    ax1.axvline(0.67, color="#ff7f0e", linestyle=":", linewidth=1.2,
                label="β = 0.67 (balanced)")

    ax1.set_xlabel("β (Energy vs Time Weight)", fontsize=12)
    ax1.set_ylabel("Mean Energy (kWh)", fontsize=12, color="#2ca02c")
    ax2.set_ylabel("Mean Travel Time (h)", fontsize=12, color="#1f77b4")
    ax1.set_title("Fig. 4(b) — Sensitivity Analysis on β",
                  fontsize=11, fontweight="bold")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    _save(fig, "fig4b_beta_sensitivity.png")


# ── Fig. 5a & 5b — Scaling with EV count ─────────────────────────────────────

def plot_ev_scaling(scaling_records: list):
    """
    Paper Fig. 5(a) and 5(b):
      X = No. of EVs, Y = mean energy (a) / mean travel time (b)
    """
    n_evs    = [r["n_evs"]       for r in scaling_records]
    energies = [r["energy_mean"] for r in scaling_records]
    times    = [r["time_mean"]   for r in scaling_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(n_evs, energies, "o-", color="#2ca02c", linewidth=1.8, markersize=6)
    ax1.set_xlabel("No. of EVs", fontsize=12)
    ax1.set_ylabel("Mean Energy Consumption (kWh)", fontsize=12)
    ax1.set_title("Fig. 5(a) — No. of EVs vs Energy", fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.3, linestyle="--")
    for sp in ax1.spines.values():
        sp.set_linestyle("--")

    ax2.plot(n_evs, times, "s-", color="#1f77b4", linewidth=1.8, markersize=6)
    ax2.set_xlabel("No. of EVs", fontsize=12)
    ax2.set_ylabel("Mean Travel Time (h)", fontsize=12)
    ax2.set_title("Fig. 5(b) — No. of EVs vs Travel Time", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.3, linestyle="--")
    for sp in ax2.spines.values():
        sp.set_linestyle("--")

    fig.suptitle("EV Fleet Scaling Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig5_ev_scaling.png")


# ── Fig. 6 — Network graph with optimal paths ─────────────────────────────────

def plot_network_with_paths(G, result, cs_nodes, graph_label: str,
                             congestion: float = 0.50,
                             node_color_junc="#4C72B0", node_color_cs="#55A868"):
    """Overlay 5 optimal EV routes on the road network."""
    import networkx as nx

    pos = nx.spring_layout(G, seed=42, k=1.8)
    colours = ["#d62728", "#ff7f0e", "#9467bd", "#17becf", "#2ca02c"]

    cs_set     = set(cs_nodes)
    node_colors = [node_color_cs if n in cs_set else node_color_junc
                   for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Background network
    nx.draw_networkx_edges(G, pos, edge_color="#cccccc", width=0.8,
                           ax=ax, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=120, ax=ax, alpha=0.9)

    # Draw each EV's path
    for ev_i, route in enumerate(result["final_routes"]):
        path_edges = [(route[k], route[k+1])
                      for k in range(len(route)-1) if k+1 < len(route)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                               edge_color=colours[ev_i % 5],
                               width=2.5, ax=ax, alpha=0.85,
                               label=f"EV {ev_i+1}")

    # Labels for CS nodes
    cs_labels = {n: f"CS{i}" for i, n in enumerate(cs_nodes)}
    nx.draw_networkx_labels(G, pos, labels=cs_labels,
                            font_size=7, font_color="white", ax=ax)

    ax.set_title(
        f"Optimal EV Routes — {graph_label.upper()} Network\n"
        f"Congestion = {int(congestion*100)}%  |  "
        f"{len(result['final_routes'])} EVs  |  Bipartite CS Allocation",
        fontsize=11, fontweight="bold"
    )
    handles = [mpatches.Patch(color=colours[i], label=f"EV {i+1}")
               for i in range(len(result["final_routes"]))]
    handles += [mpatches.Patch(color=node_color_cs,   label="Charging Station"),
                mpatches.Patch(color=node_color_junc, label="Junction")]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.axis("off")
    plt.tight_layout()
    _save(fig, f"fig6_routes_{graph_label}_cong{int(congestion*100)}.png")


# ── Bonus: Energy bar chart per EV at each congestion level ───────────────────

def plot_energy_per_ev(results: dict, graph_label: str, bipartite: bool):
    """Bar chart: energy per EV at 25/50/100% congestion."""
    cong_keys = [f"{graph_label}_cong{int(c*100)}_{'bipartite' if bipartite else 'no_bipartite'}"
                 for c in CONGESTION_LEVELS]
    cong_labels = [f"{int(c*100)}%" for c in CONGESTION_LEVELS]

    n_evs = len(results[cong_keys[0]]["energy_per_ev"]) if cong_keys[0] in results else 5
    x     = np.arange(n_evs)
    width = 0.25
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for ci, (key, label) in enumerate(zip(cong_keys, cong_labels)):
        if key not in results:
            continue
        energies = results[key]["energy_per_ev"]
        ax.bar(x + ci * width, energies, width,
               label=f"Congestion {label}", color=colours[ci], alpha=0.85,
               edgecolor="white")

    ax.set_xlabel("EV Index", fontsize=12)
    ax.set_ylabel("Energy Consumption (kWh)", fontsize=12)
    bip_str = "With Bipartite" if bipartite else "Without Bipartite"
    ax.set_title(f"Energy per EV — {graph_label.upper()} Network ({bip_str})",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"EV {i+1}" for i in range(n_evs)])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fname = f"energy_per_ev_{graph_label}_{'bip' if bipartite else 'nobip'}.png"
    _save(fig, fname)


# ── Bonus: Travel time bar chart per EV ───────────────────────────────────────

def plot_time_per_ev(results: dict, graph_label: str, bipartite: bool):
    """Bar chart: travel time per EV at 25/50/100% congestion."""
    cong_keys = [f"{graph_label}_cong{int(c*100)}_{'bipartite' if bipartite else 'no_bipartite'}"
                 for c in CONGESTION_LEVELS]
    cong_labels = [f"{int(c*100)}%" for c in CONGESTION_LEVELS]
    n_evs = len(results[cong_keys[0]]["time_per_ev"]) if cong_keys[0] in results else 5
    x     = np.arange(n_evs)
    width = 0.25
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for ci, (key, label) in enumerate(zip(cong_keys, cong_labels)):
        if key not in results:
            continue
        times_h = results[key]["time_per_ev"] / 3600.0
        ax.bar(x + ci * width, times_h, width,
               label=f"Congestion {label}", color=colours[ci], alpha=0.85,
               edgecolor="white")

    ax.set_xlabel("EV Index", fontsize=12)
    ax.set_ylabel("Travel Time (h)", fontsize=12)
    bip_str = "With Bipartite" if bipartite else "Without Bipartite"
    ax.set_title(f"Travel Time per EV — {graph_label.upper()} Network ({bip_str})",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"EV {i+1}" for i in range(n_evs)])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fname = f"time_per_ev_{graph_label}_{'bip' if bipartite else 'nobip'}.png"
    _save(fig, fname)


# ── Bonus: Summary dashboard (energy + time side-by-side for both networks) ───

def plot_comparison_dashboard(results_wide: dict, results_dense: dict):
    """Side-by-side energy comparison: wide vs dense, bipartite, 50% congestion."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (results, label) in zip(axes, [
        (results_wide,  "Wide Network"),
        (results_dense, "Dense Network"),
    ]):
        key = f"{'wide' if 'wide' in label.lower() else 'dense'}_cong50_bipartite"
        if key not in results:
            ax.set_visible(False)
            continue
        energies = results[key]["energy_per_ev"]
        n_evs    = len(energies)
        colours  = ["#d62728", "#ff7f0e", "#9467bd", "#17becf", "#2ca02c"]
        ax.bar(range(n_evs), energies,
               color=colours[:n_evs], edgecolor="white", alpha=0.88)
        ax.set_xticks(range(n_evs))
        ax.set_xticklabels([f"EV {i+1}" for i in range(n_evs)])
        ax.set_ylabel("Energy (kWh)", fontsize=11)
        ax.set_title(f"{label} — Energy at 50% Congestion (Bipartite)",
                     fontsize=11, fontweight="bold")
        ax.axhline(np.mean(energies), color="#555", linestyle="--",
                   label=f"Mean = {np.mean(energies):.2f} kWh")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Energy Comparison: Wide vs Dense Network (Bipartite CS Allocation)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "comparison_wide_vs_dense.png")
