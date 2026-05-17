"""
qlearning/run_qlearning.py
==========================
Main entry point for the Q-learning EV routing pipeline.

Steps:
  1. Load SG-GAN generated wide + dense graphs (outputs/data/ + fig2 outputs)
  2. Identify CS nodes (top-degree nodes in each graph)
  3. Run Q-learning for all scenarios (congestion × bipartite)
  4. Generate all Tables (I–VII) and Figures (3b, 4a, 4b, 5, 6)

Run with:
  conda run -n base python -m qlearning.run_qlearning
  (from the BTP_fromscratch directory)
"""

import os
import sys
import json
import pickle
import time
import numpy as np
import networkx as nx

# ── Make sure project root is on path ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlearning.config   import OUTPUT, EV, CS, QLEARN, CONGESTION_LEVELS
from qlearning.simulate import (
    run_all_scenarios, run_beta_sensitivity, run_scaling, _merged_params
)
from qlearning.pi_ddqn import run_piddqn_scenario
from qlearning.tables   import (
    table1, table2, table345, table6, table7
)
from qlearning.plots    import (
    plot_cumulative_reward, plot_3d_congestion, plot_beta_sensitivity,
    plot_ev_scaling, plot_network_with_paths,
    plot_energy_per_ev, plot_time_per_ev, plot_comparison_dashboard,
)


# ── Output directories ────────────────────────────────────────────────────────
for d in OUTPUT.values():
    os.makedirs(d, exist_ok=True)


# ── 1. Load graph data ────────────────────────────────────────────────────────

def load_graph_data():
    """Load real OSM graph + matrices from main pipeline's output."""
    data_dir = "outputs/data"
    with open(os.path.join(data_dir, "graph.pkl"), "rb") as f:
        G_raw = pickle.load(f)
    G_real = nx.Graph(G_raw) if isinstance(G_raw, (nx.MultiGraph, nx.MultiDiGraph)) else G_raw

    dist_mat = np.load(os.path.join(data_dir, "dist_mat.npy"))
    time_mat = np.load(os.path.join(data_dir, "time_mat.npy"))
    embeddings = np.load(os.path.join(data_dir, "embeddings.npy"))
    return G_real, dist_mat, time_mat, embeddings


def generate_network(embeddings, dist_mat, time_mat,
                     target_edges: int, seed: int):
    """Reproduce generate_fig2.py's generation logic inline."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_fig2",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "generate_fig2.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    gen_adj, thr = mod.generate_adj(embeddings, target_edges, seed)
    gen_dist, gen_time = mod.build_weight_matrices(gen_adj, dist_mat, time_mat, seed)
    G = nx.from_numpy_array(gen_adj)
    return G, gen_adj, gen_dist, gen_time, thr


def identify_cs_nodes(G, n_cs: int) -> list:
    """
    CS nodes = top-n_cs nodes by degree (busiest intersections become stations).
    Paper: 7 CS nodes in the network.
    """
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    return sorted_nodes[:n_cs]


# ── 2. Energy per 100 km helper ───────────────────────────────────────────────

def mean_energy_kWh_per_100km(results_dict: dict, graph_label: str,
                               dist_mat: np.ndarray) -> float:
    """Average kWh/100km over all congestion levels with bipartite."""
    vals = []
    for cong in CONGESTION_LEVELS:
        key = f"{graph_label}_cong{int(cong*100)}_bipartite"
        res = results_dict.get(key)
        if res is None:
            continue
        v_ms = EV["v_kmh"] / 3.6
        for energy, time_s in zip(res["energy_per_ev"], res["time_per_ev"]):
            dist_m = time_s * v_ms
            if dist_m > 0:
                vals.append(energy / (dist_m / 1000.0) * 100.0)
    return float(np.mean(vals)) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 65)
    print("  Q-Learning EV Routing — SG-GAN Road Network (Connaught Place)")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading graph data ...")
    G_real, dist_mat, time_mat, embeddings = load_graph_data()
    print(f"    Real graph: {G_real.number_of_nodes()} nodes, "
          f"{G_real.number_of_edges()} edges")

    # ── Generate wide and dense graphs ────────────────────────────────────────
    print("\n[2] Generating Wide & Dense networks ...")
    G_wide,  _, dist_wide,  time_wide,  thr_w = generate_network(
        embeddings, dist_mat, time_mat, target_edges=38, seed=42)
    G_dense, _, dist_dense, time_dense, thr_d = generate_network(
        embeddings, dist_mat, time_mat, target_edges=76, seed=99)

    cs_wide  = identify_cs_nodes(G_wide,  CS["n_cs"])
    cs_dense = identify_cs_nodes(G_dense, CS["n_cs"])

    print(f"    Wide  graph : {G_wide.number_of_nodes()} nodes, "
          f"{G_wide.number_of_edges()} edges  (threshold={thr_w:.4f})")
    print(f"    Dense graph : {G_dense.number_of_nodes()} nodes, "
          f"{G_dense.number_of_edges()} edges  (threshold={thr_d:.4f})")
    print(f"    CS nodes (wide) : {cs_wide}")
    print(f"    CS nodes (dense): {cs_dense}")

    # ── Tables I & II (static from config) ───────────────────────────────────
    print("\n[3] Generating Tables I and II ...")
    table1()
    table2()

    # ── Q-learning: WIDE network — all congestion × bipartite combos ──────────
    print("\n[4] Running Q-learning on WIDE network ...")
    print(f"    ({len(CONGESTION_LEVELS) * 2} scenarios × "
          f"{QLEARN['n_epochs']} epochs each — may take a few minutes)")
    results_wide = run_all_scenarios(G_wide, dist_wide, time_wide,
                                     cs_wide, "wide", seed=42)

    # ── Q-learning: DENSE network ─────────────────────────────────────────────
    print("\n[5] Running Q-learning on DENSE network ...")
    results_dense = run_all_scenarios(G_dense, dist_dense, time_dense,
                                      cs_dense, "dense", seed=99)

    # ── Tables III, IV, V ─────────────────────────────────────────────────────
    print("\n[6] Generating Tables III–V ...")
    table345(results_wide,  dist_wide,  "wide",  "III", bipartite=False)
    table345(results_wide,  dist_wide,  "wide",  "IV",  bipartite=True)
    table345(results_dense, dist_dense, "dense", "V",   bipartite=True)

    # ── Table VI — Energy comparison ──────────────────────────────────────────
    table6(results_wide, dist_wide)

    # ── Table VII — Method comparison ─────────────────────────────────────────
    our_e = mean_energy_kWh_per_100km(results_wide, "wide", dist_wide)
    
    print("\n[6.5] Running PI-DDQN for Method Comparison ...")
    pi_ddqn_res = run_piddqn_scenario(G_wide, dist_wide, time_wide, cs_wide, congestion_level=0.5, use_bipartite=True, seed=42)
    # Average energy per 100km for PI-DDQN
    vals_pi = []
    v_ms = EV["v_kmh"] / 3.6
    for energy, time_s in zip(pi_ddqn_res["energy_per_ev"], pi_ddqn_res["time_per_ev"]):
        dist_m = time_s * v_ms
        if dist_m > 0:
            vals_pi.append(energy / (dist_m / 1000.0) * 100.0)
    pi_ddqn_e = float(np.mean(vals_pi)) if vals_pi else 0.0
    
    table7(our_e, pi_ddqn_e)

    # ── Fig. 3b — Cumulative reward ───────────────────────────────────────────
    print("\n[7] Plotting Fig. 3b — Cumulative Reward ...")
    # Use wide-50% bipartite rewards for Fig. 3b
    key_3b = "wide_cong50_bipartite"
    plot_cumulative_reward(results_wide[key_3b]["rewards_per_ep"],
                           n_evs=EV["n_evs"])

    # ── Fig. 4a — 3D congestion plot ──────────────────────────────────────────
    print("[8] Plotting Fig. 4a — 3D Congestion ...")
    cong_data = []
    labels    = ["C (25%)", "B (50%)", "A (100%)"]
    for cong, lbl in zip(CONGESTION_LEVELS, labels):
        key = f"wide_cong{int(cong*100)}_bipartite"
        res = results_wide[key]
        cong_data.append({
            "cong_pct":    cong * 100,
            "energy_mean": float(res["energy_per_ev"].mean()),
            "time_mean":   float(res["time_per_ev"].mean() / 3600),
            "label":       lbl,
        })
    plot_3d_congestion(cong_data)

    # ── Fig. 4b — β sensitivity ───────────────────────────────────────────────
    print("[9] Running β sensitivity (Fig. 4b) ...")
    beta_recs = run_beta_sensitivity(G_wide, dist_wide, time_wide,
                                     cs_wide, seed=42)
    plot_beta_sensitivity(beta_recs)

    # ── Fig. 5 — EV scaling ───────────────────────────────────────────────────
    print("[10] Running EV scaling (Fig. 5) ...")
    scaling_recs = run_scaling(G_wide, dist_wide, time_wide, cs_wide, seed=42)
    plot_ev_scaling(scaling_recs)

    # ── Fig. 6 — Network with optimal paths ───────────────────────────────────
    print("[11] Plotting Fig. 6 — Routes overlay ...")
    for cong in [0.25, 0.50, 1.00]:
        key_w = f"wide_cong{int(cong*100)}_bipartite"
        key_d = f"dense_cong{int(cong*100)}_bipartite"
        plot_network_with_paths(G_wide,  results_wide[key_w],  cs_wide,
                                "wide",  cong)
        plot_network_with_paths(G_dense, results_dense[key_d], cs_dense,
                                "dense", cong)

    # ── Bonus charts ──────────────────────────────────────────────────────────
    print("[12] Plotting bonus energy/time charts ...")
    plot_energy_per_ev(results_wide,  "wide",  bipartite=False)
    plot_energy_per_ev(results_wide,  "wide",  bipartite=True)
    plot_energy_per_ev(results_dense, "dense", bipartite=True)
    plot_time_per_ev(results_wide,    "wide",  bipartite=True)
    plot_time_per_ev(results_dense,   "dense", bipartite=True)
    plot_comparison_dashboard(results_wide, results_dense)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "wide_network": {
            "nodes":  G_wide.number_of_nodes(),
            "edges":  G_wide.number_of_edges(),
            "cs_nodes": cs_wide,
            "threshold": thr_w,
            "avg_energy_kWh_100km": our_e,
        },
        "dense_network": {
            "nodes":  G_dense.number_of_nodes(),
            "edges":  G_dense.number_of_edges(),
            "cs_nodes": cs_dense,
            "threshold": thr_d,
        },
        "congestion_levels": [int(c*100) for c in CONGESTION_LEVELS],
        "qlearning_params": {k: v for k, v in QLEARN.items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    summary_path = os.path.join(OUTPUT["root"], "qlearning_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ✅  Pipeline complete!")
    print(f"  ⏱️   Runtime: {summary['runtime_s']:.1f}s")
    print(f"  ⚡  Our Q-learning: {our_e:.2f} kWh/100km  (DRL [22]: 15.70)")
    print(f"  📁  Tables : {OUTPUT['tables']}/")
    print(f"  📊  Plots  : {OUTPUT['plots']}/")
    print("=" * 65)
    print("\nFiles generated:")
    for root, _, files in os.walk(OUTPUT["root"]):
        for fn in sorted(files):
            print(f"  {os.path.join(root, fn)}")


if __name__ == "__main__":
    main()
