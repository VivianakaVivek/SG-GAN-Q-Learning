"""
qlearning/tables.py
===================
Generate all paper tables as formatted text files + CSV.

Tables produced:
  Table I   — Physical parameters (from config)
  Table II  — Q-learning hyperparameters (from config)
  Table III — Optimal paths WITHOUT bipartite (wide network, 3 congestion levels)
  Table IV  — Optimal paths WITH bipartite    (wide network, 3 congestion levels)
  Table V   — Optimal paths WITH bipartite    (dense network, 3 congestion levels)
  Table VI  — Energy comparison: proposed vs DRL baseline [22]
  Table VII — EV routing method comparison
"""

import os
import csv
import numpy as np
import networkx as nx

from qlearning.config import (
    PHYSICS, EV, CS, QLEARN, CONGESTION_LEVELS, OUTPUT
)


def _mkdirs():
    os.makedirs(OUTPUT["tables"], exist_ok=True)


def _write_csv(rows, headers, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def _write_txt(lines, path):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Table I ───────────────────────────────────────────────────────────────────

def table1():
    _mkdirs()
    rows = [
        ["η  (efficiency)",          PHYSICS["eta"],      "dimensionless"],
        ["ρ  (air density)",          PHYSICS["rho"],      "kg/m³"],
        ["Cd (drag coefficient)",     PHYSICS["Cd"],       "dimensionless"],
        ["A  (frontal area)",         PHYSICS["A"],        "m²"],
        ["Cr (rolling resistance)",   PHYSICS["Cr"],       "dimensionless"],
        ["m  (vehicle mass)",         PHYSICS["m"],        "kg"],
        ["g  (gravity)",              PHYSICS["g"],        "m/s²"],
        ["θ  (road incline)",         PHYSICS["theta"],    "rad"],
        ["v  (cruise speed)",         EV["v_kmh"],         "km/h"],
        ["B_max (battery capacity)",  EV["battery_kWh"],   "kWh"],
        ["Energy buffer",             EV["energy_buffer_kWh"], "kWh"],
        ["η_s (charging rate)",       CS["eta_s"],         "kW"],
        ["C_j (charger capacity)",    CS["charger_cap"],   "chargers/CS"],
        ["λ  (EV arrival rate)",      CS["lambda_arr"],    "EVs/hour"],
    ]
    headers = ["Parameter", "Value", "Unit"]
    _write_csv(rows, headers, os.path.join(OUTPUT["tables"], "table1_parameters.csv"))
    lines = ["TABLE I — Parameters Related to Energy Calculation", "=" * 55]
    lines += [f"  {r[0]:<35} {str(r[1]):<10} {r[2]}" for r in rows]
    _write_txt(lines, os.path.join(OUTPUT["tables"], "table1_parameters.txt"))
    print(f"[Table I]  Saved")


# ── Table II ──────────────────────────────────────────────────────────────────

def table2():
    _mkdirs()
    rows = [
        ["α  (learning rate)",         QLEARN["alpha"]],
        ["γ  (discount factor)",        QLEARN["gamma"]],
        ["ε  (initial exploration)",    QLEARN["epsilon"]],
        ["ε_min",                       QLEARN["epsilon_min"]],
        ["ε decay",                     QLEARN["epsilon_decay"]],
        ["N epochs",                    QLEARN["n_epochs"]],
        ["β  (energy/time weight)",     QLEARN["beta"]],
        ["R_base",                      QLEARN["R_base"]],
        ["K_obj",                       QLEARN["K_obj"]],
        ["K_congestion",                QLEARN["K_congestion"]],
        ["K_bonus",                     QLEARN["K_bonus"]],
        ["No. of EVs (experiment)",     EV["n_evs"]],
        ["No. of EVs (total fleet)",    EV["n_evs_total"]],
        ["No. of CS nodes",             CS["n_cs"]],
    ]
    headers = ["Hyperparameter", "Value"]
    _write_csv(rows, headers, os.path.join(OUTPUT["tables"], "table2_hyperparams.csv"))
    lines = ["TABLE II — Hyperparameters of the Proposed Strategy", "=" * 50]
    lines += [f"  {r[0]:<38} {r[1]}" for r in rows]
    _write_txt(lines, os.path.join(OUTPUT["tables"], "table2_hyperparams.txt"))
    print(f"[Table II] Saved")


# ── Tables III, IV, V helper ──────────────────────────────────────────────────

def _path_string(route: list) -> str:
    """Format node list as 'N0→N1→...→Nk'."""
    return " → ".join(str(n) for n in route)


def _energy_kWh_per_100km(energy_kWh: float, total_dist_m: float) -> float:
    if total_dist_m < 1:
        return 0.0
    return energy_kWh / (total_dist_m / 1000.0) * 100.0


def table345(results: dict, dist_mat: np.ndarray,
             graph_label: str, table_num: str,
             bipartite: bool, cong_label: str = None):
    """
    Build one table (III, IV, or V) for a given graph/bipartite combination.

    `results` is the output of simulate.run_qlearning(...).
    """
    _mkdirs()
    congestion_keys = [k for k in CONGESTION_LEVELS]

    rows_all = []
    for cong in CONGESTION_LEVELS:
        key = f"{graph_label}_cong{int(cong*100)}_{'bipartite' if bipartite else 'no_bipartite'}"
        res = results.get(key)
        if res is None:
            continue

        for ev_i, (route, energy, time_s) in enumerate(
            zip(res["final_routes"], res["energy_per_ev"], res["time_per_ev"])
        ):
            # Compute path distance
            dist_m = sum(
                dist_mat[route[k], route[k+1]]
                for k in range(len(route)-1)
                if k+1 < len(route)
            )
            rows_all.append([
                f"EV {ev_i+1}",
                f"{int(cong*100)}%",
                _path_string(route),
                f"{energy:.4f}",
                f"{time_s/3600:.4f}",
                f"{_energy_kWh_per_100km(energy, dist_m):.2f}",
            ])

    headers = ["EV", "Congestion", "Optimal Path", "Energy (kWh)",
               "Time (h)", "Energy (kWh/100km)"]
    bip_str  = "bipartite" if bipartite else "no_bipartite"
    fname    = f"table{table_num}_{graph_label}_{bip_str}"
    _write_csv(rows_all, headers, os.path.join(OUTPUT["tables"], fname + ".csv"))

    title = (f"TABLE {table_num} — Optimal Paths at Various Congestion "
             f"{'WITH' if bipartite else 'WITHOUT'} Bipartite Graph "
             f"[{graph_label.upper()}]")
    lines = [title, "=" * len(title)]
    lines.append(f"  {'EV':<6} {'Cong':<7} {'Energy':>10} {'Time(h)':>9} {'kWh/100km':>11}  Path")
    lines.append("-" * 100)
    for r in rows_all:
        lines.append(
            f"  {r[0]:<6} {r[1]:<7} {r[3]:>10} {r[4]:>9} {r[5]:>11}  {r[2]}"
        )
    _write_txt(lines, os.path.join(OUTPUT["tables"], fname + ".txt"))
    print(f"[Table {table_num}] Saved ({graph_label}, {'bipartite' if bipartite else 'no bipartite'})")


# ── Table VI — Energy comparison (proposed vs [22]) ───────────────────────────

def table6(results_wide: dict, dist_mat: np.ndarray):
    """
    Compare average energy (kWh/100km) with/without bipartite and vs DRL [22].
    Paper: without=18.41, with=15.64 (15.05% savings), DRL=15.7.
    """
    _mkdirs()
    rows = []
    for cong in CONGESTION_LEVELS:
        key_nb = f"wide_cong{int(cong*100)}_no_bipartite"
        key_b  = f"wide_cong{int(cong*100)}_bipartite"
        res_nb = results_wide.get(key_nb)
        res_b  = results_wide.get(key_b)

        def avg_e100(res):
            if res is None:
                return 0.0
            energies = res["energy_per_ev"]
            total_e  = float(energies.mean())
            # Approximate 100km distance from time and speed
            v_ms = EV["v_kmh"] / 3.6
            dist_m_avg = float(np.mean(res["time_per_ev"])) * v_ms
            return _energy_kWh_per_100km(total_e, dist_m_avg)

        e_nb = avg_e100(res_nb)
        e_b  = avg_e100(res_b)
        saving = (e_nb - e_b) / (e_nb + 1e-9) * 100
        rows.append([f"{int(cong*100)}%", f"{e_nb:.2f}", f"{e_b:.2f}",
                     f"{saving:.2f}%", "15.70"])   # DRL [22] = 15.7 kWh/100km

    headers = ["Congestion", "No Bipartite (kWh/100km)", "Bipartite (kWh/100km)",
               "Savings", "DRL [22] (kWh/100km)"]
    _write_csv(rows, headers, os.path.join(OUTPUT["tables"], "table6_comparison.csv"))
    lines = ["TABLE VI — Comparison of Energy Consumption vs Congestion (Proposed vs [22])",
             "=" * 80]
    lines.append(f"  {'Cong':<10} {'No Bipartite':>14} {'Bipartite':>12} {'Savings':>10} {'DRL [22]':>10}")
    lines.append("-" * 60)
    for r in rows:
        lines.append(f"  {r[0]:<10} {r[1]:>14} {r[2]:>12} {r[3]:>10} {r[4]:>10}")
    _write_txt(lines, os.path.join(OUTPUT["tables"], "table6_comparison.txt"))
    print("[Table VI] Saved")


# ── Table VII — Method comparison ─────────────────────────────────────────────

def table7(our_energy_kWh_100km: float, pi_ddqn_energy: float):
    """Reproduce Table VII from paper with our computed value."""
    _mkdirs()
    rows = [
        ["Clustering [28]",         "19.56", "Cannot adapt to new traffic", "—"],
        ["A* Search [29]",          "18.60", "Static graph; no real-time", "~1 min"],
        ["Dijkstra [30]",           "18.60", "Static graph; no real-time", "~1 min"],
        ["MILP [31]",               "18.60", "Not scalable",               "~10 min"],
        ["PSO [32]",                "21.60", "Suboptimal convergence",      "~30 min"],
        ["SARSA [33]",              "36.80", "High sample complexity",      "~2 hrs (500 ep)"],
        ["DRL [22]",                "15.70", "Requires long training",      "~5 hrs (6000 ep)"],
        ["Proposed Q-learning",     f"{our_energy_kWh_100km:.2f}",
                                    "Decentralised, real-time adaptive", "~3.5 hrs (10k ep)"],
        ["Proposed PI-DDQN",        f"{pi_ddqn_energy:.2f}",
                                    "Proportional-Integral target DDQN", "~10 min (1.5k ep)"],
    ]
    headers = ["Method", "Avg Energy (kWh/100km)", "Limitation", "Convergence Time"]
    _write_csv(rows, headers, os.path.join(OUTPUT["tables"], "table7_method_comparison.csv"))
    lines = ["TABLE VII — Comparison of Different EV Routing Methods", "=" * 80]
    lines.append(f"  {'Method':<25} {'Energy':>10} {'Time':>20}  Notes")
    lines.append("-" * 80)
    for r in rows:
        lines.append(f"  {r[0]:<25} {r[1]:>10} {r[3]:>20}  {r[2]}")
    _write_txt(lines, os.path.join(OUTPUT["tables"], "table7_method_comparison.txt"))
    print("[Table VII] Saved")
