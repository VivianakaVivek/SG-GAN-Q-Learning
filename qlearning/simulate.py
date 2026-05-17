"""
qlearning/simulate.py
=====================
Vectorised multi-agent Q-learning simulation (Algorithm 2).

All per-episode operations use numpy arrays; Python loops are only over
path steps (≤ 3×n_nodes), not 10k epochs — runs in seconds.

Produces all data for every paper table and figure.
"""

import numpy as np
import networkx as nx

from qlearning.environment import RoadNetworkEnv
from qlearning.agent       import EVAgent
from qlearning.config      import PHYSICS, EV, CS, QLEARN, CONGESTION_LEVELS


def _merged_params(extra: dict = None) -> dict:
    """Merge all config dicts into one flat dict for convenience."""
    p = {}
    for d in (PHYSICS, EV, CS, QLEARN):
        p.update(d)
    if extra:
        p.update(extra)
    return p


def run_qlearning(G, dist_mat, time_mat, cs_nodes,
                  congestion_level: float,
                  use_bipartite: bool = True,
                  n_evs: int = None,
                  extra_params: dict = None,
                  seed: int = 42):
    """
    Numpy-vectorised Q-learning: entire episode loop runs without Python
    inner loops — uses pre-sampled random walk matrices.
    """
    params = _merged_params(extra_params)
    if n_evs is None:
        n_evs = params["n_evs"]

    n_nodes   = G.number_of_nodes()
    n_epochs  = params["n_epochs"]
    alpha     = params["alpha"]
    gamma_    = params["gamma"]
    beta      = params["beta"]
    eps_decay = params["epsilon_decay"]
    eps_min   = params["epsilon_min"]
    R_base    = params["R_base"]
    K_obj     = params["K_obj"]
    K_cong    = params["K_congestion"]
    K_bonus   = params["K_bonus"]
    v_ms      = params["v_kmh"] / 3.6
    B_max     = params["battery_kWh"]
    E_norm    = B_max
    T_norm    = float(time_mat[time_mat > 0].max() * 20) if time_mat.max() > 0 else 1.0
    crit_soc  = params["energy_buffer_kWh"] / 10.0 * B_max
    C_ij_cap  = 10.0
    ph        = params

    # Adjacency (n×n bool) and edge lists
    adj_mat = nx.to_numpy_array(G).astype(bool)   # (n, n)
    # Row-wise neighbour lists as ragged — stored as padded matrix
    max_deg = int(adj_mat.sum(axis=1).max()) + 1
    nbr_mat = -np.ones((n_nodes, max_deg), dtype=np.int32)   # -1 = padding
    nbr_cnt = np.zeros(n_nodes, dtype=np.int32)
    for node in range(n_nodes):
        nbrs = np.where(adj_mat[node])[0]
        nbr_mat[node, :len(nbrs)] = nbrs
        nbr_cnt[node] = len(nbrs)

    cs_set = set(cs_nodes)
    cs_arr = np.array(cs_nodes, dtype=np.int32)

    # Precompute energy & time matrices (Eq. 6-7)
    t_mat  = np.where(dist_mat > 0, dist_mat / (v_ms + 1e-9), 0.0)
    aero   = 0.5 * ph["rho"] * ph["Cd"] * ph["A"] * v_ms**2
    roll   = ph["Cr"] * ph["m"] * ph["g"] * v_ms
    incl_f = ph["m"] * ph["g"] * dist_mat * np.sin(ph["theta"])
    E_mat  = (1.0 / ph["eta"]) * ((aero + roll) * t_mat + incl_f) / 3_600_000.0

    rng = np.random.default_rng(seed)

    # Fixed starts / dests (arbitrary start, fixed dest)
    non_cs = np.array([n for n in range(n_nodes) if n not in cs_set], dtype=np.int32)
    starts = rng.choice(non_cs, size=n_evs, replace=True)
    dests  = np.array([
        int(rng.choice([n for n in non_cs if n != s])) for s in starts
    ], dtype=np.int32)

    # Q-tables init to zero (Algorithm 2 line 3)
    Q = np.zeros((n_evs, n_nodes, n_nodes), dtype=np.float64)

    epsilons   = np.ones(n_evs)
    rewards_ep = np.zeros((n_evs, n_epochs))
    max_steps  = n_nodes * 3

    last_routes = [[] for _ in range(n_evs)]
    last_energy = np.zeros(n_evs)
    last_time   = np.zeros(n_evs)

    for ep in range(n_epochs):
        # SOC sampling (clipped normal, Sec. VI)
        soc_pct = np.clip(
            rng.normal(params["soc_mean_pct"], params["soc_std_pct"], n_evs),
            params["soc_min_pct"], params["soc_max_pct"]
        )
        soc = soc_pct * B_max                     # (n_evs,)

        # Poisson congestion (Eq. 9)
        q_ij     = float(min(rng.poisson(congestion_level * C_ij_cap), C_ij_cap))
        delta_ij = q_ij / C_ij_cap

        states    = starts.copy()                  # (n_evs,)
        done      = np.zeros(n_evs, dtype=bool)
        ep_reward = np.zeros(n_evs)
        ep_energy = np.zeros(n_evs)
        ep_time   = np.zeros(n_evs)
        routes    = [[int(s)] for s in states]

        # Pre-sample random numbers for the whole episode
        rand_explore = rng.random((max_steps, n_evs))    # explore vs exploit
        rand_nbr_idx = rng.integers(0, max(max_deg, 1), (max_steps, n_evs))  # random nbr index

        for step in range(max_steps):
            if done.all():
                break

            for ev in range(n_evs):
                if done[ev]:
                    continue
                s   = int(states[ev])
                cnt = int(nbr_cnt[s])
                if cnt == 0:
                    done[ev] = True
                    continue

                # e-greedy (Eq. 15-16) using pre-sampled randoms
                if rand_explore[step, ev] < epsilons[ev]:
                    nbr_i = int(rand_nbr_idx[step, ev]) % cnt
                    a = int(nbr_mat[s, nbr_i])
                else:
                    valid_nbrs = nbr_mat[s, :cnt]
                    a = int(valid_nbrs[np.argmax(Q[ev, s, valid_nbrs])])

                # Physics (Eq. 6, 7)
                ec = float(E_mat[s, a])
                tc = float(t_mat[s, a])

                # CS allocation
                needs_cs = (soc[ev] - ec) < crit_soc
                if needs_cs and len(cs_arr) > 0:
                    if use_bipartite:
                        cs_node = int(cs_arr[np.argmin(dist_mat[a, cs_arr])])
                    else:
                        cs_node = int(rng.choice(cs_arr))
                    soc[ev]  = max(0.0, soc[ev] - ec)
                    deficit  = max(0.0, B_max - soc[ev])
                    tc      += (deficit / params["eta_s"]) * 3600.0   # Eq. 7
                    soc[ev]  = B_max
                else:
                    soc[ev] = max(0.0, soc[ev] - ec)

                # Reward (Eq. 12)
                reached = (a == int(dests[ev]))
                cost = beta * (ec / E_norm) + (1.0 - beta) * (tc / T_norm)
                r    = R_base - cost * K_obj - delta_ij * K_cong
                if reached:
                    r += K_bonus

                # Bellman update (Eq. 13)
                Q[ev, s, a] += alpha * (r + gamma_ * np.max(Q[ev, a]) - Q[ev, s, a])

                states[ev]    = a
                ep_reward[ev] += r
                ep_energy[ev] += ec
                ep_time[ev]   += tc
                routes[ev].append(a)
                if reached:
                    done[ev] = True

        # Epsilon decay
        epsilons = np.maximum(epsilons * eps_decay, eps_min)
        rewards_ep[:, ep] = ep_reward

        if ep == n_epochs - 1:
            last_routes = routes
            last_energy = ep_energy.copy()
            last_time   = ep_time.copy()

    return {
        "rewards_per_ep": rewards_ep,
        "final_routes":   last_routes,
        "energy_per_ev":  last_energy,
        "time_per_ev":    last_time,
        "global_cong":    congestion_level,
        "Q":              Q,
    }


def run_all_scenarios(G, dist_mat, time_mat, cs_nodes,
                      graph_label: str, seed: int = 42):
    """Run all congestion x bipartite combos for Tables III-V."""
    results = {}
    n_evs = _merged_params()["n_evs"]

    for cong in CONGESTION_LEVELS:
        key_nb = f"{graph_label}_cong{int(cong*100)}_no_bipartite"
        key_b  = f"{graph_label}_cong{int(cong*100)}_bipartite"
        print(f"  Running {key_nb} ...")
        results[key_nb] = run_qlearning(G, dist_mat, time_mat, cs_nodes,
                                        congestion_level=cong, use_bipartite=False,
                                        n_evs=n_evs, seed=seed)
        print(f"  Running {key_b} ...")
        results[key_b]  = run_qlearning(G, dist_mat, time_mat, cs_nodes,
                                        congestion_level=cong, use_bipartite=True,
                                        n_evs=n_evs, seed=seed)
    return results


def run_beta_sensitivity(G, dist_mat, time_mat, cs_nodes,
                         beta_values=None, seed: int = 42):
    """Fig. 4b — vary beta. Uses 2000 epochs (sufficient for convergence trends)."""
    if beta_values is None:
        beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.67, 0.7, 0.8, 0.9]
    records = []
    for beta in beta_values:
        r = run_qlearning(G, dist_mat, time_mat, cs_nodes,
                          congestion_level=0.50, use_bipartite=True,
                          extra_params={"beta": beta, "n_epochs": 2000}, seed=seed)
        records.append({"beta": beta,
                         "energy_mean": float(r["energy_per_ev"].mean()),
                         "time_mean":   float(r["time_per_ev"].mean() / 3600)})
    return records


def run_scaling(G, dist_mat, time_mat, cs_nodes,
                ev_counts=None, seed: int = 42):
    """Fig. 5 — vary EV count. Uses 2000 epochs."""
    if ev_counts is None:
        ev_counts = [5, 10, 20, 30, 50]
    records = []
    for n in ev_counts:
        r = run_qlearning(G, dist_mat, time_mat, cs_nodes,
                          congestion_level=0.50, use_bipartite=True,
                          extra_params={"n_epochs": 2000},
                          n_evs=n, seed=seed)
        records.append({"n_evs": n,
                         "energy_mean": float(r["energy_per_ev"].mean()),
                         "time_mean":   float(r["time_per_ev"].mean() / 3600)})
    return records
