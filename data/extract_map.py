"""
data/extract_map.py
Downloads and preprocesses the OSM road network for the configured location.
"""

import osmnx as ox
import networkx as nx
import numpy as np
import pickle
import os


def download_graph(config: dict) -> nx.Graph:
    """Download and simplify the road network from OpenStreetMap.
    Supports two modes:
      - use_point=True : download a circular area around (lat, lon) with given radius_m
      - use_point=False: download by place name polygon (default)
    """
    map_cfg = config["map"]
    use_point = map_cfg.get("use_point", False)

    if use_point:
        lat = map_cfg["lat"]
        lon = map_cfg["lon"]
        radius = map_cfg["radius_m"]
        print(f"[OSM] Downloading road network around "
              f"({lat}, {lon}) radius={radius}m  [{map_cfg['place_name']}]")
        G = ox.graph_from_point(
            (lat, lon),
            dist=radius,
            network_type=map_cfg["network_type"],
            simplify=map_cfg["simplify"],
        )
    else:
        print(f"[OSM] Downloading road network for: {map_cfg['place_name']}")
        G = ox.graph_from_place(
            map_cfg["place_name"],
            network_type=map_cfg["network_type"],
            simplify=map_cfg["simplify"],
        )

    G = ox.convert.to_undirected(G)
    print(f"[OSM] Raw graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def subsample_graph(G: nx.Graph, max_nodes: int) -> nx.Graph:
    """
    Extract a dense connected subgraph of exactly max_nodes nodes.

    Strategy — Greedy Densest Subgraph:
      1. Start with the highest-degree node.
      2. At each step, add the unvisited neighbor that brings the MOST
         new edges into the current subgraph.
      3. Repeat until max_nodes reached.

    This produces a mesh-like, densely-connected topology — matching the
    paper's "mesh-type road network" — unlike BFS which creates a tree.
    """
    if G.number_of_nodes() <= max_nodes:
        return G

    # Work on largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G_cc = G.subgraph(largest_cc).copy()

    # Seed: highest-degree node
    seed = max(G_cc.degree(), key=lambda x: x[1])[0]
    selected = {seed}

    while len(selected) < max_nodes:
        # Candidate nodes: neighbors of any already-selected node
        candidates = set()
        for node in selected:
            candidates.update(G_cc.neighbors(node))
        candidates -= selected

        if not candidates:
            # No more neighbors — fall back to any remaining node
            remaining = set(G_cc.nodes()) - selected
            if not remaining:
                break
            candidates = remaining

        # Pick candidate that adds the most edges into `selected`
        best_node = max(
            candidates,
            key=lambda v: sum(1 for nb in G_cc.neighbors(v) if nb in selected)
        )
        selected.add(best_node)

    H = G_cc.subgraph(selected).copy()
    print(f"[OSM] Dense subgraph: {H.number_of_nodes()} nodes, "
          f"{H.number_of_edges()} edges  "
          f"(avg degree={2*H.number_of_edges()/H.number_of_nodes():.2f})")
    return H


def build_matrices(G: nx.Graph, config: dict):
    """
    Build adjacency and edge weight matrices (distance, time) from the graph.

    Returns:
        nodes_list  : ordered list of node IDs
        node_idx    : mapping node_id -> matrix index
        adj         : (n x n) adjacency matrix
        dist_mat    : (n x n) distance matrix  [meters]
        time_mat    : (n x n) travel time matrix [seconds]
    """
    graph_cfg = config["graph"]
    default_speed = graph_cfg["default_speed_kph"]

    nodes_list = list(G.nodes())
    n = len(nodes_list)
    node_idx = {node: i for i, node in enumerate(nodes_list)}

    adj = np.zeros((n, n), dtype=np.float32)
    dist_mat = np.zeros((n, n), dtype=np.float32)
    time_mat = np.zeros((n, n), dtype=np.float32)

    for u, v, data in G.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        dist = float(data.get("length", 1.0))  # meters

        # Speed: OSM may give a list or a single value
        speed_raw = data.get("speed_kph", default_speed)
        if isinstance(speed_raw, list):
            speed = float(speed_raw[0])
        else:
            try:
                speed = float(speed_raw)
            except (ValueError, TypeError):
                speed = float(default_speed)

        travel_time = dist / (speed * 1000.0 / 3600.0)  # seconds

        adj[i, j] = adj[j, i] = 1.0
        dist_mat[i, j] = dist_mat[j, i] = dist
        time_mat[i, j] = time_mat[j, i] = travel_time

    return nodes_list, node_idx, adj, dist_mat, time_mat


def save_graph_data(G, nodes_list, node_idx, adj, dist_mat, time_mat, config: dict):
    """Persist processed graph data to disk."""
    data_dir = config["output"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "graph.pkl"), "wb") as f:
        pickle.dump(G, f)

    np.save(os.path.join(data_dir, "adj.npy"), adj)
    np.save(os.path.join(data_dir, "dist_mat.npy"), dist_mat)
    np.save(os.path.join(data_dir, "time_mat.npy"), time_mat)

    with open(os.path.join(data_dir, "nodes_list.pkl"), "wb") as f:
        pickle.dump(nodes_list, f)
    with open(os.path.join(data_dir, "node_idx.pkl"), "wb") as f:
        pickle.dump(node_idx, f)

    print(f"[Data] Saved graph data to '{data_dir}/'")


def load_graph_data(config: dict):
    """Load persisted graph data from disk."""
    data_dir = config["output"]["data_dir"]

    with open(os.path.join(data_dir, "graph.pkl"), "rb") as f:
        G = pickle.load(f)
    adj       = np.load(os.path.join(data_dir, "adj.npy"))
    dist_mat  = np.load(os.path.join(data_dir, "dist_mat.npy"))
    time_mat  = np.load(os.path.join(data_dir, "time_mat.npy"))

    with open(os.path.join(data_dir, "nodes_list.pkl"), "rb") as f:
        nodes_list = pickle.load(f)
    with open(os.path.join(data_dir, "node_idx.pkl"), "rb") as f:
        node_idx = pickle.load(f)

    print(f"[Data] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, nodes_list, node_idx, adj, dist_mat, time_mat


def prepare_graph(config: dict, force_reload: bool = False):
    """
    Full pipeline: download → subsample → build matrices → save/load.
    Skips download if data already exists (unless force_reload=True).
    """
    data_dir = config["output"]["data_dir"]
    graph_pkl = os.path.join(data_dir, "graph.pkl")

    if os.path.exists(graph_pkl) and not force_reload:
        print("[Data] Found cached graph data, loading from disk...")
        return load_graph_data(config)

    G = download_graph(config)

    max_nodes = config["graph"].get("max_nodes")
    if max_nodes:
        G = subsample_graph(G, max_nodes)

    nodes_list, node_idx, adj, dist_mat, time_mat = build_matrices(G, config)
    save_graph_data(G, nodes_list, node_idx, adj, dist_mat, time_mat, config)
    return G, nodes_list, node_idx, adj, dist_mat, time_mat
