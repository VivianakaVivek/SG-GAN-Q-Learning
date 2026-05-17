"""
models/embeddings.py
Node embedding using Node2Vec or spectral decomposition.
All parameters are read from config.
"""

import numpy as np
import networkx as nx
from node2vec import Node2Vec
from sklearn.preprocessing import normalize


def compute_node2vec_embeddings(G: nx.Graph, config: dict) -> np.ndarray:
    """
    Compute Node2Vec embeddings for all nodes in G.

    Returns:
        embeddings : (n x embedding_dim) float32 numpy array,
                     ordered by G.nodes()
    """
    emb_cfg = config["embedding"]
    print(f"[Embedding] Running Node2Vec on {G.number_of_nodes()} nodes ...")

    node2vec = Node2Vec(
        G,
        dimensions=emb_cfg["dim"],
        walk_length=emb_cfg["walk_length"],
        num_walks=emb_cfg["num_walks"],
        p=emb_cfg["p"],
        q=emb_cfg["q"],
        workers=emb_cfg["workers"],
        quiet=True,
    )
    model = node2vec.fit(window=10, min_count=1, epochs=emb_cfg["epochs"])

    nodes_list = list(G.nodes())
    embeddings = np.array(
        [model.wv[str(node)] for node in nodes_list], dtype=np.float32
    )
    embeddings = normalize(embeddings, norm="l2")
    print(f"[Embedding] Done. Shape: {embeddings.shape}")
    return embeddings


def compute_spectral_embeddings(G: nx.Graph, config: dict) -> np.ndarray:
    """
    Compute spectral (Laplacian eigenvector) embeddings for all nodes.

    Returns:
        embeddings : (n x embedding_dim) float32 numpy array
    """
    emb_cfg = config["embedding"]
    dim = emb_cfg["dim"]
    n = G.number_of_nodes()
    k = min(dim, n - 1)

    print(f"[Embedding] Computing spectral embeddings (k={k}) ...")
    L = nx.normalized_laplacian_matrix(G).toarray().astype(np.float32)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Take eigenvectors corresponding to the k smallest non-zero eigenvalues
    embeddings = eigenvectors[:, 1 : k + 1]

    # Pad to full dim if graph is small
    if embeddings.shape[1] < dim:
        pad = np.zeros((n, dim - embeddings.shape[1]), dtype=np.float32)
        embeddings = np.hstack([embeddings, pad])

    embeddings = normalize(embeddings.astype(np.float32), norm="l2")
    print(f"[Embedding] Done. Shape: {embeddings.shape}")
    return embeddings


def get_embeddings(G: nx.Graph, config: dict) -> np.ndarray:
    """Dispatch to the correct embedding method based on config."""
    method = config["embedding"]["method"]
    if method == "node2vec":
        return compute_node2vec_embeddings(G, config)
    elif method == "spectral":
        return compute_spectral_embeddings(G, config)
    else:
        raise ValueError(f"Unknown embedding method: '{method}'. Choose 'node2vec' or 'spectral'.")
