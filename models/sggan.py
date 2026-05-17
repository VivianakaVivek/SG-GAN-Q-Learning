"""
models/sggan.py
Spatial Graph GAN — Generator and Discriminator architectures.
All hyperparameters are passed through config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier_init(module: nn.Module):
    """Apply Xavier (Glorot) uniform initialization to all linear layers."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class GraphGenerator(nn.Module):
    """
    Generator G(v | vc; θ_G).
    Takes a noise vector and produces an embedding for a synthetic neighbor node.
    """

    def __init__(self, noise_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Use LayerNorm instead of BatchNorm so we handle batch_size=1 (single-neighbor nodes)
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        _xavier_init(self)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z : (batch, noise_dim)  →  (batch, output_dim)"""
        return self.net(z)


class GraphDiscriminator(nn.Module):
    """
    Discriminator D(v, vc; θ_D).
    Receives a candidate neighbor embedding v and the center vertex embedding vc,
    and outputs the probability that the pair is a real edge.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        _xavier_init(self)

    def forward(self, v: torch.Tensor, vc: torch.Tensor) -> torch.Tensor:
        """
        v  : (batch, embedding_dim)  — neighbor embeddings
        vc : (batch, embedding_dim)  — center vertex embedding (broadcast)
        → (batch, 1) probability
        """
        return self.net(torch.cat([v, vc], dim=-1))


def build_sggan(config: dict, device: torch.device):
    """
    Instantiate Generator and Discriminator from config and move to device.

    Returns:
        generator     : GraphGenerator
        discriminator : GraphDiscriminator
    """
    sg_cfg = config["sggan"]
    emb_dim = config["embedding"]["dim"]

    generator = GraphGenerator(
        noise_dim=sg_cfg["noise_dim"],
        hidden_dim=sg_cfg["hidden_dim"],
        output_dim=sg_cfg["output_dim"],
    ).to(device)

    discriminator = GraphDiscriminator(
        embedding_dim=emb_dim,
        hidden_dim=sg_cfg["hidden_dim"],
    ).to(device)

    return generator, discriminator
