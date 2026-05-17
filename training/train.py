"""
training/train.py
Full SG-GAN adversarial training loop — parameters matched to the base paper.

Key paper specs implemented:
  - LR = 0.0001 for first 50 iterations, then 0.0002  (Section III-A)
  - Adam: σ1=0.5, σ2=0.999                            (Section III-A)
  - Gradient clipping: max norm 5.0                   (Section III-A)
  - Warm-up pre-training: up to 10 epochs             (Section III-A)
  - 200 total training iterations                     (Section VI)
  - Accuracy tracking: Generator & Discriminator      (Fig. 3a)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.sggan import build_sggan


def _make_optimizers(generator, discriminator, config: dict, lr: float):
    """Create Adam optimizers with paper-exact betas."""
    sg_cfg = config["sggan"]
    betas = (sg_cfg["adam_beta1"], sg_cfg["adam_beta2"])
    opt_G = torch.optim.Adam(generator.parameters(),     lr=lr, betas=betas)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    return opt_G, opt_D


def _set_lr(optimizer, lr: float):
    """Update learning rate in-place for all param groups."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Binary classification accuracy: how often (logit > 0.5) == label."""
    preds = (logits >= 0.5).float()
    return (preds == labels).float().mean().item()


def pretrain(generator, discriminator, embeddings, adj, config, device):
    """
    Warm-up phase (paper §III-A):
      - Discriminator: binary classification (real vs random fake).
      - Generator: MSE reconstruction towards real neighbor embeddings.
    Uses warmup LR.
    """
    sg_cfg = config["sggan"]
    pretrain_epochs = sg_cfg["pretrain_epochs"]
    noise_dim       = sg_cfg["noise_dim"]
    clip            = sg_cfg["grad_clip_norm"]
    lr_warmup       = sg_cfg["lr_warmup"]

    opt_G, opt_D = _make_optimizers(generator, discriminator, config, lr_warmup)
    criterion = nn.BCELoss()
    mse       = nn.MSELoss()
    n         = len(embeddings)

    print(f"[Pretrain] Warming up for {pretrain_epochs} epochs "
          f"(lr={lr_warmup}) ...")

    for epoch in range(pretrain_epochs):
        g_losses, d_losses = [], []

        for vc_idx in range(n):
            neighbors = np.where(adj[vc_idx] > 0)[0]
            if len(neighbors) == 0:
                continue
            k = len(neighbors)
            vc_emb = embeddings[vc_idx].unsqueeze(0)
            v_real = embeddings[neighbors]                 # (k, emb_dim)

            # Discriminator warm-up
            opt_D.zero_grad()
            real_labels = torch.ones(k, 1, device=device)
            loss_d_real = criterion(
                discriminator(v_real, vc_emb.expand(k, -1)), real_labels
            )
            z      = torch.randn(k, noise_dim, device=device)
            v_fake = generator(z).detach()
            fake_labels = torch.zeros(k, 1, device=device)
            loss_d_fake = criterion(
                discriminator(v_fake, vc_emb.expand(k, -1)), fake_labels
            )
            loss_D = loss_d_real + loss_d_fake
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip)
            opt_D.step()
            d_losses.append(loss_D.item())

            # Generator warm-up (reconstruction)
            opt_G.zero_grad()
            z     = torch.randn(k, noise_dim, device=device)
            v_gen = generator(z)
            loss_G = mse(v_gen, v_real.detach())
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip)
            opt_G.step()
            g_losses.append(loss_G.item())

        print(
            f"  [Pretrain {epoch+1}/{pretrain_epochs}] "
            f"D_loss={np.mean(d_losses):.4f}  G_loss={np.mean(g_losses):.6f}"
        )


def train_sggan(embeddings_np, adj, config: dict, device: torch.device):
    """
    Full adversarial training loop matching the paper (Algorithm 1).

    LR schedule (paper §III-A):
      - Epochs 1–50  : lr = lr_warmup  (0.0001)
      - Epochs 51–200: lr = learning_rate (0.0002)

    Tracks accuracy (as in paper Fig. 3a):
      - Generator accuracy    = how often D(G(z)) >= 0.5  (fooling D)
      - Discriminator accuracy= how often D classifies real/fake correctly

    Returns:
        generator, discriminator, history
            history keys: loss_G, loss_D, acc_G, acc_D
    """
    sg_cfg        = config["sggan"]
    epochs        = sg_cfg["epochs"]
    g_steps       = sg_cfg["g_steps"]
    d_steps       = sg_cfg["d_steps"]
    noise_dim     = sg_cfg["noise_dim"]
    clip          = sg_cfg["grad_clip_norm"]
    lr_warmup     = sg_cfg["lr_warmup"]
    lr_final      = sg_cfg["learning_rate"]
    lr_switch     = sg_cfg["lr_warmup_iters"]   # epoch at which to switch LR
    log_every     = config["output"]["log_interval"]

    generator, discriminator = build_sggan(config, device)

    # Start with warmup LR
    opt_G, opt_D = _make_optimizers(generator, discriminator, config, lr_warmup)
    criterion = nn.BCELoss()

    embeddings = torch.tensor(embeddings_np, dtype=torch.float32, device=device)
    n          = embeddings.shape[0]

    # Warm-up pre-training
    pretrain(generator, discriminator, embeddings, adj, config, device)

    history = {"loss_G": [], "loss_D": [], "acc_G": [], "acc_D": []}

    print(f"\n[Train] Adversarial training for {epochs} epochs ...")
    print(f"        LR: {lr_warmup} (epochs 1-{lr_switch}) → "
          f"{lr_final} (epochs {lr_switch+1}-{epochs})")

    for epoch in tqdm(range(epochs), desc="SG-GAN Training"):

        # ── LR schedule (paper: switch after first 50 iterations) ────────────
        current_lr = lr_warmup if epoch < lr_switch else lr_final
        _set_lr(opt_G, current_lr)
        _set_lr(opt_D, current_lr)

        epoch_g_losses, epoch_d_losses = [], []
        epoch_g_accs,   epoch_d_accs   = [], []

        for vc_idx in range(n):
            neighbors = np.where(adj[vc_idx] > 0)[0]
            if len(neighbors) == 0:
                continue
            k      = len(neighbors)
            vc_emb = embeddings[vc_idx].unsqueeze(0)
            vc_exp = vc_emb.expand(k, -1)
            v_real = embeddings[neighbors]

            # ── Discriminator steps ──────────────────────────────────────────
            for _ in range(d_steps):
                opt_D.zero_grad()

                # Real edges — label smoothing: use 0.9 instead of 1.0
                # (prevents D from being over-confident, matches paper balance)
                real_labels = torch.full((k, 1), 0.9, device=device)
                d_real_out  = discriminator(v_real, vc_exp)
                loss_d_real = criterion(d_real_out, real_labels)

                # Fake edges — hard 0 labels
                z      = torch.randn(k, noise_dim, device=device)
                v_fake = generator(z).detach()
                fake_labels = torch.zeros(k, 1, device=device)
                d_fake_out  = discriminator(v_fake, vc_exp)
                loss_d_fake = criterion(d_fake_out, fake_labels)

                loss_D = loss_d_real + loss_d_fake
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip)
                opt_D.step()
                epoch_d_losses.append(loss_D.item())

                # Discriminator accuracy: correct on both real and fake
                real_hard = torch.ones(k, 1, device=device)
                acc_d = (_accuracy(d_real_out, real_hard) +
                         _accuracy(d_fake_out, fake_labels)) / 2.0
                epoch_d_accs.append(acc_d)

            # ── Generator steps ──────────────────────────────────────────────
            for _ in range(g_steps):
                opt_G.zero_grad()
                z      = torch.randn(k, noise_dim, device=device)
                v_fake = generator(z)
                # G wants D to output 1 (real) for its samples
                real_labels = torch.ones(k, 1, device=device)
                d_out       = discriminator(v_fake, vc_exp)
                loss_G      = criterion(d_out, real_labels)
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), clip)
                opt_G.step()
                epoch_g_losses.append(loss_G.item())

                # Generator accuracy: how often D(G(z)) >= 0.5 (fooling D)
                acc_g = _accuracy(d_out, real_labels)
                epoch_g_accs.append(acc_g)


        mean_g_loss = float(np.mean(epoch_g_losses)) if epoch_g_losses else 0.0
        mean_d_loss = float(np.mean(epoch_d_losses)) if epoch_d_losses else 0.0
        mean_g_acc  = float(np.mean(epoch_g_accs))   if epoch_g_accs  else 0.0
        mean_d_acc  = float(np.mean(epoch_d_accs))   if epoch_d_accs  else 0.0

        history["loss_G"].append(mean_g_loss)
        history["loss_D"].append(mean_d_loss)
        history["acc_G"].append(mean_g_acc)
        history["acc_D"].append(mean_d_acc)

        if (epoch + 1) % log_every == 0:
            tqdm.write(
                f"  Epoch [{epoch+1:3d}/{epochs}] "
                f"lr={current_lr:.4f}  "
                f"G_loss={mean_g_loss:.4f}  D_loss={mean_d_loss:.4f}  "
                f"G_acc={mean_g_acc*100:.1f}%  D_acc={mean_d_acc*100:.1f}%"
            )

    print("[Train] Training complete.")
    return generator, discriminator, history


def save_models(generator, discriminator, history, config: dict):
    model_dir = config["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    torch.save(generator.state_dict(),
               os.path.join(model_dir, "generator.pth"))
    torch.save(discriminator.state_dict(),
               os.path.join(model_dir, "discriminator.pth"))
    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Models] Saved to '{model_dir}/'")


def load_models(config: dict, device: torch.device):
    from models.sggan import build_sggan
    model_dir = config["output"]["model_dir"]
    generator, discriminator = build_sggan(config, device)
    generator.load_state_dict(
        torch.load(os.path.join(model_dir, "generator.pth"), map_location=device)
    )
    discriminator.load_state_dict(
        torch.load(os.path.join(model_dir, "discriminator.pth"), map_location=device)
    )
    generator.eval()
    discriminator.eval()
    print(f"[Models] Loaded from '{model_dir}/'")
    return generator, discriminator
