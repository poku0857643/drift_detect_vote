"""
GATConv Drift Detector
=======================
Graph Attention Network drift detector for graph-structured inputs.
Uses a 2-layer GATConv encoder trained via Graph AutoEncoder (GAE)
self-supervised reconstruction loss on the reference graph.

Four drift signals (in priority order)
---------------------------------------
1. MMD²        — Maximum Mean Discrepancy between ref/prod node embeddings
2. Attention KS — KS test on per-edge attention weight distributions (α_uv)
3. Cosine dist  — Mean embedding vector shift
4. GAE loss     — Reconstruction loss delta vs reference baseline

Loaded once at lifespan startup from a saved checkpoint file.
Never refitted per-request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────── #
# Model
# ─────────────────────────────────────────────────────────────────── #

class GATEncoder(nn.Module):
    """
    Two-layer Graph Attention Network encoder.

    Layer 1: GATConv(in → hidden, heads=H, concat=True)  → (N, hidden×H)
    Layer 2: GATConv(hidden×H → hidden, heads=1, concat=False) → (N, hidden)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(
            in_channels, hidden_dim, heads=heads, concat=True, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1, (ei1, aw1) = self.conv1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2, (ei2, aw2) = self.conv2(x1, edge_index, return_attention_weights=True)

        if return_attention_weights:
            return x2, [(ei1, aw1), (ei2, aw2)]
        return x2


# ─────────────────────────────────────────────────────────────────── #
# Checkpoint
# ─────────────────────────────────────────────────────────────────── #

@dataclass
class GATCheckpoint:
    model_state:         dict
    ref_embeddings:      torch.Tensor   # (N_ref, hidden_dim)
    ref_attention_flat:  torch.Tensor   # flattened α_uv across all edges/heads
    ref_edge_index:      torch.Tensor   # reference graph edges for GAE loss
    ref_recon_loss:      float
    in_channels:         int
    hidden_dim:          int
    heads:               int

    def save(self, path: str | Path) -> None:
        torch.save(self.__dict__, str(path))
        logger.info("Checkpoint saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "GATCheckpoint":
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(**data)


# ─────────────────────────────────────────────────────────────────── #
# Loss helpers
# ─────────────────────────────────────────────────────────────────── #

def _gae_loss(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
) -> torch.Tensor:
    """BCE reconstruction loss on positive and randomly sampled negative edges."""
    pos = (z[edge_index[0]] * z[edge_index[1]]).sum(-1).sigmoid()
    neg = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(-1).sigmoid()
    return (-torch.log(pos + 1e-9).mean() + -torch.log(1 - neg + 1e-9).mean()) / 2


def _mmd2(X: torch.Tensor, Y: torch.Tensor, max_n: int = 512) -> float:
    """MMD² with RBF kernel; median heuristic sets bandwidth."""
    if len(X) > max_n:
        X = X[torch.randperm(len(X))[:max_n]]
    if len(Y) > max_n:
        Y = Y[torch.randperm(len(Y))[:max_n]]

    XY = torch.cat([X, Y])
    dists = torch.cdist(XY, XY)
    median = dists[dists > 0].median()
    g = 1.0 / (2.0 * median ** 2 + 1e-9)

    def k(A, B):
        d = A.unsqueeze(1) - B.unsqueeze(0)
        return torch.exp(-g * d.pow(2).sum(-1))

    n, m = len(X), len(Y)
    return float((k(X, X).sum() / (n * n) - 2 * k(X, Y).sum() / (n * m) + k(Y, Y).sum() / (m * m)).item())


# ─────────────────────────────────────────────────────────────────── #
# Detector
# ─────────────────────────────────────────────────────────────────── #

class GATDriftDetector:
    """
    Wraps a trained GATEncoder and computes four drift signals per request.

    Alert thresholds (from CLAUDE.md):
        MMD²  > 0.05
        KS p  < 0.05
    """

    MMD_THRESHOLD = 0.05
    KS_P_THRESHOLD = 0.05

    def __init__(self, checkpoint: GATCheckpoint) -> None:
        self.model = GATEncoder(
            checkpoint.in_channels, checkpoint.hidden_dim, checkpoint.heads
        )
        self.model.load_state_dict(checkpoint.model_state)
        self.model.eval()

        self._ref_emb        = checkpoint.ref_embeddings.detach()
        self._ref_attn_flat  = checkpoint.ref_attention_flat.detach().numpy()
        self._ref_edge_index = checkpoint.ref_edge_index
        self._ref_recon_loss = checkpoint.ref_recon_loss

    @torch.no_grad()
    def detect(self, prod_graph: Data) -> dict:
        prod_emb, prod_attn = self.model(
            prod_graph.x, prod_graph.edge_index, return_attention_weights=True
        )
        prod_attn_flat = torch.cat([aw.flatten() for _, aw in prod_attn]).numpy()

        # 1 — MMD²
        mmd2 = _mmd2(self._ref_emb, prod_emb)

        # 2 — Attention weight KS test
        ks_stat, ks_p = scipy_stats.ks_2samp(self._ref_attn_flat, prod_attn_flat)

        # 3 — Cosine distance on mean embeddings
        ref_mean  = F.normalize(self._ref_emb.mean(0, keepdim=True), dim=1)
        prod_mean = F.normalize(prod_emb.mean(0, keepdim=True), dim=1)
        cos_dist  = float(1.0 - (ref_mean * prod_mean).sum().item())

        # 4 — GAE reconstruction loss delta
        neg_ei      = negative_sampling(prod_graph.edge_index, prod_emb.size(0))
        prod_recon  = float(_gae_loss(prod_emb, prod_graph.edge_index, neg_ei).item())
        recon_delta = prod_recon - self._ref_recon_loss

        drift_detected = mmd2 > self.MMD_THRESHOLD or ks_p < self.KS_P_THRESHOLD

        return {
            "drift_detected": drift_detected,
            "signals": {
                "mmd2":                     round(mmd2, 6),
                "attention_ks_stat":        round(float(ks_stat), 6),
                "attention_ks_p":           round(float(ks_p), 6),
                "cosine_distance":          round(cos_dist, 6),
                "gae_reconstruction_loss":  round(prod_recon, 6),
                "gae_reconstruction_delta": round(recon_delta, 6),
            },
            "thresholds": {
                "mmd2":  self.MMD_THRESHOLD,
                "ks_p":  self.KS_P_THRESHOLD,
            },
            "ref_recon_loss": round(self._ref_recon_loss, 6),
        }


# ─────────────────────────────────────────────────────────────────── #
# Training
# ─────────────────────────────────────────────────────────────────── #

def train_gat(
    reference_graph: Data,
    hidden_dim: int = 128,
    heads: int = 8,
    dropout: float = 0.1,
    epochs: int = 300,
    lr: float = 5e-3,
    patience: int = 30,
    device: str = "cpu",
) -> GATCheckpoint:
    """
    Train GATEncoder on reference_graph via GAE self-supervised loss.

    Returns a GATCheckpoint ready to be saved or loaded into GATDriftDetector.
    """
    in_channels = reference_graph.x.size(1)
    graph = reference_graph.to(device)

    model = GATEncoder(in_channels, hidden_dim, heads, dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_loss   = float("inf")
    best_state  = None
    no_improve  = 0

    logger.info(
        "Training GATEncoder — nodes=%d  edges=%d  features=%d  epochs=%d",
        graph.num_nodes, graph.num_edges, in_channels, epochs,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        optimiser.zero_grad()
        z, _ = model(graph.x, graph.edge_index, return_attention_weights=True)
        neg_ei = negative_sampling(graph.edge_index, graph.num_nodes)
        loss = _gae_loss(z, graph.edge_index, neg_ei)
        loss.backward()
        optimiser.step()
        scheduler.step()

        loss_val = loss.item()
        if loss_val < best_loss - 1e-6:
            best_loss  = loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            logger.info("  epoch %3d / %d  loss=%.5f  best=%.5f", epoch, epochs, loss_val, best_loss)

        if no_improve >= patience:
            logger.info("  early stop at epoch %d", epoch)
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        ref_emb, ref_attn = model(graph.x, graph.edge_index, return_attention_weights=True)
        neg_ei = negative_sampling(graph.edge_index, graph.num_nodes)
        ref_recon = float(_gae_loss(ref_emb, graph.edge_index, neg_ei).item())

    ref_attn_flat = torch.cat([aw.detach().flatten() for _, aw in ref_attn])

    logger.info("Training complete — best_loss=%.5f  ref_recon=%.5f", best_loss, ref_recon)

    return GATCheckpoint(
        model_state        = best_state,
        ref_embeddings     = ref_emb.detach().cpu(),
        ref_attention_flat = ref_attn_flat.cpu(),
        ref_edge_index     = graph.edge_index.cpu(),
        ref_recon_loss     = ref_recon,
        in_channels        = in_channels,
        hidden_dim         = hidden_dim,
        heads              = heads,
    )