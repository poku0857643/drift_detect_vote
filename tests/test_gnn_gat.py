"""Tests for the GATConv detector: GATEncoder, train_gat, GATDriftDetector."""
from __future__ import annotations

import torch
import pytest
from torch_geometric.data import Data

from app.detectors.gnn_gat import GATEncoder, GATCheckpoint, GATDriftDetector, train_gat, _mmd2


# ─────────────────────────────────────────────────────────────────── #
# Fixtures
# ─────────────────────────────────────────────────────────────────── #

@pytest.fixture(scope="module")
def small_graph():
    """A minimal 40-node, 4-feature graph for fast tests."""
    torch.manual_seed(0)
    n, f = 40, 4
    x = torch.randn(n, f)
    # Simple ring + random edges
    src = list(range(n)) + list(torch.randint(0, n, (60,)).tolist())
    dst = [(i + 1) % n for i in range(n)] + list(torch.randint(0, n, (60,)).tolist())
    ei = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=ei, num_nodes=n)


@pytest.fixture(scope="module")
def trained_checkpoint(small_graph):
    """Train a tiny GATEncoder and return its checkpoint (runs once per module)."""
    return train_gat(
        reference_graph=small_graph,
        hidden_dim=16,
        heads=2,
        epochs=20,
        lr=1e-2,
        patience=10,
        device="cpu",
    )


# ─────────────────────────────────────────────────────────────────── #
# GATEncoder
# ─────────────────────────────────────────────────────────────────── #

class TestGATEncoder:
    def test_output_shape(self, small_graph):
        model = GATEncoder(in_channels=4, hidden_dim=16, heads=2)
        model.eval()
        with torch.no_grad():
            out = model(small_graph.x, small_graph.edge_index)
        assert out.shape == (40, 16)

    def test_attention_weights_returned(self, small_graph):
        model = GATEncoder(in_channels=4, hidden_dim=16, heads=2)
        model.eval()
        with torch.no_grad():
            out, attn = model(small_graph.x, small_graph.edge_index, return_attention_weights=True)
        assert len(attn) == 2
        _, aw1 = attn[0]
        _, aw2 = attn[1]
        assert aw1.ndim == 2   # (E, heads)
        assert aw2.ndim == 2


# ─────────────────────────────────────────────────────────────────── #
# MMD²
# ─────────────────────────────────────────────────────────────────── #

def test_mmd2_identical_zero():
    X = torch.randn(50, 8)
    assert _mmd2(X, X) == pytest.approx(0.0, abs=1e-3)


def test_mmd2_large_shift():
    X = torch.randn(50, 8)
    Y = torch.randn(50, 8) + 10.0
    assert _mmd2(X, Y) > 0.05


# ─────────────────────────────────────────────────────────────────── #
# GATCheckpoint
# ─────────────────────────────────────────────────────────────────── #

class TestGATCheckpoint:
    def test_fields_populated(self, trained_checkpoint):
        ckpt = trained_checkpoint
        assert ckpt.in_channels == 4
        assert ckpt.hidden_dim == 16
        assert ckpt.heads == 2
        assert ckpt.ref_embeddings.shape[1] == 16
        assert ckpt.ref_recon_loss > 0.0

    def test_save_load_roundtrip(self, trained_checkpoint, tmp_path):
        path = tmp_path / "ckpt.pt"
        trained_checkpoint.save(str(path))
        loaded = GATCheckpoint.load(str(path))
        assert loaded.in_channels == trained_checkpoint.in_channels
        assert loaded.hidden_dim  == trained_checkpoint.hidden_dim
        assert torch.allclose(loaded.ref_embeddings, trained_checkpoint.ref_embeddings)


# ─────────────────────────────────────────────────────────────────── #
# GATDriftDetector
# ─────────────────────────────────────────────────────────────────── #

class TestGATDriftDetector:
    def test_no_drift_same_graph(self, trained_checkpoint, small_graph):
        detector = GATDriftDetector(trained_checkpoint)
        result = detector.detect(small_graph)
        # Identical graph → MMD should be ~0, KS p should be high → no drift
        assert not result["drift_detected"]

    def test_drift_on_shifted_graph(self, trained_checkpoint, small_graph):
        detector = GATDriftDetector(trained_checkpoint)
        shifted = Data(
            x=small_graph.x + 10.0,
            edge_index=small_graph.edge_index,
        )
        result = detector.detect(shifted)
        assert result["drift_detected"]

    def test_all_signals_present(self, trained_checkpoint, small_graph):
        detector = GATDriftDetector(trained_checkpoint)
        result = detector.detect(small_graph)
        for key in ("mmd2", "attention_ks_stat", "attention_ks_p",
                    "cosine_distance", "gae_reconstruction_loss",
                    "gae_reconstruction_delta"):
            assert key in result["signals"]

    def test_thresholds_in_result(self, trained_checkpoint, small_graph):
        detector = GATDriftDetector(trained_checkpoint)
        result = detector.detect(small_graph)
        assert "mmd2" in result["thresholds"]
        assert "ks_p"  in result["thresholds"]