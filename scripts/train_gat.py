"""
GAT Encoder Training Script
============================
Train the GATConv encoder on a reference graph and save a checkpoint.

The checkpoint is later loaded at API startup (see main.py lifespan).

Usage
-----
# Train on a local .pt graph file
python scripts/train_gat.py \\
    --graph   data/reference_graph.pt \\
    --out     checkpoints/gat_checkpoint.pt \\
    --hidden  128 --heads 8 --epochs 300

# Train on a synthetic demo graph (no data file required)
python scripts/train_gat.py --demo --out checkpoints/gat_checkpoint.pt

# Train and upload checkpoint to GCS
python scripts/train_gat.py \\
    --graph   data/reference_graph.pt \\
    --out     gs://your-bucket/checkpoints/gat_checkpoint.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.detectors.gnn_gat import train_gat, GATCheckpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────── #
# Synthetic demo graph
# ─────────────────────────────────────────────────────────────────── #

def make_demo_graph(n_nodes: int = 200, n_features: int = 16, seed: int = 42) -> Data:
    """
    Generate a synthetic graph for demo/testing purposes.

    Structure: two Gaussian clusters connected by inter-cluster edges,
    mimicking a career graph where two job families are bridged by
    transition roles.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    half = n_nodes // 2

    # Two feature clusters
    x_a = torch.randn(half,          n_features, generator=rng)
    x_b = torch.randn(n_nodes - half, n_features, generator=rng) + 2.0
    x   = torch.cat([x_a, x_b], dim=0)

    # Intra-cluster edges (random k-NN style)
    edges = []
    k = 5
    for cluster_start, cluster_size in [(0, half), (half, n_nodes - half)]:
        for i in range(cluster_start, cluster_start + cluster_size):
            peers = torch.randint(cluster_start, cluster_start + cluster_size, (k,), generator=rng)
            for j in peers.tolist():
                if i != j:
                    edges += [[i, j], [j, i]]

    # Inter-cluster edges (sparse bridges)
    n_bridges = n_nodes // 10
    src = torch.randint(0, half, (n_bridges,), generator=rng)
    dst = torch.randint(half, n_nodes, (n_bridges,), generator=rng)
    for s, d in zip(src.tolist(), dst.tolist()):
        edges += [[s, d], [d, s]]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, num_nodes=n_nodes)


# ─────────────────────────────────────────────────────────────────── #
# GCS helpers
# ─────────────────────────────────────────────────────────────────── #

def load_from_gcs(gcs_path: str) -> Data:
    from google.cloud import storage
    import io
    bucket_name, blob_path = gcs_path[5:].split("/", 1)
    client = storage.Client()
    blob   = client.bucket(bucket_name).blob(blob_path)
    buf    = io.BytesIO(blob.download_as_bytes())
    return torch.load(buf, map_location="cpu", weights_only=False)


def save_to_gcs(checkpoint: GATCheckpoint, gcs_path: str) -> None:
    from google.cloud import storage
    import io
    bucket_name, blob_path = gcs_path[5:].split("/", 1)
    client = storage.Client()
    buf    = io.BytesIO()
    checkpoint.save(buf)
    buf.seek(0)
    client.bucket(bucket_name).blob(blob_path).upload_from_file(buf)
    logger.info("Checkpoint uploaded → %s", gcs_path)


# ─────────────────────────────────────────────────────────────────── #
# Main
# ─────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GATConv encoder for drift detection")
    parser.add_argument("--graph",   type=str, default=None, help="Path to reference graph .pt (local or gs://)")
    parser.add_argument("--out",     type=str, default="checkpoints/gat_checkpoint.pt")
    parser.add_argument("--hidden",  type=int, default=128)
    parser.add_argument("--heads",   type=int, default=8)
    parser.add_argument("--epochs",  type=int, default=300)
    parser.add_argument("--lr",      type=float, default=5e-3)
    parser.add_argument("--patience",type=int, default=30)
    parser.add_argument("--demo",    action="store_true", help="Use synthetic demo graph")
    parser.add_argument("--device",  type=str, default="cpu")
    args = parser.parse_args()

    # Load or generate reference graph
    if args.demo or args.graph is None:
        logger.info("Using synthetic demo graph (200 nodes, 16 features)")
        graph = make_demo_graph()
    elif args.graph.startswith("gs://"):
        logger.info("Loading reference graph from GCS: %s", args.graph)
        graph = load_from_gcs(args.graph)
    else:
        logger.info("Loading reference graph from: %s", args.graph)
        graph = torch.load(args.graph, map_location="cpu", weights_only=False)

    logger.info(
        "Reference graph — nodes=%d  edges=%d  features=%d",
        graph.num_nodes, graph.num_edges, graph.x.size(1),
    )

    # Train
    checkpoint = train_gat(
        reference_graph=graph,
        hidden_dim=args.hidden,
        heads=args.heads,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
    )

    # Save
    out = args.out
    if out.startswith("gs://"):
        save_to_gcs(checkpoint, out)
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        checkpoint.save(out)

    logger.info("Done — checkpoint saved to %s", out)
    logger.info(
        "Summary: ref_recon_loss=%.5f  ref_embedding_shape=%s",
        checkpoint.ref_recon_loss,
        tuple(checkpoint.ref_embeddings.shape),
    )


if __name__ == "__main__":
    main()