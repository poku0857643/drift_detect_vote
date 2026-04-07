"""
Create Reference Graph
=======================
Converts your tabular data (CSV, Parquet, or NumPy .npy) into a
PyTorch Geometric Data object and saves it as a .pt file ready for
scripts/train_gat.py.

Graph construction
------------------
Each **row** in your data becomes a **node**.
Edges are built using k-nearest neighbours on the feature vectors
(cosine similarity), connecting each node to its k most similar peers.
This captures the relational structure of your data — e.g. similar
career profiles, similar ECG patterns.

Usage
-----
# From a CSV file
python scripts/create_graph.py --input data/reference.csv --out data/reference_graph.pt

# From a Parquet file, picking specific columns
python scripts/create_graph.py \\
    --input   data/reference.parquet \\
    --columns age salary tenure score \\
    --out     data/reference_graph.pt

# From a NumPy array
python scripts/create_graph.py --input data/reference.npy --out data/reference_graph.pt

# Control graph density
python scripts/create_graph.py --input data/reference.csv --k 10 --out data/reference_graph.pt

Options
-------
--input    Path to data file (.csv / .parquet / .npy)
--out      Output path for the .pt graph file  (default: data/reference_graph.pt)
--columns  Columns to use as node features (CSV/Parquet only). Default: all numeric.
--k        Number of nearest neighbours per node (default: 5)
--sample   Max rows to use. Larger graphs slow training. Recommended: ≤ 5000.
--seed     Random seed for reproducibility (default: 42)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────── #
# Data loading
# ─────────────────────────────────────────────────────────────────── #

def load_features(
    path: str,
    columns: list[str] | None,
    sample: int | None,
    seed: int,
) -> np.ndarray:
    """Load data and return a float32 feature matrix (N × F)."""
    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    suffix = p.suffix.lower()

    if suffix == ".npy":
        X = np.load(str(p)).astype(np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

    elif suffix == ".csv":
        import csv
        import io
        # minimal CSV reader — avoids pandas dependency at this step
        try:
            import pandas as pd
            df = pd.read_csv(p)
            if columns:
                df = df[columns]
            X = df.select_dtypes(include="number").values.astype(np.float32)
        except ImportError:
            logger.warning("pandas not installed — reading CSV with stdlib csv module")
            with open(p) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            header = list(rows[0].keys()) if rows else []
            cols   = columns or header
            X = np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float32)

    elif suffix in (".parquet", ".pq"):
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas is required to read Parquet files: pip install pandas pyarrow")
            sys.exit(1)
        df = pd.read_parquet(p, columns=columns or None)
        X = df.select_dtypes(include="number").values.astype(np.float32)

    else:
        logger.error("Unsupported file format: %s  (supported: .csv .parquet .npy)", suffix)
        sys.exit(1)

    logger.info("Loaded  %d rows × %d features from %s", X.shape[0], X.shape[1], path)

    # Normalise: z-score per feature
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    # Optional sample
    if sample and sample < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=sample, replace=False)
        X   = X[idx]
        logger.info("Sampled to %d rows (--sample %d)", len(X), sample)

    return X


# ─────────────────────────────────────────────────────────────────── #
# Graph construction
# ─────────────────────────────────────────────────────────────────── #

def build_knn_graph(X: np.ndarray, k: int) -> torch.Tensor:
    """
    Build a k-NN graph from feature matrix X using cosine similarity.
    Returns edge_index tensor of shape (2, E).
    """
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(X) - 1)
    logger.info("Building k-NN graph  k=%d  n=%d ...", k, len(X))

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    src_list, dst_list = [], []
    for i, neighbours in enumerate(indices):
        for j in neighbours[1:]:          # skip self (index 0)
            src_list += [i, int(j)]       # undirected: add both directions
            dst_list += [int(j), i]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    logger.info(
        "Graph built — %d nodes  %d edges  (%.1f avg degree)",
        len(X), edge_index.size(1), edge_index.size(1) / len(X),
    )
    return edge_index


# ─────────────────────────────────────────────────────────────────── #
# Main
# ─────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert tabular data to a PyTorch Geometric graph")
    parser.add_argument("--input",   required=True,  help="Path to .csv / .parquet / .npy data file")
    parser.add_argument("--out",     default="data/reference_graph.pt")
    parser.add_argument("--columns", nargs="+",      help="Column names to use (CSV/Parquet only)")
    parser.add_argument("--k",       type=int, default=5,    help="Nearest neighbours per node")
    parser.add_argument("--sample",  type=int, default=None, help="Max rows to keep")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    # Load
    X = load_features(args.input, args.columns, args.sample, args.seed)

    if len(X) < 10:
        logger.error("Need at least 10 rows to build a meaningful graph (got %d).", len(X))
        sys.exit(1)

    # Build graph
    edge_index = build_knn_graph(X, k=args.k)
    x          = torch.tensor(X, dtype=torch.float)
    graph      = Data(x=x, edge_index=edge_index, num_nodes=len(X))

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, str(out))
    logger.info("Graph saved → %s", out)
    logger.info(
        "  nodes=%d  edges=%d  features=%d",
        graph.num_nodes, graph.num_edges, graph.x.size(1),
    )
    logger.info(
        "\nNext step — train the GATConv encoder:\n"
        "  python scripts/train_gat.py --graph %s --out checkpoints/gat_checkpoint.pt",
        out,
    )


if __name__ == "__main__":
    main()