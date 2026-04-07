# DriftDetect

> A three-model voting ensemble for production ML drift detection — covering graph-structured, time-series, and tabular data — with GCP Cloud Monitoring integration.

---

## The Problem This System Solves

Production ML models degrade silently. The root causes fall into three distinct categories that **no single detector can cover**:

| Drift type | What changes | Classical blind spot |
|------------|-------------|----------------------|
| **Covariate drift** | Individual feature distributions P(X) shift | Detectable by KS/PSI, but only for marginal distributions |
| **Structural / relational drift** | Relationships between entities shift — P(X, A) — e.g. career graph topology, ECG waveform morphology | KS/PSI **cannot** detect this — they see only flat feature vectors |
| **Outlier pattern drift** | Novel anomaly clusters appear in production that weren't in training | Statistical tests miss subtle multi-feature combinations |

This system is specifically designed for domains where **relational structure matters** — career progression graphs, physiological monitoring (ECG), and any ML pipeline where entities are connected and their relationships encode meaning. A statistical test on individual features will not catch it when the *pattern of connections* changes, only when individual values change.

### Why one detector is not enough

Each detector has a systematic blind spot:

- A **statistical test** will pass when the marginal distributions look stable but the joint structure has shifted — a classic false negative in graph data.
- A **GNN** produces low drift scores when individual node features shift without topological change — a false negative for pure covariate drift.
- A **Random Forest anomaly detector** can miss gradual cluster migration that doesn't produce obvious outliers.

The voting ensemble exists to **combine independent evidence** — requiring agreement across at least two of three orthogonal methods dramatically reduces both false positives and false negatives.

---

## System Architecture

```
Your Application
      │
      ├── tabular / feature vectors ──►  POST /api/v1/report
      └── graph data (nodes + edges) ──► POST /drift/gnn
                                              │
                    ┌─────────────────────────▼──────────────────────────────┐
                    │           DriftDetect API  (GCP Cloud Run)             │
                    │                                                        │
                    │  ┌─────────────────────────────────────────────────┐  │
                    │  │              Voting Ensemble                     │  │
                    │  │   majority | unanimous | any                    │  │
                    │  │                                                  │  │
                    │  │  ┌───────────┐  ┌─────────────┐  ┌──────────┐  │  │
                    │  │  │   GNN     │  │ Statistical  │  │  Random  │  │  │
                    │  │  │           │  │              │  │  Forest  │  │  │
                    │  │  │ GATConv   │  │  KS test     │  │          │  │  │
                    │  │  │ MMD²      │  │  PSI         │  │ Isolation│  │  │
                    │  │  │ Attention │  │  Wasserstein │  │ Forest   │  │  │
                    │  │  │ KS test   │  │  Bonferroni  │  │ + LOF    │  │  │
                    │  │  │ GAE loss  │  │  Fisher      │  │ + SHAP   │  │  │
                    │  │  └─────┬─────┘  └──────┬───────┘  └────┬─────┘  │  │
                    │  │        │  vote          │  vote         │  vote  │  │
                    │  │        └────────────────┴───────────────┘        │  │
                    │  │                         │                         │  │
                    │  │              ┌──────────▼──────────┐              │  │
                    │  │              │   Final Decision     │              │  │
                    │  │              │   confidence score   │              │  │
                    │  │              │   feature attribution│              │  │
                    │  │              └──────────┬──────────┘              │  │
                    │  └─────────────────────────┼─────────────────────────┘  │
                    │              ┌─────────────┼──────────────┐             │
                    │              ▼             ▼              ▼             │
                    │       ┌────────────┐ ┌──────────┐ ┌────────────────┐   │
                    │       │ SQLite DB  │ │  HTML    │ │  GCP Cloud     │   │
                    │       │            │ │  Report  │ │  Monitoring    │   │
                    │       │ drift_runs │ │          │ │                │   │
                    │       │ detector_  │ │ verdict  │ │ custom.google  │   │
                    │       │ results    │ │ per-feat │ │ apis.com/      │   │
                    │       │ feature_   │ │ scores   │ │ drift/         │   │
                    │       │ metrics    │ │ opt.     │ │ {metric_name}  │   │
                    │       │ optim_     │ │ recom.   │ │                │   │
                    │       │ prediction │ └──────────┘ │ (async, via    │   │
                    │       └────────────┘              │  BackgroundTask│   │
                    │                                   └────────────────┘   │
                    └────────────────────────────────────────────────────────┘
                              │
                              ▼
                    drift_detected, ensemble_score,
                    per-detector votes, feature attribution,
                    predicted improvement %, HTML report URL,
                    GCP metrics written flag
```

---

## How the Voting System Works and Why

The three detectors approach drift from entirely different angles. Their combination is not arbitrary — each one plugs the blind spot of the others.

### Detector 1 — GNN (Graph Attention Network)

**What it detects:** Structural and relational drift — changes in how entities relate to each other, not just changes in individual values.

**How it works:**
- Builds a feature-correlation graph (or uses the input graph directly for graph data)
- GATConv propagates node statistics through learned attention-weighted edges
- Computes four signals in priority order:
  1. **MMD²** — Maximum Mean Discrepancy between reference and production node embeddings (threshold: > 0.05)
  2. **Attention weight KS test** — distribution shift in per-edge α_uv attention weights (threshold: p < 0.05)
  3. **Cosine distance** — mean embedding vector shift (informational)
  4. **GAE reconstruction loss** — unsupervised concept drift via graph autoencoder

**Why it works where classical methods fail:** When career graph topology changes (new job categories emerge, promotion paths shift) or ECG waveform relationships change, individual feature distributions may stay stable. GATConv detects the relational shift through its attention mechanism — a signal that is fundamentally invisible to KS/PSI.

---

### Detector 2 — Statistical (KS + PSI + Wasserstein)

**What it detects:** Marginal distribution shift — individual features changing their distribution shape, range, or density.

**How it works:**
- **Kolmogorov–Smirnov test** per feature with Bonferroni correction for multiple comparisons
- **Population Stability Index (PSI)** — industry-standard binned probability shift (flag if > 0.20)
- **Wasserstein distance** — earth mover's distance normalised by reference std (flag if > 0.20)
- Fisher's method combines per-feature evidence into a single p-value
- A feature is flagged if *any* of the three tests fires

**Why it's in the ensemble:** Fast, interpretable, and highly sensitive to univariate changes. Acts as a canary — if any single feature is distributing differently, this catches it immediately. Provides the per-feature breakdown needed for root cause analysis.

---

### Detector 3 — Random Forest (Isolation Forest + LOF)

**What it detects:** Multivariate anomaly pattern drift — novel combinations of features appearing in production that weren't in training, and local cluster migration.

**How it works:**
- **Isolation Forest** trained on reference data defines the "normal" manifold. Anomaly score distributions (reference vs production) compared via KS test.
- **Local Outlier Factor** (LOF, novelty mode) catches local density shifts that IF misses — triggered when production samples are dense in regions sparse in reference.
- **Marginal SHAP attribution** — each feature's contribution to drift scored by measuring how much the IF score shifts when that feature is replaced with its reference mean.

**Why it's in the ensemble:** Catches the case where no individual feature has shifted, but the *combination* has. A patient whose individual ECG metrics are within normal range but whose joint feature pattern has never been seen before will score high anomaly but low KS — only RF catches this.

---

### Why the Vote Reduces Bias

```
                    GNN        Statistical     Random Forest
                     │               │               │
Structural drift     ✓               ✗               ✗   ← only GNN catches it
Covariate drift      ✗               ✓               ✗   ← only Stat catches it
Multivariate drift   ✗               ✗               ✓   ← only RF catches it
All three            ✓               ✓               ✓   ← clear drift signal
Two of three         ✓ / ✗           ✓ / ✗          ✓ / ✗ ← majority verdict
```

With `majority` strategy (default), drift is confirmed when ≥ 2 detectors agree. This means:
- **False positives** are suppressed — a statistical fluke in one detector is overruled.
- **False negatives** are suppressed — a blind spot in one detector is covered by another.
- **Confidence** is quantified as `votes_for_drift / 3`, giving a calibrated signal for alert severity.

---

## Target Data

This system is designed for ML pipelines operating on:

| Domain | Data type | Why this system fits |
|--------|-----------|---------------------|
| **Career / HR analytics** | Career progression graphs, skill graphs, job transition networks | GNN detects when graph topology shifts; RF catches novel career patterns |
| **Clinical / physiological monitoring** | ECG signals, vital sign time series, patient pathway graphs | GNN detects waveform relationship shifts; Statistical catches baseline drift |
| **Recommendation systems** | User–item interaction graphs, embedding spaces | GNN detects preference graph topology changes; all three validate feature drift |
| **Any tabular ML model** | Feature vectors at inference time | Statistical + RF cover all standard covariate and concept drift patterns |
| **Sensor / IoT networks** | Multi-sensor readings with inter-sensor correlations | GNN detects correlation structure shift; Statistical catches per-sensor drift |

The system is **label-free** — it requires only the input feature distributions, not ground-truth labels. This makes it usable in production where labels arrive with a delay or not at all.

---

## Setup

```bash
git clone <repo-url> && cd drift_detect
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

| URL | Description |
|-----|-------------|
| http://127.0.0.1:8000/docs | Swagger UI — all endpoints |
| http://127.0.0.1:8000/api/v1/metrics/dashboard | HTML metrics dashboard |

### GCP configuration (Cloud Run)

```bash
export GCS_REFERENCE_GRAPH_PATH="gs://your-bucket/reference_graph.pt"
export GCP_PROJECT_ID="your-gcp-project"
```

The service account running the container needs only `roles/monitoring.metricWriter`. GCP metrics are written asynchronously via `BackgroundTasks` — they never block the API response.

---

## API Reference

### Tabular drift detection

**`POST /api/v1/detect`** — run ensemble, no persistence

**`POST /api/v1/report`** — run ensemble, persist to SQLite, generate HTML report

```json
{
  "reference":     [[25, 50000, 0.8], ...],
  "current":       [[45, 90000, 0.3], ...],
  "feature_names": ["age", "salary", "engagement"],
  "strategy":      "majority",
  "threshold":     0.05
}
```

**`GET /api/v1/report/{run_id}/html`** — view rendered HTML report

---

### Graph drift detection

**`POST /drift/gnn`** — GATConv four-signal detector with GCP logging

```json
{
  "node_features": [[0.2, 1.4, ...], ...],
  "edge_index":    [[0, 1, 2, ...], [1, 2, 0, ...]]
}
```

Response:
```json
{
  "drift_detected": true,
  "signals": {
    "mmd2": 0.071,
    "attention_ks_stat": 0.43,
    "attention_ks_p": 0.002,
    "cosine_distance": 0.18,
    "gae_reconstruction_loss": 0.94
  },
  "thresholds": { "mmd2": 0.05, "ks_p": 0.05 },
  "gcp_logged": true
}
```

GCP custom metrics written (async):
- `custom.googleapis.com/drift/mmd2`
- `custom.googleapis.com/drift/attention_ks_stat`
- `custom.googleapis.com/drift/cosine_distance`
- `custom.googleapis.com/drift/gae_reconstruction_loss`
- `custom.googleapis.com/drift/drift_detected`

---

### Metrics & dashboard

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/metrics` | Paginated run history |
| `GET` | `/api/v1/metrics/performance` | Drift rate, detector agreement, avg improvement |
| `GET` | `/api/v1/metrics/dashboard` | Interactive HTML dashboard |
| `GET` | `/api/v1/metrics/{run_id}` | Single run detail |

---

## Integrating Into Your Pipeline

```python
import httpx, pandas as pd

reference = pd.read_parquet("training_data.parquet").sample(500).values.tolist()
current   = pd.read_parquet("production_window.parquet").sample(500).values.tolist()

result = httpx.post("https://your-cloud-run-url/api/v1/report", json={
    "reference": reference,
    "current":   current,
    "feature_names": ["age", "salary", "engagement"],
    "strategy": "majority"
}).json()

if result["drift_detected"]:
    print(result["optimization"]["recommendation"])
    # → retrain, alert, investigate top drifted features
```

---

## Project Structure

```
app/
├── detectors/
│   ├── base.py              # Abstract DriftDetector + DetectorResult
│   ├── gnn.py               # Numpy GCN — tabular feature-correlation graph
│   ├── gnn_gat.py           # GATConv — graph-structured inputs (PyTorch Geometric)
│   ├── statistical.py       # KS + PSI + Wasserstein
│   └── isolation.py         # Isolation Forest + LOF + marginal SHAP
├── ensemble/
│   └── voting.py            # VotingEnsemble (majority / unanimous / any)
├── gcp/
│   └── monitoring.py        # GCP Cloud Monitoring metric writer
├── db/
│   ├── models.py            # SQLAlchemy ORM — 4 tables
│   └── crud.py              # save_run, list_runs, get_performance_summary
├── reports/
│   ├── optimizer.py         # Predicted improvement % from drift correction
│   └── renderer.py          # Self-contained HTML report
└── routers/
    ├── drift.py             # /api/v1/detect, /report, /report/{id}/html
    ├── drift_gnn.py         # /drift/gnn  (GATConv + GCP)
    └── metrics.py           # /api/v1/metrics/*
```