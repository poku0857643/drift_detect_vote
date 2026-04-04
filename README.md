# DriftDetect API

> Detect when your production data has shifted away from what your model was trained on — before it silently degrades.

---

## The Problem

Machine learning models are trained on a snapshot of data. Once deployed, the real world keeps changing:

- Customer demographics shift over time
- Sensor readings drift due to hardware wear
- Market behaviour changes after economic events
- Upstream pipelines introduce subtle bugs or schema changes

Most teams only discover this **after** their model's accuracy has already dropped — often days or weeks later. By then the damage (bad predictions, lost revenue, incorrect decisions) is done.

**DriftDetect** monitors your data continuously and raises an alert the moment the distribution starts to change, giving you time to retrain before performance degrades.

---

## How It Works

```
Your Application
      │
      │  POST /api/v1/report
      │  { reference: [...], current: [...] }
      ▼
┌─────────────────────────────────────────────────────────┐
│                    DriftDetect API                      │
│                                                         │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│   │     GNN     │  │ Statistical  │  │  Isolation  │  │
│   │  Detector   │  │  Detector    │  │   Forest    │  │
│   │             │  │              │  │  Detector   │  │
│   │ Correlation │  │  KS test per │  │  Anomaly    │  │
│   │ graph +     │  │  feature +   │  │  score      │  │
│   │ GCN message │  │  Bonferroni  │  │  shift      │  │
│   │ passing     │  │  correction  │  │  (KS test)  │  │
│   └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  │
│          │  DRIFT?        │  DRIFT?          │ DRIFT?  │
│          ▼                ▼                  ▼         │
│   ┌────────────────────────────────────────────────┐   │
│   │              Voting Ensemble                   │   │
│   │   strategy: majority | unanimous | any         │   │
│   │   confidence = votes_for_drift / 3             │   │
│   └────────────────────────┬───────────────────────┘   │
│                            │                           │
│              ┌─────────────┴──────────────┐            │
│              ▼                            ▼            │
│   ┌──────────────────┐       ┌────────────────────┐    │
│   │   SQLite DB      │       │   HTML Report      │    │
│   │                  │       │                    │    │
│   │ drift_runs       │       │ • Verdict banner   │    │
│   │ detector_results │       │ • Per-detector     │    │
│   │ feature_metrics  │       │   cards            │    │
│   │ optimization_    │       │ • Feature drift    │    │
│   │   predictions    │       │   table            │    │
│   └──────────────────┘       │ • Optimization     │    │
│                              │   recommendation   │    │
│                              └────────────────────┘    │
└─────────────────────────────────────────────────────────┘
      │
      │  Response: drift_detected, ensemble_score,
      │            votes, per-feature scores,
      │            predicted improvement %, HTML url
      ▼
Your Application
(trigger alert / retrain / log)
```

---

## Purpose

DriftDetect solves three things at once:

| Goal | How |
|------|-----|
| **Early warning** | Catches distribution shift before model accuracy visibly drops |
| **Root cause** | Pinpoints which features drifted, not just that drift occurred |
| **Action guidance** | Predicts the % performance improvement you'd gain by retraining |

---

## The Three Detectors

Each detector approaches drift from a different angle. Requiring agreement across all three reduces false positives.

### 1. GNN (Graph Neural Network)
Builds a feature-correlation graph where each feature is a node and edges connect correlated features. Multi-layer graph convolution (GCN) propagates statistical node features (mean, std, quantiles, skew, kurtosis) through the reference correlation structure. Drift = the mean per-node embedding distance between reference and current exceeds a bootstrapped threshold.

*Best at:* structural changes — when the relationships **between** features shift, not just individual feature distributions.

### 2. Statistical (KS Test + Bonferroni)
Runs a Kolmogorov–Smirnov two-sample test on every feature independently. Applies Bonferroni correction for multiple comparisons and combines p-values via Fisher's method.

*Best at:* univariate distribution shifts — when one or a few features change their distribution shape.

### 3. Isolation Forest
Trains an Isolation Forest on the reference data to define what "normal" looks like. Compares the anomaly score distributions of reference vs current via KS test. Per-feature attribution via score perturbation.

*Best at:* multivariate outlier patterns — when unusual combinations of feature values appear in the current data.

---

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd drift_detect

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn main:app --reload
```

Swagger UI: **http://127.0.0.1:8000/docs**
Dashboard:  **http://127.0.0.1:8000/api/v1/metrics/dashboard**

---

## Integrating Into Your Existing Workflow

### Option A — Scheduled batch check (recommended)

Run a drift check on a rolling window of recent production data against your training baseline, on a schedule (e.g. daily cron).

```python
import httpx
import pandas as pd

# Load your training baseline and recent production window
reference = pd.read_parquet("training_data.parquet").sample(500).values.tolist()
current   = pd.read_parquet("production_last_7d.parquet").sample(500).values.tolist()

response = httpx.post("http://localhost:8000/api/v1/report", json={
    "reference": reference,
    "current":   current,
    "feature_names": ["age", "income", "score", "tenure"],
    "strategy": "majority",
    "threshold": 0.05
})

result = response.json()

if result["drift_detected"]:
    print(f"DRIFT DETECTED — score: {result['ensemble_score']:.3f}")
    print(f"Recommendation: {result['optimization']['recommendation']}")
    print(f"HTML report: http://localhost:8000{result['html_url']}")
    # trigger_retraining_pipeline()
    # send_slack_alert(result)
```

### Option B — Real-time inference monitoring

Accumulate a buffer of recent predictions and compare to your training distribution at each inference batch.

```python
from collections import deque
import httpx

REFERENCE = load_training_features()   # your baseline, loaded once
BUFFER    = deque(maxlen=200)          # rolling window of recent inputs

def predict(features: list[float]) -> float:
    BUFFER.append(features)

    # Check drift every 200 samples
    if len(BUFFER) == 200:
        resp = httpx.post("http://localhost:8000/api/v1/detect", json={
            "reference": REFERENCE,
            "current":   list(BUFFER),
        }).json()

        if resp["drift_detected"]:
            log_warning("Input drift detected", score=resp["ensemble_score"])

    return model.predict([features])[0]
```

### Option C — CI/CD data validation gate

Block model promotion if the new training dataset has drifted significantly from the last production baseline.

```bash
# In your CI pipeline (e.g. GitHub Actions, GitLab CI)
python scripts/check_drift.py \
  --reference data/baseline.parquet \
  --current   data/new_training.parquet \
  --threshold 0.05 \
  --fail-on-drift
```

```python
# scripts/check_drift.py
import sys, httpx, pandas as pd, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--reference")
parser.add_argument("--current")
parser.add_argument("--threshold", type=float, default=0.05)
parser.add_argument("--fail-on-drift", action="store_true")
args = parser.parse_args()

ref = pd.read_parquet(args.reference).sample(300).values.tolist()
cur = pd.read_parquet(args.current).sample(300).values.tolist()

result = httpx.post("http://localhost:8000/api/v1/detect", json={
    "reference": ref, "current": cur, "threshold": args.threshold
}).json()

print(f"Drift detected: {result['drift_detected']}  score: {result['ensemble_score']:.3f}")
if result["drift_detected"] and args.fail_on_drift:
    sys.exit(1)   # fail the pipeline
```

---

## API Reference

### `POST /api/v1/detect`
Run the three-model ensemble. Returns voting result. **Nothing is stored.**

```json
// Request
{
  "reference":     [[...], [...]],   // baseline samples (≥5 rows)
  "current":       [[...], [...]],   // new samples to check (≥5 rows)
  "feature_names": ["f1", "f2"],     // optional — auto-named if omitted
  "strategy":      "majority",       // "majority" | "unanimous" | "any"
  "threshold":     0.05              // significance level (default 0.05)
}

// Response
{
  "drift_detected":  true,
  "strategy":        "majority",
  "votes_for_drift": 3,
  "total_detectors": 3,
  "confidence":      1.0,
  "ensemble_score":  0.98,
  "feature_scores":  { "age": 0.41, "income": 0.38, "score": 0.21 },
  "detector_results": [ ... ]
}
```

### `POST /api/v1/report`
Same as `/detect` but also **persists to SQLite** and generates an HTML report.
Response includes `html_url` and `optimization.predicted_improvement_pct`.

### `GET /api/v1/report/{run_id}/html`
View the full HTML drift report in a browser.

### `GET /api/v1/metrics/dashboard`
Interactive HTML dashboard — drift rate over time, detector agreement, top drifted features, run history table.

### `GET /api/v1/metrics/performance`
```json
{
  "total_runs": 42,
  "drift_detected_count": 11,
  "drift_rate": 0.262,
  "detector_agreement_rate": 0.857,
  "avg_ensemble_score": 0.34,
  "avg_predicted_improvement_pct": 12.4,
  "most_drifted_features": ["income", "age", "score"]
}
```

---

## Optimization Ratio

Every `/report` response includes a predicted performance improvement estimate — how much better your model could perform if you retrained it to account for the detected drift.

| Ensemble score | Estimated improvement |
|----------------|-----------------------|
| 0.0 – 0.2      | 0 – 4 %               |
| 0.2 – 0.5      | 4 – 15 %              |
| 0.5 – 0.8      | 15 – 33 %             |
| 0.8 – 1.0      | 33 – 45 %             |

This is a calibrated heuristic, not a guarantee — treat it as a prioritisation signal.