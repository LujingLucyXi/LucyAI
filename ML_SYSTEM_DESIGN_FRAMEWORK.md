# ML System Design Framework

> A production-ready, end-to-end guide for building, deploying, and operating Machine Learning systems at scale.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1 – Problem Definition & Analysis](#phase-1--problem-definition--analysis)
3. [Phase 2 – Data Architecture](#phase-2--data-architecture)
4. [Phase 3 – Model Development](#phase-3--model-development)
5. [Phase 4 – Model Deployment](#phase-4--model-deployment)
6. [Phase 5 – Monitoring & Observability](#phase-5--monitoring--observability)
7. [Phase 6 – Feedback Loop & Iteration](#phase-6--feedback-loop--iteration)
8. [Technology Selection Guide](#technology-selection-guide)
9. [Best Practices & Anti-Patterns](#best-practices--anti-patterns)
10. [Repository Structure](#repository-structure)
11. [Quick Start](#quick-start)

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                       ML SYSTEM LIFECYCLE                             │
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐    │
│  │   Problem    │───▶│    Data      │───▶│  Model Development   │    │
│  │  Definition  │    │ Architecture │    │  (train/eval/tune)   │    │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘    │
│                                                     │                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────▼───────────┐    │
│  │   Feedback   │◀───│  Monitoring  │◀───│  Model Deployment    │    │
│  │     Loop     │    │Observability │    │  (serving/A-B test)  │    │
│  └──────────────┘    └──────────────┘    └──────────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘
```

The framework treats ML as a **continuous engineering discipline**, not a one-shot modelling exercise.  Each phase feeds into the next, and the monitoring + feedback loop closes the cycle.

---

## Phase 1 – Problem Definition & Analysis

### Business Objectives

Before writing a single line of code, align with stakeholders on:

| Question | Example Answer |
|----------|---------------|
| What decision will the model support? | Flag fraudulent transactions in real time |
| What is the current baseline? | Manual review catches 40% of fraud |
| What is the acceptable error type? | False negatives (missed fraud) cost more than false positives |
| What is the required latency? | < 100 ms per request |

### Success Metrics

Choose metrics that reflect **business value**, not just statistical performance:

| Metric | Formula | When to Use |
|--------|---------|-------------|
| Precision | TP / (TP + FP) | Cost of false positives is high |
| Recall | TP / (TP + FN) | Cost of false negatives is high |
| F1 | 2·P·R / (P+R) | Balance precision and recall |
| AUC-ROC | Area under ROC curve | Ranking quality matters |
| Accuracy | (TP+TN) / N | Classes are balanced |
| MRR / NDCG | — | Ranking / recommendation tasks |

### Constraints Checklist

- [ ] **Latency**: p50 / p95 / p99 targets defined
- [ ] **Throughput**: requests-per-second budget
- [ ] **Cost**: inference + training budget per month
- [ ] **Explainability**: regulatory or UX requirement?
- [ ] **Fairness**: protected attributes identified?
- [ ] **Privacy**: PII data handling and compliance (GDPR, CCPA)

### Data Requirements

```
┌─────────────────────────────────────────────────────┐
│  Data Requirement Assessment                        │
│                                                     │
│  ✔ Minimum labelled samples   → ~10 × features     │
│  ✔ Class balance              → define threshold   │
│  ✔ Feature availability       → prod vs. train gap │
│  ✔ Label latency              → days / weeks?      │
│  ✔ Historical coverage        → seasonality?       │
└─────────────────────────────────────────────────────┘
```

---

## Phase 2 – Data Architecture

### Data Collection & Ingestion

```python
# data_pipeline.py — DataPipeline.load()
pipeline = DataPipeline(config)
df = pipeline.load("data/raw/events.csv")   # CSV / JSON / Parquet supported
```

**Ingestion patterns**:

| Pattern | Latency | Use Case |
|---------|---------|----------|
| Batch (daily/hourly) | Minutes–hours | Offline training, reporting |
| Micro-batch (Spark Streaming) | Seconds–minutes | Near-real-time features |
| Streaming (Kafka + Flink) | Milliseconds | Real-time feature computation |

### Data Storage & Versioning

```
data/
├── raw/          ← immutable, append-only
├── processed/    ← train.csv, val.csv, test.csv
└── features/     ← feature store snapshots
```

**Key principle**: Always version datasets alongside model artifacts.  Use a tool such as DVC, Delta Lake, or LakeFS for reproducibility.

### Data Quality & Validation

```python
# data_pipeline.py — DataValidator
validator = DataValidator(config)
report = validator.validate(df)
# report = {"passed": True/False, "warnings": [...], "errors": [...]}
```

Checks performed automatically:

1. Required columns present
2. Target column exists
3. Missing value rate per column
4. Duplicate row detection
5. Empty dataset guard

### Feature Engineering

```python
# data_pipeline.py — FeatureEngineer
engineer = FeatureEngineer(config)
train_transformed = engineer.fit_transform(train_df)   # fit + transform
val_transformed   = engineer.transform(val_df)          # transform only (no leakage!)
```

**Transformations applied**:
- Numeric: median imputation → outlier clipping (±3σ) → standard scaling
- Categorical: `__MISSING__` fill → label encoding

> ⚠️ **Anti-pattern**: Fitting the scaler/encoder on the full dataset before splitting causes **data leakage**. Always fit on training data only.

---

## Phase 3 – Model Development

### Exploratory Data Analysis

Before training, always:

```python
import pandas as pd
df.describe(include="all")        # distribution summary
df.isnull().mean().sort_values()  # missing rate
df["target"].value_counts()       # class balance
df.corr()                         # feature correlation
```

### Model Selection Decision Tree

```
Is the target continuous?
├── YES → Regression
│         ├── Linear (interpretable, fast)
│         ├── Gradient Boosting (high accuracy)
│         └── Neural Network (complex patterns)
└── NO  → Classification
          ├── Binary?
          │   ├── Logistic Regression (baseline)
          │   ├── Random Forest (robust)
          │   └── Gradient Boosting (state-of-the-art tabular)
          └── Multiclass? → One-vs-Rest or Softmax
```

### Training Pipeline

```python
# model_training.py — ModelTrainer
config = ModelConfig(
    model_type="random_forest",
    cv_folds=5,
    hyperparameter_search="grid",
    hyperparameters={
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
        }
    }
)
trainer = ModelTrainer(config)
trainer.train(train_df, val_df=val_df)
```

### Hyperparameter Tuning

| Strategy | Pros | Cons |
|----------|------|------|
| Grid Search | Exhaustive, reproducible | Expensive for large grids |
| Random Search | Faster, good coverage | Non-deterministic |
| Bayesian Optimisation (Optuna) | Sample-efficient | More complex setup |

### Model Evaluation

```python
metrics = trainer.evaluate(test_df, split_name="test")
# {"accuracy": 0.94, "precision": 0.93, "recall": 0.95, "f1": 0.94, "auc": 0.98}
```

---

## Phase 4 – Model Deployment

### Model Versioning & Persistence

```python
# Save model + metadata
model_dir = trainer.save(version="v1.2.0")
# Produces: models/v1.2.0/model.pkl  +  models/v1.2.0/metadata.json

# Load model
model, meta = ModelTrainer.load(model_dir)
```

### FastAPI Serving Application

```python
# model_serving.py — create_app()
from model_serving import create_app, ServingConfig

config = ServingConfig(port=8000, workers=4)
app = create_app(config)
# uvicorn model_serving:app --host 0.0.0.0 --port 8000
```

**Available endpoints**:

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Liveness probe |
| GET  | `/ready` | Readiness probe (503 if no model) |
| POST | `/predict` | Single-sample prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/models/{version}/activate` | Promote version to active |
| POST | `/models/{version}/canary` | Enable canary routing |
| DELETE | `/models/canary` | Disable canary routing |

### A/B Testing & Canary Deployments

```python
registry = ModelRegistry(serving_config)
registry.register("v1", stable_model)
registry.register("v2", new_model)

registry.set_active("v1")     # 90 % of traffic
registry.set_canary("v2")     # 10 % of traffic (configurable)
```

```
Traffic routing:
  ┌────────┐   90%   ┌──────────┐
  │Request │────────▶│ Model v1 │ (stable)
  │        │   10%   ├──────────┤
  └────────┘────────▶│ Model v2 │ (canary)
                     └──────────┘
```

### Deployment Checklist

- [ ] Model registered in registry with version tag
- [ ] Canary traffic percentage configured
- [ ] Health + readiness endpoints verified
- [ ] Rollback procedure documented
- [ ] Load test performed (target QPS)
- [ ] Latency SLA validated (p99 < threshold)

---

## Phase 5 – Monitoring & Observability

### Performance Metrics Tracking

```python
# monitoring.py — MonitoringSystem
monitor = MonitoringSystem(config)
monitor.metrics.record("f1", 0.92, tags={"env": "prod", "version": "v1"})
monitor.metrics.latest("f1")   # → 0.92
```

### Data Drift Detection

```python
monitor.data_drift.set_reference(train_features_df)

# Later, in production:
result = monitor.check_data_drift(production_batch_df)
# result = {"drift_detected": True, "columns": {"age": {"ks_statistic": 0.3, "p_value": 0.001, "drifted": True}, ...}}
```

Two-sample **Kolmogorov-Smirnov test** per numeric feature at configurable significance level (default 0.05).

**Population Stability Index (PSI)**:

| PSI Value | Interpretation |
|-----------|---------------|
| < 0.10    | No significant shift |
| 0.10–0.25 | Moderate shift — investigate |
| > 0.25    | Significant shift — retrain |

```python
psi = DataDriftDetector.psi(reference_arr, production_arr)
```

### Model Drift Detection

```python
monitor.model_drift.set_baseline(0.94)       # post-training F1
monitor.check_performance("f1", live_f1)     # call each evaluation cycle
# Fires a CRITICAL alert if relative drop > 10 % (configurable)
```

### Alert Mechanisms

```python
def send_to_slack(level, message, meta):
    import requests
    requests.post(webhook_url, json={"text": f"[{level.upper()}] {message}"})

monitor.alerts.register_handler(send_to_slack)
```

---

## Phase 6 – Feedback Loop & Iteration

### Continuous Improvement Cycle

```
Performance degrades
        │
        ▼
Monitor fires alert
        │
        ├── Data drift?  → Re-ingest fresh data, re-run pipeline
        ├── Label drift? → Collect new labels, audit annotation
        └── Concept drift? → Update features or model architecture
        │
        ▼
Retraining pipeline triggered
        │
        ▼
Shadow deployment (serve new model, log predictions, don't act)
        │
        ▼
A/B test: route 10% traffic to new model
        │
        ▼
Compare metrics over N days
        │
        ├── New model wins → promote to active, retire old
        └── New model loses → roll back, investigate
```

### Retraining Triggers

```python
# example_workflow.py — check_retraining_trigger()
retrain_needed = check_retraining_trigger(monitor)
if retrain_needed:
    # Kick off retraining job (Airflow DAG, GitHub Actions, etc.)
    pass
```

Automatic triggers include:

| Trigger | Default Threshold |
|---------|-----------------|
| Performance drop | > 10 % relative |
| Data drift (PSI) | > 0.25 |
| Scheduled retraining | Weekly / monthly |
| Volume change | ± 30 % of expected traffic |

---

## Technology Selection Guide

### Data Layer

```
Data Volume < 10 GB  → pandas + local parquet
Data Volume 10–500 GB → Spark / Dask on cluster
Data Volume > 500 GB  → distributed lake (Delta / Iceberg)

Feature Store:
  OSS / low-cost → Feast (offline) + Redis (online)
  Managed       → Databricks Feature Store, Vertex AI Feature Store
```

### Model Training

```
Tabular data       → scikit-learn, XGBoost, LightGBM
Images             → PyTorch / TensorFlow + transfer learning
Text               → HuggingFace Transformers
Time Series        → Prophet, TFT, N-HiTS
Experiment Tracking → MLflow, Weights & Biases, Neptune
```

### Model Serving

```
Low latency (< 50 ms) → FastAPI + uvicorn (this framework)
Batch inference       → Spark MLlib predict, AWS Batch
GPU serving           → Triton Inference Server
Managed               → SageMaker, Vertex AI, Azure ML
```

### Monitoring

```
Custom lightweight  → This framework (monitoring.py)
OSS comprehensive   → Evidently AI, Grafana + Prometheus
Managed             → AWS Model Monitor, Arize AI, WhyLabs
```

---

## Best Practices & Anti-Patterns

### ✅ Best Practices

1. **Separate concerns**: keep data, training, serving, and monitoring as independent modules.
2. **Version everything**: datasets, features, models, and configs.
3. **Avoid data leakage**: fit preprocessors only on training data.
4. **Test your pipeline**: unit-test every transformation and serving endpoint.
5. **Monitor both data and model drift**: they require different responses.
6. **Define rollback procedures** before every deployment.
7. **Use shadow mode first**: deploy a new model silently before routing live traffic.
8. **Log predictions in production** for future labelled datasets.

### ❌ Anti-Patterns

| Anti-Pattern | Problem | Fix |
|---|---|---|
| Fitting scaler on full dataset | Data leakage | Fit only on train split |
| Shipping model without monitoring | Silent degradation | Add drift + performance checks |
| No model versioning | Can't roll back | Tag every artifact |
| Single evaluation metric | Misleading on imbalanced data | Use precision, recall, F1, AUC |
| Manual hyperparameter guessing | Suboptimal performance | Grid/random/Bayesian search |
| Training-serving skew | Feature mismatch in prod | Shared feature logic in feature store |
| No canary deployment | Big-bang risk | Always A/B test before full promotion |
| Retraining from scratch every time | Slow, expensive | Consider warm-starting or fine-tuning |

---

## Repository Structure

```
LucyAI/
├── config.py               ← Centralised configuration (DataConfig, ModelConfig, …)
├── data_pipeline.py        ← DataValidator, FeatureEngineer, DataPipeline
├── model_training.py       ← ModelTrainer (train, tune, evaluate, save/load)
├── model_serving.py        ← FastAPI app, ModelRegistry, A/B routing
├── monitoring.py           ← MetricStore, drift detectors, AlertManager
├── utils.py                ← Logging setup, timer decorator, helpers
├── example_workflow.py     ← End-to-end demo script
├── tests/
│   ├── __init__.py
│   └── test_ml_framework.py  ← 51 unit tests (pytest)
├── data/
│   ├── raw/
│   └── processed/
├── models/                 ← Versioned model artifacts
├── monitoring/
│   └── metrics/
└── ML_SYSTEM_DESIGN_FRAMEWORK.md   ← This document
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install scikit-learn pandas numpy fastapi uvicorn pydantic httpx pytest scipy
```

### 2. Run the example workflow

```bash
python example_workflow.py
```

### 3. Run unit tests

```bash
pytest tests/test_ml_framework.py -v
```

### 4. Start the serving API

```python
import uvicorn
from model_serving import create_app, ServingConfig
from model_training import ModelTrainer

# Load your trained model
trainer, meta = ModelTrainer.load("models/v1")

# Build and configure the app
app = create_app(ServingConfig())
app.state.registry.register("v1", trainer)

uvicorn.run(app, host="0.0.0.0", port=8000)
```

Then call the API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_0": 0.5, "feature_1": -1.2}}'
```

### 5. Customise the configuration

```python
from config import MLSystemConfig, ModelConfig

config = MLSystemConfig.for_environment("production")
config.model.model_type = "gradient_boosting"
config.serving.canary_traffic_percent = 5.0
config.monitoring.model_retraining_trigger_threshold = 0.05
```

---

*Framework designed for Python 3.9+. Tested with scikit-learn 1.x, FastAPI 0.100+, pandas 2.x/3.x.*
