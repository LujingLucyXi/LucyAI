"""
End-to-end example workflow demonstrating the ML system design framework.

Runs a complete pipeline:
1. Synthetic data generation
2. Data pipeline (validation → feature engineering → splitting)
3. Model training with hyperparameter tuning
4. Model serving (local A/B routing demonstration)
5. Monitoring setup and drift check
6. Retraining trigger logic
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# ---------------------------------------------------------------------------
# Resolve imports regardless of working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from config import DataConfig, MLSystemConfig, ModelConfig, MonitoringConfig
from data_pipeline import DataPipeline
from model_serving import ModelRegistry, ServingConfig, create_app
from model_training import ModelTrainer
from monitoring import MonitoringSystem
from utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# 1.  Data generation helper
# ---------------------------------------------------------------------------


def generate_synthetic_dataset(n_samples: int = 500) -> pd.DataFrame:
    """Generate a labelled binary-classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y
    # Introduce a small categorical feature for the pipeline to encode
    df["category"] = np.random.choice(["A", "B", "C"], size=n_samples)
    return df


# ---------------------------------------------------------------------------
# 2.  Data pipeline
# ---------------------------------------------------------------------------


def run_data_pipeline(df: pd.DataFrame, config: MLSystemConfig):
    logger.info("=== Step 1: Data Pipeline ===")
    pipeline = DataPipeline(config.data)
    train, val, test = pipeline.process(df)
    logger.info(
        "Splits — train: %d, val: %d, test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# 3.  Model training
# ---------------------------------------------------------------------------


def run_model_training(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: MLSystemConfig,
    tmp_dir: str,
):
    logger.info("=== Step 2: Model Training ===")
    model_config = config.model
    model_config.model_registry_path = os.path.join(tmp_dir, "models")

    trainer = ModelTrainer(model_config)
    trainer.train(train, val_df=val)
    test_metrics = trainer.evaluate(test, split_name="test")
    logger.info("Test metrics: %s", test_metrics)

    model_dir = trainer.save(version="v1")
    return trainer, model_dir, test_metrics


# ---------------------------------------------------------------------------
# 4.  Model serving demonstration
# ---------------------------------------------------------------------------


def run_serving_demo(trainer: ModelTrainer, train: pd.DataFrame, config: MLSystemConfig):
    logger.info("=== Step 3: Model Serving Demo ===")
    serving_cfg = config.serving
    registry = ModelRegistry(serving_cfg)
    registry.register("v1", trainer.model)

    # Simulate a single prediction using feature names from the training set
    model, version = registry.get_model()
    target_col = config.data.target_column
    feature_cols = [c for c in train.columns if c != target_col]
    sample = {col: float(train[col].iloc[0]) for col in feature_cols}
    sample_df = pd.DataFrame([sample])
    pred = model.predict(sample_df)[0]
    logger.info("Sample prediction (version=%s): %s", version, pred)

    # Register a second model version as canary
    registry.register("v2", trainer.model)  # same weights for illustration
    registry.set_canary("v2")
    logger.info("Canary version 'v2' registered with %.0f%% traffic.", serving_cfg.canary_traffic_percent)

    return registry


# ---------------------------------------------------------------------------
# 5.  Monitoring
# ---------------------------------------------------------------------------


def run_monitoring(
    trainer: ModelTrainer,
    test: pd.DataFrame,
    test_metrics: dict,
    config: MLSystemConfig,
    tmp_dir: str,
):
    logger.info("=== Step 4: Monitoring ===")
    mon_config = MonitoringConfig(
        metrics_path=os.path.join(tmp_dir, "monitoring"),
        drift_detection_window=50,
        model_retraining_trigger_threshold=0.10,
    )
    monitor = MonitoringSystem(mon_config)

    # Register a logging alert handler
    def log_alert(level: str, message: str, meta: dict) -> None:
        logger.warning("[ALERT][%s] %s", level.upper(), message)

    monitor.alerts.register_handler(log_alert)

    # Set baseline performance and reference data
    monitor.model_drift.set_baseline(test_metrics.get("f1", 0.8))
    target_col = config.data.target_column
    feature_df = test.drop(columns=[target_col], errors="ignore")
    monitor.data_drift.set_reference(feature_df)

    # Simulate production traffic with slight noise
    monitor.metrics.record("request_count", 1000)
    for _ in range(20):
        monitor.check_performance("f1", test_metrics.get("f1", 0.8) - 0.02)

    # Simulate a drifted production batch
    drifted_df = feature_df.copy() * 3 + 5
    drift_result = monitor.check_data_drift(drifted_df)
    logger.info("Drift detected: %s", drift_result["drift_detected"])

    return monitor


# ---------------------------------------------------------------------------
# 6.  Retraining trigger
# ---------------------------------------------------------------------------


def check_retraining_trigger(monitor: MonitoringSystem) -> bool:
    """Return True if any critical alert indicates retraining is needed."""
    critical = [a for a in monitor.alerts.alerts if a["level"] == "critical"]
    if critical:
        logger.warning(
            "Retraining trigger fired — %d critical alert(s).", len(critical)
        )
        return True
    logger.info("No retraining needed at this time.")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting end-to-end ML framework workflow.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = MLSystemConfig(
            data=DataConfig(
                target_column="target",
                processed_data_path=os.path.join(tmp_dir, "data/processed"),
            ),
            model=ModelConfig(
                model_type="random_forest",
                model_registry_path=os.path.join(tmp_dir, "models"),
                cv_folds=3,
                hyperparameters={
                    "random_forest": {
                        "n_estimators": [50, 100],
                        "max_depth": [5, 10],
                    }
                },
            ),
        )

        df = generate_synthetic_dataset(n_samples=500)
        train, val, test = run_data_pipeline(df, config)
        trainer, model_dir, test_metrics = run_model_training(
            train, val, test, config, tmp_dir
        )
        registry = run_serving_demo(trainer, train, config)
        monitor = run_monitoring(trainer, test, test_metrics, config, tmp_dir)
        retrain = check_retraining_trigger(monitor)

        logger.info(
            "Workflow complete. Retraining triggered: %s", retrain
        )


if __name__ == "__main__":
    main()
