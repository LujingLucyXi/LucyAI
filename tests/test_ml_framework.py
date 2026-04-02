"""
Unit tests for the ML system design framework.

Covers:
- Data pipeline validation and feature engineering
- Model training, evaluation, and persistence
- Model serving endpoints (FastAPI)
- Monitoring: metric store, drift detection, alerting
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Resolve imports regardless of working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DataConfig, ModelConfig, MonitoringConfig, ServingConfig
from data_pipeline import DataPipeline, DataValidator, FeatureEngineer
from model_serving import ModelRegistry, create_app
from model_training import ModelTrainer
from monitoring import (
    AlertManager,
    DataDriftDetector,
    MetricStore,
    ModelDriftDetector,
    MonitoringSystem,
)
from utils import flatten_dict, setup_logger

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small labelled DataFrame for generic tests."""
    X, y = make_classification(
        n_samples=200, n_features=8, n_informative=4, random_state=0
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(8)])
    df["target"] = y
    return df


@pytest.fixture()
def data_config(tmp_path) -> DataConfig:
    return DataConfig(
        processed_data_path=str(tmp_path / "processed"),
        target_column="target",
        test_split=0.2,
        validation_split=0.1,
        random_seed=42,
    )


@pytest.fixture()
def model_config(tmp_path) -> ModelConfig:
    return ModelConfig(
        model_type="random_forest",
        model_registry_path=str(tmp_path / "models"),
        cv_folds=3,
        hyperparameters={},  # Skip tuning for speed
    )


@pytest.fixture()
def trained_model(sample_df, model_config) -> tuple:
    """Returns ``(trainer, train_df, test_df)`` with a fitted model."""
    X = sample_df.drop(columns=["target"])
    y = sample_df["target"]
    split = int(0.8 * len(sample_df))
    train = pd.concat([X.iloc[:split], y.iloc[:split]], axis=1)
    test = pd.concat([X.iloc[split:], y.iloc[split:]], axis=1)
    trainer = ModelTrainer(model_config)
    trainer.train(train)
    return trainer, train, test


# ===========================================================================
# 1. Data pipeline
# ===========================================================================


class TestDataValidator:
    def test_passes_valid_dataframe(self, sample_df, data_config):
        validator = DataValidator(data_config)
        report = validator.validate(sample_df)
        assert report["passed"] is True
        assert report["errors"] == []

    def test_detects_missing_target(self, sample_df, data_config):
        df_no_target = sample_df.drop(columns=["target"])
        validator = DataValidator(data_config)
        report = validator.validate(df_no_target)
        assert report["passed"] is False
        assert any("target" in e for e in report["errors"])

    def test_detects_required_column_missing(self, sample_df):
        cfg = DataConfig(required_columns=["nonexistent_col"])
        validator = DataValidator(cfg)
        report = validator.validate(sample_df)
        assert report["passed"] is False
        assert any("nonexistent_col" in e for e in report["errors"])

    def test_detects_empty_dataframe(self, data_config):
        empty = pd.DataFrame(columns=["feat_0", "target"])
        validator = DataValidator(data_config)
        report = validator.validate(empty)
        assert report["passed"] is False

    def test_warns_on_high_missing(self, data_config):
        df = pd.DataFrame(
            {"feat_0": [np.nan] * 100, "target": [0] * 100}
        )
        validator = DataValidator(data_config)
        report = validator.validate(df)
        # High missing rate should produce a warning
        assert any("feat_0" in w for w in report["warnings"])


class TestFeatureEngineer:
    def test_fit_transform_returns_same_shape(self, sample_df, data_config):
        engineer = FeatureEngineer(data_config)
        result = engineer.fit_transform(sample_df)
        # Should have same number of rows; target column is preserved
        assert len(result) == len(sample_df)

    def test_transform_without_fit_raises(self, sample_df, data_config):
        engineer = FeatureEngineer(data_config)
        with pytest.raises(RuntimeError, match="fit_transform"):
            engineer.transform(sample_df)

    def test_handles_categorical_column(self, sample_df, data_config):
        df = sample_df.copy()
        df["cat"] = np.random.choice(["X", "Y", "Z"], size=len(df))
        engineer = FeatureEngineer(data_config)
        result = engineer.fit_transform(df)
        # Categorical column should be encoded to numeric
        assert pd.api.types.is_numeric_dtype(result["cat"])

    def test_transform_unseen_category(self, sample_df, data_config):
        df_train = sample_df.copy()
        df_train["cat"] = "A"
        engineer = FeatureEngineer(data_config)
        engineer.fit_transform(df_train)

        df_prod = sample_df.copy()
        df_prod["cat"] = "UNSEEN"
        # Should not raise; unknown categories are mapped to a known class
        result = engineer.transform(df_prod)
        assert "cat" in result.columns


class TestDataPipeline:
    def test_process_returns_three_splits(self, sample_df, data_config):
        pipeline = DataPipeline(data_config)
        train, val, test = pipeline.process(sample_df)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_split_sizes_roughly_correct(self, sample_df, data_config):
        pipeline = DataPipeline(data_config)
        train, val, test = pipeline.process(sample_df)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_df)
        # Test should be ~20 %
        assert abs(len(test) / total - 0.2) < 0.05

    def test_raises_on_missing_target(self, sample_df, data_config):
        df_no_target = sample_df.drop(columns=["target"])
        pipeline = DataPipeline(data_config)
        with pytest.raises(ValueError, match="validation failed"):
            pipeline.process(df_no_target)

    def test_save_splits_creates_files(self, sample_df, data_config):
        pipeline = DataPipeline(data_config)
        train, val, test = pipeline.process(sample_df)
        pipeline.save_splits(train, val, test)
        base = data_config.processed_data_path
        assert os.path.exists(os.path.join(base, "train.csv"))
        assert os.path.exists(os.path.join(base, "val.csv"))
        assert os.path.exists(os.path.join(base, "test.csv"))

    def test_load_raises_on_unsupported_format(self, data_config):
        pipeline = DataPipeline(data_config)
        with pytest.raises(ValueError, match="Unsupported"):
            pipeline.load("data.xyz")


# ===========================================================================
# 2. Model training
# ===========================================================================


class TestModelTrainer:
    def test_train_returns_fitted_model(self, trained_model):
        trainer, train, _ = trained_model
        assert trainer.model is not None

    def test_evaluate_returns_metrics(self, trained_model):
        trainer, _, test = trained_model
        metrics = trainer.evaluate(test)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_before_train_raises(self, model_config, sample_df):
        trainer = ModelTrainer(model_config)
        with pytest.raises(RuntimeError, match="train"):
            trainer.evaluate(sample_df)

    def test_save_creates_files(self, trained_model, tmp_path):
        trainer, _, _ = trained_model
        trainer.config.model_registry_path = str(tmp_path / "registry")
        model_dir = trainer.save(version="v1")
        assert os.path.exists(os.path.join(model_dir, "model.pkl"))
        assert os.path.exists(os.path.join(model_dir, "metadata.json"))

    def test_load_roundtrip(self, trained_model, tmp_path):
        trainer, _, test = trained_model
        trainer.config.model_registry_path = str(tmp_path / "registry")
        model_dir = trainer.save(version="v1")
        model, meta = ModelTrainer.load(model_dir)
        assert model is not None
        assert meta["model_type"] == "random_forest"

    def test_cv_scores_populated_after_train(self, trained_model):
        trainer, _, _ = trained_model
        assert len(trainer.cv_scores) > 0

    def test_unknown_model_type_raises(self, model_config, sample_df):
        model_config.model_type = "not_a_real_model"
        trainer = ModelTrainer(model_config)
        X = sample_df.drop(columns=["target"])
        y = sample_df["target"]
        split = int(0.8 * len(sample_df))
        train = pd.concat([X.iloc[:split], y.iloc[:split]], axis=1)
        with pytest.raises(ValueError, match="Unknown model type"):
            trainer.train(train)


# ===========================================================================
# 3. Model serving
# ===========================================================================


@pytest.fixture()
def serving_app(trained_model):
    """FastAPI test client with a pre-loaded model."""
    trainer, _, _ = trained_model
    config = ServingConfig(enable_ab_testing=False)
    app = create_app(config)
    app.state.registry.register("v1", trainer.model)
    return TestClient(app)


class TestServingEndpoints:
    def test_health_returns_ok(self, serving_app):
        resp = serving_app.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_ready_returns_200_when_model_loaded(self, serving_app):
        resp = serving_app.get("/ready")
        assert resp.status_code == 200

    def test_predict_returns_prediction(self, serving_app):
        features = {f"feat_{i}": float(np.random.randn()) for i in range(8)}
        resp = serving_app.post("/predict", json={"features": features})
        assert resp.status_code == 200
        body = resp.json()
        assert "prediction" in body
        assert body["model_version"] == "v1"

    def test_predict_batch(self, serving_app):
        records = [
            {f"feat_{i}": float(np.random.randn()) for i in range(8)}
            for _ in range(5)
        ]
        resp = serving_app.post(
            "/predict/batch", json={"records": records}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 5

    def test_batch_exceeds_limit_returns_400(self, serving_app):
        config = ServingConfig(max_batch_size=3, enable_ab_testing=False)
        app = create_app(config)
        # No model loaded → expect 503 before size check; load one first.
        trainer_fixture = serving_app.app.state.registry._models["v1"]
        app.state.registry.register("v1", trainer_fixture)
        client = TestClient(app)
        records = [{"x": 1}] * 10
        resp = client.post("/predict/batch", json={"records": records})
        assert resp.status_code == 400

    def test_health_no_model_loaded(self):
        config = ServingConfig(enable_ab_testing=False)
        app = create_app(config)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is False

    def test_ready_no_model_returns_503(self):
        config = ServingConfig(enable_ab_testing=False)
        app = create_app(config)
        client = TestClient(app)
        resp = client.get("/ready")
        assert resp.status_code == 503

    def test_activate_model(self, serving_app):
        resp = serving_app.post("/models/v1/activate")
        assert resp.status_code == 200

    def test_activate_missing_version_returns_404(self, serving_app):
        resp = serving_app.post("/models/missing/activate")
        assert resp.status_code == 404


# ===========================================================================
# 4. Monitoring
# ===========================================================================


@pytest.fixture()
def mon_config(tmp_path) -> MonitoringConfig:
    return MonitoringConfig(
        metrics_path=str(tmp_path / "metrics"),
        drift_detection_window=10,
        model_retraining_trigger_threshold=0.10,
        drift_significance_level=0.05,
    )


@pytest.fixture()
def reference_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"a": rng.normal(0, 1, 200), "b": rng.normal(5, 2, 200)}
    )


class TestMetricStore:
    def test_record_and_query(self, mon_config):
        store = MetricStore(mon_config)
        store.record("accuracy", 0.9)
        store.record("accuracy", 0.85)
        results = store.query("accuracy")
        assert len(results) == 2
        assert results[0]["value"] == pytest.approx(0.9)

    def test_latest_returns_most_recent(self, mon_config):
        store = MetricStore(mon_config)
        store.record("f1", 0.7)
        store.record("f1", 0.8)
        assert store.latest("f1") == pytest.approx(0.8)

    def test_latest_returns_none_for_unknown(self, mon_config):
        store = MetricStore(mon_config)
        assert store.latest("unknown_metric") is None

    def test_metrics_persisted_to_csv(self, mon_config):
        store = MetricStore(mon_config)
        store.record("precision", 0.75)
        csv_path = os.path.join(mon_config.metrics_path, "metrics.csv")
        assert os.path.isfile(csv_path)


class TestDataDriftDetector:
    def test_no_drift_same_distribution(self, mon_config, reference_df):
        detector = DataDriftDetector(mon_config)
        detector.set_reference(reference_df)
        rng = np.random.default_rng(1)
        prod = pd.DataFrame(
            {"a": rng.normal(0, 1, 100), "b": rng.normal(5, 2, 100)}
        )
        result = detector.detect(prod)
        assert result["drift_detected"] is False

    def test_drift_different_distribution(self, mon_config, reference_df):
        detector = DataDriftDetector(mon_config)
        detector.set_reference(reference_df)
        # Shift mean drastically to guarantee drift
        shifted = reference_df + 1000
        result = detector.detect(shifted)
        assert result["drift_detected"] is True

    def test_detect_without_reference_raises(self, mon_config, reference_df):
        detector = DataDriftDetector(mon_config)
        with pytest.raises(RuntimeError, match="set_reference"):
            detector.detect(reference_df)

    def test_psi_zero_for_identical(self, reference_df):
        vals = reference_df["a"].values
        psi = DataDriftDetector.psi(vals, vals)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_psi_positive_for_shifted(self, reference_df):
        vals = reference_df["a"].values
        shifted = vals + 10
        psi = DataDriftDetector.psi(vals, shifted)
        assert psi > 0


class TestModelDriftDetector:
    def test_no_drift_stable_performance(self, mon_config):
        detector = ModelDriftDetector(mon_config)
        detector.set_baseline(0.9)
        result = detector.update(0.88)  # 2.2% drop — under 10% threshold
        assert result["drift_detected"] is False

    def test_drift_detected_on_large_drop(self, mon_config):
        detector = ModelDriftDetector(mon_config)
        detector.set_baseline(0.9)
        result = detector.update(0.7)  # 22% drop — exceeds 10% threshold
        assert result["drift_detected"] is True

    def test_no_drift_without_baseline(self, mon_config):
        detector = ModelDriftDetector(mon_config)
        result = detector.update(0.5)
        assert result["drift_detected"] is False


class TestAlertManager:
    def test_alert_fires_handler(self, mon_config):
        manager = AlertManager(mon_config)
        received: list = []
        manager.register_handler(
            lambda lvl, msg, meta: received.append((lvl, msg))
        )
        manager.alert("warning", "Test alert")
        assert len(received) == 1
        assert received[0] == ("warning", "Test alert")

    def test_alerts_property_accumulates(self, mon_config):
        manager = AlertManager(mon_config)
        manager.alert("info", "A")
        manager.alert("critical", "B")
        assert len(manager.alerts) == 2

    def test_broken_handler_does_not_propagate(self, mon_config):
        manager = AlertManager(mon_config)

        def bad_handler(lvl, msg, meta):
            raise RuntimeError("handler error")

        manager.register_handler(bad_handler)
        # Should not raise
        manager.alert("info", "Should not blow up")
        assert len(manager.alerts) == 1


class TestMonitoringSystem:
    def test_record_prediction_with_latency(self, mon_config):
        monitor = MonitoringSystem(mon_config)
        monitor.record_prediction({"x": 1}, 1, latency_ms=12.5)
        assert monitor.metrics.latest("prediction_latency_ms") == pytest.approx(12.5)

    def test_check_performance_records_metric(self, mon_config):
        monitor = MonitoringSystem(mon_config)
        monitor.model_drift.set_baseline(0.9)
        monitor.check_performance("accuracy", 0.88)
        assert monitor.metrics.latest("accuracy") is not None

    def test_check_data_drift_fires_alert_on_drift(
        self, mon_config, reference_df
    ):
        monitor = MonitoringSystem(mon_config)
        monitor.data_drift.set_reference(reference_df)
        shifted = reference_df + 1000
        received: list = []
        monitor.alerts.register_handler(
            lambda lvl, msg, meta: received.append(lvl)
        )
        monitor.check_data_drift(shifted)
        assert "warning" in received


# ===========================================================================
# 5. Utility tests
# ===========================================================================


class TestUtils:
    def test_flatten_dict_simple(self):
        nested = {"a": {"b": 1, "c": 2}, "d": 3}
        flat = flatten_dict(nested)
        assert flat == {"a.b": 1, "a.c": 2, "d": 3}

    def test_flatten_dict_deep(self):
        nested = {"x": {"y": {"z": 42}}}
        flat = flatten_dict(nested)
        assert flat == {"x.y.z": 42}

    def test_clamp(self):
        from utils import clamp
        assert clamp(5.0, 0.0, 10.0) == pytest.approx(5.0)
        assert clamp(-1.0, 0.0, 10.0) == pytest.approx(0.0)
        assert clamp(15.0, 0.0, 10.0) == pytest.approx(10.0)
        assert clamp(0.0, 0.0, 0.0) == pytest.approx(0.0)

