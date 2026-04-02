"""
Model training pipeline for the ML system design framework.

Covers:
- Model selection and initialisation
- Hyperparameter tuning (grid / random search)
- Cross-validated training
- Model evaluation and persistence
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from config import ModelConfig
from utils import ensure_directory, setup_logger, timer

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Registry of supported model types
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Any] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}


def _build_estimator(config: ModelConfig) -> Any:
    """Instantiate a base estimator from the configuration."""
    cls = MODEL_REGISTRY.get(config.model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model type '{config.model_type}'. "
            f"Supported types: {list(MODEL_REGISTRY)}"
        )
    kwargs: Dict[str, Any] = {}
    if config.model_type in ("random_forest", "gradient_boosting"):
        kwargs["random_state"] = 42
    if config.model_type == "logistic_regression":
        kwargs["max_iter"] = 1000
    return cls(**kwargs)


class ModelTrainer:
    """
    Trains, tunes, and evaluates a scikit-learn classification model.

    Parameters
    ----------
    config:
        Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model: Optional[Any] = None
        self.best_params: Dict[str, Any] = {}
        self.cv_scores: np.ndarray = np.array([])
        self._log = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timer
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> Any:
        """
        Train a model with optional hyperparameter tuning.

        Parameters
        ----------
        train_df:
            Training DataFrame that includes the target column.
        val_df:
            Optional validation DataFrame used for post-training evaluation.

        Returns
        -------
        sklearn estimator
            The fitted model.
        """
        target = self.config.target_column if hasattr(self.config, "target_column") else "target"
        X_train = train_df.drop(columns=[target], errors="ignore")
        y_train = train_df[target]

        base_estimator = _build_estimator(self.config)
        param_grid = self.config.hyperparameters.get(self.config.model_type, {})

        if param_grid:
            self.model, self.best_params = self._tune(
                base_estimator, X_train, y_train, param_grid
            )
        else:
            self._log.info(
                "No hyperparameter grid defined for '%s'; training with defaults.",
                self.config.model_type,
            )
            self.model = base_estimator.fit(X_train, y_train)

        # Cross-validation on training data
        self.cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=self.config.cv_folds,
            scoring=self.config.scoring_metric,
        )
        self._log.info(
            "CV %s: %.4f ± %.4f",
            self.config.scoring_metric,
            self.cv_scores.mean(),
            self.cv_scores.std(),
        )

        if val_df is not None:
            self.evaluate(val_df, split_name="validation")

        return self.model

    def evaluate(
        self, df: pd.DataFrame, split_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate the fitted model on *df* and log the metrics.

        Returns
        -------
        dict
            Metric name → value mapping.
        """
        if self.model is None:
            raise RuntimeError("Call train() before evaluate().")

        target = self.config.target_column if hasattr(self.config, "target_column") else "target"
        X = df.drop(columns=[target], errors="ignore")
        y_true = df[target]
        y_pred = self.model.predict(X)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
        }

        # AUC — only for binary classification with predict_proba support
        if hasattr(self.model, "predict_proba") and len(np.unique(y_true)) == 2:
            y_prob = self.model.predict_proba(X)[:, 1]
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))

        self._log.info(
            "[%s] %s",
            split_name,
            "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )
        return metrics

    def save(self, version: str = "latest") -> str:
        """
        Persist the fitted model and its metadata.

        Returns
        -------
        str
            Path to the saved model directory.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call train() first.")

        model_dir = os.path.join(self.config.model_registry_path, version)
        ensure_directory(model_dir)

        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "wb") as fh:
            pickle.dump(self.model, fh)

        meta = {
            "model_type": self.config.model_type,
            "version": version,
            "best_params": self.best_params,
            "cv_mean": float(self.cv_scores.mean()) if len(self.cv_scores) else None,
            "cv_std": float(self.cv_scores.std()) if len(self.cv_scores) else None,
        }
        meta_path = os.path.join(model_dir, "metadata.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        self._log.info("Model saved to '%s'.", model_dir)
        return model_dir

    @staticmethod
    def load(model_dir: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a previously saved model.

        Parameters
        ----------
        model_dir:
            Path returned by :meth:`save`.

        Returns
        -------
        tuple
            ``(estimator, metadata_dict)``
        """
        model_path = os.path.join(model_dir, "model.pkl")
        meta_path = os.path.join(model_dir, "metadata.json")

        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        with open(meta_path) as fh:
            meta = json.load(fh)

        logger.info("Model loaded from '%s'.", model_dir)
        return model, meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tune(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """Run hyperparameter search and return the best estimator + params."""
        self._log.info(
            "Starting %s hyperparameter search for '%s'.",
            self.config.hyperparameter_search,
            self.config.model_type,
        )

        search_cls = (
            GridSearchCV
            if self.config.hyperparameter_search == "grid"
            else RandomizedSearchCV
        )
        search_kwargs: Dict[str, Any] = dict(
            estimator=estimator,
            param_grid=param_grid if self.config.hyperparameter_search == "grid" else None,
            scoring=self.config.scoring_metric,
            cv=self.config.cv_folds,
            n_jobs=-1,
            refit=True,
        )
        if self.config.hyperparameter_search != "grid":
            search_kwargs["param_distributions"] = search_kwargs.pop("param_grid")
            search_kwargs["n_iter"] = self.config.n_trials
            search_kwargs["random_state"] = 42

        searcher = search_cls(**search_kwargs)
        searcher.fit(X, y)

        self._log.info(
            "Best %s: %.4f | params: %s",
            self.config.scoring_metric,
            searcher.best_score_,
            searcher.best_params_,
        )
        return searcher.best_estimator_, searcher.best_params_
