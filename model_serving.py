"""
Model serving module for the ML system design framework.

Provides a FastAPI application with:
- Single-sample and batch prediction endpoints
- A/B testing and canary deployment support
- Health and readiness checks
- Model versioning
"""

from __future__ import annotations

import logging
import pickle
import random
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import ServingConfig
from utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """Schema for a single prediction request."""

    features: Dict[str, Any] = Field(
        ..., description="Feature name → value mapping."
    )
    request_id: Optional[str] = Field(
        None, description="Optional client-supplied request identifier."
    )


class BatchPredictionRequest(BaseModel):
    """Schema for a batch prediction request."""

    records: List[Dict[str, Any]] = Field(
        ..., description="List of feature dictionaries."
    )
    request_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """Schema for a prediction response."""

    prediction: Any
    probability: Optional[float] = None
    model_version: str
    request_id: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Schema for a batch prediction response."""

    predictions: List[Any]
    probabilities: Optional[List[Optional[float]]] = None
    model_version: str
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Model registry / cache
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    In-process model registry supporting multiple named versions.

    Parameters
    ----------
    config:
        Serving configuration.
    """

    def __init__(self, config: ServingConfig) -> None:
        self.config = config
        self._models: Dict[str, Any] = {}
        self._active_version: Optional[str] = None
        self._canary_version: Optional[str] = None
        self._log = logging.getLogger(__name__)

    # ------------------------------------------------------------------

    def register(self, version: str, model: Any) -> None:
        """Register *model* under *version*."""
        if len(self._models) >= self.config.model_cache_size:
            oldest = next(iter(self._models))
            self._log.info("Evicting cached model version '%s'.", oldest)
            del self._models[oldest]
        self._models[version] = model
        if self._active_version is None:
            self._active_version = version
        self._log.info("Registered model version '%s'.", version)

    def load_from_file(self, path: str, version: str) -> None:
        """Load a pickled model from *path* and register it as *version*."""
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        self.register(version, model)

    def set_active(self, version: str) -> None:
        if version not in self._models:
            raise KeyError(f"Version '{version}' is not registered.")
        self._active_version = version
        self._log.info("Active model version set to '%s'.", version)

    def set_canary(self, version: Optional[str]) -> None:
        if version is not None and version not in self._models:
            raise KeyError(f"Canary version '{version}' is not registered.")
        self._canary_version = version
        self._log.info("Canary model version set to '%s'.", version)

    def get_model(self) -> tuple[Any, str]:
        """
        Return ``(model, version)`` according to A/B / canary routing.

        If a canary version is configured and A/B testing is enabled, traffic
        is routed to the canary with probability
        ``config.canary_traffic_percent / 100``.
        """
        if (
            self.config.enable_ab_testing
            and self._canary_version is not None
            and random.random() < self.config.canary_traffic_percent / 100
        ):
            version = self._canary_version
        else:
            version = self._active_version

        if version is None:
            raise RuntimeError("No model version is registered.")
        return self._models[version], version

    @property
    def active_version(self) -> Optional[str]:
        return self._active_version

    @property
    def is_ready(self) -> bool:
        return self._active_version is not None


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app(config: Optional[ServingConfig] = None) -> FastAPI:
    """
    Build and return the FastAPI serving application.

    Parameters
    ----------
    config:
        Serving configuration.  Uses defaults when *None*.
    """
    if config is None:
        config = ServingConfig()

    registry = ModelRegistry(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        logger.info("ML serving application started.")
        yield
        logger.info("ML serving application shutting down.")

    app = FastAPI(
        title="LucyAI ML Model Serving",
        description="Production model serving with A/B testing and canary deployments.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Expose registry so tests / integration code can inject models.
    app.state.registry = registry

    # ------------------------------------------------------------------ #
    # Routes                                                               #
    # ------------------------------------------------------------------ #

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        """Liveness probe — always returns 200 if the process is alive."""
        return HealthResponse(
            status="ok",
            model_loaded=registry.is_ready,
            model_version=registry.active_version,
        )

    @app.get("/ready", tags=["ops"])
    async def ready() -> dict:
        """Readiness probe — 200 only when a model is loaded."""
        if not registry.is_ready:
            raise HTTPException(status_code=503, detail="No model loaded.")
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    async def predict(request: PredictionRequest) -> PredictionResponse:
        """Run a single-sample prediction."""
        if not registry.is_ready:
            raise HTTPException(status_code=503, detail="No model loaded.")

        model, version = registry.get_model()
        df = pd.DataFrame([request.features])

        try:
            prediction = model.predict(df)[0]
            probability: Optional[float] = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df)[0]
                probability = float(np.max(proba))
        except Exception as exc:
            logger.error("Prediction error: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return PredictionResponse(
            prediction=_to_python(prediction),
            probability=probability,
            model_version=version,
            request_id=request.request_id,
        )

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["inference"],
    )
    async def predict_batch(
        request: BatchPredictionRequest,
    ) -> BatchPredictionResponse:
        """Run batch predictions (up to ``max_batch_size`` samples)."""
        if not registry.is_ready:
            raise HTTPException(status_code=503, detail="No model loaded.")
        if len(request.records) > config.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.records)} exceeds limit {config.max_batch_size}.",
            )

        model, version = registry.get_model()
        df = pd.DataFrame(request.records)

        try:
            predictions = model.predict(df).tolist()
            probabilities: Optional[List[Optional[float]]] = None
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(df)
                probabilities = [float(np.max(row)) for row in probas]
        except Exception as exc:
            logger.error("Batch prediction error: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return BatchPredictionResponse(
            predictions=[_to_python(p) for p in predictions],
            probabilities=probabilities,
            model_version=version,
            request_id=request.request_id,
        )

    @app.post("/models/{version}/activate", tags=["model-management"])
    async def activate_model(version: str) -> dict:
        """Promote *version* to the active (production) slot."""
        try:
            registry.set_active(version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"message": f"Version '{version}' is now active."}

    @app.post("/models/{version}/canary", tags=["model-management"])
    async def set_canary(version: str) -> dict:
        """Route canary traffic to *version*."""
        try:
            registry.set_canary(version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "message": f"Canary set to '{version}' "
            f"({config.canary_traffic_percent}% traffic)."
        }

    @app.delete("/models/canary", tags=["model-management"])
    async def clear_canary() -> dict:
        """Disable canary routing."""
        registry.set_canary(None)
        return {"message": "Canary routing disabled."}

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_python(value: Any) -> Any:
    """Convert numpy scalars to native Python types for JSON serialisation."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
