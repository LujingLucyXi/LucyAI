"""
Configuration management for the ML system design framework.
Centralizes all configuration parameters for data pipelines, model training,
serving, and monitoring.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    feature_store_path: str = "data/features"
    validation_split: float = 0.1
    test_split: float = 0.2
    random_seed: int = 42
    missing_value_threshold: float = 0.3  # Drop columns with > 30% missing
    outlier_std_threshold: float = 3.0
    categorical_max_cardinality: int = 50
    required_columns: List[str] = field(default_factory=list)
    target_column: str = "target"


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str = "random_forest"  # random_forest, gradient_boosting, logistic_regression
    model_registry_path: str = "models"
    experiment_name: str = "default_experiment"
    n_trials: int = 20  # Hyperparameter tuning trials
    cv_folds: int = 5
    scoring_metric: str = "f1"
    early_stopping_rounds: int = 10
    hyperparameter_search: str = "grid"  # grid or random
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "gradient_boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
        },
    })


@dataclass
class ServingConfig:
    """Configuration for model serving."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_batch_size: int = 100
    request_timeout: float = 5.0  # seconds
    model_cache_size: int = 3
    enable_ab_testing: bool = True
    canary_traffic_percent: float = 10.0
    health_check_interval: int = 30  # seconds


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and observability."""
    metrics_path: str = "monitoring/metrics"
    alert_email: Optional[str] = None
    alert_slack_webhook: Optional[str] = None
    drift_detection_window: int = 1000  # samples
    drift_significance_level: float = 0.05
    performance_degradation_threshold: float = 0.05  # 5% relative drop
    data_quality_check_interval: int = 3600  # seconds
    model_retraining_trigger_threshold: float = 0.10  # 10% performance drop
    log_predictions: bool = True
    metrics_retention_days: int = 90


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = "logs/ml_system.log"
    max_log_size_mb: int = 100
    backup_count: int = 5


@dataclass
class MLSystemConfig:
    """Top-level configuration aggregating all subsystem configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    environment: str = "development"  # development, staging, production

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def for_environment(cls, env: str) -> "MLSystemConfig":
        """Create a configuration appropriate for the given environment."""
        config = cls(environment=env)
        if env == "production":
            config.serving.workers = 8
            config.monitoring.log_predictions = True
            config.logging.log_level = "WARNING"
        elif env == "staging":
            config.serving.workers = 2
            config.monitoring.log_predictions = True
            config.logging.log_level = "INFO"
        return config
