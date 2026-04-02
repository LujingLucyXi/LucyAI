"""
Data pipeline module for the ML system design framework.

Covers:
- Data loading and ingestion
- Data validation and quality checks
- Feature engineering and preprocessing
- Train/validation/test splitting
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import DataConfig
from utils import ensure_directory, setup_logger, timer

logger = setup_logger(__name__)


class DataValidator:
    """Validates a DataFrame against configured quality rules."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self._log = logging.getLogger(__name__)

    def validate(self, df: pd.DataFrame) -> Dict[str, object]:
        """
        Run all quality checks and return a report.

        Parameters
        ----------
        df:
            Raw input DataFrame.

        Returns
        -------
        dict
            Validation report with keys ``passed``, ``warnings``, and ``errors``.
        """
        warnings: List[str] = []
        errors: List[str] = []

        # 1. Required columns
        for col in self.config.required_columns:
            if col not in df.columns:
                errors.append(f"Required column '{col}' is missing.")

        # 2. Target column
        if self.config.target_column not in df.columns:
            errors.append(
                f"Target column '{self.config.target_column}' is missing."
            )

        # 3. Missing-value rate
        for col in df.columns:
            missing_rate = df[col].isna().mean()
            if missing_rate > self.config.missing_value_threshold:
                warnings.append(
                    f"Column '{col}' has {missing_rate:.1%} missing values "
                    f"(threshold: {self.config.missing_value_threshold:.1%})."
                )

        # 4. Duplicate rows
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            warnings.append(f"Dataset contains {n_dupes} duplicate rows.")

        # 5. Empty dataset
        if len(df) == 0:
            errors.append("Dataset is empty.")

        passed = len(errors) == 0
        report = {"passed": passed, "warnings": warnings, "errors": errors}
        self._log.info(
            "Validation %s — %d error(s), %d warning(s).",
            "passed" if passed else "FAILED",
            len(errors),
            len(warnings),
        )
        return report


class FeatureEngineer:
    """Applies feature engineering transformations to a DataFrame."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self._scaler = StandardScaler()
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timer
    def fit_transform(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Fit on *df* and return the transformed DataFrame."""
        target_col = target_col or self.config.target_column
        feature_df = df.drop(columns=[target_col], errors="ignore")
        self._identify_column_types(feature_df)
        transformed = self._apply_transformations(feature_df, fit=True)
        self._fitted = True
        if target_col in df.columns:
            transformed[target_col] = df[target_col].values
        return transformed

    @timer
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform *df* using previously fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        target_col = self.config.target_column
        feature_df = df.drop(columns=[target_col], errors="ignore")
        transformed = self._apply_transformations(feature_df, fit=False)
        if target_col in df.columns:
            transformed[target_col] = df[target_col].values
        return transformed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _identify_column_types(self, df: pd.DataFrame) -> None:
        self._numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self._categorical_cols = df.select_dtypes(
            include=["str", "object", "category"]
        ).columns.tolist()

    def _apply_transformations(
        self, df: pd.DataFrame, fit: bool
    ) -> pd.DataFrame:
        result = df.copy()

        # --- Missing value imputation ---
        for col in self._numeric_cols:
            if col in result.columns:
                if fit:
                    fill_value = result[col].median()
                else:
                    col_idx = self._numeric_cols.index(col)
                    if hasattr(self._scaler, "mean_") and col_idx < len(self._scaler.mean_):
                        fill_value = self._scaler.mean_[col_idx]
                    else:
                        fill_value = result[col].median()
                result[col] = result[col].fillna(fill_value)

        for col in self._categorical_cols:
            if col in result.columns:
                result[col] = result[col].fillna("__MISSING__")

        # --- Outlier clipping ---
        for col in self._numeric_cols:
            if col in result.columns:
                mean = result[col].mean()
                std = result[col].std()
                threshold = self.config.outlier_std_threshold
                result[col] = result[col].clip(
                    mean - threshold * std, mean + threshold * std
                )

        # --- Encoding ---
        for col in self._categorical_cols:
            if col not in result.columns:
                continue
            if col not in self._label_encoders:
                self._label_encoders[col] = LabelEncoder()
            le = self._label_encoders[col]
            if fit:
                result[col] = le.fit_transform(result[col].astype(str))
            else:
                known = set(le.classes_)
                result[col] = result[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                result[col] = le.transform(result[col])

        # --- Scaling ---
        if self._numeric_cols:
            numeric_present = [
                c for c in self._numeric_cols if c in result.columns
            ]
            if numeric_present:
                if fit:
                    result[numeric_present] = self._scaler.fit_transform(
                        result[numeric_present]
                    )
                else:
                    result[numeric_present] = self._scaler.transform(
                        result[numeric_present]
                    )

        return result


class DataPipeline:
    """
    Orchestrates the full data pipeline:
    load → validate → engineer features → split.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.validator = DataValidator(config)
        self.engineer = FeatureEngineer(config)
        self._log = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timer
    def load(self, source: str) -> pd.DataFrame:
        """
        Load data from *source*.

        Supports CSV, JSON, and Parquet files, as well as existing DataFrames
        passed as a string label (used in testing).
        """
        self._log.info("Loading data from: %s", source)
        ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
        loaders = {
            "csv": pd.read_csv,
            "json": pd.read_json,
            "parquet": pd.read_parquet,
        }
        loader = loaders.get(ext)
        if loader is None:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                f"Supported formats: {list(loaders)}"
            )
        df = loader(source)
        self._log.info("Loaded %d rows × %d columns.", *df.shape)
        return df

    @timer
    def process(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run validation, feature engineering, and splitting.

        Returns
        -------
        tuple
            ``(train_df, val_df, test_df)``
        """
        # Validate
        report = self.validator.validate(df)
        if not report["passed"]:
            raise ValueError(
                "Data validation failed:\n" + "\n".join(report["errors"])
            )
        for warning in report["warnings"]:
            self._log.warning(warning)

        # Drop high-missing columns
        high_missing = [
            col
            for col in df.columns
            if df[col].isna().mean() > self.config.missing_value_threshold
            and col != self.config.target_column
        ]
        if high_missing:
            self._log.info("Dropping high-missing columns: %s", high_missing)
            df = df.drop(columns=high_missing)

        # Split first (fit engineer on train only to avoid leakage)
        train_val, test = train_test_split(
            df,
            test_size=self.config.test_split,
            random_state=self.config.random_seed,
        )
        val_size_adjusted = self.config.validation_split / (
            1 - self.config.test_split
        )
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=self.config.random_seed,
        )

        # Feature engineering
        train = self.engineer.fit_transform(train)
        val = self.engineer.transform(val)
        test = self.engineer.transform(test)

        self._log.info(
            "Split — train: %d, val: %d, test: %d",
            len(train),
            len(val),
            len(test),
        )
        return train, val, test

    def save_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """Persist train/val/test splits to the configured processed data path."""
        ensure_directory(self.config.processed_data_path)
        train.to_csv(
            f"{self.config.processed_data_path}/train.csv", index=False
        )
        val.to_csv(f"{self.config.processed_data_path}/val.csv", index=False)
        test.to_csv(f"{self.config.processed_data_path}/test.csv", index=False)
        self._log.info(
            "Splits saved to '%s'.", self.config.processed_data_path
        )
