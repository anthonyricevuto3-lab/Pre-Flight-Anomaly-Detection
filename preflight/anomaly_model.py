"""Robust median/MAD anomaly detector.

A lightweight, dependency-free detector. A feature value is considered
anomalous when it deviates from the training median by more than
``mad_threshold`` scaled MADs. A row is flagged when at least
``min_features_over_threshold`` of its features are anomalous.

NASA traceability:
* NASA-STD-8739.8 - deterministic model; identical inputs yield identical
  outputs, supporting verification and reproducibility.
* Power of Ten Rule 5 - runtime assertions guard every method.
* Power of Ten Rule 6 - model state is encapsulated, not global.
* Power of Ten Rule 7 - all inputs are validated before use.

Requirements satisfied: SR-010, SR-011, SR-014, SR-015.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .config import DEFAULT_MAD_THRESHOLD, DEFAULT_MIN_FEATURES_OVER_THRESHOLD
from .robust_stats import median, median_absolute_deviation

# Detector label convention, matching common anomaly-detection APIs.
INLIER_LABEL: int = 1
OUTLIER_LABEL: int = -1


class RobustAnomalyDetector:
    """Median/MAD anomaly detector over a fixed set of named features."""

    def __init__(
        self,
        feature_names: Sequence[str],
        mad_threshold: float = DEFAULT_MAD_THRESHOLD,
        min_features_over_threshold: int = DEFAULT_MIN_FEATURES_OVER_THRESHOLD,
    ) -> None:
        assert len(feature_names) > 0, "at least one feature is required"
        assert mad_threshold > 0.0, "mad_threshold must be positive"
        assert min_features_over_threshold >= 1, "min_features_over_threshold must be >= 1"
        assert min_features_over_threshold <= len(feature_names), (
            "min_features_over_threshold cannot exceed the number of features"
        )

        self._feature_names: Tuple[str, ...] = tuple(feature_names)
        self._mad_threshold: float = float(mad_threshold)
        self._min_features_over_threshold: int = int(min_features_over_threshold)
        # name -> (median, scaled_mad)
        self._statistics: Dict[str, Tuple[float, float]] = {}
        self._is_fitted: bool = False

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Return the ordered tuple of feature names."""
        return self._feature_names

    def fit(self, samples: List[List[float]]) -> "RobustAnomalyDetector":
        """Estimate per-feature median and MAD from training samples.

        Args:
            samples: A non-empty list of equal-length numeric rows.

        Returns:
            RobustAnomalyDetector: ``self``, fitted.
        """
        assert isinstance(samples, list), "samples must be a list"
        assert len(samples) > 0, "training data must not be empty"

        columns = list(zip(*samples))
        assert len(columns) == len(self._feature_names), "feature count mismatch in training data"

        for name, column in zip(self._feature_names, columns):
            column_values = list(column)
            center = median(column_values)
            spread = median_absolute_deviation(column_values, center)
            self._statistics[name] = (center, spread)

        self._is_fitted = True
        assert len(self._statistics) == len(self._feature_names), "statistics incompletely populated"
        return self

    def feature_statistics(self) -> Dict[str, Tuple[float, float]]:
        """Return a copy of the fitted (median, MAD) statistics per feature."""
        assert self._is_fitted, "detector must be fitted before reading statistics"
        assert len(self._statistics) == len(self._feature_names), "statistics incompletely populated"
        return dict(self._statistics)

    def classify_row(self, row: Sequence[float]) -> int:
        """Classify a single feature row.

        Args:
            row: A sequence of feature values in ``feature_names`` order.

        Returns:
            int: ``OUTLIER_LABEL`` (-1) if anomalous, else ``INLIER_LABEL`` (1).
        """
        assert self._is_fitted, "detector must be fitted before prediction"
        assert len(row) == len(self._feature_names), "row length does not match feature count"

        exceedances = 0
        for name, value in zip(self._feature_names, row):
            center, spread = self._statistics[name]
            if abs(float(value) - center) > self._mad_threshold * spread:
                exceedances += 1

        return OUTLIER_LABEL if exceedances >= self._min_features_over_threshold else INLIER_LABEL

    def predict(self, samples: List[List[float]]) -> List[int]:
        """Classify a batch of feature rows.

        Args:
            samples: A list of feature rows.

        Returns:
            List[int]: One label per input row, in order.
        """
        assert self._is_fitted, "detector must be fitted before prediction"
        assert isinstance(samples, list), "samples must be a list"

        labels = [self.classify_row(row) for row in samples]
        assert len(labels) == len(samples), "prediction count must match input count"
        return labels
