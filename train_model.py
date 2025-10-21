"""Train a lightweight, dependency-free anomaly model (IsolationForest replacement)."""
from __future__ import annotations

# Diagnostic imports and early environment checks to help debug missing packages
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional
import csv
import math
import pickle

# Prefer joblib if available (more efficient for large models); fall back to pickle.
try:
    import joblib as _joblib  # type: ignore
    print("DEBUG: joblib available -> True")
except Exception:
    _joblib = None  # type: ignore
    print("DEBUG: joblib available -> False")

# Local defaults replacing the missing `anomaly_detection` module.
REQUIRED_FEATURES: List[str] = ["rpm", "temperature", "pressure", "voltage"]
DEFAULT_MODEL_PATH = Path("models") / "isolation_forest_model.pkl"
DATA_PATH = Path("airplane_data.csv")


def _save_model(obj, path: Path) -> None:
    """Persist the trained model to disk. Uses joblib if available, else pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _joblib is not None:
        _joblib.dump(obj, path)
    else:
        with path.open("wb") as fh:
            pickle.dump(obj, fh)


def _median(values: List[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of empty list")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _mad(values: List[float], med: Optional[float] = None) -> float:
    # Median Absolute Deviation (normalized with 1.4826 to approximate std under normality)
    if med is None:
        med = _median(values)
    abs_dev = [abs(v - med) for v in values]
    raw_mad = _median(abs_dev)
    return 1.4826 * raw_mad


class SimpleAnomalyModel:
    """
    A tiny, dependency-free replacement for IsolationForest.

    It learns per-feature robust statistics (median & MAD) and flags points as
    anomalous when a configurable proportion of features exceed a MAD threshold.
    This trades algorithmic sophistication for zero heavy dependencies while
    remaining practical and deterministic for embedded/edge deployments.

    API intentionally mirrors common scikit-learn patterns: fit(), predict().
    """

    def __init__(
        self,
        feature_names: List[str],
        mad_threshold: float = 3.0,
        min_features_over_threshold: int = 1,
    ) -> None:
        """
        :param feature_names: ordered list of feature names.
        :param mad_threshold: how many MADs from median to treat as outlier per feature.
        :param min_features_over_threshold: minimum number of features that must exceed
               the per-feature threshold to flag the row as an anomaly.
        """
        self.feature_names = feature_names
        self.mad_threshold = float(mad_threshold)
        self.min_features_over_threshold = int(min_features_over_threshold)
        self._stats: Dict[str, Tuple[float, float]] = {}  # name -> (median, mad)

    def fit(self, X: List[List[float]]) -> "SimpleAnomalyModel":
        if not X:
            raise ValueError("Training data is empty")
        cols = list(zip(*X))  # column-wise
        if len(cols) != len(self.feature_names):
            raise ValueError("Feature count mismatch during fit")

        for name, col in zip(self.feature_names, cols):
            col_list = list(col)
            med = _median(col_list)
            mad = _mad(col_list, med=med)
            # Prevent degenerate zero-MAD; use small epsilon so thresholding still works
            if mad <= 1e-12:
                mad = 1e-6
            self._stats[name] = (med, mad)
        return self

    def decision_function(self, X: List[List[float]]) -> List[float]:
        """
        Returns a simple anomaly score: higher means more normal (negative = more anomalous).
        Score here is the negative sum of per-feature (|x - med| / (mad * thresh)),
        clipped so that values within threshold contribute up to 1.0 each; beyond threshold
        they contribute >1, making the total more negative.
        """
        scores: List[float] = []
        for row in X:
            if len(row) != len(self.feature_names):
                raise ValueError("Feature count mismatch during decision_function")
            parts = []
            for (name, x) in zip(self.feature_names, row):
                med, mad = self._stats[name]
                z = abs(x - med) / (mad * self.mad_threshold)
                parts.append(z)  # z > 1 => over threshold
            # More over-threshold features => larger sum => more anomalous.
            # Negate so that more normal -> closer to 0 or positive; anomalous -> more negative.
            scores.append(-sum(parts))
        return scores

    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Mimic IsolationForest's predict: 1 for inliers, -1 for outliers.
        A row is an outlier if count(|x - med| > mad_threshold * mad) >= min_features_over_threshold.
        """
        labels: List[int] = []
        for row in X:
            over = 0
            for (name, x) in zip(self.feature_names, row):
                med, mad = self._stats[name]
                if abs(x - med) > self.mad_threshold * mad:
                    over += 1
            labels.append(-1 if over >= self.min_features_over_threshold else 1)
        return labels


def _read_required_columns_from_csv(
    path: Path, required_cols: List[str]
) -> List[List[float]]:
    """
    Read required columns as floats from a CSV file. Rows with missing/invalid values
    in required columns are skipped (with a DEBUG note).
    """
    rows: List[List[float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in required_cols if c not in header]
        if missing:
            raise ValueError(
                "Training data is missing required columns: " + ", ".join(sorted(missing))
            )

        for i, rec in enumerate(reader, start=1):
            try:
                row = [float(rec[col]) for col in required_cols]
                # Filter out NaN/inf rows
                if any(math.isnan(v) or math.isinf(v) for v in row):
                    print(f"DEBUG: skipping row {i} due to NaN/inf")
                    continue
                rows.append(row)
            except Exception:
                print(f"DEBUG: skipping row {i} due to parse error in required columns")
                continue
    if not rows:
        raise ValueError("No valid rows found after parsing CSV")
    return rows


def main() -> None:
    """Train and persist a dependency-free anomaly model from CSV REQUIRED_FEATURES.

    - Validates the presence of DATA_PATH.
    - Ensures all REQUIRED_FEATURES exist in the training CSV.
    - Trains a robust-statistics-based anomaly model with fixed params.
    - Saves the model to DEFAULT_MODEL_PATH.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    print(f"DEBUG: reading CSV -> {DATA_PATH}")
    X = _read_required_columns_from_csv(DATA_PATH, REQUIRED_FEATURES)
    print(f"DEBUG: loaded {len(X)} valid rows with {len(REQUIRED_FEATURES)} features")

    # Train the lightweight model. Hyperparameters chosen to be conservative.
    model = SimpleAnomalyModel(
        feature_names=REQUIRED_FEATURES,
        mad_threshold=3.0,               # ~3σ under normality
        min_features_over_threshold=1,   # flag if any single feature is extreme
    ).fit(X)

    _save_model(model, DEFAULT_MODEL_PATH)
    print(f"Model trained and saved to {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
