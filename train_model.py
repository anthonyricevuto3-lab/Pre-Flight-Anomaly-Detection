"""Train a lightweight, dependency-free anomaly model (IsolationForest replacement)."""
from __future__ import annotations

# Diagnostic imports and early environment checks to help debug missing packages
import sys
import random
import time
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


def _randomize_csv_file(csv_path: Path, output_path: Path = None) -> Path:
    """
    Create a randomized version of the CSV file with variations in the data.
    
    :param csv_path: Path to the original CSV file
    :param output_path: Path for the randomized CSV (if None, overwrites original)
    :return: Path to the randomized CSV file
    """
    if output_path is None:
        output_path = csv_path
    
    rows = []
    header = None
    
    # Read the original CSV
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    if not rows:
        raise ValueError("CSV file is empty")
    
    print(f"DEBUG: Read {len(rows)} rows from original CSV")
    
    # Randomize the data
    randomized_rows = []
    
    for i, row in enumerate(rows):
        new_row = row.copy()
        
        # Add timestamp variation (±5 seconds)
        if 'timestamp' in new_row:
            original_time = float(new_row['timestamp'])
            time_variation = random.uniform(-5, 5)
            new_row['timestamp'] = str(max(0, original_time + time_variation))
        
        # Add variations to numerical columns
        for col in ['rpm', 'temperature', 'pressure', 'voltage']:
            if col in new_row:
                try:
                    original_value = float(new_row[col])
                    # Add 3-8% random variation
                    variation_factor = random.uniform(0.03, 0.08)
                    variation = random.uniform(-variation_factor, variation_factor) * original_value
                    new_value = original_value + variation
                    
                    # Round to appropriate decimal places
                    if col == 'rpm':
                        new_value = round(new_value)
                    else:
                        new_value = round(new_value, 1)
                    
                    new_row[col] = str(new_value)
                except ValueError:
                    pass  # Skip if can't convert to float
        
        randomized_rows.append(new_row)
    
    # Shuffle the rows (except header)
    random.shuffle(randomized_rows)
    
    # Add some duplicate rows with slight variations (10% more data)
    additional_rows = []
    num_additional = max(1, len(randomized_rows) // 10)
    
    for _ in range(num_additional):
        base_row = random.choice(randomized_rows).copy()
        
        # Add slight variations to the duplicate
        for col in ['rpm', 'temperature', 'pressure', 'voltage']:
            if col in base_row:
                try:
                    original_value = float(base_row[col])
                    small_variation = random.uniform(-0.02, 0.02) * original_value
                    new_value = original_value + small_variation
                    
                    if col == 'rpm':
                        new_value = round(new_value)
                    else:
                        new_value = round(new_value, 1)
                    
                    base_row[col] = str(new_value)
                except ValueError:
                    pass
        
        additional_rows.append(base_row)
    
    # Combine original randomized rows with additional variations
    all_rows = randomized_rows + additional_rows
    random.shuffle(all_rows)  # Final shuffle
    
    # Write the randomized CSV
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"DEBUG: Created randomized CSV with {len(all_rows)} rows at {output_path}")
    return output_path


def _backup_original_csv(csv_path: Path) -> Path:
    """Create a backup of the original CSV file if it doesn't exist."""
    backup_path = csv_path.with_suffix('.csv.original')
    
    if not backup_path.exists():
        print(f"DEBUG: Creating backup of original CSV at {backup_path}")
        with csv_path.open("r") as src, backup_path.open("w") as dst:
            dst.write(src.read())
    
    return backup_path


def _randomize_data(rows: List[List[float]], randomization_factor: float = 0.1) -> List[List[float]]:
    """
    Apply randomization to the training data to create variations.
    
    :param rows: Original data rows
    :param randomization_factor: Amount of noise to add (as fraction of value)
    :return: Randomized data rows
    """
    randomized_rows = []
    
    for row in rows:
        new_row = []
        for value in row:
            # Add random noise: value +/- (randomization_factor * value)
            noise = random.uniform(-randomization_factor, randomization_factor) * value
            new_value = value + noise
            new_row.append(new_value)
        randomized_rows.append(new_row)
    
    return randomized_rows


def _shuffle_and_sample_data(rows: List[List[float]], sample_ratio: float = 0.8) -> List[List[float]]:
    """
    Shuffle the data and optionally sample a subset for training.
    
    :param rows: Original data rows
    :param sample_ratio: Fraction of data to use for training (0.0 to 1.0)
    :return: Shuffled and sampled data
    """
    # Make a copy and shuffle
    shuffled_rows = rows.copy()
    random.shuffle(shuffled_rows)
    
    # Sample a subset if requested
    if 0 < sample_ratio < 1.0:
        sample_size = int(len(shuffled_rows) * sample_ratio)
        shuffled_rows = shuffled_rows[:sample_size]
    
    return shuffled_rows


def _augment_data(rows: List[List[float]], augmentation_factor: int = 2) -> List[List[float]]:
    """
    Create additional synthetic data points based on existing data.
    
    :param rows: Original data rows
    :param augmentation_factor: How many times to multiply the dataset
    :return: Augmented dataset
    """
    if augmentation_factor <= 1:
        return rows
    
    augmented_rows = rows.copy()
    
    for _ in range(augmentation_factor - 1):
        # Create variations of existing data
        for row in rows:
            new_row = []
            for value in row:
                # Add small random variation (5% of original value)
                variation = random.uniform(-0.05, 0.05) * value
                new_value = value + variation
                new_row.append(new_value)
            augmented_rows.append(new_row)
    
    return augmented_rows


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
    - Creates a backup of the original CSV file.
    - Randomizes the CSV file with new data variations.
    - Ensures all REQUIRED_FEATURES exist in the training CSV.
    - Trains a robust-statistics-based anomaly model with fixed params.
    - Saves the model to DEFAULT_MODEL_PATH.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    # Set random seed based on current time for different results each run
    random.seed(int(time.time()))
    print(f"DEBUG: Random seed set to: {int(time.time())}")

    # Create backup of original CSV file (only once)
    backup_path = _backup_original_csv(DATA_PATH)
    
    # Randomize the CSV file itself
    print("DEBUG: Randomizing CSV file...")
    randomized_csv_path = _randomize_csv_file(DATA_PATH)
    
    print(f"DEBUG: reading randomized CSV -> {randomized_csv_path}")
    X = _read_required_columns_from_csv(randomized_csv_path, REQUIRED_FEATURES)
    print(f"DEBUG: loaded {len(X)} valid rows with {len(REQUIRED_FEATURES)} features from randomized CSV")

    # Apply additional in-memory randomization techniques
    print("DEBUG: Applying additional data processing...")
    
    # 1. Shuffle the data
    random.shuffle(X)
    print(f"DEBUG: Data shuffled")
    
    # 2. Add slight noise for more variation
    X = _randomize_data(X, randomization_factor=0.02)  # 2% additional noise
    print(f"DEBUG: Applied 2% additional randomization noise")

    # Train the lightweight model. Hyperparameters chosen to be conservative.
    model = SimpleAnomalyModel(
        feature_names=REQUIRED_FEATURES,
        mad_threshold=3.0,               # ~3σ under normality
        min_features_over_threshold=1,   # flag if any single feature is extreme
    ).fit(X)

    _save_model(model, DEFAULT_MODEL_PATH)
    print(f"Model trained and saved to {DEFAULT_MODEL_PATH}")
    print(f"Model trained on {len(X)} randomized data points")
    print(f"Original CSV backed up to: {backup_path}")


if __name__ == "__main__":
    main()
