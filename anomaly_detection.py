"""Utility functions for loading the Isolation Forest model and detecting anomalies.

These helpers are shared by both the command line scripts and the Azure Functions
entry point so that we keep the anomaly detection logic in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "isolation_forest_model.pkl"
REQUIRED_FEATURES = ["rpm", "temperature", "pressure", "voltage"]


_model_cache: Optional[object] = None
_model_path_cache: Optional[Path] = None


def load_model(model_path: Optional[Path] = None):
    """Load and cache the Isolation Forest model.

    Parameters
    ----------
    model_path:
        Optional override of the path to the model. When not provided the
        default ``isolation_forest_model.pkl`` that lives in the repository
        root is used.
    """

    global _model_cache, _model_path_cache

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Isolation Forest model not found at {path}")

    if _model_cache is None or _model_path_cache != path:
        _model_cache = joblib.load(path)
        _model_path_cache = path

    return _model_cache


def _to_dataframe(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Convert a sequence of dict-like records into a pandas ``DataFrame``.

    Any additional columns that are not strictly required are preserved so that
    they can be echoed back in the anomaly report.
    """

    frame = pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("No records supplied for anomaly detection")

    missing = [feature for feature in REQUIRED_FEATURES if feature not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(sorted(missing))
        )

    return frame


def detect_anomalies_from_records(
    records: Iterable[Mapping[str, object]],
    *,
    model_path: Optional[Path] = None,
) -> List[MutableMapping[str, object]]:
    """Identify anomalous records from an iterable of mapping objects.

    Parameters
    ----------
    records:
        Iterable of dictionaries (or dictionary-like objects) containing at
        least the keys specified in ``REQUIRED_FEATURES``.
    model_path:
        Optional override for where to load the trained model from.

    Returns
    -------
    List[MutableMapping[str, object]]
        The subset of records that were classified as anomalies. The returned
        mappings include the original keys so any metadata (for example a
        ``timestamp``) is preserved.
    """

    frame = _to_dataframe(records)
    model = load_model(model_path=model_path)

    predictions = model.predict(frame[REQUIRED_FEATURES])
    anomalies: List[MutableMapping[str, object]] = []
    for prediction, record in zip(predictions, frame.to_dict(orient="records")):
        if prediction == -1:
            anomalies.append(record)

    return anomalies


def detect_anomalies_from_csv(
    csv_path: Path,
    *,
    model_path: Optional[Path] = None,
) -> List[MutableMapping[str, object]]:
    """Convenience wrapper that reads sensor data from a CSV file."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV data file not found at {path}")

    frame = pd.read_csv(path)
    return detect_anomalies_from_records(frame.to_dict(orient="records"), model_path=model_path)
