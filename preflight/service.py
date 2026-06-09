"""Application service layer (business logic orchestration).

Coordinates data loading, model fitting, and report construction. This
layer is transport-agnostic: it contains no HTTP types, which keeps it
independently unit-testable (NASA-STD-8739.8: verification).

NASA traceability:
* Power of Ten Rule 4 - each routine is short and single-purpose.
* Power of Ten Rule 6 - no module-level mutable state; the detector is
  built locally per request.
* Power of Ten Rule 7 - inputs validated; caller faults raise ValidationError.

Requirements satisfied: SR-003, SR-004, SR-005, SR-024.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .anomaly_model import OUTLIER_LABEL, RobustAnomalyDetector
from .config import (
    DEFAULT_MAD_THRESHOLD,
    FEATURE_UNITS,
    MAX_READINGS_PER_REQUEST,
    REQUIRED_FEATURES,
)
from .errors import ValidationError
from .robust_stats import median, median_absolute_deviation

_LOGGER = logging.getLogger(__name__)

Reading = Dict[str, Union[float, int, str]]

# Example payload returned to callers on validation failure to aid recovery.
_EXAMPLE_READING: Dict[str, float] = {
    "rpm": 1500.0,
    "temperature": 75.0,
    "pressure": 3000.0,
    "voltage": 28.0,
}


def build_detector(csv_path: Path) -> RobustAnomalyDetector:
    """Load training data and return a fitted detector.

    Args:
        csv_path: Path to the training data CSV.

    Returns:
        RobustAnomalyDetector: A fitted detector.
    """
    from .data_repository import load_training_samples

    assert isinstance(csv_path, Path), "csv_path must be a Path"
    samples = load_training_samples(csv_path)
    assert len(samples) > 0, "training data must not be empty"
    return RobustAnomalyDetector(REQUIRED_FEATURES).fit(samples)


def analyze_training_data(csv_path: Path) -> Dict[str, object]:
    """Classify every training row and summarize the results.

    Implements the read-only (GET) analysis behavior.

    Args:
        csv_path: Path to the training data CSV.

    Returns:
        Dict[str, object]: A JSON-serializable analysis report.
    """
    from .data_repository import load_training_samples

    assert isinstance(csv_path, Path), "csv_path must be a Path"

    samples = load_training_samples(csv_path)
    detector = RobustAnomalyDetector(REQUIRED_FEATURES).fit(samples)
    predictions = detector.predict(samples)
    assert len(predictions) == len(samples), "prediction/sample length mismatch"

    ranges = _operating_ranges(samples)
    anomalies, normal = _split_by_label(samples, predictions)
    return _analysis_report(samples, ranges, anomalies, normal)


def detect_readings(readings: List[Reading], csv_path: Path) -> Dict[str, object]:
    """Classify caller-supplied readings against a freshly fitted detector.

    Implements the (POST) detection behavior.

    Args:
        readings: A list of reading objects.
        csv_path: Path to the training data CSV.

    Returns:
        Dict[str, object]: A JSON-serializable detection report.

    Raises:
        ValidationError: If no supplied reading is valid.
    """
    assert isinstance(readings, list), "readings must be a list"
    assert len(readings) <= MAX_READINGS_PER_REQUEST, "reading count exceeds request cap"

    detector = build_detector(csv_path)
    processed: List[Reading] = []
    anomalous: List[Reading] = []
    validation_errors: List[str] = []

    for reading in readings[:MAX_READINGS_PER_REQUEST]:  # Power of Ten Rule 2.
        row, error = _validate_reading(reading)
        if error:
            validation_errors.append(error)
            continue
        processed.append(reading)
        if detector.classify_row(row) == OUTLIER_LABEL:
            anomalous.append(reading)

    if not processed:
        raise ValidationError(
            "No valid readings could be processed",
            details={
                "required_features": list(REQUIRED_FEATURES),
                "validation_errors": validation_errors,
                "example_valid_request": _EXAMPLE_READING,
            },
        )

    _LOGGER.info("Processed %d readings; %d anomalous", len(processed), len(anomalous))
    return _detection_report(processed, anomalous)


def _validate_reading(reading: Reading) -> Tuple[List[float], str]:
    """Validate one reading and extract its feature row.

    Args:
        reading: A candidate reading object.

    Returns:
        Tuple[List[float], str]: ``(row, "")`` on success, or
        ``([], error_message)`` on failure. Exactly one is meaningful.
    """
    if not isinstance(reading, dict):
        return [], "Reading must be a JSON object"

    missing = [name for name in REQUIRED_FEATURES if name not in reading]
    if missing:
        return [], f"Missing required features: {missing}"

    try:
        row = [float(reading[name]) for name in REQUIRED_FEATURES]
    except (ValueError, TypeError):
        return [], "All feature values must be numeric"
    return row, ""


def _operating_ranges(samples: List[List[float]]) -> Dict[str, Dict[str, object]]:
    """Compute per-feature normal operating ranges from training samples."""
    assert isinstance(samples, list), "samples must be a list"
    assert len(samples) > 0, "samples must not be empty"

    columns = list(zip(*samples))
    ranges: Dict[str, Dict[str, object]] = {}
    for name, column in zip(REQUIRED_FEATURES, columns):
        column_values = list(column)
        center = median(column_values)
        spread = median_absolute_deviation(column_values, center)
        ranges[name] = {
            "min": round(center - DEFAULT_MAD_THRESHOLD * spread, 2),
            "max": round(center + DEFAULT_MAD_THRESHOLD * spread, 2),
            "median": round(center, 2),
            "unit": FEATURE_UNITS[name],
        }
    return ranges


def _split_by_label(
    samples: List[List[float]],
    predictions: List[int],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Partition classified samples into anomalous and normal record lists."""
    assert len(samples) == len(predictions), "samples and predictions must align"

    anomalies: List[Dict[str, object]] = []
    normal: List[Dict[str, object]] = []
    for index, (row, label) in enumerate(zip(samples, predictions)):
        record = {
            "reading_id": index + 1,
            "data": {name: round(value, 2) for name, value in zip(REQUIRED_FEATURES, row)},
            "status": "ANOMALY" if label == OUTLIER_LABEL else "NORMAL",
        }
        (anomalies if label == OUTLIER_LABEL else normal).append(record)
    return anomalies, normal


def _analysis_report(
    samples: List[List[float]],
    ranges: Dict[str, Dict[str, object]],
    anomalies: List[Dict[str, object]],
    normal: List[Dict[str, object]],
) -> Dict[str, object]:
    """Assemble the GET analysis report payload."""
    assert len(samples) > 0, "samples must not be empty"
    assert len(anomalies) + len(normal) == len(samples), "partition must cover all samples"

    total = len(samples)
    return {
        "message": "Complete Anomaly Analysis of Training Data",
        "data_source": f"Analysis of {total} readings from the training data file",
        "normal_operating_ranges": ranges,
        "summary": {
            "total_readings": total,
            "anomalies_detected": len(anomalies),
            "normal_readings": len(normal),
            "anomaly_percentage": round((len(anomalies) / total) * 100, 2),
        },
        "detected_anomalies": anomalies,
        "normal_readings": normal[:10],
        "note": (
            "Normal ranges are derived from the training data using the "
            "Median Absolute Deviation (MAD). Values outside these ranges "
            "are flagged as anomalies."
        ),
    }


def _detection_report(
    processed: List[Reading],
    anomalous: List[Reading],
) -> Dict[str, object]:
    """Assemble the POST detection report payload."""
    assert len(processed) > 0, "at least one reading must have been processed"
    assert len(anomalous) <= len(processed), "anomalies cannot exceed processed readings"

    if anomalous:
        result = (
            f"ANOMALIES DETECTED: {len(anomalous)} out of {len(processed)} "
            "readings flagged as anomalous"
        )
    else:
        result = (
            f"NO ANOMALIES DETECTED: All {len(processed)} readings are "
            "within normal parameters"
        )
    return {
        "analysis_result": result,
        "anomalies": [_format_anomaly(item) for item in anomalous],
        "anomalous_data": anomalous,
        "total_readings_analyzed": len(processed),
    }


def _format_anomaly(reading: Reading) -> str:
    """Render a single anomalous reading as a readable summary string."""
    assert isinstance(reading, dict), "reading must be a dict"
    assert all(name in reading for name in REQUIRED_FEATURES), "reading missing required features"
    return ", ".join(f"{name}={reading[name]}" for name in REQUIRED_FEATURES)
