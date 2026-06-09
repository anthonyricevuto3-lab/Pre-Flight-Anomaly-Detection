"""Verification tests for the service orchestration layer.

Verifies: SR-003, SR-004, SR-005, SR-024, SR-040.
"""

from pathlib import Path

import pytest

from preflight import service
from preflight.errors import ValidationError

_CSV = (
    "timestamp,rpm,temperature,pressure,voltage\n"
    "1,1500,75.0,3000.0,28.0\n"
    "2,1510,75.5,3010.0,28.1\n"
    "3,1490,74.5,2990.0,27.9\n"
    "4,1505,75.2,3005.0,28.0\n"
)


def _write_csv(tmp_path: Path) -> Path:
    path = tmp_path / "airplane_data.csv"
    path.write_text(_CSV, encoding="utf-8")
    return path


def test_analyze_training_data_reports_all_rows(tmp_path):
    report = service.analyze_training_data(_write_csv(tmp_path))
    assert report["summary"]["total_readings"] == 4
    assert set(report["normal_operating_ranges"]) == {
        "rpm",
        "temperature",
        "pressure",
        "voltage",
    }


def test_detect_readings_flags_clear_outlier(tmp_path):
    path = _write_csv(tmp_path)
    readings = [{"rpm": 99999, "temperature": 75.0, "pressure": 3000.0, "voltage": 28.0}]
    report = service.detect_readings(readings, path)
    assert report["total_readings_analyzed"] == 1
    assert "ANOMALIES DETECTED" in report["analysis_result"]


def test_detect_readings_passes_nominal_reading(tmp_path):
    path = _write_csv(tmp_path)
    readings = [{"rpm": 1500, "temperature": 75.0, "pressure": 3000.0, "voltage": 28.0}]
    report = service.detect_readings(readings, path)
    assert report["anomalous_data"] == []


def test_detect_readings_rejects_all_invalid(tmp_path):
    path = _write_csv(tmp_path)
    with pytest.raises(ValidationError) as exc_info:
        service.detect_readings([{"rpm": 1500}], path)
    assert exc_info.value.details is not None
    assert "required_features" in exc_info.value.details


def test_detect_readings_skips_invalid_keeps_valid(tmp_path):
    path = _write_csv(tmp_path)
    readings = [
        {"rpm": 1500, "temperature": 75.0, "pressure": 3000.0, "voltage": 28.0},
        {"rpm": "oops", "temperature": 75.0, "pressure": 3000.0, "voltage": 28.0},
    ]
    report = service.detect_readings(readings, path)
    assert report["total_readings_analyzed"] == 1
