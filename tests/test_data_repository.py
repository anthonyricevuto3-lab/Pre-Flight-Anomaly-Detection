"""Verification tests for the training-data repository.

Verifies: SR-021, SR-022, SR-023.
"""

from pathlib import Path

import pytest

from preflight.data_repository import (
    _extract_feature_row,
    augment_samples,
    load_training_samples,
)
from preflight.errors import TrainingDataError

_VALID_CSV = "timestamp,rpm,temperature,pressure,voltage\n1,1500,75.0,3000.0,28.0\n2,1600,80.0,3100.0,29.0\n"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_valid_csv(tmp_path):
    path = _write(tmp_path, _VALID_CSV)
    samples = load_training_samples(path)
    assert samples == [[1500.0, 75.0, 3000.0, 28.0], [1600.0, 80.0, 3100.0, 29.0]]


def test_missing_file_raises_training_data_error(tmp_path):
    with pytest.raises(TrainingDataError):
        load_training_samples(tmp_path / "does_not_exist.csv")


def test_empty_after_filtering_raises(tmp_path):
    path = _write(tmp_path, "timestamp,rpm,temperature,pressure,voltage\n1,,,,\n")
    with pytest.raises(TrainingDataError):
        load_training_samples(path)


def test_extract_rejects_non_finite():
    record = {"rpm": "nan", "temperature": "75", "pressure": "3000", "voltage": "28"}
    assert _extract_feature_row(record) is None


def test_extract_rejects_missing_field():
    record = {"rpm": "1500", "temperature": "75", "pressure": "3000"}
    assert _extract_feature_row(record) is None


def test_augment_is_deterministic_for_seed():
    samples = [[1500.0, 75.0, 3000.0, 28.0]]
    assert augment_samples(samples, seed=42) == augment_samples(samples, seed=42)


def test_augment_rejects_invalid_variation():
    with pytest.raises(AssertionError):
        augment_samples([[1.0]], seed=1, variation=1.5)
