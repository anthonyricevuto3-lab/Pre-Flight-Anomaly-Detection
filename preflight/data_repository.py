"""Training-data access layer.

Reads and validates the training CSV and optionally produces a
deterministic, augmented copy for sensitivity analysis. Data access is
isolated here so the rest of the system is independent of storage format.

NASA traceability:
* Power of Ten Rule 2 - every loop is bounded by ``MAX_TRAINING_ROWS``.
* Power of Ten Rule 7 - return values and parsed fields are validated;
  malformed rows are rejected, not silently coerced.
* NASA-STD-8739.8 - augmentation is seeded and therefore reproducible.

Requirements satisfied: SR-021, SR-022, SR-023.
"""

from __future__ import annotations

import csv
import logging
import math
import random
from pathlib import Path
from typing import List, Optional

from .config import MAX_TRAINING_ROWS, REQUIRED_FEATURES
from .errors import TrainingDataError

_LOGGER = logging.getLogger(__name__)


def load_training_samples(csv_path: Path) -> List[List[float]]:
    """Load and validate training samples from a CSV file.

    Args:
        csv_path: Path to a CSV file containing the required feature columns.

    Returns:
        List[List[float]]: Validated feature rows in ``REQUIRED_FEATURES`` order.

    Raises:
        TrainingDataError: If the file is missing or contains no valid rows.
    """
    assert isinstance(csv_path, Path), "csv_path must be a Path"

    if not csv_path.exists():
        raise TrainingDataError(f"Training data file not found: {csv_path.name}")

    samples: List[List[float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, record in enumerate(reader):
            if index >= MAX_TRAINING_ROWS:  # Power of Ten Rule 2: bounded loop.
                _LOGGER.warning("Training row cap (%d) reached; remaining rows ignored", MAX_TRAINING_ROWS)
                break
            row = _extract_feature_row(record)
            if row is not None:
                samples.append(row)

    if not samples:
        raise TrainingDataError("No valid training samples were found in the data file")

    assert len(samples) <= MAX_TRAINING_ROWS, "loaded rows exceed configured cap"
    return samples


def _extract_feature_row(record: dict) -> Optional[List[float]]:
    """Parse one CSV record into a validated feature row.

    Args:
        record: A mapping of column name to raw string value.

    Returns:
        Optional[List[float]]: The parsed row, or ``None`` if the record is
        missing fields or contains non-finite values.
    """
    assert record is not None, "record must not be None"
    assert isinstance(record, dict), "record must be a mapping"

    try:
        row = [float(record[name]) for name in REQUIRED_FEATURES]
    except (KeyError, ValueError, TypeError):
        return None

    for value in row:
        if math.isnan(value) or math.isinf(value):
            return None
    return row


def augment_samples(
    samples: List[List[float]],
    seed: int,
    variation: float = 0.05,
) -> List[List[float]]:
    """Return a deterministic, slightly perturbed copy of ``samples``.

    Each value is scaled by a pseudo-random factor in
    ``[1 - variation, 1 + variation]``. Because the generator is seeded, the
    output is fully reproducible for a given ``seed`` (NASA-STD-8739.8:
    repeatable verification).

    Args:
        samples: Source feature rows.
        seed: Seed for the local pseudo-random generator.
        variation: Maximum fractional perturbation (0.0 - 1.0).

    Returns:
        List[List[float]]: A new, perturbed list of rows.
    """
    assert isinstance(samples, list), "samples must be a list"
    assert len(samples) > 0, "cannot augment an empty sample set"
    assert 0.0 <= variation < 1.0, "variation must be in [0.0, 1.0)"

    generator = random.Random(seed)
    augmented: List[List[float]] = []
    for row in samples[:MAX_TRAINING_ROWS]:  # Power of Ten Rule 2: bounded loop.
        augmented.append([value + generator.uniform(-variation, variation) * value for value in row])

    assert len(augmented) == min(len(samples), MAX_TRAINING_ROWS), "augmentation lost rows"
    return augmented
