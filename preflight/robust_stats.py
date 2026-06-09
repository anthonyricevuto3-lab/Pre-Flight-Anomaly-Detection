"""Robust statistics primitives (median, Median Absolute Deviation).

These pure functions form the numerical core of the detector. They are
deliberately small, side-effect free, and individually verifiable.

NASA traceability:
* Power of Ten Rule 4 - each routine fits well within a single page.
* Power of Ten Rule 5 - at least two runtime assertions per function check
  pre-conditions and post-conditions.
* Power of Ten Rule 7 - parameters are validated before use.
* NASA-STD-8739.8 - deterministic, testable numerical behavior.

Requirements satisfied: SR-012, SR-013.
"""

from __future__ import annotations

import math
from typing import List, Optional

from .config import MIN_MAD

# Scale factor that makes the MAD a consistent estimator of the standard
# deviation for normally distributed data.
MAD_SCALE: float = 1.4826


def median(values: List[float]) -> float:
    """Return the median of a non-empty list of finite numbers.

    Args:
        values: A non-empty list of finite floats.

    Returns:
        float: The median value.

    Raises:
        AssertionError: If pre-conditions are violated.
    """
    assert isinstance(values, list), "values must be a list"
    assert len(values) > 0, "cannot compute median of an empty sequence"

    ordered = sorted(values)
    count = len(ordered)
    midpoint = count // 2
    if count % 2 == 1:
        result = float(ordered[midpoint])
    else:
        result = 0.5 * (ordered[midpoint - 1] + ordered[midpoint])

    assert not math.isnan(result), "median resolved to NaN"
    assert not math.isinf(result), "median resolved to infinity"
    return result


def median_absolute_deviation(values: List[float], center: Optional[float] = None) -> float:
    """Return the scaled, floored Median Absolute Deviation of ``values``.

    The result is multiplied by ``MAD_SCALE`` to approximate a standard
    deviation and is floored at ``MIN_MAD`` to prevent a degenerate
    zero-spread feature (Power of Ten Rule 7: guard against unsafe values).

    Args:
        values: A non-empty list of finite floats.
        center: Optional precomputed median; computed if omitted.

    Returns:
        float: The scaled MAD, guaranteed to be >= ``MIN_MAD``.
    """
    assert isinstance(values, list), "values must be a list"
    assert len(values) > 0, "cannot compute MAD of an empty sequence"

    center_value = median(values) if center is None else float(center)
    deviations = [abs(value - center_value) for value in values]
    raw_mad = median(deviations)
    scaled = MAD_SCALE * raw_mad
    result = scaled if scaled > MIN_MAD else MIN_MAD

    assert result >= MIN_MAD, "MAD must not fall below the configured floor"
    assert not math.isinf(result), "MAD resolved to infinity"
    return float(result)
