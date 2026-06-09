"""Verification tests for robust statistics primitives.

Verifies: SR-012, SR-013.
"""

import pytest

from preflight.config import MIN_MAD
from preflight.robust_stats import MAD_SCALE, median, median_absolute_deviation


def test_median_odd_length():
    assert median([3.0, 1.0, 2.0]) == 2.0


def test_median_even_length_is_average_of_middle_pair():
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_median_rejects_empty_sequence():
    with pytest.raises(AssertionError):
        median([])


def test_median_rejects_non_list():
    with pytest.raises(AssertionError):
        median((1.0, 2.0))  # type: ignore[arg-type]


def test_mad_uses_supplied_center_and_scale():
    values = [10.0, 12.0, 14.0]
    center = median(values)
    expected = MAD_SCALE * median([abs(v - center) for v in values])
    assert median_absolute_deviation(values, center) == pytest.approx(expected)


def test_mad_is_floored_for_zero_spread():
    # All-identical data has zero raw spread; result must be floored.
    assert median_absolute_deviation([5.0, 5.0, 5.0]) == MIN_MAD


def test_mad_rejects_empty_sequence():
    with pytest.raises(AssertionError):
        median_absolute_deviation([])
