"""Verification tests for the robust anomaly detector.

Verifies: SR-010, SR-011, SR-014, SR-015.
"""

import pytest

from preflight.anomaly_model import INLIER_LABEL, OUTLIER_LABEL, RobustAnomalyDetector

FEATURES = ("a", "b")


def _fitted_detector(threshold=3.0, min_over=1):
    samples = [[10.0, 100.0] for _ in range(9)]
    samples.append([10.0, 100.0])
    detector = RobustAnomalyDetector(
        FEATURES, mad_threshold=threshold, min_features_over_threshold=min_over
    )
    return detector.fit(samples)


def test_constructor_rejects_invalid_threshold():
    with pytest.raises(AssertionError):
        RobustAnomalyDetector(FEATURES, mad_threshold=0.0)


def test_constructor_rejects_min_over_exceeding_feature_count():
    with pytest.raises(AssertionError):
        RobustAnomalyDetector(FEATURES, min_features_over_threshold=3)


def test_predict_before_fit_raises():
    detector = RobustAnomalyDetector(FEATURES)
    with pytest.raises(AssertionError):
        detector.predict([[1.0, 2.0]])


def test_inlier_classified_as_normal():
    detector = _fitted_detector()
    assert detector.classify_row([10.0, 100.0]) == INLIER_LABEL


def test_clear_outlier_classified_as_anomaly():
    detector = _fitted_detector()
    assert detector.classify_row([10_000.0, 100.0]) == OUTLIER_LABEL


def test_classify_row_rejects_wrong_length():
    detector = _fitted_detector()
    with pytest.raises(AssertionError):
        detector.classify_row([1.0])


def test_predict_is_deterministic():
    detector = _fitted_detector()
    batch = [[10.0, 100.0], [10_000.0, 100.0]]
    assert detector.predict(batch) == detector.predict(batch)


def test_feature_statistics_returns_copy():
    detector = _fitted_detector()
    stats = detector.feature_statistics()
    stats["a"] = (0.0, 0.0)
    assert detector.feature_statistics()["a"] != (0.0, 0.0)
