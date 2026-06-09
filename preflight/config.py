"""Centralized configuration constants and runtime settings.

All tunable parameters are defined here so that data scope is minimized
and configuration is traceable to a single location.

NASA traceability:
* Power of Ten Rule 6 - declare data at the smallest possible scope; this
  module is the single source of truth for shared constants.
* NPR 7150.2D - configuration management and traceable design parameters.
* Secure Coding - externalized configuration (no hard-coded secrets);
  cross-origin policy is configurable rather than implicitly permissive.

Requirements satisfied: SR-001, SR-002, SR-030, SR-031.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

# [SR-001] The system shall evaluate exactly these sensor features, in this
# fixed order. The order is significant: feature rows are positional.
REQUIRED_FEATURES: Tuple[str, ...] = ("rpm", "temperature", "pressure", "voltage")

# [SR-002] Human-readable engineering units reported with operating ranges.
FEATURE_UNITS: Dict[str, str] = {
    "rpm": "revolutions per minute",
    "temperature": "degrees Celsius",
    "pressure": "PSI (pounds per square inch)",
    "voltage": "volts",
}

# [SR-010] Default robust-detector parameters.
DEFAULT_MAD_THRESHOLD: float = 3.0
DEFAULT_MIN_FEATURES_OVER_THRESHOLD: int = 1

# [SR-011] Lower bound applied to the Median Absolute Deviation to avoid a
# degenerate zero-spread feature producing divide-by-near-zero behavior.
MIN_MAD: float = 1e-6

# [SR-020] Fixed upper bounds on iteration counts (Power of Ten Rule 2:
# every loop must have a statically provable upper bound). These also serve
# as denial-of-service guards on untrusted request and file sizes.
MAX_TRAINING_ROWS: int = 100_000
MAX_READINGS_PER_REQUEST: int = 10_000

# Service identity reported by the health endpoint.
SERVICE_NAME: str = "Pre-Flight Anomaly Detection"
SERVICE_VERSION: str = "2.0.0"

_TRAINING_DATA_FILENAME: str = "airplane_data.csv"


def training_data_path() -> Path:
    """Return the resolved path to the training data file.

    The ``TRAINING_DATA_PATH`` environment variable overrides the default
    location, which keeps the deployment configurable and avoids the
    fragile multi-path filesystem search used previously.

    Returns:
        Path: An absolute path to the training data CSV file.
    """
    override = os.environ.get("TRAINING_DATA_PATH")
    if override:
        return Path(override).resolve()
    return (Path(__file__).resolve().parent.parent / _TRAINING_DATA_FILENAME).resolve()


def allowed_origin() -> str:
    """Return the configured CORS allowed origin.

    Defaults to ``*`` for backward compatibility with the public demo, but
    is overridable via the ``ALLOWED_ORIGINS`` environment variable so that
    production deployments can restrict cross-origin access (Secure Coding:
    least privilege).

    Returns:
        str: The value to use for ``Access-Control-Allow-Origin``.
    """
    return os.environ.get("ALLOWED_ORIGINS", "*")
