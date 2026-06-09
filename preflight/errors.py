"""Domain-specific exception hierarchy.

A small, explicit exception hierarchy lets the HTTP boundary map failures
to the correct status codes without inspecting message strings, and lets
internal code signal precise, recoverable conditions.

NASA traceability:
* NASA-STD-8739.8 - off-nominal behavior is handled explicitly.
* Secure Coding - distinguishes client (4xx) from server (5xx) faults so
  that internal details are not leaked to callers.

Requirements satisfied: SR-040, SR-041.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class PreflightError(Exception):
    """Base class for all anomaly-detection domain errors."""


class ValidationError(PreflightError):
    """Raised when caller-supplied input fails validation (maps to HTTP 400).

    Args:
        message: A safe, human-readable description of the validation fault.
        details: Optional structured, non-sensitive context for the caller.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        assert isinstance(message, str), "message must be a string"
        assert details is None or isinstance(details, dict), "details must be a dict or None"
        super().__init__(message)
        self.details: Optional[Dict[str, Any]] = details


class TrainingDataError(PreflightError):
    """Raised when training data is missing or unusable (maps to HTTP 500)."""
