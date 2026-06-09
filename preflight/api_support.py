"""HTTP boundary helpers (CORS, JSON serialization, error sanitization).

Centralizes response construction so that every endpoint emits consistent
headers and so that error responses never leak internal detail.

NASA traceability:
* Secure Coding - error responses return controlled messages only; raw
  exception text and stack traces are never returned to the caller.
* NPR 7150.2D - single, reusable interface contract for responses.

Requirements satisfied: SR-032, SR-041, SR-042.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import azure.functions as func

from .config import allowed_origin

_LOGGER = logging.getLogger(__name__)


def cors_headers() -> Dict[str, str]:
    """Return the standard CORS headers for all responses."""
    return {
        "Access-Control-Allow-Origin": allowed_origin(),
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Accept",
    }


def json_response(payload: Dict[str, Any], status_code: int = 200) -> func.HttpResponse:
    """Build a JSON HTTP response with CORS headers.

    Args:
        payload: A JSON-serializable mapping.
        status_code: A valid HTTP status code.

    Returns:
        func.HttpResponse: The serialized response.
    """
    assert isinstance(payload, dict), "payload must be a dict"
    assert 100 <= status_code <= 599, "status_code must be a valid HTTP status code"

    return func.HttpResponse(
        json.dumps(payload, indent=2),
        mimetype="application/json",
        status_code=status_code,
        headers=cors_headers(),
    )


def error_response(
    message: str,
    status_code: int,
    details: Optional[Dict[str, Any]] = None,
) -> func.HttpResponse:
    """Build a sanitized JSON error response.

    Only the supplied ``message`` and optional ``details`` (which the caller
    must ensure are non-sensitive) are returned. Internal exception text is
    deliberately excluded (Secure Coding: no information disclosure).

    Args:
        message: A safe, human-readable error description.
        status_code: An HTTP error status code (>= 400).
        details: Optional non-sensitive structured context.

    Returns:
        func.HttpResponse: The serialized error response.
    """
    assert isinstance(message, str), "message must be a string"
    assert 400 <= status_code <= 599, "error status_code must be >= 400"

    body: Dict[str, Any] = {"error": message}
    if details is not None:
        body["details"] = details
    return json_response(body, status_code=status_code)


def no_content_response() -> func.HttpResponse:
    """Return a 204 No Content response with CORS headers (CORS preflight)."""
    return func.HttpResponse(status_code=204, headers=cors_headers())
