"""Azure Functions HTTP entry point for Pre-Flight Anomaly Detection.

This module is intentionally thin: it only translates between HTTP and the
transport-agnostic :mod:`preflight.service` layer. All numerical logic,
validation, and report construction live in the ``preflight`` package.

NASA traceability:
* NPR 7150.2D - layered design; the interface boundary is isolated from
  application logic.
* Secure Coding - the previous ``/debug`` endpoint (which exposed the
  filesystem, environment variables, and interpreter details) has been
  removed; unexpected errors return a generic message while full detail is
  logged server-side only.
* Power of Ten Rule 7 - return values from the service layer are validated
  by the service layer's own assertions; handlers validate request shape.

Requirements satisfied: SR-003, SR-004, SR-005, SR-033, SR-041.
"""

import datetime
import logging
from typing import List

import azure.functions as func

from preflight import api_support, config, service
from preflight.errors import TrainingDataError, ValidationError

_LOGGER = logging.getLogger(__name__)

app = func.FunctionApp()


@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Liveness/health endpoint for monitoring (SR-033)."""
    _LOGGER.info("Health check requested")
    payload = {
        "status": "healthy",
        "service": config.SERVICE_NAME,
        "version": config.SERVICE_VERSION,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return api_support.json_response(payload)


@app.route(
    route="detect_anomalies",
    auth_level=func.AuthLevel.ANONYMOUS,
    methods=["GET", "POST", "OPTIONS"],
)
def detect_anomalies(req: func.HttpRequest) -> func.HttpResponse:
    """Anomaly-detection endpoint.

    * OPTIONS - CORS preflight (204).
    * GET     - analyze the entire training data set (SR-004).
    * POST    - classify caller-supplied readings (SR-005).
    """
    _LOGGER.info("detect_anomalies invoked: method=%s", req.method)
    if req.method == "OPTIONS":
        return api_support.no_content_response()
    if req.method == "GET":
        return _handle_get_analysis()
    return _handle_post_detection(req)


def _handle_get_analysis() -> func.HttpResponse:
    """Run and serialize the full training-data analysis (GET)."""
    try:
        report = service.analyze_training_data(config.training_data_path())
        return api_support.json_response(report)
    except TrainingDataError:
        _LOGGER.exception("Training data unavailable during analysis")
        return api_support.error_response("Training data is currently unavailable", 500)
    except Exception:  # noqa: BLE001 - last-resort guard; detail logged, not returned.
        _LOGGER.exception("Unexpected error during training-data analysis")
        return api_support.error_response("Internal server error", 500)


def _handle_post_detection(req: func.HttpRequest) -> func.HttpResponse:
    """Validate, classify, and serialize caller-supplied readings (POST)."""
    try:
        readings = _parse_readings(req)
    except ValidationError as exc:
        return api_support.error_response(str(exc), 400, details=exc.details)

    try:
        report = service.detect_readings(readings, config.training_data_path())
        return api_support.json_response(report)
    except ValidationError as exc:
        return api_support.error_response(str(exc), 400, details=exc.details)
    except TrainingDataError:
        _LOGGER.exception("Training data unavailable during detection")
        return api_support.error_response("Training data is currently unavailable", 500)
    except Exception:  # noqa: BLE001 - last-resort guard; detail logged, not returned.
        _LOGGER.exception("Unexpected error during anomaly detection")
        return api_support.error_response("Internal server error", 500)


def _parse_readings(req: func.HttpRequest) -> List[dict]:
    """Parse and normalize the request body into a list of readings.

    Args:
        req: The incoming HTTP request.

    Returns:
        List[dict]: One or more reading objects.

    Raises:
        ValidationError: If the body is not valid JSON of the expected shape,
            or if it exceeds the configured per-request reading cap.
    """
    try:
        body = req.get_json()
    except ValueError:
        raise ValidationError("Request body must be valid JSON containing sensor readings")

    if isinstance(body, dict):
        readings = [body]
    elif isinstance(body, list):
        readings = body
    else:
        raise ValidationError(
            "Request body must be a single reading object or an array of readings"
        )

    if len(readings) > config.MAX_READINGS_PER_REQUEST:
        raise ValidationError(
            "Too many readings in a single request",
            details={"maximum_readings": config.MAX_READINGS_PER_REQUEST},
        )
    return readings
