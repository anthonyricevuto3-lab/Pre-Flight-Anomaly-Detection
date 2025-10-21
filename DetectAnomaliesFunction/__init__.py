"""HTTP-triggered Azure Function that evaluates aircraft telemetry for anomalies."""
from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any, Dict, Iterable, List, Mapping

import azure.functions as func

from anomaly_detection import REQUIRED_FEATURES, detect_anomalies_from_records


def _normalize_payload(data: Any) -> List[Mapping[str, Any]]:
    """Convert the body payload into a list of record dictionaries."""

    if isinstance(data, Mapping):
        if "readings" in data:
            payload = data["readings"]
        else:
            # Interpret the mapping itself as a single reading.
            payload = [data]
    else:
        payload = data

    if isinstance(payload, Mapping):
        payload = [payload]

    if not isinstance(payload, Iterable):
        raise ValueError("Request body must be a JSON object or array of readings")

    readings: List[Mapping[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError("Each reading must be a JSON object with telemetry fields")
        readings.append(item)

    return readings


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        payload = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Request body must be valid JSON."}),
            mimetype="application/json",
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        readings = _normalize_payload(payload)
        anomalies = detect_anomalies_from_records(readings)
    except ValueError as exc:
        return func.HttpResponse(
            json.dumps({"error": str(exc), "required_features": REQUIRED_FEATURES}),
            mimetype="application/json",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    except FileNotFoundError as exc:
        return func.HttpResponse(
            json.dumps({"error": str(exc)}),
            mimetype="application/json",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return func.HttpResponse(
            json.dumps({"error": f"Unexpected error: {exc}"}),
            mimetype="application/json",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    response_body: Dict[str, Any] = {
        "anomalyCount": len(anomalies),
        "anomalies": anomalies,
        "requiredFeatures": REQUIRED_FEATURES,
    }
    if len(anomalies) == 0:
        response_body["message"] = "Everything is clear for take off"

    return func.HttpResponse(
        json.dumps(response_body),
        mimetype="application/json",
        status_code=HTTPStatus.OK,
    )
