import azure.functions as func
import json
import logging
from typing import Dict, List, Union

from anomaly_detection import detect_anomalies_from_records

app = func.FunctionApp()

@app.route(route="detect_anomalies", auth_level=func.AuthLevel.FUNCTION)
def detect_anomalies(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing anomaly detection request')

    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            "Request body must be valid JSON containing sensor readings",
            status_code=400
        )

    # Expect either a single reading or a list of readings
    readings: List[Dict[str, Union[float, int, str]]]
    if isinstance(req_body, dict):
        readings = [req_body]
    elif isinstance(req_body, list):
        readings = req_body
    else:
        return func.HttpResponse(
            "Request body must be a single reading object or an array of readings",
            status_code=400
        )

    try:
        anomalies = detect_anomalies_from_records(readings)
        response = {
            "anomalies_detected": len(anomalies) > 0,
            "anomalous_readings": anomalies,
            "total_readings": len(readings),
        }
        return func.HttpResponse(
            json.dumps(response),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error processing anomaly detection: {str(e)}")
        return func.HttpResponse(
            f"Error processing readings: {str(e)}",
            status_code=500
        )