import azure.functions as func
import json
import logging
import datetime
from typing import Dict, List, Union

from anomaly_detection import detect_anomalies_from_records

app = func.FunctionApp()

@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint for monitoring"""
    logging.info('Health check request received')
    
    health_data = {
        "status": "healthy",
        "service": "Pre-Flight Anomaly Detection",
        "version": "1.0.0",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    
    return func.HttpResponse(
        json.dumps(health_data, indent=2),
        mimetype="application/json",
        status_code=200
    )

@app.route(route="detect_anomalies", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET", "POST"])
def detect_anomalies(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing anomaly detection request')

    # Handle GET request for testing
    if req.method == "GET":
        sample_data = {
            "message": "Pre-Flight Anomaly Detection API is running!",
            "endpoints": {
                "POST /api/detect_anomalies": "Submit sensor readings for anomaly detection",
                "GET /api/detect_anomalies": "Get API information and sample data"
            },
            "sample_request": {
                "method": "POST",
                "body": {
                    "rpm": 1500,
                    "temperature": 75.0,
                    "pressure": 3000.0,
                    "voltage": 28.0
                }
            },
            "function_url": f"https://pre-fligt-anomaly-detection.azurewebsites.net/api/detect_anomalies"
        }
        return func.HttpResponse(
            json.dumps(sample_data, indent=2),
            mimetype="application/json",
            status_code=200
        )

    # Handle POST request for anomaly detection
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