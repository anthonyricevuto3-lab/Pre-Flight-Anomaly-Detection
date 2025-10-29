import azure.functions as func
import json
import logging
import datetime
import tempfile
import os
import csv
import random
import time
import math
import pickle
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional

# Required features for anomaly detection
REQUIRED_FEATURES = ["rpm", "temperature", "pressure", "voltage"]

# Training data file path
DATA_PATH = Path(__file__).resolve().parent / "airplane_data.csv"

def _median(values: List[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of empty list")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])

def _mad(values: List[float], med: Optional[float] = None) -> float:
    # Median Absolute Deviation (normalized with 1.4826 to approximate std under normality)
    if med is None:
        med = _median(values)
    abs_dev = [abs(v - med) for v in values]
    raw_mad = _median(abs_dev)
    return 1.4826 * raw_mad

class SimpleAnomalyModel:
    """
    A tiny, dependency-free replacement for IsolationForest.
    """

    def __init__(
        self,
        feature_names: List[str],
        mad_threshold: float = 3.0,
        min_features_over_threshold: int = 1,
    ) -> None:
        self.feature_names = feature_names
        self.mad_threshold = float(mad_threshold)
        self.min_features_over_threshold = int(min_features_over_threshold)
        self._stats: Dict[str, Tuple[float, float]] = {}  # name -> (median, mad)

    def fit(self, X: List[List[float]]) -> "SimpleAnomalyModel":
        if not X:
            raise ValueError("Training data is empty")
        cols = list(zip(*X))  # column-wise
        if len(cols) != len(self.feature_names):
            raise ValueError("Feature count mismatch during fit")

        for name, col in zip(self.feature_names, cols):
            col_list = list(col)
            med = _median(col_list)
            mad = _mad(col_list, med=med)
            # Prevent degenerate zero-MAD; use small epsilon so thresholding still works
            if mad <= 1e-12:
                mad = 1e-6
            self._stats[name] = (med, mad)
        return self

    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Mimic IsolationForest's predict: 1 for inliers, -1 for outliers.
        """
        labels: List[int] = []
        for row in X:
            over = 0
            for (name, x) in zip(self.feature_names, row):
                med, mad = self._stats[name]
                if abs(x - med) > self.mad_threshold * mad:
                    over += 1
            labels.append(-1 if over >= self.min_features_over_threshold else 1)
        return labels

def _read_and_randomize_csv(csv_path: Path) -> List[List[float]]:
    """Read CSV and apply randomization for training data variation."""
    rows = []
    
    # Read the original CSV
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Extract required features as floats
                data_row = [float(row[col]) for col in REQUIRED_FEATURES]
                # Filter out NaN/inf rows
                if any(math.isnan(v) or math.isinf(v) for v in data_row):
                    continue
                rows.append(data_row)
            except (ValueError, KeyError):
                continue  # Skip rows with missing or invalid data
    
    if not rows:
        raise ValueError("No valid training data found")
    
    # Apply randomization to create variations
    randomized_rows = []
    for row in rows:
        new_row = []
        for value in row:
            # Add 3-8% random variation
            variation_factor = random.uniform(0.03, 0.08)
            variation = random.uniform(-variation_factor, variation_factor) * value
            new_value = value + variation
            new_row.append(new_value)
        randomized_rows.append(new_row)
    
    # Shuffle the rows
    random.shuffle(randomized_rows)
    
    # Add some duplicate rows with slight variations (10% more data)
    additional_rows = []
    num_additional = max(1, len(randomized_rows) // 10)
    
    for _ in range(num_additional):
        base_row = random.choice(randomized_rows).copy()
        # Add slight variations to the duplicate
        for i in range(len(base_row)):
            original_value = base_row[i]
            small_variation = random.uniform(-0.02, 0.02) * original_value
            base_row[i] = original_value + small_variation
        additional_rows.append(base_row)
    
    # Combine and final shuffle
    all_rows = randomized_rows + additional_rows
    random.shuffle(all_rows)
    
    return all_rows

def train_fresh_model() -> SimpleAnomalyModel:
    """Train a new model with randomized data."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
    
    # Set random seed based on current time for different results each run
    random.seed(int(time.time()))
    
    # Read and randomize training data
    training_data = _read_and_randomize_csv(DATA_PATH)
    
    # Train the model
    model = SimpleAnomalyModel(
        feature_names=REQUIRED_FEATURES,
        mad_threshold=3.0,
        min_features_over_threshold=1,
    ).fit(training_data)
    
    return model

# Global model cache
_current_model: Optional[SimpleAnomalyModel] = None

def get_fresh_model() -> SimpleAnomalyModel:
    """Get a freshly trained model with randomized data."""
    global _current_model
    _current_model = train_fresh_model()
    return _current_model


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
        # Train a fresh model with randomized data for each request
        logging.info("Training fresh model with randomized data...")
        fresh_model = get_fresh_model()
        
        # Convert readings to the format expected by our model
        processed_readings = []
        anomalous_readings = []
        
        for reading in readings:
            try:
                # Validate that all required features are present
                missing_features = [f for f in REQUIRED_FEATURES if f not in reading]
                if missing_features:
                    raise ValueError(f"Missing required features: {missing_features}")
                
                # Extract feature values in the correct order
                feature_values = [float(reading[feature]) for feature in REQUIRED_FEATURES]
                
                # Use the fresh model to predict
                prediction = fresh_model.predict([feature_values])[0]
                
                # If prediction is -1, it's an anomaly
                if prediction == -1:
                    anomalous_readings.append(reading)
                
                processed_readings.append(reading)
                
            except (ValueError, KeyError) as e:
                logging.warning(f"Skipping invalid reading: {e}")
                continue
        
        logging.info(f"Model retrained and processed {len(processed_readings)} readings")
        
        # Create clean output showing data source and anomaly results
        if len(anomalous_readings) > 0:
            anomaly_message = f"ANOMALIES DETECTED: {len(anomalous_readings)} out of {len(processed_readings)} readings flagged as anomalous"
            
            # Format each anomaly with its values
            anomaly_details = []
            for i, anomaly in enumerate(anomalous_readings, 1):
                anomaly_text = f"Anomaly {i}: "
                anomaly_values = []
                for feature in REQUIRED_FEATURES:
                    anomaly_values.append(f"{feature}={anomaly[feature]}")
                anomaly_text += ", ".join(anomaly_values)
                anomaly_details.append(anomaly_text)
            
            response = {
                "data_source": f"Training data read from: {DATA_PATH.name}",
                "analysis_result": anomaly_message,
                "anomalies": anomaly_details,
                "anomalous_data": anomalous_readings,
                "total_readings_analyzed": len(processed_readings)
            }
        else:
            response = {
                "data_source": f"Training data read from: {DATA_PATH.name}",
                "analysis_result": f"NO ANOMALIES DETECTED: All {len(processed_readings)} readings are within normal parameters",
                "anomalies": [],
                "anomalous_data": [],
                "total_readings_analyzed": len(processed_readings)
            }
        
        return func.HttpResponse(
            json.dumps(response, indent=2),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error processing anomaly detection: {str(e)}")
        return func.HttpResponse(
            f"Error processing readings: {str(e)}",
            status_code=500
        )