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
    global DATA_PATH
    
    logging.info(f"Looking for training data at: {DATA_PATH}")
    logging.info(f"Current working directory: {Path.cwd()}")
    logging.info(f"Function file location: {Path(__file__).resolve().parent}")
    
    # List files in the function directory for debugging
    try:
        function_dir = Path(__file__).resolve().parent
        files_in_dir = list(function_dir.iterdir())
        logging.info(f"Files in function directory: {[f.name for f in files_in_dir]}")
    except Exception as e:
        logging.error(f"Error listing files in function directory: {e}")
    
    if not DATA_PATH.exists():
        # Try alternative paths
        alternative_paths = [
            Path.cwd() / "airplane_data.csv",
            Path("/home/site/wwwroot/airplane_data.csv"),
            Path("./airplane_data.csv")
        ]
        
        for alt_path in alternative_paths:
            logging.info(f"Trying alternative path: {alt_path}")
            if alt_path.exists():
                logging.info(f"Found training data at alternative path: {alt_path}")
                DATA_PATH = alt_path
                break
        else:
            error_msg = f"Training data not found at {DATA_PATH} or any alternative paths: {alternative_paths}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    logging.info(f"Using training data from: {DATA_PATH}")
    
    # Set random seed based on current time for different results each run
    random.seed(int(time.time()))
    
    try:
        # Read and randomize training data
        training_data = _read_and_randomize_csv(DATA_PATH)
        logging.info(f"Successfully loaded {len(training_data)} training samples")
    except Exception as e:
        logging.error(f"Error reading/randomizing CSV data: {e}")
        raise
    
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

@app.route(route="debug", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
def debug_function(req: func.HttpRequest) -> func.HttpResponse:
    """Debug function to explore Azure file system and locate airplane_data.csv"""
    import os
    import sys
    from pathlib import Path
    
    debug_info = {
        "message": "Debug information for Azure deployment",
        "current_working_directory": os.getcwd(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "script_location": __file__,
        "script_parent": str(Path(__file__).parent),
        "environment_variables": {
            "HOME": os.environ.get("HOME", "Not set"),
            "WEBSITE_SITE_NAME": os.environ.get("WEBSITE_SITE_NAME", "Not set"),
            "FUNCTIONS_WORKER_RUNTIME": os.environ.get("FUNCTIONS_WORKER_RUNTIME", "Not set"),
            "AZURE_FUNCTIONS_ENVIRONMENT": os.environ.get("AZURE_FUNCTIONS_ENVIRONMENT", "Not set"),
            "PWD": os.environ.get("PWD", "Not set"),
        },
        "file_system_exploration": {},
        "csv_file_search": []
    }
    
    # Explore common locations
    locations_to_check = [
        os.getcwd(),
        str(Path(__file__).parent),
        "/home/site/wwwroot",
        "/tmp",
        os.path.expanduser("~"),
        "/",
        str(Path(__file__).parent / "data"),
        str(Path(__file__).parent / "src" / "data"),
    ]
    
    for location in locations_to_check:
        try:
            if os.path.exists(location):
                files = os.listdir(location)
                debug_info["file_system_exploration"][location] = {
                    "exists": True,
                    "files": files[:20],  # Limit to first 20 files
                    "total_files": len(files)
                }
                
                # Check for CSV files specifically
                csv_files = [f for f in files if f.endswith('.csv')]
                if csv_files:
                    debug_info["csv_file_search"].extend([f"{location}/{f}" for f in csv_files])
            else:
                debug_info["file_system_exploration"][location] = {"exists": False}
        except Exception as e:
            debug_info["file_system_exploration"][location] = {"error": str(e)}
    
    # Try to find airplane_data.csv specifically
    search_patterns = ["airplane_data.csv", "*airplane*", "*.csv"]
    for pattern in search_patterns:
        try:
            import glob
            matches = glob.glob(f"/**/{pattern}", recursive=True)
            if matches:
                debug_info[f"search_{pattern}"] = matches[:10]  # Limit results
        except Exception as e:
            debug_info[f"search_{pattern}_error"] = str(e)
    
    return func.HttpResponse(
        json.dumps(debug_info, indent=2),
        mimetype="application/json",
        status_code=200
    )

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

@app.route(route="detect_anomalies", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET", "POST", "OPTIONS"])
def detect_anomalies(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing anomaly detection request')
    
    # CORS headers for GitHub Pages
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
    }
    
    # Handle OPTIONS preflight request
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=200, headers=headers)

    # Handle GET request - show all anomalies from CSV data
    if req.method == "GET":
        try:
            # Train model and get all CSV data
            model = train_fresh_model()
            
            # Read the CSV data again to analyze all rows
            csv_data = _read_and_randomize_csv(DATA_PATH)
            
            # Calculate normal ranges from the training data
            rpm_values = [row[0] for row in csv_data]
            temp_values = [row[1] for row in csv_data]
            pressure_values = [row[2] for row in csv_data]
            voltage_values = [row[3] for row in csv_data]
            
            # Calculate median and MAD for normal ranges
            rpm_median = _median(rpm_values)
            rpm_mad = _mad(rpm_values, rpm_median)
            temp_median = _median(temp_values)
            temp_mad = _mad(temp_values, temp_median)
            pressure_median = _median(pressure_values)
            pressure_mad = _mad(pressure_values, pressure_median)
            voltage_median = _median(voltage_values)
            voltage_mad = _mad(voltage_values, voltage_median)
            
            # Define normal ranges (within 3 MAD of median)
            normal_ranges = {
                "rpm": {
                    "min": round(rpm_median - (3 * rpm_mad), 2),
                    "max": round(rpm_median + (3 * rpm_mad), 2),
                    "median": round(rpm_median, 2),
                    "unit": "revolutions per minute"
                },
                "temperature": {
                    "min": round(temp_median - (3 * temp_mad), 2),
                    "max": round(temp_median + (3 * temp_mad), 2),
                    "median": round(temp_median, 2),
                    "unit": "degrees Celsius"
                },
                "pressure": {
                    "min": round(pressure_median - (3 * pressure_mad), 2),
                    "max": round(pressure_median + (3 * pressure_mad), 2),
                    "median": round(pressure_median, 2),
                    "unit": "PSI (pounds per square inch)"
                },
                "voltage": {
                    "min": round(voltage_median - (3 * voltage_mad), 2),
                    "max": round(voltage_median + (3 * voltage_mad), 2),
                    "median": round(voltage_median, 2),
                    "unit": "volts"
                }
            }
            
            # Predict anomalies for all data
            predictions = model.predict(csv_data)
            
            # Prepare response with all anomalies
            all_anomalies = []
            normal_readings = []
            
            for i, (row, prediction) in enumerate(zip(csv_data, predictions)):
                reading_data = {
                    "rpm": round(row[0], 2),
                    "temperature": round(row[1], 2), 
                    "pressure": round(row[2], 2),
                    "voltage": round(row[3], 2)
                }
                
                if prediction == -1:  # Anomaly detected
                    all_anomalies.append({
                        "reading_id": i + 1,
                        "data": reading_data,
                        "status": "ANOMALY"
                    })
                else:
                    normal_readings.append({
                        "reading_id": i + 1,
                        "data": reading_data,
                        "status": "NORMAL"
                    })
            
            response_data = {
                "message": "Complete Anomaly Analysis of Training Data",
                "data_source": f"Analysis of {len(csv_data)} readings from airplane_data.csv",
                "normal_operating_ranges": normal_ranges,
                "summary": {
                    "total_readings": len(csv_data),
                    "anomalies_detected": len(all_anomalies),
                    "normal_readings": len(normal_readings),
                    "anomaly_percentage": round((len(all_anomalies) / len(csv_data)) * 100, 2)
                },
                "detected_anomalies": all_anomalies,
                "normal_readings": normal_readings[:10],  # Show first 10 normal readings
                "note": f"Normal ranges calculated using Median Absolute Deviation (MAD). Values outside these ranges are flagged as anomalies."
            }
            
            return func.HttpResponse(
                json.dumps(response_data, indent=2),
                mimetype="application/json",
                status_code=200,
                headers=headers
            )
            
        except Exception as e:
            logging.error(f"Error analyzing CSV data: {e}")
            return func.HttpResponse(
                json.dumps({"error": f"Error analyzing data: {str(e)}"}),
                mimetype="application/json",
                status_code=500,
                headers=headers
            )

    # Handle POST request for anomaly detection
    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Request body must be valid JSON containing sensor readings"}),
            mimetype="application/json",
            status_code=400,
            headers=headers
        )

    # Expect either a single reading or a list of readings
    readings: List[Dict[str, Union[float, int, str]]]
    if isinstance(req_body, dict):
        readings = [req_body]
    elif isinstance(req_body, list):
        readings = req_body
    else:
        return func.HttpResponse(
            json.dumps({"error": "Request body must be a single reading object or an array of readings"}),
            mimetype="application/json",
            status_code=400,
            headers=headers
        )

    try:
        # Train a fresh model with randomized data for each request
        logging.info("Training fresh model with randomized data...")
        try:
            fresh_model = get_fresh_model()
        except FileNotFoundError as e:
            logging.error(f"Training data file not found: {e}")
            error_response = {
                "error": "Training data file not found",
                "details": str(e),
                "data_source_expected": str(DATA_PATH),
                "message": "The airplane_data.csv file is missing from the deployment"
            }
            return func.HttpResponse(
                json.dumps(error_response, indent=2),
                mimetype="application/json",
                status_code=500,
                headers=headers
            )
        except Exception as e:
            logging.error(f"Error training model: {e}")
            error_response = {
                "error": "Model training failed",
                "details": str(e)
            }
            return func.HttpResponse(
                json.dumps(error_response, indent=2),
                mimetype="application/json",
                status_code=500,
                headers=headers
            )
        
        # Convert readings to the format expected by our model
        processed_readings = []
        anomalous_readings = []
        validation_errors = []
        
        for reading in readings:
            try:
                # Validate that all required features are present
                missing_features = [f for f in REQUIRED_FEATURES if f not in reading]
                if missing_features:
                    error_msg = f"Missing required features: {missing_features}. Required: {REQUIRED_FEATURES}"
                    validation_errors.append(error_msg)
                    continue
                
                # Extract feature values in the correct order
                feature_values = [float(reading[feature]) for feature in REQUIRED_FEATURES]
                
                # Use the fresh model to predict
                prediction = fresh_model.predict([feature_values])[0]
                
                # If prediction is -1, it's an anomaly
                if prediction == -1:
                    anomalous_readings.append(reading)
                
                processed_readings.append(reading)
                
            except (ValueError, KeyError) as e:
                validation_errors.append(f"Invalid reading: {e}")
                continue
        
        # If no readings were processed, return an error with validation details
        if len(processed_readings) == 0:
            error_response = {
                "error": "No valid readings could be processed",
                "required_features": REQUIRED_FEATURES,
                "validation_errors": validation_errors,
                "example_valid_request": {
                    "rpm": 1500,
                    "temperature": 75.0,
                    "pressure": 3000.0,
                    "voltage": 28.0
                }
            }
            return func.HttpResponse(
                json.dumps(error_response, indent=2),
                mimetype="application/json",
                status_code=400
            )
        
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
            status_code=200,
            headers=headers
        )
    except Exception as e:
        logging.error(f"Error processing anomaly detection: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Error processing readings: {str(e)}"}),
            mimetype="application/json",
            status_code=500,
            headers=headers
        )