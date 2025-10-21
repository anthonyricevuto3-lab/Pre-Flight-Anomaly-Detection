# Pre-Flight Anomaly Detection

This project provides a lightweight anomaly detection system for pre-flight aircraft sensor readings. It uses a robust statistical approach that is suitable for edge deployment, with minimal dependencies.

## Features

- Trains a robust statistical model using Median Absolute Deviation (MAD)
- Detects anomalies in sensor readings (rpm, temperature, pressure, voltage)
- Provides both CLI and Azure Functions interfaces
- Designed for edge deployment with minimal dependencies

## Quick Start

1. Set up Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Train the model:

```bash
python train_model.py
```

3. Test anomaly detection locally:

```bash
python detect_anomalies.py airplane_data.csv
```

4. Run the Azure Function locally:

```bash
func start
```

## API Usage

Send POST requests to the `/api/detect_anomalies` endpoint with JSON data:

```json
{
    "rpm": 1500,
    "temperature": 75.0,
    "pressure": 3000,
    "voltage": 28.0
}
```

You can also send multiple readings as an array.

## Directory Structure

- `train_model.py`: Trains the anomaly detection model
- `detect_anomalies.py`: CLI tool for anomaly detection
- `function_app.py`: Azure Functions entry point
- `anomaly_detection.py`: Core anomaly detection logic
- `airplane_data.csv`: Sample training data
- `models/`: Directory for trained models

## Development

1. Make sure to run `train_model.py` first to generate the model
2. Test the CLI with sample data
3. Run unit tests if available
4. Start the Function app locally for API testing

## Deployment

1. Train the model locally first
2. Deploy to Azure Functions using Azure CLI or VS Code
3. Make sure to include the model file in your deployment

## Requirements

See `requirements.txt` for full dependencies. Key requirements:

- Python 3.8+
- azure-functions
- pandas
- joblib (optional, for efficient model persistence)