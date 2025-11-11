# Pre-Flight Anomaly Detection

This project provides a lightweight anomaly detection system for pre-flight aircraft sensor readings. It uses a robust statistical approach (median + MAD) and is designed to run as a small Azure Function with minimal external dependencies.

## Highlights

- Dynamic model training: the Azure Function trains a small robust model on each request using randomized/augmented CSV training data. This avoids a separate training step and keeps deployment simple.
- Detects anomalies for key sensors: `rpm`, `temperature`, `pressure`, `voltage`.
- Lightweight and dependency-friendly for edge and cloud deployments.

## Quick Start (local)

1. Set up a Python virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # on PowerShell
pip install -r requirements.txt
```

2. Run the Function locally:

```powershell
# Start the Functions host (Azure Functions Core Tools required)
func start
```

3. Test the API with a POST to `/api/detect_anomalies` (JSON body with rpm, temperature, pressure, voltage). The function will train a small model dynamically and return detected anomalies plus normal ranges.

## Demo Frontend

A small demo frontend is included at `frontend/index.html`. You can open it locally in a browser (or serve with a static server) to send sample payloads to the function and view responses.

- Open `frontend/index.html` in your browser, or serve it from a static host.
- Edit `frontend/script.js` to set `FUNCTION_URL` if your function is deployed remotely (defaults to `/api/detect_anomalies`).

## API Usage

Send POST requests to the `/api/detect_anomalies` endpoint with JSON data (example):

```json
{
    "rpm": 1500,
    "temperature": 75.0,
    "pressure": 3000,
    "voltage": 28.0
}
```

The function replies with a JSON body containing anomaly flags, detected anomaly details, and calculated normal ranges.

## Notes on Training

- The project no longer requires a separate `train_model.py` step in CI — training is handled dynamically in the function. This simplifies CI/CD and avoids missing-file errors.

## Repository / Demo

See the GitHub repo for code and deployment details: https://github.com/anthonyricevuto3-lab/Pre-Flight-Anomaly-Detection

If you'd like, I can also add a small GitHub Pages site to host the demo front-end publicly.