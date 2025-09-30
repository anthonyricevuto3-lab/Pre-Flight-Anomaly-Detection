# Airplane Anomaly Detection

This project provides tooling for training an Isolation Forest model on pre-flight
sensor readings and serving the anomaly detection logic via either a command-line
script or an Azure Functions HTTP endpoint.

## Project structure

- `airplane_data.csv` – sample telemetry used for local experimentation and model training.
- `anomaly_detection.py` – shared helper functions for loading the trained model and running predictions.
- `train_model.py` – trains the Isolation Forest model and saves it to `isolation_forest_model.pkl`.
- `detect_anomalies.py` – CLI entry point that reports anomalies found in a CSV file.
- `DetectAnomaliesFunction/` – Azure Functions HTTP trigger that performs anomaly detection on JSON payloads.
- `host.json`, `.funcignore`, `requirements.txt` – configuration required to publish the Function App.

## Python environment

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Training the model

```bash
python train_model.py
```

The script expects `airplane_data.csv` to contain the following columns:
`rpm`, `temperature`, `pressure`, `voltage`, and any metadata you want to preserve
(such as `timestamp`). The trained model is stored at `isolation_forest_model.pkl`.

## Detecting anomalies from the CLI

```bash
python detect_anomalies.py            # Uses airplane_data.csv by default
python detect_anomalies.py custom.csv # Analyse a different CSV file
```

The command prints any rows classified as anomalies. If none are detected the
script reports "Everything is clear for take off".

## Running the Azure Function locally

1. Install the [Azure Functions Core Tools](https://learn.microsoft.com/azure/azure-functions/functions-run-local) for your platform.
2. Copy `local.settings.json.example` to `local.settings.json` and update any
   settings as needed.
3. Start the Function host:

   ```bash
   func start
   ```

4. Send sample data to the HTTP endpoint:

   ```bash
   curl \
     -X POST http://localhost:7071/api/detect \
     -H "Content-Type: application/json" \
     -d '{
           "readings": [
             {"timestamp": 1, "rpm": 4200, "temperature": 350, "pressure": 30.1, "voltage": 27.5},
             {"timestamp": 2, "rpm": 5200, "temperature": 410, "pressure": 25.0, "voltage": 23.0}
           ]
         }'
   ```

   The response includes the number of anomalies detected and echoes the
   offending records.

## Deploying to Azure Functions

1. Create a Function App resource (Python stack) and a storage account in your
   Azure subscription.
2. Ensure the Function App is configured with the **Python 3.10** runtime (or the
   version that matches your local environment).
3. Publish the project using the Azure Functions Core Tools:

   ```bash
   func azure functionapp publish <YOUR_FUNCTION_APP_NAME>
   ```

   The `.funcignore` file prevents local-only assets (such as `airplane_data.csv`
   and Git metadata) from being uploaded.

4. Once deployed, issue POST requests to `https://<YOUR_FUNCTION_APP_NAME>.azurewebsites.net/api/detect`
   with a JSON body matching the structure shown above. The function returns a
   JSON response describing any anomalies and includes a friendly message when
   none are detected.

## Notes for production deployments

- Store the trained model file (`isolation_forest_model.pkl`) in a secure
  location, such as Azure Blob Storage, and configure the Function to load it at
  startup.
- Consider using [Azure Functions managed identities](https://learn.microsoft.com/azure/app-service/overview-managed-identity) to access the model or other sensitive configuration.
- Monitor execution using Application Insights and adjust the Isolation Forest
  contamination rate as you collect more telemetry.
