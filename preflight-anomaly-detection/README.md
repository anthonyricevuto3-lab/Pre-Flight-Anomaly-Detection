# Preflight Anomaly Detection

This project is designed to detect anomalies in airplane data using an isolation forest model. The main components of the project are outlined below.

## Project Structure

```
preflight-anomaly-detection
├── src
│   ├── detect_anomalies.py       # Main logic for detecting anomalies
│   ├── __init__.py               # Marks the directory as a Python package
│   ├── models
│   │   └── isolation_forest_model.pkl  # Trained isolation forest model
│   └── data
│       └── airplane_data.csv      # Dataset for anomaly detection
├── scripts
│   ├── create_venv.sh             # Shell script to create a virtual environment (Unix)
│   └── create_venv.ps1            # PowerShell script to create a virtual environment (Windows)
├── requirements.txt                # Python dependencies for the project
├── runtime.txt                     # Python runtime version for Azure deployment
├── .gitignore                      # Files and directories to ignore by Git
├── .vscode
│   └── settings.json               # VS Code settings for the project
├── azure-pipelines.yml             # Azure DevOps pipeline configuration
└── README.md                       # Project documentation

## CI / Python runtime

This project now targets Python 3.11 for deployment and CI. The `runtime.txt` file has been updated to `3.11.9` and the Azure pipeline (`azure-pipelines.yml`) was updated to use Python 3.11 via the `UsePythonVersion` task and to verify the installed version during the run.
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd preflight-anomaly-detection
   ```

2. **Create a virtual environment:**
   - For Unix-like systems:
     ```
     ./scripts/create_venv.sh
     ```
   - For Windows systems:
     ```
     ./scripts/create_venv.ps1
     ```

3. **Activate the virtual environment:**
   - For Unix-like systems:
     ```
     source venv/bin/activate
     ```
   - For Windows systems:
     ```
     .\venv\Scripts\activate
     ```

4. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

5. **Run the anomaly detection script:**
   ```
   python src/detect_anomalies.py
   ```

## Usage

The `detect_anomalies.py` script will load the airplane data and the trained isolation forest model, check for required columns, and identify any anomalies in the data. Anomalies will be printed to the console.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.