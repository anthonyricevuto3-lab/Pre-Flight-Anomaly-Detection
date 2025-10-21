import pandas as pd
import joblib
import sys
import os


def resource_path(filename: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)


def main():
    try:
        data_file = resource_path('data/airplane_data.csv')
        model_file = resource_path('models/isolation_forest_model.pkl')

        if not os.path.exists(data_file):
            print(f"ERROR: {data_file} file not found")
            sys.exit(1)

        if not os.path.exists(model_file):
            print(f"ERROR: {model_file} file not found")
            sys.exit(1)

        df = pd.read_csv(data_file)
        model = joblib.load(model_file)

        required_columns = ['rpm', 'temperature', 'pressure', 'voltage', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"ERROR: Missing required columns in data: {missing_columns}")
            sys.exit(1)

        X = df[['rpm', 'temperature', 'pressure', 'voltage']]
        preds = model.predict(X)

        anomalies_found = []

        for i, row in df.iterrows():
            if preds[i] == -1:
                anomalies_found.append({
                    'timestamp': row['timestamp'],
                    'rpm': row['rpm'],
                    'temperature': row['temperature'],
                    'pressure': row['pressure'],
                    'voltage': row['voltage']
                })

        if anomalies_found:
            print("ANOMALIES DETECTED:")
            for anomaly in anomalies_found:
                print(
                    f"ERROR at [{anomaly['timestamp']}s]: Abnormal readings - "
                    f"RPM = {anomaly['rpm']}, Temp = {anomaly['temperature']}, "
                    f"Pressure = {anomaly['pressure']}, Voltage = {anomaly['voltage']}")
        else:
            print("Everything is clear for take off")

    except FileNotFoundError as e:
        print(f"ERROR: Required file not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

main()