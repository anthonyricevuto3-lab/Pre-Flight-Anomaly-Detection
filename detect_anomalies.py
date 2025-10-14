
import pandas as pd
import joblib
import sys
import os

try:
    # Load data and model
    if not os.path.exists('airplane_data.csv'):
        print("ERROR: airplane_data.csv file not found")
        sys.exit(1)
    
    if not os.path.exists('isolation_forest_model.pkl'):
        print("ERROR: isolation_forest_model.pkl file not found")
        sys.exit(1)
    
    df = pd.read_csv('airplane_data.csv')
    model = joblib.load('isolation_forest_model.pkl')
    
    # Validate required columns exist
    required_columns = ['rpm', 'temperature', 'pressure', 'voltage', 'timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns in data: {missing_columns}")
        sys.exit(1)
    
    X = df[['rpm', 'temperature', 'pressure', 'voltage']]
    preds = model.predict(X)  # -1 for anomaly, 1 for normal
    
    # Check for anomalies and only display errors
    anomalies_found = []
    
    for i, row in df.iterrows():
        if preds[i] == -1:  # Anomaly detected
            anomalies_found.append({
                'timestamp': row['timestamp'],
                'rpm': row['rpm'],
                'temperature': row['temperature'],
                'pressure': row['pressure'],
                'voltage': row['voltage']
            })
    
    # Display results
    if anomalies_found:
        print("ANOMALIES DETECTED:")
        for anomaly in anomalies_found:
            print(f"ERROR at [{anomaly['timestamp']}s]: Abnormal readings - "
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
