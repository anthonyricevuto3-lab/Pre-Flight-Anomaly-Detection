"""Test the updated function with improved anomaly display format."""

import requests
import json
import time

# Azure Function URL
FUNCTION_URL = "https://pre-fligt-anomaly-detection-fxenbdg2ced7hrg8.westus3-01.azurewebsites.net/api/detect_anomalies"

def test_anomaly_display():
    """Test the new anomaly display format."""
    
    print("Testing updated anomaly display format...")
    print("=" * 60)
    
    # Test 1: Normal readings (should show NO ANOMALIES)
    print("\nTest 1: Normal readings")
    normal_readings = [
        {"rpm": 1500, "temperature": 75.0, "pressure": 3000.0, "voltage": 28.0},
        {"rpm": 1600, "temperature": 78.0, "pressure": 3100.0, "voltage": 29.0}
    ]
    
    try:
        response = requests.post(
            FUNCTION_URL,
            json=normal_readings,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Data Source: {result.get('data_source', 'Unknown')}")
            print(f"Result: {result.get('analysis_result', 'Unknown')}")
            if result.get('anomalies'):
                for anomaly in result['anomalies']:
                    print(f"  {anomaly}")
        else:
            print(f"Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Mix of normal and anomalous readings
    print("\nTest 2: Mix of normal and anomalous readings")
    mixed_readings = [
        {"rpm": 1500, "temperature": 75.0, "pressure": 3000.0, "voltage": 28.0},  # Normal
        {"rpm": 2500, "temperature": 95.0, "pressure": 4500.0, "voltage": 35.0},  # Anomalous
        {"rpm": 3000, "temperature": 105.0, "pressure": 5000.0, "voltage": 40.0}  # Very anomalous
    ]
    
    try:
        response = requests.post(
            FUNCTION_URL,
            json=mixed_readings,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Data Source: {result.get('data_source', 'Unknown')}")
            print(f"Result: {result.get('analysis_result', 'Unknown')}")
            if result.get('anomalies'):
                for anomaly in result['anomalies']:
                    print(f"  {anomaly}")
        else:
            print(f"Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Single anomalous reading
    print("\nTest 3: Single anomalous reading")
    single_anomaly = {"rpm": 4000, "temperature": 120.0, "pressure": 6000.0, "voltage": 45.0}
    
    try:
        response = requests.post(
            FUNCTION_URL,
            json=single_anomaly,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Data Source: {result.get('data_source', 'Unknown')}")
            print(f"Result: {result.get('analysis_result', 'Unknown')}")
            if result.get('anomalies'):
                for anomaly in result['anomalies']:
                    print(f"  {anomaly}")
        else:
            print(f"Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_anomaly_display()