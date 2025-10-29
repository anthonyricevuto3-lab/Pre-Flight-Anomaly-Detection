"""Test the deployed Azure Function with dynamic model retraining."""

import requests
import json
import time

# Azure Function URL
FUNCTION_URL = "https://pre-fligt-anomaly-detection-fxenbdg2ced7hrg8.westus3-01.azurewebsites.net/api/detect_anomalies"

def test_health_check():
    """Test the health endpoint."""
    health_url = "https://pre-fligt-anomaly-detection-fxenbdg2ced7hrg8.westus3-01.azurewebsites.net/api/health"
    try:
        response = requests.get(health_url, timeout=30)
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {json.dumps(response.json(), indent=2)}")
            return True
    except Exception as e:
        print(f"Health check failed: {e}")
    return False

def test_anomaly_detection():
    """Test the anomaly detection with model retraining."""
    
    # Test data
    test_readings = [
        {
            "rpm": 1500,
            "temperature": 75.0,
            "pressure": 3000.0,
            "voltage": 28.0
        },
        {
            "rpm": 2500,  # Potentially anomalous - high RPM
            "temperature": 95.0,  # High temperature
            "pressure": 4500.0,  # High pressure
            "voltage": 35.0  # High voltage
        }
    ]
    
    print("Testing anomaly detection with model retraining...")
    print(f"Sending {len(test_readings)} readings to {FUNCTION_URL}")
    
    try:
        response = requests.post(
            FUNCTION_URL,
            json=test_readings,
            headers={"Content-Type": "application/json"},
            timeout=60  # Longer timeout since we're training a model
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Anomaly detection successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Check if model info is included
            if "model_info" in result and result["model_info"].get("freshly_trained"):
                print("✓ Model was freshly trained with randomized data!")
                return True
            else:
                print("⚠ Model info missing or model not freshly trained")
                return False
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing anomaly detection: {e}")
        return False

def test_multiple_requests():
    """Test multiple requests to verify each one retrains the model."""
    print("\nTesting multiple requests to verify model retraining...")
    
    test_reading = {
        "rpm": 1500,
        "temperature": 75.0,
        "pressure": 3000.0,
        "voltage": 28.0
    }
    
    training_timestamps = []
    
    for i in range(3):
        print(f"\nRequest {i+1}:")
        try:
            response = requests.post(
                FUNCTION_URL,
                json=test_reading,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if "model_info" in result:
                    timestamp = result["model_info"].get("training_timestamp")
                    training_timestamps.append(timestamp)
                    print(f"  Training timestamp: {timestamp}")
                    print(f"  Anomalies detected: {result.get('anomalies_detected', 'unknown')}")
                else:
                    print("  No model info in response")
            else:
                print(f"  Request failed: {response.status_code}")
        
        except Exception as e:
            print(f"  Error: {e}")
        
        # Small delay between requests
        time.sleep(2)
    
    # Check if timestamps are different (indicating fresh training each time)
    unique_timestamps = set(training_timestamps)
    if len(unique_timestamps) == len(training_timestamps):
        print("✓ Each request trained a fresh model (all timestamps unique)")
        return True
    else:
        print(f"⚠ Some timestamps were the same: {training_timestamps}")
        return False

def main():
    """Run all tests."""
    print("Testing deployed Azure Function with dynamic model retraining")
    print("=" * 60)
    
    # Test health check
    if not test_health_check():
        print("Health check failed, function may not be deployed yet")
        return
    
    print()
    
    # Test basic anomaly detection
    if not test_anomaly_detection():
        print("Basic anomaly detection test failed")
        return
    
    # Test multiple requests
    if not test_multiple_requests():
        print("Multiple requests test failed")
        return
    
    print("\n✓ All tests passed! The function is working correctly with dynamic model retraining.")

if __name__ == "__main__":
    main()