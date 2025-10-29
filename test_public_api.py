#!/usr/bin/env python3
"""
Test script for the deployed Azure Function public API
"""

import requests
import json
import time

# Function App URL
FUNCTION_APP_URL = "https://pre-fligt-anomaly-detection-fxenbdg2ced7hrg8.westus3-01.azurewebsites.net"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{FUNCTION_APP_URL}/api/health", timeout=30)
        print(f"Health Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Health Response:", json.dumps(response.json(), indent=2))
            return True
        else:
            print("Health Response:", response.text)
            return False
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False

def test_get_endpoint():
    """Test the GET endpoint for API information"""
    print("\nTesting GET endpoint...")
    try:
        response = requests.get(f"{FUNCTION_APP_URL}/api/detect_anomalies", timeout=30)
        print(f"GET Status Code: {response.status_code}")
        if response.status_code == 200:
            print("GET Response:", json.dumps(response.json(), indent=2))
            return True
        else:
            print("GET Response:", response.text)
            return False
    except requests.exceptions.RequestException as e:
        print(f"GET request failed: {e}")
        return False

def test_anomaly_detection():
    """Test the anomaly detection with sample data"""
    print("\nTesting anomaly detection...")
    
    # Normal flight data
    normal_data = {
        "rpm": 1500,
        "temperature": 75.0,
        "pressure": 3000.0,
        "voltage": 28.0
    }
    
    # Anomalous flight data (extreme values)
    anomalous_data = {
        "rpm": 2500,      # High RPM
        "temperature": 120.0,  # High temperature
        "pressure": 1000.0,    # Low pressure
        "voltage": 15.0        # Low voltage
    }
    
    test_cases = [
        ("Normal data", normal_data),
        ("Anomalous data", anomalous_data)
    ]
    
    for test_name, data in test_cases:
        print(f"\n--- Testing {test_name} ---")
        try:
            response = requests.post(
                f"{FUNCTION_APP_URL}/api/detect_anomalies",
                json=data,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("Response:", json.dumps(result, indent=2))
                print(f"Anomalies detected: {result.get('anomalies_detected', 'N/A')}")
            else:
                print("Response:", response.text)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

def main():
    """Main testing function"""
    print("=" * 60)
    print("Testing Pre-Flight Anomaly Detection Azure Function")
    print("=" * 60)
    
    # Wait a bit for deployment to complete
    print("Waiting 30 seconds for deployment to complete...")
    time.sleep(30)
    
    # Run tests
    health_ok = test_health_endpoint()
    get_ok = test_get_endpoint()
    
    if health_ok and get_ok:
        test_anomaly_detection()
    else:
        print("\nBasic endpoints failed. Deployment may still be in progress.")
        print("Please wait a few minutes and try again.")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()