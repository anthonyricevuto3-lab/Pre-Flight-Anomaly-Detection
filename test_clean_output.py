"""Test the updated function with cleaner output format."""

import json
import sys
from pathlib import Path

# Add the current directory to Python path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

# Import the training function from our modified function_app
from function_app import get_fresh_model, REQUIRED_FEATURES, DATA_PATH

def test_clean_output():
    """Test the updated clean output format."""
    print("Testing clean output format...")
    
    try:
        # Train a fresh model
        fresh_model = get_fresh_model()
        print(f"✓ Fresh model trained from: {DATA_PATH.name}")
        
        # Test normal reading
        normal_reading = [1500, 75.0, 3000.0, 28.0]
        prediction = fresh_model.predict([normal_reading])[0]
        
        # Test anomalous reading  
        anomalous_reading = [2500, 95.0, 4500.0, 35.0]
        prediction2 = fresh_model.predict([anomalous_reading])[0]
        
        print(f"Normal reading prediction: {prediction} (1=normal, -1=anomaly)")
        print(f"Anomalous reading prediction: {prediction2} (1=normal, -1=anomaly)")
        
        # Simulate the clean output format
        test_readings = [
            {"rpm": 1500, "temperature": 75.0, "pressure": 3000.0, "voltage": 28.0},
            {"rpm": 2500, "temperature": 95.0, "pressure": 4500.0, "voltage": 35.0}
        ]
        
        anomalous_readings = []
        processed_readings = []
        
        for reading in test_readings:
            feature_values = [float(reading[feature]) for feature in REQUIRED_FEATURES]
            prediction = fresh_model.predict([feature_values])[0]
            
            if prediction == -1:
                anomalous_readings.append(reading)
            
            processed_readings.append(reading)
        
        # Create the clean output
        if len(anomalous_readings) > 0:
            anomaly_message = f"ANOMALIES DETECTED: {len(anomalous_readings)} out of {len(processed_readings)} readings flagged as anomalous"
        else:
            anomaly_message = f"NO ANOMALIES DETECTED: All {len(processed_readings)} readings are within normal parameters"
        
        clean_output = {
            "data_source": f"Training data read from: {DATA_PATH.name}",
            "analysis_result": anomaly_message,
            "anomalous_readings": anomalous_readings,
            "total_readings_analyzed": len(processed_readings)
        }
        
        print("\nClean Output Format:")
        print(json.dumps(clean_output, indent=2))
        
        print("\n✓ Clean output test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_clean_output()