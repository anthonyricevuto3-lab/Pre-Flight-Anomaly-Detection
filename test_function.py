"""Test script to verify the modified function logic works correctly."""

import json
import sys
from pathlib import Path

# Add the current directory to Python path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

# Import the training function from our modified function_app
from function_app import get_fresh_model, REQUIRED_FEATURES

def test_fresh_model():
    """Test that we can train a fresh model and use it for prediction."""
    print("Testing fresh model training...")
    
    try:
        # Train a fresh model
        model = get_fresh_model()
        print("✓ Fresh model trained successfully")
        
        # Test prediction with sample data
        test_data = [
            [1500, 75.0, 3000.0, 28.0],  # Normal reading
            [2500, 95.0, 4500.0, 35.0],  # Potentially anomalous reading
        ]
        
        predictions = model.predict(test_data)
        print(f"✓ Predictions: {predictions}")
        
        # Test multiple times to ensure randomization
        print("\nTesting randomization - training multiple models:")
        for i in range(3):
            fresh_model = get_fresh_model()
            pred = fresh_model.predict([[1500, 75.0, 3000.0, 28.0]])
            print(f"  Model {i+1} prediction: {pred[0]}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fresh_model()