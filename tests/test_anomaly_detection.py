"""Test anomaly detection functionality."""
import pytest
from pathlib import Path
import sys

# Add parent directory to path so we can import our module
sys.path.append(str(Path(__file__).parent.parent))

from anomaly_detection import load_model, detect_anomalies_from_records

def test_load_model():
    """Test that we can load the model."""
    model = load_model()
    assert model is not None
    assert hasattr(model, 'predict')

def test_detect_anomalies():
    """Test anomaly detection with normal and anomalous data."""
    # Normal reading
    normal_reading = {
        "rpm": 1500,
        "temperature": 75.0,
        "pressure": 3000,
        "voltage": 28.0
    }
    
    # Anomalous reading (high temperature and pressure)
    anomalous_reading = {
        "rpm": 1750,
        "temperature": 95.0,
        "pressure": 5100,
        "voltage": 29.0
    }
    
    # Test normal reading
    anomalies = detect_anomalies_from_records([normal_reading])
    assert len(anomalies) == 0, "Normal reading should not be flagged as anomalous"
    
    # Test anomalous reading
    anomalies = detect_anomalies_from_records([anomalous_reading])
    assert len(anomalies) > 0, "Anomalous reading should be detected"

def test_missing_features():
    """Test that we properly handle missing features."""
    incomplete_reading = {
        "rpm": 1500,
        "temperature": 75.0
        # missing pressure and voltage
    }
    
    with pytest.raises(ValueError):
        detect_anomalies_from_records([incomplete_reading])