"""Train the Isolation Forest model used for anomaly detection."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from anomaly_detection import DEFAULT_MODEL_PATH, REQUIRED_FEATURES

DATA_PATH = Path('airplane_data.csv')


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing = [feature for feature in REQUIRED_FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(
            'Training data is missing required columns: ' + ', '.join(sorted(missing))
        )

    X = df[REQUIRED_FEATURES]
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)

    joblib.dump(model, DEFAULT_MODEL_PATH)
    print(f'Model trained and saved to {DEFAULT_MODEL_PATH}')


if __name__ == '__main__':
    main()
