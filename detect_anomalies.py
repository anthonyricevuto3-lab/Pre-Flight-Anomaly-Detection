"""CLI utility to report anomalies from the sample data set."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from anomaly_detection import detect_anomalies_from_csv


def main(argv: List[str] | None = None) -> int:
    data_path = Path('airplane_data.csv') if not argv else Path(argv[0])

    try:
        anomalies = detect_anomalies_from_csv(data_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"ERROR: {exc}")
        return 1

    if anomalies:
        print('ANOMALIES DETECTED:')
        for anomaly in anomalies:
            timestamp = anomaly.get('timestamp', 'unknown')
            print(
                'ERROR at [{}s]: Abnormal readings - '.format(timestamp)
                + ', '.join(
                    f"{key.upper()} = {anomaly[key]}"
                    for key in ['rpm', 'temperature', 'pressure', 'voltage']
                    if key in anomaly
                )
            )
    else:
        print('Everything is clear for take off')

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
