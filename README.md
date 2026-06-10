# Pre-Flight Anomaly Detection

A lightweight anomaly-detection service for pre-flight aircraft sensor
readings. It uses a robust, dependency-free statistical model (median +
Median Absolute Deviation) and runs as an Azure Function.

This codebase is structured to follow NASA software engineering guidance:
NPR 7150.2D (Software Engineering Requirements), NASA-STD-8739.8B (Software
Assurance and Software Safety), NASA-HDBK-2203 (the SWEHB), the NASA Secure
Coding Portal (NPR 7150.2D §3.11), and the NASA/JPL "Power of Ten" rules.
Coding-standard conformance and static analysis (SWE-134/SWE-135) are
enforced with `flake8` and `mypy`. See the [`docs/`](docs/) directory.

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/SOFTWARE_REQUIREMENTS.md`](docs/SOFTWARE_REQUIREMENTS.md) | Uniquely identified, traceable requirements (NPR 7150.2D). |
| [`docs/CODING_STANDARDS.md`](docs/CODING_STANDARDS.md) | Power of Ten rules adapted to Python + project conventions. |
| [`docs/SOFTWARE_ASSURANCE.md`](docs/SOFTWARE_ASSURANCE.md) | V&V approach, architecture record, secure-coding controls. |
| [`docs/NASA_COMPLIANCE_MATRIX.md`](docs/NASA_COMPLIANCE_MATRIX.md) | Item-by-item disposition against every referenced NASA directive, standard, handbook, and category. |

## Architecture

```
function_app.py            # Thin HTTP entry point (Azure Functions)
preflight/                 # Application package
  config.py                #   constants & configuration (single source of truth)
  errors.py                #   typed exception hierarchy
  robust_stats.py          #   median / MAD primitives
  anomaly_model.py         #   RobustAnomalyDetector
  data_repository.py       #   CSV loading & seeded augmentation
  service.py               #   orchestration & report construction
  api_support.py           #   CORS, JSON, sanitized error responses
tests/                     # Verification suite (pytest)
docs/                      # NASA-aligned engineering documentation
airplane_data.csv          # Training data
```

The HTTP layer is isolated from the numerical/business logic, which is
transport-agnostic and independently unit-testable.

## Quick Start (local)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
func start                      # Azure Functions Core Tools required
```

## Verification

```powershell
pip install -r requirements.txt
flake8 preflight function_app.py tests   # coding standard + static analysis (SWE-134/135)
mypy preflight function_app.py           # static type analysis (SWE-135)
pytest                                    # unit tests, zero-warning policy (-W error)
```

All three checks run as a gate in the deployment CI workflow.

## API

### `GET /api/detect_anomalies`
Analyzes the entire training data set and returns normal operating ranges,
a per-feature summary, and the detected anomalies.

### `POST /api/detect_anomalies`
Classifies one or more caller-supplied readings.

```json
{
  "rpm": 1500,
  "temperature": 75.0,
  "pressure": 3000,
  "voltage": 28.0
}
```

A single object or an array of objects is accepted. Up to
`MAX_READINGS_PER_REQUEST` (10,000) readings per request are processed.

### `GET /api/health`
Returns service name, version, and a UTC timestamp.

## Configuration

| Environment variable | Default | Purpose |
|----------------------|---------|---------|
| `TRAINING_DATA_PATH` | `<repo>/airplane_data.csv` | Override training data location. |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origin. Set to an explicit origin in production. |

## Security Notes

- The previous `/debug` endpoint (which exposed the filesystem, environment
  variables, and interpreter details) has been **removed** per NASA Secure
  Coding guidance.
- Error responses are sanitized; internal detail is logged server-side only.
- For production, set `ALLOWED_ORIGINS` and require authentication.

## Demo Frontend

A demo frontend is included at [`frontend/index.html`](frontend/index.html).
Set `FUNCTION_URL` in `frontend/script.js` to point at your deployed function.

## Repository

https://github.com/anthonyricevuto3-lab/Pre-Flight-Anomaly-Detection
