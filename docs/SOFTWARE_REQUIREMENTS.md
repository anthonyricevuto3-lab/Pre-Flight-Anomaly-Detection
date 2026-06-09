# Software Requirements Specification (SRS)

**Project:** Pre-Flight Anomaly Detection
**Classification (self-assessed):** NPR 7150.2D Class D / E (research & technology, non safety-critical demonstrator)
**Reference standards (verified against official sources, June 2026):**

- **NPR 7150.2D** — NASA Software Engineering Requirements (effective 2022-03-08; mandatory) — https://nodis3.gsfc.nasa.gov/displayDir.cfm?Internal_ID=N_PR_7150_002D_
- **NASA-STD-8739.8B** — Software Assurance and Software Safety (rev. B, 2022-09-08; ACTIVE/mandatory) — https://standards.nasa.gov/standard/nasa/nasa-std-87398
- **NASA-HDBK-2203** — NASA Software Engineering and Assurance Handbook (SWEHB, Ver. D) — https://swehb.nasa.gov/
- NASA Secure Coding Portal guidance (NPR 7150.2D §3.11 Software Cybersecurity) — https://sma.nasa.gov/news/articles/newsitem/2015/07/31/secure-coding-portal
- NASA/JPL "Power of Ten" rules for safety-critical code

> **Note on NASA-STD-8739.9 (Software Formal Inspections):** This standard is
> **INACTIVE** (cancelled; last revision 2013/2016). Its intent is now carried
> by **NPR 7150.2D §5.3 (Software Peer Reviews/Inspections)** and the SWEHB.
> The inspection checklist in `docs/CODING_STANDARDS.md` is aligned to §5.3.

## NPR 7150.2D Requirement (SWE) Coverage

| NPR 7150.2D activity | SWE ref. | How this project addresses it |
|----------------------|----------|-------------------------------|
| §4.1 Software Requirements | SWE-050 | This document (uniquely identified, testable requirements). |
| §4.2 Architecture / §4.3 Design | SWE-057/058 | Layered architecture in `docs/SOFTWARE_ASSURANCE.md` §4 and module docstrings. |
| §4.4 Implementation — coding standard | SWE-134 | `docs/CODING_STANDARDS.md`; enforced by `flake8` (`setup.cfg`). |
| §4.4 Implementation — static analysis | SWE-135 | `flake8` + `mypy`, run in CI and locally. |
| §4.5 Testing | SWE-061/062/066 | `tests/` (pytest), each test traced to a requirement ID. |
| §3.11 Software Cybersecurity | SWE-156/157/207 | Secure-coding controls table in `docs/SOFTWARE_ASSURANCE.md` §5. |
| §3.12 Bi-Directional Traceability | SWE-052 | Requirement ↔ code ↔ test tables below. |
| §5.3 Peer Reviews/Inspections | SWE-087/088 | Inspection checklist in `docs/CODING_STANDARDS.md`. |
| §5.1 Configuration Management | SWE-079 | Git version control; pinned dependencies; `.gitignore` excludes artifacts. |

> This document captures the requirements implemented by the system and
> provides bidirectional traceability between each requirement, the code
> that implements it, and the test that verifies it. This satisfies the
> NPR 7150.2D expectation that software requirements be documented,
> uniquely identified, and traceable.

---

## 1. Functional Requirements

| ID      | Requirement | Implemented in | Verified by |
|---------|-------------|----------------|-------------|
| SR-001  | The system shall evaluate exactly the features `rpm`, `temperature`, `pressure`, `voltage`, in that fixed order. | `preflight/config.py` (`REQUIRED_FEATURES`) | `test_service.py::test_analyze_training_data_reports_all_rows` |
| SR-002  | The system shall report engineering units alongside normal operating ranges. | `preflight/config.py` (`FEATURE_UNITS`) | `test_service.py` |
| SR-003  | The system shall expose a single `detect_anomalies` endpoint supporting GET, POST, and OPTIONS. | `function_app.py` | manual / integration |
| SR-004  | On GET, the system shall classify every training row and return a summary with normal operating ranges. | `preflight/service.py::analyze_training_data` | `test_service.py::test_analyze_training_data_reports_all_rows` |
| SR-005  | On POST, the system shall classify each caller-supplied reading and report detected anomalies. | `preflight/service.py::detect_readings` | `test_service.py::test_detect_readings_*` |
| SR-033  | The system shall expose a `health` endpoint reporting service name, version, and UTC timestamp. | `function_app.py::health_check` | manual / integration |

## 2. Algorithm Requirements

| ID      | Requirement | Implemented in | Verified by |
|---------|-------------|----------------|-------------|
| SR-010  | The detector shall flag a feature as anomalous when it deviates from the training median by more than `mad_threshold` scaled MADs. | `preflight/anomaly_model.py` | `test_anomaly_model.py::test_clear_outlier_classified_as_anomaly` |
| SR-011  | The detector shall flag a row when at least `min_features_over_threshold` features are anomalous. | `preflight/anomaly_model.py` | `test_anomaly_model.py` |
| SR-012  | The system shall compute the median of a non-empty finite sequence. | `preflight/robust_stats.py::median` | `test_robust_stats.py::test_median_*` |
| SR-013  | The system shall compute a scaled MAD, floored at `MIN_MAD` to prevent zero-spread degeneracy. | `preflight/robust_stats.py::median_absolute_deviation` | `test_robust_stats.py::test_mad_*` |
| SR-014  | The detector shall be deterministic: identical inputs shall yield identical outputs. | `preflight/anomaly_model.py` | `test_anomaly_model.py::test_predict_is_deterministic` |
| SR-015  | The detector shall not expose internal mutable state by reference. | `preflight/anomaly_model.py::feature_statistics` | `test_anomaly_model.py::test_feature_statistics_returns_copy` |

## 3. Data Requirements

| ID      | Requirement | Implemented in | Verified by |
|---------|-------------|----------------|-------------|
| SR-021  | The system shall load training samples from a configurable CSV path. | `preflight/data_repository.py::load_training_samples` | `test_data_repository.py::test_load_valid_csv` |
| SR-022  | The system shall reject malformed or non-finite training rows rather than coercing them. | `preflight/data_repository.py::_extract_feature_row` | `test_data_repository.py::test_extract_*` |
| SR-023  | When data augmentation is used, it shall be seeded and reproducible. | `preflight/data_repository.py::augment_samples` | `test_data_repository.py::test_augment_is_deterministic_for_seed` |
| SR-024  | The system shall derive normal operating ranges from training data using median ± `mad_threshold`·MAD. | `preflight/service.py::_operating_ranges` | `test_service.py` |

## 4. Safety, Security, and Robustness Requirements

| ID      | Requirement | Implemented in | Verified by |
|---------|-------------|----------------|-------------|
| SR-020  | Every loop shall have a statically provable upper bound (`MAX_TRAINING_ROWS`, `MAX_READINGS_PER_REQUEST`). | `preflight/config.py`, `data_repository.py`, `service.py` | code inspection |
| SR-030  | All shared constants shall be defined in a single configuration module. | `preflight/config.py` | code inspection |
| SR-031  | Cross-origin policy shall be configurable, not hard-coded. | `preflight/config.py::allowed_origin` | code inspection |
| SR-032  | All HTTP responses shall carry consistent CORS headers from a single helper. | `preflight/api_support.py` | code inspection |
| SR-040  | Caller input faults shall be reported as HTTP 400; internal faults as HTTP 500. | `preflight/errors.py`, `function_app.py` | `test_service.py::test_detect_readings_rejects_all_invalid` |
| SR-041  | Error responses shall not disclose stack traces, exception text, filesystem paths, or environment details. | `function_app.py`, `preflight/api_support.py::error_response` | code inspection |
| SR-042  | Full diagnostic detail shall be logged server-side only. | `function_app.py` (`_LOGGER.exception`) | code inspection |
| SR-050  | All source shall pass static analysis (`flake8`, `mypy`) with zero findings (SWE-135). | entire `preflight/` + `function_app.py`; config in `setup.cfg` | CI static-analysis step |
| SR-051  | All source shall conform to the project coding standard (SWE-134). | `docs/CODING_STANDARDS.md`; enforced by `flake8` (incl. mccabe complexity ≤ 10) | CI static-analysis step |

## 5. Removed Capability (Security Disposition)

| ID      | Disposition |
|---------|-------------|
| SR-090  | The legacy `/debug` endpoint, which exposed the working directory, environment variables, interpreter version, and recursive filesystem listings, has been **removed** as a violation of NASA Secure Coding guidance (information disclosure / least privilege). |

---

## 6. Traceability Summary

Every requirement above maps to (a) an implementing module and (b) a
verification test or documented inspection. The automated portion of the
verification evidence is produced by:

```powershell
flake8 preflight function_app.py tests   # SWE-134, SWE-135
mypy preflight function_app.py           # SWE-135
pytest                                    # SWE-061 (zero-warning policy)
```

All three run in the deployment CI workflow as a verification gate, so code
that does not satisfy these requirements cannot be deployed.
