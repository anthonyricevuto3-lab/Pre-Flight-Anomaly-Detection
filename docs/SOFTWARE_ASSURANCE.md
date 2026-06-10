# Software Assurance Plan

**Project:** Pre-Flight Anomaly Detection
**Basis:** NASA-STD-8739.8B (Software Assurance and Software Safety, rev. B,
2022), NPR 7150.2D (NASA Software Engineering Requirements), and
NASA-HDBK-2203 (SWEHB). Inspection guidance follows NPR 7150.2D §5.3;
NASA-STD-8739.9 (Software Formal Inspections) is inactive and is not relied
upon as an active basis.

This plan documents how the software is assured to meet its requirements
throughout its lifecycle, scaled to a small research/demonstrator project.

---

## 1. Lifecycle Overview (NPR 7150.2D)

| Phase | Artifact |
|-------|----------|
| Requirements | `docs/SOFTWARE_REQUIREMENTS.md` (uniquely identified, traceable) |
| Design | Layered package architecture (see §4) documented in module docstrings |
| Implementation | `preflight/` package + `function_app.py`, per `docs/CODING_STANDARDS.md` |
| Verification | `tests/` (pytest), run under a zero-warning policy |
| Maintenance | Inspection checklist in `docs/CODING_STANDARDS.md`; CI gate guards regressions |
| Operations/Retirement | Operated as an Azure Function; retirement = remove the deployment and archive the repository (NPR 7150.2D §4.6) |

## 2. Verification & Validation

- **Static analysis** (NPR 7150.2D SWE-135): `flake8` (pycodestyle, pyflakes,
  mccabe complexity ≤ 10) and `mypy` (type checking) report zero findings.
- **Coding-standard conformance** (SWE-134): enforced by `flake8` per
  `setup.cfg` and `docs/CODING_STANDARDS.md`.
- **Unit verification** (SWE-061): `pytest` executes the `tests/` suite. Each
  test references the requirement ID(s) it verifies.
- **Zero-warning policy**: `pytest.ini` sets `-W error`, so any runtime
  warning fails the build (Power of Ten Rule 10).
- **Determinism**: detection results are reproducible (SR-014); augmentation
  is seeded (SR-023). This makes verification repeatable.

Run the full verification gate locally:

```powershell
pip install -r requirements.txt
flake8 preflight function_app.py tests
mypy preflight function_app.py
pytest
```

The same three checks run in the deployment CI workflow; deployment is
blocked unless all pass.

## 3. Software Safety & Defensive Design

- **Bounded resource use** (SR-020): all loops over external/untrusted input
  are capped, providing both safety and denial-of-service resistance.
- **Runtime assertions** (Power of Ten Rule 5): pre/post-conditions are
  checked in the numerical core.
- **Explicit off-nominal handling**: a typed exception hierarchy
  (`preflight/errors.py`) separates caller faults (HTTP 400) from internal
  faults (HTTP 500).
- **Fail-safe MAD floor** (SR-013): prevents divide-by-near-zero on
  zero-variance features.

## 4. Architecture (Design Record)

```
HTTP boundary          function_app.py      (routing, request parsing)
                       api_support.py       (CORS, JSON, error sanitization)
-------------------------------------------------------------------------
Application logic       service.py          (orchestration, reports)
-------------------------------------------------------------------------
Domain core            anomaly_model.py     (detector)
                       robust_stats.py      (median / MAD)
                       data_repository.py   (CSV access, augmentation)
-------------------------------------------------------------------------
Cross-cutting          config.py            (constants, configuration)
                       errors.py            (exception hierarchy)
```

The domain core and application logic contain no HTTP types and are
independently unit-testable.

## 5. Secure Coding Assurance (NASA Secure Coding Portal)

| Concern | Control |
|---------|---------|
| Information disclosure | Error responses are sanitized; full detail logged server-side only (SR-041, SR-042). |
| Excessive exposure | The legacy `/debug` endpoint was removed (SR-090). |
| Input validation | Request bodies and CSV fields are validated; invalid data rejected (SR-022, SR-040). |
| Least privilege | CORS origin is configurable via `ALLOWED_ORIGINS` (SR-031). |
| Resource exhaustion | Per-request and per-file iteration caps (SR-020). |

## 6. Residual Risks / Notes

- This is a demonstrator and has **not** undergone Independent Verification
  and Validation (IV&V). It is not certified for operational flight use.
- The default CORS origin remains `*` for the public demo; production
  deployments should set `ALLOWED_ORIGINS` to an explicit allow-list.
- Authentication is `ANONYMOUS` by design for the demo; a production system
  should require authentication/authorization.
