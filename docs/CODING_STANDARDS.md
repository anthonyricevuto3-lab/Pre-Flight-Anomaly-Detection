# Coding Standards

**Project:** Pre-Flight Anomaly Detection
**Basis:** NASA/JPL "Power of Ten" rules, NPR 7150.2D (SWE-134 coding
standard), NASA Secure Coding Portal guidance (NPR 7150.2D §3.11), and the
NASA Software Engineering and Assurance Handbook (NASA-HDBK-2203 / SWEHB).

The original "Power of Ten" rules were written for C. Below, each rule is
restated, adapted to Python, and mapped to how this codebase complies.

---

## The Power of Ten — Python Adaptation

### Rule 1 — Restrict to simple control flow
- No recursion; no `goto`-equivalents.
- Control flow is limited to straight-line code, simple `if`/`for`, and
  early returns. **Compliant.**

### Rule 2 — All loops shall have a fixed upper bound
- Every iteration over external/untrusted data is sliced or counter-capped
  by `MAX_TRAINING_ROWS` / `MAX_READINGS_PER_REQUEST` (see
  `data_repository.py`, `service.py`). **Compliant.**

### Rule 3 — Do not use dynamic allocation after initialization
- Python manages memory automatically. The intent is honored by capping
  collection sizes (Rule 2) so memory use is bounded. **Compliant (in spirit).**

### Rule 4 — Keep functions short (a single page, ~60 lines)
- Functions are decomposed so each fits on a single screen. The former
  ~280-line request handler is split across `function_app.py` and the
  `preflight` service helpers. **Compliant.**

### Rule 5 — Use a minimum of two runtime assertions per function
- Core functions assert pre-conditions and post-conditions. See
  `robust_stats.py`, `anomaly_model.py`, `service.py`, `api_support.py`.
- HTTP boundary handlers validate request shape explicitly and raise
  `ValidationError` rather than asserting on untrusted input (asserts may be
  disabled with `-O`; untrusted input must always be checked). **Compliant.**

### Rule 6 — Restrict data to the smallest possible scope
- The previous module-level mutable globals (`_current_model`, reassigned
  `DATA_PATH`) are eliminated. The detector is built locally per request and
  all constants live in `config.py`. **Compliant.**

### Rule 7 — Check the return value of non-void functions; validate parameters
- Parameters are validated at function entry. Parsed CSV fields and request
  bodies are validated, and invalid data is rejected rather than coerced.
  **Compliant.**

### Rule 8 — Limit preprocessor / metaprogramming use
- No `exec`, no dynamic `eval`, no monkey-patching, minimal decorators
  (only the Azure routing decorators). **Compliant.**

### Rule 9 — Limit pointer/reference indirection
- No function-pointer-style indirection in hot paths; accessor methods
  return copies of internal state (`feature_statistics`). **Compliant.**

### Rule 10 — Compile clean with all warnings and static analysis enabled
- The test suite runs under `-W error` (warnings are failures, see
  `pytest.ini`).
- Static analysis is enforced with **`flake8`** (pycodestyle + pyflakes +
  mccabe complexity ≤ 10) and **`mypy`** (full type checking), configured in
  `setup.cfg`. Both report zero findings and run in CI (NPR 7150.2D
  SWE-134/SWE-135). **Compliant.**

---

## Additional Project Conventions

1. **Type hints** are required on all public functions and methods.
2. **Docstrings** are required on every module, public class, and public
   function, and shall reference the requirement IDs they satisfy where
   applicable.
3. **Layering**: transport (HTTP) code lives only in `function_app.py` and
   `api_support.py`; numerical and business logic must remain
   transport-agnostic and independently testable.
4. **Secure error handling**: never return raw exception text, stack traces,
   filesystem paths, or environment data to a caller; log detail
   server-side and return a generic message (NASA Secure Coding).
5. **Determinism**: detection logic must be reproducible; any randomness
   must be explicitly seeded.

---

## Peer Review / Inspection Checklist (NPR 7150.2D §5.3)

> NASA-STD-8739.9 (Software Formal Inspections) is **inactive**; its intent is
> now covered by NPR 7150.2D §5.3 (Software Peer Reviews/Inspections), to which
> this checklist is aligned.

Changes should be reviewed against this checklist before merge:

- [ ] Requirements traceability updated in `docs/SOFTWARE_REQUIREMENTS.md`.
- [ ] New/changed functions have ≥2 assertions or explicit input validation.
- [ ] No function exceeds ~60 lines.
- [ ] All loops over external data are bounded.
- [ ] No new module-level mutable state.
- [ ] No sensitive data in error responses.
- [ ] `flake8` and `mypy` report zero findings.
- [ ] Tests added/updated; `pytest` passes with zero warnings.
