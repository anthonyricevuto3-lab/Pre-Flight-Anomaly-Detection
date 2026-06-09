"""Pre-Flight Anomaly Detection application package.

This package implements a pre-flight aircraft sensor anomaly-detection
service. It is structured in accordance with NASA software engineering
guidance:

* NPR 7150.2D  - Software Engineering Requirements (modular design,
  requirements traceability, documentation).
* NASA-STD-8739.8 - Software Assurance and Software Safety (defensive
  design, verification, deterministic behavior).
* NASA Secure Coding Portal - input validation, least privilege, no
  information disclosure in error responses.
* NASA/JPL "Power of Ten" rules for safety-critical code (bounded loops,
  restricted data scope, runtime assertions, checked return values).

Each public module documents the requirement identifiers it satisfies.
Requirement identifiers (e.g. SR-001) are defined in
``docs/SOFTWARE_REQUIREMENTS.md``.
"""

from .config import SERVICE_NAME, SERVICE_VERSION

__all__ = ["SERVICE_NAME", "SERVICE_VERSION"]
