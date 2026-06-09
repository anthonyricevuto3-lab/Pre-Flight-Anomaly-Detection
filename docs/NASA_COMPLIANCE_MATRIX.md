# NASA Compliance Matrix

This document walks through **every** NASA directive, standard, handbook,
requirement area, policy category, and rule/resource category identified in
the project's reference set, and records an honest disposition for each.

**Disposition legend**

| Symbol | Meaning |
|--------|---------|
| ✅ Implemented | Addressed in this repository. |
| 🟡 Partial | Addressed in a manner scaled to the project; see note. |
| ⚪ N/A (tailored) | Not applicable to this software; tailored out per NPR 7150.2D §2.2. |

**Project classification.** Self-assessed **NPR 7150.2D Class D/E** (research &
technology / non–safety-critical demonstrator), per NPR 7150.2D Appendix D.
Tailoring of management- and safety-process requirements is performed under
NPR 7150.2D §2.2 (Principles Related to Tailoring). This is a demonstrator and
has **not** undergone Independent Verification & Validation (IV&V).

**Verification gate (objective evidence).** `flake8` (0 findings) + `mypy`
(0 issues, 9 files) + `pytest` (27 passed, `-W error`). Run in CI before deploy.

---

## 1. NPR 7150.2D — NASA Software Engineering Requirements

Mandatory for NASA employees; Office of the Chief Engineer; effective
2022-03-08, expires 2027-03-08.

### Preface & Front Matter

| Area | Disp. | Notes |
|------|-------|-------|
| P.1 Purpose | ✅ | Project goal and scope stated in `README.md` and `docs/`. |
| P.2 Applicability | ✅ | Class D/E self-assessment recorded (this file, §Project classification). |
| P.3 Authority | ⚪ | Agency governance clause; no project artifact required. |
| P.4 Applicable Documents and Forms | ✅ | Reference set listed in `docs/SOFTWARE_REQUIREMENTS.md`. |
| P.5 Measurement/Verification | ✅ | Verification gate + `docs/SOFTWARE_ASSURANCE.md` §2; metrics in §8 below. |
| P.6 Cancellation | ⚪ | Directive lifecycle clause; not a project artifact. |

### Chapter 1 — Introduction

| Area | Disp. | Notes |
|------|-------|-------|
| 1.1 Overview | ✅ | `README.md`. |
| 1.2 Hierarchy of NASA SW Documents | ✅ | Reference hierarchy captured in `docs/SOFTWARE_REQUIREMENTS.md`. |
| 1.3 Document Structure | ✅ | `docs/` set: requirements, design/assurance, coding standards, this matrix. |

### Chapter 2 — Roles, Responsibilities, Tailoring

| Area | Disp. | Notes |
|------|-------|-------|
| 2.1 Roles and Responsibilities | 🟡 | Single-developer demonstrator; developer holds engineering + assurance roles. Recorded here. |
| 2.2 Principles Related to Tailoring | ✅ | Tailoring basis for Class D/E documented (this file). |

### Chapter 3 — Software Management

| Area | SWE | Disp. | Notes |
|------|-----|-------|-------|
| 3.1 Software Life Cycle Planning | SWE-013 | 🟡 | Lightweight iterative lifecycle described in `docs/SOFTWARE_ASSURANCE.md` §1. |
| 3.2 Software Cost Estimation | SWE-015 | ⚪ | No budgeted cost baseline for a personal demonstrator. |
| 3.3 Software Schedules | SWE-016 | ⚪ | No formal schedule baseline. |
| 3.4 Software Training | SWE-017 | ⚪ | Single developer; no training program required. |
| 3.5 Software Classification Assessments | SWE-020 | ✅ | Class D/E self-assessment (this file). |
| 3.6 SA and Independent V&V | SWE-022/039 | 🟡 | Software Assurance performed (`docs/SOFTWARE_ASSURANCE.md`); **IV&V not performed** (out of scope for a demonstrator — declared residual risk R-05). |
| 3.7 Safety-Critical Software | SWE-023 | ⚪ | Not safety-critical: no control authority, no hazard interface. Determination recorded (this file, §SA determination). |
| 3.8 Automatic Generation of Source Code | SWE-146 | ⚪ | No auto-generated code in the product. |
| 3.9 Software Development Processes/Practices | SWE-033 | ✅ | Defined coding standard + static analysis + tests + reviews (`docs/CODING_STANDARDS.md`). |
| 3.10 Software Reuse | SWE-027 | ✅ | Only reuse is the pinned `azure-functions` runtime binding; rationale in `requirements.txt`. |
| 3.11 Software Cybersecurity | SWE-156/157/207 | ✅ | Secure-coding controls (`docs/SOFTWARE_ASSURANCE.md` §5); debug endpoint removed; errors sanitized; input validated; CWE notes §7 below. |
| 3.12 Software Bi-Directional Traceability | SWE-052 | ✅ | Requirement ↔ code ↔ test tables in `docs/SOFTWARE_REQUIREMENTS.md`. |

### Chapter 4 — Software Engineering (Life Cycle)

| Area | SWE | Disp. | Notes |
|------|-----|-------|-------|
| 4.1 Software Requirements | SWE-050 | ✅ | `docs/SOFTWARE_REQUIREMENTS.md` (SR-001…SR-051), uniquely identified & testable. |
| 4.2 Software Architecture | SWE-057 | ✅ | Layered architecture record (`docs/SOFTWARE_ASSURANCE.md` §4). |
| 4.3 Software Design | SWE-058 | ✅ | Module docstrings document each unit's design and responsibilities. |
| 4.4 Software Implementation | SWE-060/134/135 | ✅ | Coding standard (`docs/CODING_STANDARDS.md`), enforced by `flake8`+`mypy` (`setup.cfg`). |
| 4.5 Software Testing | SWE-061/062/066 | ✅ | `tests/` (27 tests), each traced to a requirement; zero-warning policy. |
| 4.6 Operations, Maintenance, Retirement | SWE-075 | 🟡 | Maintenance via inspection checklist + CI; retirement = remove deployment & repo archive. Noted in `docs/SOFTWARE_ASSURANCE.md`. |

### Chapter 5 — Supporting Life Cycle

| Area | SWE | Disp. | Notes |
|------|-----|-------|-------|
| 5.1 Software Configuration Management | SWE-079 | ✅ | Git VCS; pinned dependencies; `.gitignore`/`.funcignore` exclude artifacts; versioned (`SERVICE_VERSION`). |
| 5.2 Software Risk Management | SWE-086 | ✅ | Risk register in §9 below. |
| 5.3 Software Peer Reviews/Inspections | SWE-087/088 | ✅ | Inspection checklist in `docs/CODING_STANDARDS.md` (aligned to §5.3; NASA-STD-8739.9 inactive). |
| 5.4 Software Measurements | SWE-090/091 | ✅ | Measurement snapshot in §8 below. |
| 5.5 Non-conformance / Defect Management | SWE-201 | 🟡 | Defects tracked via Git history + GitHub Issues; CI gate prevents regressions. Process noted §10. |

### Chapter 6 — Recommended Documentation

| Area | Disp. | Notes |
|------|-------|-------|
| 6.1 Software Engineering Products | ✅ | Products produced: requirements, design/assurance, coding standard, test suite, this matrix. |
| 6.2 Software Engineering Product Content | 🟡 | Content scaled to Class D/E; full §6.2 templates not produced (tailored). |

### Appendices

| Area | Disp. | Notes |
|------|-------|-------|
| A Definitions | ✅ | Key terms defined inline in docstrings/docs (anomaly, MAD, inlier/outlier). |
| B Acronyms | ✅ | SA, IV&V, MAD, CORS, CWE, V&V expanded on first use across docs. |
| C Requirements Mapping Matrix | ✅ | This document + `docs/SOFTWARE_REQUIREMENTS.md` tables. |
| D Software Classifications | ✅ | Class D/E selected (this file). |
| E References | ✅ | `docs/SOFTWARE_REQUIREMENTS.md` reference list with URLs. |

### Safety-Critical Software (SA) Determination

Per NPR 7150.2D §3.7 / NASA-STD-8739.8B §4.2, this software is **not
safety-critical**: it has no command/control authority over flight or ground
hardware, performs no real-time hazard mitigation, and its output is advisory
analysis only. No software hazard causes were identified (8739.8B Table 2).

---

## 2. NASA-STD-8739.8B — Software Assurance and Software Safety

Active, mandatory; OSMA; discipline 8000.

| Area | Disp. | Notes |
|------|-------|-------|
| 1 Scope | ✅ | Assurance scope stated in `docs/SOFTWARE_ASSURANCE.md`. |
| 1.1 Document Purpose | ✅ | Plan purpose stated. |
| 1.2 Applicability | ✅ | Applied at Class D/E scale. |
| 1.3 Documentation and Deliverables | ✅ | Assurance plan + this matrix are the deliverables. |
| 1.4 Request for Relief | ⚪ | No formal relief process for a demonstrator; tailoring recorded instead. |
| 2 Applicable/Reference Documents | ✅ | See applicable-documents dispositions below. |
| 2.3 Order of Precedence | ⚪ | Conflict-resolution clause; none encountered. |
| 3 Acronyms and Definitions | ✅ | Expanded across docs. |
| 4 SA & Software Safety Requirements | 🟡 | Core SA activities performed (analysis, V&V, traceability); full 8739.8B matrix is agency-internal and tailored. |
| 4.1 Software Assurance Description | ✅ | `docs/SOFTWARE_ASSURANCE.md`. |
| 4.2 Safety-Critical Determination | ✅ | Determination recorded (§1 above). |
| 4.3 SA & Safety Requirements | 🟡 | Tailored to Class D/E. |
| 4.4 Independent V&V | ⚪ | IV&V not performed (residual risk R-05). |
| 4.5 Tailoring | ✅ | Tailoring rationale recorded. |
| Appendix A — Hazard Development | ⚪ | No software-related hazards (non–safety-critical). |
| Table 1 — SA Requirements Mapping | 🟡 | Represented by this matrix at project scale. |
| Table 2 — SW causes in hazard analysis | ⚪ | No hazards identified. |

### 8739.8B Applicable / Reference Documents

| Document | Disp. | Notes |
|----------|-------|-------|
| NPR 1400.1 — Directives & Charters | ⚪ | Agency directive governance; not a software artifact. |
| NPR 7120.5 — Space Flight Program/Project Mgmt | ⚪ | No flight program. |
| NPR 7120.10 — Technical Standards for Programs | ⚪ | No program standards-tailoring authority. |
| NPR 7150.2 — Software Engineering Requirements | ✅ | Primary basis (§1 above). |
| NPR 8000.4 — Agency Risk Management | 🟡 | Project-level risk register §9. |
| NPR 8715.3 — General Safety Program | ⚪ | No physical-safety scope. |
| NASA-HDBK-2203 — SW Engineering Handbook | ✅ | Guidance source (§4 below). |
| NPD 2810.1 — Information Security Policy | 🟡 | Reflected via secure-coding controls (§3.11). |
| NPD 8720.1 — Reliability & Maintainability Policy | ⚪ | Hardware R&M policy. |
| NPR 1441.1 — Records Management | ⚪ | No agency records obligation for a demo. |
| NPR 2810.1 — Security of Information Technology | 🟡 | Secure-coding + least-privilege CORS reflect intent. |
| NPR 7123.1 — Systems Engineering | 🟡 | Lightweight SE: requirements→design→impl→test→trace. |
| NASA-STD-1006 — Space System Protection | ⚪ | No space system. |
| NASA-STD-7009 — Models and Simulations | 🟡 | The detector is a statistical model; determinism/reproducibility & documented assumptions partially align (credibility intent). |
| NASA-HDBK-8709.22 — SMA Acronyms | ⚪ | Reference glossary only. |
| NASA-HDBK-8739.23 — Complex Electronics | ⚪ | Hardware/electronics. |
| NASA-GB-8719.13 — SW Safety Guidebook | ⚪ | Software-safety guidance; N/A (non–safety-critical). |

---

## 3. NASA-STD-8739.9 — Software Formal Inspections

**Status: INACTIVE / cancelled** (last rev. 2013/2016). Not relied upon as an
active basis. Inspection intent is satisfied via **NPR 7150.2D §5.3** and the
peer-review checklist in `docs/CODING_STANDARDS.md`.

---

## 4. NASA-HDBK-2203 — Software Engineering & Assurance Handbook (SWEHB)

Guidance handbook supporting NPR 7150.2D and NASA-STD-8739.8B.

| SWEHB area | Disp. | Notes |
|------------|-------|-------|
| Book A — Introduction | ✅ | Used as guidance for this matrix. |
| Book B — Institutional Requirements | ⚪ | Center/institutional scope; N/A to a personal demo. |
| Book C — Project Software Requirements | ✅ | Guided `docs/SOFTWARE_REQUIREMENTS.md`. |
| Book D — Topics | ✅ | Guided coding-standard and assurance topics. |
| Book E — Tools, References, Terms | ✅ | Tools (`flake8`/`mypy`/`pytest`) + references applied. |
| Book F — SPAN (NASA-only) | ⚪ | NASA-internal; inaccessible/inapplicable. |
| NPR 7150.2D guidance | ✅ | Applied (§1). |
| NASA-STD-8739.8B guidance | ✅ | Applied (§2). |
| Multicore/Concurrent/Partitioned V&V | ⚪ | Single-threaded request handlers; no concurrency model. |
| Early Software Definition & Maturity | ✅ | Requirements defined before implementation. |
| Rigorous Defect Disposition | ✅ | CI gate + review checklist disposition defects. |
| Comprehensive V&V Execution | 🟡 | Unit V&V executed; system/IV&V tailored out. |
| Validated Testbeds | 🟡 | Local `func` host + pytest fixtures serve as the test environment. |
| Early Fault Management Definition | ✅ | Off-nominal handling defined (typed errors, sanitized responses). |
| Hazard-Driven Validation | ⚪ | No hazards (non–safety-critical). |

---

## 5. NASA Secure Coding Portal — Rules / Resources

| Area | Disp. | Notes |
|------|-------|-------|
| Secure Coding rules | ✅ | Encoded in `docs/CODING_STANDARDS.md` (secure error handling, validation). |
| Secure Coding guidelines | ✅ | Applied (least privilege, defense in depth). |
| Secure Coding tools | ✅ | `flake8`/`mypy` static analysis; `pytest`. |
| Secure Coding resources | ✅ | Referenced in docs. |
| Secure Coding requirements | ✅ | Mapped to SR-031/041/042/050. |
| Secure Coding & Standards tutorial | ⚪ | Training content (NASA-internal). |
| Java / C / C++ secure coding | ⚪ | Project is Python; equivalent practices applied. |
| Secure coding standards | ✅ | Project coding standard incorporates them. |
| Defense-in-depth mitigation | ✅ | Input validation + bounded resources + sanitized output + least-privilege CORS. |
| Vulnerability elimination | ✅ | Removed `/debug` info-disclosure endpoint; no `eval`/`exec`. |
| Secure practices across SDLC | ✅ | Secure design → implementation → static analysis → CI gate. |
| Discussion Forum / Vulnerability Updates / Videos / Links | ⚪ | NASA-internal community resources. |

### Secure-coding weaknesses (CWE) explicitly considered

| CWE | Weakness | Mitigation |
|-----|----------|-----------|
| CWE-200/209 | Information exposure / error message leakage | Errors sanitized; detail logged server-side only (SR-041/042). |
| CWE-20 | Improper input validation | All readings & CSV fields validated (SR-022/040). |
| CWE-400 | Uncontrolled resource consumption | Per-request & per-file iteration caps (SR-020). |
| CWE-489/215 | Active debug code / debug info exposure | `/debug` endpoint removed (SR-090). |
| CWE-942 | Overly permissive CORS | `ALLOWED_ORIGINS` configurable (SR-031). |
| CWE-95 | Eval injection | No `eval`/`exec`/dynamic code. |

---

## 6. NODIS — Directive / Procedure Categories

The NODIS library and its numeric classification are agency document-control
categories, not engineering requirements on a single application.

| Category | Disp. | Notes |
|----------|-------|-------|
| NASA-Wide Directives / Other / Historical / Regulations / Procurement Library | ⚪ | Agency document repositories; no project artifact. |
| 1000–6999, 8000–9999 series (Org, Legal, HR, Property, Procurement, Transport, Program Mgmt, Financial, Audits) | ⚪ | Outside the scope of an engineering demonstrator. |
| **7000–7999 — Program Formulation** | ✅ | The directive that governs this software, **NPR 7150.2D**, lives in this series and is the primary basis (§1). |

---

## 7. 8000 — Safety, Quality, Reliability, Maintainability Standards

Of the listed 8000-series documents, only the software standard applies; the
remainder govern physical hardware, workmanship, safety, metrology, or orbital
operations and are **N/A** to this software.

| Standard | Disp. |
|----------|-------|
| **NASA-STD-8739.8 — Software Assurance and Software Safety** | ✅ Applied (§2). |
| NASA-HDBK-8709.22 — SMA Acronyms/Definitions | ⚪ Reference glossary only. |
| NASA-HDBK-8709.24 — Safety Culture Handbook | ⚪ N/A (no org safety program). |
| NASA-HDBK-8709.25 — Human Factors Handbook | ⚪ N/A. |
| NASA-HDBK-8715.26 — Nuclear Flight Safety | ⚪ N/A. |
| NASA-HDBK-8719.14 — Limiting Orbital Debris | ⚪ N/A. |
| NASA-HDBK-8739.18 — Problems/Nonconformances/Anomalies | 🟡 Defect process §10 reflects intent at project scale. |
| NASA-HDBK-8739.19-2/-3/-4 — Measurement Quality Assurance | ⚪ N/A (no physical measurement equipment). |
| NASA-HDBK-8739.21 — ESD Control Workmanship | ⚪ N/A. |
| NASA-STD-8719.11 — Fire Protection & Life Safety | ⚪ N/A. |
| NASA-STD-8719.12 — Explosives/Propellants/Pyrotechnics | ⚪ N/A. |
| NASA-STD-8719.14 — Limiting Orbital Debris | ⚪ N/A. |
| NASA-STD-8719.17 — Pressure Vessels & Systems | ⚪ N/A. |
| NASA-STD-8719.24 (+ANNEX) — Payload Safety | ⚪ N/A. |
| NASA-STD-8719.25 — Range Flight Safety | ⚪ N/A. |
| NASA-STD-8719.26 — Non-Code Metallic Pressure Vessels | ⚪ N/A. |
| NASA-STD-8719.27 — Planetary Protection | ⚪ N/A. |
| NASA-STD-8719.28 — Wind Tunnel Model Systems | ⚪ N/A. |
| NASA-STD-8719.29 — Human-Rating | ⚪ N/A. |
| NASA-STD-8719.9 — Lifting Standard | ⚪ N/A. |
| NASA-STD-8729.1 — Reliability & Maintainability (Spaceflight) | ⚪ N/A (hardware R&M). |
| NASA-STD-8739.1 — Polymeric Application Workmanship | ⚪ N/A. |
| NASA-STD-8739.10 — EEE Parts Assurance | ⚪ N/A. |
| NASA-STD-8739.12 — Metrology & Calibration | ⚪ N/A. |
| NASA-STD-8739.14 — Fastener Procurement | ⚪ N/A. |
| NASA-STD-8739.4 — Crimping/Cabling Workmanship | ⚪ N/A. |
| NASA-STD-8739.5 — Fiber Optic Workmanship | ⚪ N/A. |
| NASA-STD-8739.6 — Workmanship Implementation | ⚪ N/A. |

---

## 8. Software Measurements (NPR 7150.2D §5.4)

Snapshot captured 2026-06-09:

| Metric | Value |
|--------|-------|
| Source files (product) | 9 |
| Source files incl. tests | 14 |
| Total lines of code | 894 (≈729 product + 165 test) |
| Unit tests | 27 (all passing) |
| Static-analysis findings (flake8) | 0 |
| Type-check issues (mypy) | 0 |
| Max cyclomatic complexity (mccabe) | ≤ 10 (enforced) |
| Largest function | < 60 logical lines (Power of Ten Rule 4) |
| Known open defects | 0 |

---

## 9. Software Risk Register (NPR 7150.2D §5.2)

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R-01 | Training CSV missing/corrupt at runtime | Low | Med | `TrainingDataError` → sanitized 500; health endpoint; CI presence. |
| R-02 | Malicious/oversized request payload | Med | Med | Input validation + `MAX_READINGS_PER_REQUEST` cap (SR-020/040). |
| R-03 | Information disclosure via errors | Low | High | Error sanitization; debug endpoint removed (SR-041/042/090). |
| R-04 | Overly permissive CORS in production | Med | Med | `ALLOWED_ORIGINS` configurable; documented to lock down (SR-031). |
| R-05 | No IV&V → undetected requirement gaps | Med | Low | Accepted residual risk for a Class D/E demonstrator; mitigated by static analysis + traceable tests. |
| R-06 | Model false negatives/positives | Med | Low | Robust median/MAD; deterministic & reproducible; documented assumptions. |
| R-07 | Anonymous auth on endpoints | Med | Med | Accepted for public demo; documented to require auth in production. |

---

## 10. Defect / Non-conformance Management (NPR 7150.2D §5.5)

- **Capture**: defects recorded as Git history and (where applicable) GitHub
  Issues.
- **Disposition**: reviewed against the §5.3 peer-review checklist; fixes land
  as new commits with accompanying tests.
- **Prevention of regression**: the CI verification gate (`flake8` + `mypy` +
  `pytest`) must pass before deployment, preventing reintroduction.
- **Current status**: 0 known open defects (see §8).
