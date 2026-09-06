"""Automated clinical reports (PROJECT_PLAN.md section 12) -- the missing
declared output identified when auditing v1.0 against the deliverable list.

Three report types, all rendered from the same LLM backend and anti-
fabrication discipline agent-orchestrator's Summarizer node uses (state only
what the structured facts say), with fact-ledger citations, exported to PDF
and FHIR DocumentReference:

- ``shift_handover.py`` -- per ward, on the real 4-hourly wall-clock boundary
  (E16).
- ``daily_summary.py`` -- per patient, per admission-relative day (R1).
- ``post_discharge_digest.py`` -- per patient, from morphed wearable
  telemetry (R7: labelled as morphed, never presented as measured).

See reports/README.md for exactly what "per ward" and "per day" mean in a
de-identified, per-patient-date-shifted dataset where there is no shared
wall-clock "now" across patients.
"""
