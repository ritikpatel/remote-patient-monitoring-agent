"""Evaluation and validation framework (PROJECT_PLAN.md section 13):
one HTML report across four axes -- prediction, alerting, latency, and RAG/
agent. Reuses Phase 4-6's real services and Phase 5's real trained model
throughout; this phase does not re-derive numbers those phases already
produced, it measures things they didn't (calibration/decision-curve
analysis, alert-level clinical metrics, end-to-end latency under load, and
retrieval/faithfulness/cost for the agent).
"""
