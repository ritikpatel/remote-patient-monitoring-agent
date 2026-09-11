"""The state every node in the graph reads and writes.

PROJECT_PLAN.md section 10: VitalsMonitor -> LabInterpreter -> RiskScorer ->
ContextRetriever -> EscalationDecider -> Summarizer, extended with DiseaseContext
(first) and CarePlanner (after the escalation decision) -- see graph.py.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # inputs
    stay_id: int
    hour: int
    patient_ref: str

    # DiseaseContext -- what this patient is being treated for. Runs FIRST because
    # three later nodes need it: RiskScorer (to ask for this disease's recalibrated
    # threshold), ContextRetriever (to scope retrieval to this admission's notes and
    # this diagnosis's guidance), and CarePlanner (which cannot recommend anything
    # sensible without knowing the condition).
    disease_context: dict[str, Any]

    # VitalsMonitor
    vitals: dict[str, Any]
    vitals_flags: list[str]

    # LabInterpreter
    abnormal_labs: list[dict[str, Any]]

    # RiskScorer -- the ONLY node allowed to produce a risk score, and it does so
    # by reading risk-engine's response, never by computing one itself (Phase 4's
    # first constraint). It now reads TWO of risk-engine's endpoints: the
    # deterministic NEWS2/SOFA path, which is what escalation is decided on, and
    # the promoted learned model, which is what severity is graded on.
    risk_score: dict[str, Any]
    ml_risk: dict[str, Any] | None
    # low | medium | high, graded by the learned model against cut-points fitted to
    # its own out-of-fold score distribution (ml/models/serving.py). None when no
    # promoted model is exported, or when the model declined to grade -- which is a
    # distinct state from "low" and must not collapse into it.
    severity: str | None
    severity_source: str

    # ContextRetriever
    context_passages: list[dict[str, Any]]

    # EscalationDecider -- `escalate` is decided by policy (risk_score alone);
    # `llm_advisory` is logged for observability but never read by the policy
    # (Phase 4's second constraint).
    escalate: bool
    escalation_reason: str
    llm_advisory: str | None

    # CarePlanner -- the disease-aware "what should be done" step. Only produces a
    # plan when the patient both escalates (deterministic policy) and grades high
    # (learned model); otherwise `care_plan` is None and `care_plan_skipped_reason`
    # says which of the two gates was not met.
    care_plan: dict[str, Any] | None
    care_plan_skipped_reason: str | None

    # Summarizer -- the one node that freely generates prose.
    summary: str

    # bookkeeping
    audit_rows: list[int]  # sequence numbers of the audit rows this run wrote
