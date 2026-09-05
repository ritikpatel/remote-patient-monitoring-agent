"""The state every node in the graph reads and writes.

PROJECT_PLAN.md section 10: VitalsMonitor -> LabInterpreter -> RiskScorer ->
ContextRetriever -> EscalationDecider -> Summarizer.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # inputs
    stay_id: int
    hour: int
    patient_ref: str

    # VitalsMonitor
    vitals: dict[str, Any]
    vitals_flags: list[str]

    # LabInterpreter
    abnormal_labs: list[dict[str, Any]]

    # RiskScorer -- the ONLY node allowed to produce a risk score, and it does so
    # by reading risk-engine's response, never by computing one itself (Phase 4's
    # first constraint).
    risk_score: dict[str, Any]

    # ContextRetriever
    context_passages: list[dict[str, Any]]

    # EscalationDecider -- `escalate` is decided by policy (risk_score alone);
    # `llm_advisory` is logged for observability but never read by the policy
    # (Phase 4's second constraint).
    escalate: bool
    escalation_reason: str
    llm_advisory: str | None

    # Summarizer -- the one node that freely generates prose.
    summary: str

    # bookkeeping
    audit_rows: list[int]  # sequence numbers of the audit rows this run wrote
