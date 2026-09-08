"""The six graph nodes, and the audit-wrapping decorator every one of them runs
through.

PROJECT_PLAN.md section 10's three constraints, enforced here (not just documented):

1. **The LLM never computes a risk score.** risk_scorer's only job is to call
   risk-engine's HTTP API and store its response verbatim in state["risk_score"] --
   there is no code path in this file that derives a score any other way.
2. **EscalationDecider is a policy engine with LLM advisory input, not the
   reverse.** escalation_decider computes `escalate` from `risk_score` ALONE,
   before it ever asks the LLM for anything; the LLM's opinion is stored in
   `llm_advisory` and is provably never read back into `escalate` -- see
   test_nodes.py's test proving the policy overrides a contrary LLM opinion.
3. **Every agent step writes {input_hash, tool_calls, output, model_id, tokens,
   latency_ms} to the audit log.** `audited` wraps every node function below;
   no node is reachable without going through it.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import duckdb
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.common.audit import AuditLogProtocol  # noqa: E402
from state import AgentState  # noqa: E402
from warehouse.news2 import escalation_reason, should_escalate  # noqa: E402

# The escalation predicate is NOT redefined here. NEWS2 has two independent triggers
# -- the ICU-recalibrated aggregate tier (E5) and RCP 2017's single-parameter red flag
# (finding F1) -- and both are defined once in warehouse/news2.py so this policy node,
# eval/alerting.py's replay, and risk-engine's response can never drift apart. An
# earlier version of this node inlined `tier == "high"`, which silently dropped the
# second trigger; see warehouse/news2.py's module docstring for the case that found it
# and the measurement that chose the replacement.
ESCALATE_ON_TIER = "high"


class LLMBackend:
    """Structural protocol matching notes_synth.backends.Backend -- reused here
    rather than re-declared, since agent-orchestrator's advisory/summary
    generation is exactly the same shape of call (system, user, max_tokens) ->
    text + token usage that notes_synth already made real against Groq."""

    name: str
    model: str

    def generate(self, system: str, user: str, max_tokens: int): ...  # noqa: D102


@dataclass
class Dependencies:
    db_path: Path
    risk_engine_client: httpx.Client
    rag_client: httpx.Client
    audit_log: AuditLogProtocol
    llm: (
        LLMBackend | None
    )  # None disables the two LLM-touching nodes' generation (falls back to a fixed note)


NodeFn = Callable[[AgentState], dict]


def audited(node_name: str, deps: Dependencies, fn: Callable[[AgentState], dict]) -> NodeFn:
    """Wraps a node function so every call writes an audit row, whether or not an
    LLM was involved -- deterministic nodes log model_id=None, tokens=0.
    """

    def wrapped(state: AgentState) -> dict:
        start = time.monotonic()
        output = fn(state)
        latency_ms = (time.monotonic() - start) * 1000
        row = deps.audit_log.record_agent_step(
            node_name,
            input_data={k: v for k, v in state.items() if k != "audit_rows"},
            output=output,
            tool_calls=output.pop("_tool_calls", []),
            model_id=output.pop("_model_id", None),
            tokens=output.pop("_tokens", 0),
            latency_ms=latency_ms,
            subject_ref=state.get("patient_ref"),
        )
        existing = state.get("audit_rows", [])
        return {**output, "audit_rows": [*existing, row.seq]}

    return wrapped


def _conn(deps: Dependencies) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(deps.db_path), read_only=True)


def vitals_monitor(deps: Dependencies) -> NodeFn:
    def run(state: AgentState) -> dict:
        conn = _conn(deps)
        try:
            row = conn.execute(
                "SELECT hr, rr, spo2, sbp, temp_c, gcs_total, fio2 FROM capstone.hourly_grid "
                "WHERE stay_id = ? AND hour = ?",
                [state["stay_id"], state["hour"]],
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return {
                "vitals": {},
                "vitals_flags": ["no hourly_grid row for this stay/hour"],
                "_tool_calls": ["warehouse.hourly_grid"],
            }
        vitals = dict(
            zip(["hr", "rr", "spo2", "sbp", "temp_c", "gcs_total", "fio2"], row, strict=True)
        )
        flags = [f"{k}={v}" for k, v in vitals.items() if v is not None]
        return {"vitals": vitals, "vitals_flags": flags, "_tool_calls": ["warehouse.hourly_grid"]}

    return run


def lab_interpreter(deps: Dependencies) -> NodeFn:
    def run(state: AgentState) -> dict:
        conn = _conn(deps)
        try:
            hadm_id = conn.execute(
                "SELECT hadm_id FROM mimiciv_icu.icustays WHERE stay_id = ?", [state["stay_id"]]
            ).fetchone()
            if hadm_id is None:
                return {"abnormal_labs": [], "_tool_calls": ["warehouse.labevents"]}
            rows = conn.execute(
                """
                SELECT di.label, l.value, l.valueuom, l.charttime
                FROM mimiciv_hosp.labevents l
                JOIN mimiciv_hosp.d_labitems di ON l.itemid = di.itemid
                WHERE l.hadm_id = ? AND l.flag = 'abnormal'
                ORDER BY l.charttime DESC LIMIT 10
                """,
                [hadm_id[0]],
            ).fetchall()
        finally:
            conn.close()
        labs = [
            {"label": lbl, "value": v, "unit": u, "charttime": str(ct)} for lbl, v, u, ct in rows
        ]
        return {"abnormal_labs": labs, "_tool_calls": ["warehouse.labevents"]}

    return run


def risk_scorer(deps: Dependencies) -> NodeFn:
    """Constraint 1: this node's entire job is relaying risk-engine's response.
    There is deliberately no arithmetic on vitals/labs here."""

    def run(state: AgentState) -> dict:
        resp = deps.risk_engine_client.get(f"/score/{state['stay_id']}/{state['hour']}")
        resp.raise_for_status()
        return {"risk_score": resp.json(), "_tool_calls": ["risk-engine./score"]}

    return run


def context_retriever(deps: Dependencies) -> NodeFn:
    def run(state: AgentState) -> dict:
        risk = state.get("risk_score", {})
        query_terms = " ".join(risk.get("reason", [])) or "deterioration risk"
        resp = deps.rag_client.get("/search", params={"q": query_terms, "k": 3})
        resp.raise_for_status()
        return {"context_passages": resp.json(), "_tool_calls": ["rag-service./search"]}

    return run


def escalation_decider(deps: Dependencies) -> NodeFn:
    def run(state: AgentState) -> dict:
        risk = state.get("risk_score", {})
        tier = risk.get("news2_tier_icu")
        max_nongcs = risk.get("max_component_nongcs")
        red_params = ",".join(risk.get("red_params") or [])
        gcs_drop = risk.get("gcs_drop", False)
        # POLICY FIRST: escalate is decided here, before the LLM is ever consulted.
        escalate = should_escalate(tier, max_nongcs, gcs_drop)
        reason = escalation_reason(tier, max_nongcs, red_params, gcs_drop)

        advisory = None
        model_id = None
        tokens = 0
        if deps.llm is not None:
            system = (
                "You advise a deterministic clinical escalation policy. Your opinion is logged "
                "for review but the escalation decision has already been made by policy and "
                "cannot be changed by you. State only whether you agree with the decision and why, "
                "in one sentence."
            )
            user = f"Risk score: {risk}. Policy decision: escalate={escalate} ({reason})."
            result = deps.llm.generate(system, user, max_tokens=150)
            advisory = result.text.strip()
            model_id = result.model
            tokens = result.input_tokens + result.output_tokens

        return {
            "escalate": escalate,
            "escalation_reason": reason,
            "llm_advisory": advisory,
            "_tool_calls": ["risk_score (policy input only)", "news2.should_escalate"]
            + (["llm.advisory"] if deps.llm else []),
            "_model_id": model_id,
            "_tokens": tokens,
        }

    return run


def summarizer(deps: Dependencies) -> NodeFn:
    """Constraint 1's other half: the LLM's job here is synthesis and explanation
    of already-computed facts, not numeric reasoning -- the prompt is built
    entirely from state that earlier deterministic nodes already produced."""

    def run(state: AgentState) -> dict:
        if deps.llm is None:
            fallback = (
                f"[no LLM configured] escalate={state.get('escalate')}: "
                f"{state.get('escalation_reason')}"
            )
            return {"summary": fallback, "_model_id": None, "_tokens": 0}
        system = (
            "You write a one-paragraph clinical summary for a clinician dashboard from the "
            "structured facts given. State only what is given; do not invent vitals, labs, or "
            "scores not present in the input."
        )
        user = (
            f"Vitals: {state.get('vitals')}\n"
            f"Abnormal labs: {state.get('abnormal_labs')}\n"
            f"Risk score: {state.get('risk_score')}\n"
            f"Escalation decision: escalate={state.get('escalate')} "
            f"({state.get('escalation_reason')})\n"
            f"Retrieved context: {[p.get('text') for p in state.get('context_passages', [])]}"
        )
        result = deps.llm.generate(system, user, max_tokens=400)
        return {
            "summary": result.text.strip(),
            "_tool_calls": ["llm.summarize"],
            "_model_id": result.model,
            "_tokens": result.input_tokens + result.output_tokens,
        }

    return run
