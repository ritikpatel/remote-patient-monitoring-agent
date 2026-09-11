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
from warehouse.disease import CHARLSON_FLAGS  # noqa: E402
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


def disease_context(deps: Dependencies) -> NodeFn:
    """What this patient is being treated for -- the first node in the graph.

    Runs before everything else because three later nodes are useless without it:
    RiskScorer reports which disease-specific threshold applied, ContextRetriever
    scopes retrieval to this admission and this diagnosis, and CarePlanner cannot
    recommend anything sensible for "a deteriorating patient" in the abstract.

    Reads `capstone.disease_context` (warehouse/disease.py). A stay with no row --
    a wearable patient with no ICU admission, say -- yields an empty dict rather
    than an exception: the rest of the graph degrades to the disease-blind
    behaviour it had before this node existed, which is the correct fallback.
    """

    def run(state: AgentState) -> dict:
        conn = _conn(deps)
        try:
            row = conn.execute(
                "SELECT hadm_id, dx_chapter, dx_title, dx_icd_code, "
                "charlson_comorbidity_index FROM capstone.disease_context "
                "WHERE stay_id = ?",
                [state["stay_id"]],
            ).fetchone()
            comorbidities: list[str] = []
            if row is not None:
                flags = conn.execute(
                    f"SELECT {', '.join(CHARLSON_FLAGS)} FROM capstone.disease_context "
                    "WHERE stay_id = ?",
                    [state["stay_id"]],
                ).fetchone()
                comorbidities = [
                    name for name, value in zip(CHARLSON_FLAGS, flags or [], strict=False) if value
                ]
        finally:
            conn.close()

        if row is None:
            return {
                "disease_context": {},
                "_tool_calls": ["warehouse.disease_context"],
            }
        hadm_id, dx_chapter, dx_title, dx_icd_code, charlson = row
        return {
            "disease_context": {
                "hadm_id": int(hadm_id) if hadm_id is not None else None,
                "dx_chapter": dx_chapter,
                "dx_title": dx_title,
                "dx_icd_code": dx_icd_code,
                "charlson_comorbidity_index": (int(charlson) if charlson is not None else None),
                "comorbidities": comorbidities,
            },
            "_tool_calls": ["warehouse.disease_context"],
        }

    return run


def risk_scorer(deps: Dependencies) -> NodeFn:
    """Constraint 1: this node's entire job is relaying risk-engine's responses.
    There is deliberately no arithmetic on vitals/labs here.

    It now relays TWO of them, and the split matters:

    * ``/score/{stay_id}/{hour}`` -- the deterministic NEWS2/SOFA path. This, and
      only this, is what EscalationDecider reads. The escalation authority is
      unchanged and still a pure function of the shared ``should_escalate``
      predicate.
    * ``/score/ml/{stay_id}/{hour}`` -- the promoted learned model, which grades
      **severity**. Severity decides whether a care plan is generated and whether a
      human is paged, not whether the alert exists.

    That division is the whole design: the learned model got a say in the chain
    without becoming the thing that decides an alert is real. A 503 from the ML
    endpoint (no model exported yet) leaves ``severity`` None and the deterministic
    path completely unaffected, which is why the exception is swallowed here rather
    than failing the run.
    """

    def run(state: AgentState) -> dict:
        resp = deps.risk_engine_client.get(f"/score/{state['stay_id']}/{state['hour']}")
        resp.raise_for_status()
        risk = resp.json()

        ml_risk = None
        tool_calls = ["risk-engine./score"]
        try:
            ml_resp = deps.risk_engine_client.post(f"/score/ml/{state['stay_id']}/{state['hour']}")
            if ml_resp.status_code == 200:
                ml_risk = ml_resp.json()
                tool_calls.append("risk-engine./score/ml")
        except httpx.HTTPError as exc:
            ml_risk = {"error": f"{type(exc).__name__}: {exc}"}

        severity = (ml_risk or {}).get("severity")
        if severity is not None:
            severity_source = "learned model (ml/models/serving.grade_severity)"
        else:
            # No promoted model, or an export predating severity cut-points. The
            # chain must not silently treat that as "low" -- an ungraded alert falls
            # back to the deterministic path's own judgement downstream.
            severity_source = "unavailable -- no graded learned-model score"
        return {
            "risk_score": risk,
            "ml_risk": ml_risk,
            "severity": severity,
            "severity_source": severity_source,
            "_tool_calls": tool_calls,
        }

    return run


def context_retriever(deps: Dependencies) -> NodeFn:
    """Two retrievals, not one, because a care plan needs two different things.

    Before this was disease-aware the node issued a single query built from the
    NEWS2 reason strings ("HR=131 contributes 2 to NEWS2") against the whole corpus.
    That has two failure modes, both of which were real: the query is a bag of
    numbers with almost no clinical vocabulary in it, and an unscoped corpus search
    returns whichever admission's discharge summary happens to use those words most
    densely -- i.e. **another patient's chart**, summarised under this patient's
    name.

    So: notes are scoped to this admission's ``hadm_id``, and the query is built
    from the diagnosis and comorbidities rather than from score arithmetic. The
    guideline half stays corpus-wide, which is the point of a guideline.
    """

    def run(state: AgentState) -> dict:
        risk = state.get("risk_score", {})
        disease = state.get("disease_context") or {}
        hadm_id = disease.get("hadm_id")

        # Clinical vocabulary first, score arithmetic second: "Sepsis, unspecified
        # organism / Infectious / renal_disease" retrieves usefully; "HR=131
        # contributes 2 to NEWS2" does not.
        disease_terms = " ".join(
            str(t)
            for t in [
                disease.get("dx_title"),
                disease.get("dx_chapter"),
                *(disease.get("comorbidities") or []),
            ]
            if t
        )
        query = (
            " ".join(filter(None, [disease_terms, " ".join(risk.get("reason", []))]))
            or "deterioration risk"
        )

        passages: list[dict] = []
        tool_calls = []
        if hadm_id is not None:
            note_resp = deps.rag_client.get(
                "/search",
                params={"q": query, "k": 3, "source": "note", "hadm_id": hadm_id},
            )
            note_resp.raise_for_status()
            passages.extend(note_resp.json())
            tool_calls.append("rag-service./search[notes,this-admission]")

        guideline_resp = deps.rag_client.get(
            "/search", params={"q": query, "k": 3, "source": "guideline"}
        )
        guideline_resp.raise_for_status()
        passages.extend(guideline_resp.json())
        tool_calls.append("rag-service./search[guidelines]")

        return {"context_passages": passages, "_tool_calls": tool_calls}

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


# Only a high-severity alert earns a care plan. Two gates, deliberately different
# in kind: `escalate` is the deterministic NEWS2 policy (does this alert exist at
# all), `severity == high` is the learned model's grade (is it bad enough to warrant
# generating and paging clinical guidance). Both must hold. A model that could
# *suppress* the escalation would violate Phase 4's second constraint; a model that
# only decides whether to attach guidance to an alert that is already real does not.
CARE_PLAN_SEVERITY = "high"


def care_planner(deps: Dependencies) -> NodeFn:
    """The disease-aware "what should be done" step.

    Everything upstream answers *is this patient deteriorating*. This node answers
    *and what does that mean for this particular patient* -- which is only a
    well-posed question once the diagnosis, the comorbidities, this admission's own
    notes and the relevant guideline passages are all in state, which is exactly
    what DiseaseContext and the rescoped ContextRetriever put there.

    Three properties the prompt enforces, because this is the node whose output
    reaches a clinician's inbox:

    1. **Grounded.** The retrieved passages are the only clinical source offered,
       and the prompt requires each recommendation to be attributable to one of
       them or to the patient's own stated observations. The passages carry
       `[F0xx]` fact ids from notes_synth's ledger, so a recommendation traceable
       to a note passage is traceable to the structured fact that note was allowed
       to state.
    2. **Not a prescription.** It proposes assessment and escalation steps -- who to
       call, what to check, what to watch -- and is explicitly barred from naming
       drug doses. This project has no clinical validation and no prescribing
       authority (PROJECT_PLAN.md section 17), and an LLM inventing a vasopressor
       dose from a TF-IDF hit is precisely the failure this constraint exists to
       make impossible.
    3. **Refusable.** If the retrieved context does not support a recommendation,
       saying so is a valid and expected output. A care plan that always produces
       five confident bullets regardless of input is not grounded, it is
       decorative.

    With no LLM configured this node returns a deterministic, non-generated plan
    assembled from the structured facts alone -- the same no-LLM fallback pattern
    `summarizer` uses. That path states what is known and recommends review; it
    never fabricates the reasoning it could not generate.
    """

    def run(state: AgentState) -> dict:
        escalate = state.get("escalate", False)
        severity = state.get("severity")
        disease = state.get("disease_context") or {}
        passages = state.get("context_passages", [])

        if not escalate:
            return {
                "care_plan": None,
                "care_plan_skipped_reason": (
                    "no care plan: the deterministic policy did not escalate"
                ),
            }
        if severity != CARE_PLAN_SEVERITY:
            graded = severity if severity is not None else "not graded (no promoted model)"
            return {
                "care_plan": None,
                "care_plan_skipped_reason": (
                    f"no care plan: alert escalated but severity is {graded}, "
                    f"below '{CARE_PLAN_SEVERITY}'"
                ),
            }

        condition = disease.get("dx_title") or "condition not coded"
        comorbidities = ", ".join(disease.get("comorbidities") or []) or "none recorded"
        citations = [
            {
                "passage_id": p.get("passage_id"),
                "source": p.get("source"),
                "fact_ids": p.get("fact_ids", []),
            }
            for p in passages
        ]

        if deps.llm is None:
            # Deterministic fallback: state what is known, recommend review, invent
            # nothing. Explicitly labelled so a reader never mistakes it for
            # generated clinical reasoning.
            plan = (
                f"[no LLM configured -- deterministic fallback] "
                f"High-severity deterioration alert for a patient with {condition} "
                f"(comorbidities: {comorbidities}). "
                f"Escalation basis: {state.get('escalation_reason')}. "
                f"Recommend urgent clinical review; {len(passages)} context passages "
                f"were retrieved and are attached for the reviewing clinician."
            )
            return {
                "care_plan": {
                    "condition": condition,
                    "severity": severity,
                    "recommended_actions": plan,
                    "citations": citations,
                    "generated": False,
                },
                "care_plan_skipped_reason": None,
                "_tool_calls": ["care_plan.deterministic_fallback"],
                "_model_id": None,
                "_tokens": 0,
            }

        system = (
            "You are assisting a clinical escalation workflow for a patient who has "
            "ALREADY been escalated by a deterministic policy. Your job is to state what "
            "should be assessed and who should be contacted, specific to this patient's "
            "condition and comorbidities.\n"
            "Rules:\n"
            "- Ground every recommendation in the retrieved context or the observations "
            "given. Do not introduce clinical claims that appear in neither.\n"
            "- Recommend assessment, monitoring and escalation steps only. Do NOT give "
            "drug doses, prescriptions, or specific therapy titrations.\n"
            "- If the retrieved context does not support a recommendation, say so "
            "plainly. Saying 'the retrieved context does not address this' is a correct "
            "answer, not a failure.\n"
            "- Write 3-5 short bullets, then one line naming the most urgent action.\n"
            "- This is a synthetic drill on de-identified data, not a real patient."
        )
        user = (
            f"Condition (primary diagnosis): {condition}\n"
            f"Diagnosis group: {disease.get('dx_chapter')}\n"
            f"Chronic comorbidities: {comorbidities}\n"
            f"Charlson index: {disease.get('charlson_comorbidity_index')}\n"
            f"Current vitals: {state.get('vitals')}\n"
            f"Abnormal labs: {state.get('abnormal_labs')}\n"
            f"Deterministic escalation: {state.get('escalation_reason')}\n"
            f"Learned-model severity: {severity} "
            f"(probability {(state.get('ml_risk') or {}).get('probability')})\n"
            f"Retrieved context:\n"
            + "\n".join(f"- [{p.get('source')}] {p.get('text')}" for p in passages)
        )
        result = deps.llm.generate(system, user, max_tokens=500)
        return {
            "care_plan": {
                "condition": condition,
                "severity": severity,
                "recommended_actions": result.text.strip(),
                "citations": citations,
                "generated": True,
            },
            "care_plan_skipped_reason": None,
            "_tool_calls": ["llm.care_plan"],
            "_model_id": result.model,
            "_tokens": result.input_tokens + result.output_tokens,
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
