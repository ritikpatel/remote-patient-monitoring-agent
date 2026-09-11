"""agent-orchestrator: the LangGraph agent graph over HTTP.

See nodes.py's module docstring for the three constraints this service enforces
(the LLM never scores; the policy engine reads the LLM, not the reverse; every
step is audited) and graph.py for the six-node graph itself.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from graph import build_graph  # noqa: E402
from nodes import Dependencies, LLMBackend  # noqa: E402
from notes_synth.backends import GroqBackend  # noqa: E402
from services.common.audit_postgres import build_audit_log  # noqa: E402
from services.common.observability import instrument_metrics, instrument_tracing  # noqa: E402
from warehouse.news2 import should_escalate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
DEFAULT_AUDIT_DB_PATH = Path(__file__).resolve().parent / "agent_audit.db"

RISK_ENGINE_URL = os.environ.get("RISK_ENGINE_URL", "http://localhost:8001")
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8004")

app = FastAPI(title="agent-orchestrator", version="0.1.0")
instrument_metrics(app)
instrument_tracing(app, "agent-orchestrator")
_deps: Dependencies | None = None


def _default_llm() -> LLMBackend | None:
    if os.environ.get("GROQ_API_KEY"):
        return GroqBackend()
    return None  # falls back to the deterministic no-LLM path in nodes.py


def get_deps() -> Dependencies:
    global _deps
    if _deps is None:
        _deps = Dependencies(
            db_path=DEFAULT_DB_PATH,
            risk_engine_client=httpx.Client(base_url=RISK_ENGINE_URL),
            rag_client=httpx.Client(base_url=RAG_SERVICE_URL),
            audit_log=build_audit_log(
                DEFAULT_AUDIT_DB_PATH, postgres_dsn=os.environ.get("AUDIT_DATABASE_URL")
            ),
            llm=_default_llm(),
        )
    return _deps


def set_deps(deps: Dependencies) -> None:
    global _deps
    _deps = deps


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "agent-orchestrator",
        "llm_configured": get_deps().llm is not None,
    }


class RunRequest(BaseModel):
    stay_id: int
    hour: int
    patient_ref: str


@app.post("/run")
def run(req: RunRequest) -> dict:
    deps = get_deps()
    graph = build_graph(deps)
    result = graph.invoke(
        {"stay_id": req.stay_id, "hour": req.hour, "patient_ref": req.patient_ref, "audit_rows": []}
    )
    return dict(result)


# `ICUStay/34547401` is the Observation contract's reference shape for a MIMIC ICU
# patient (services/contracts/observation.py). A wearable volunteer is `Subject/S05`
# and has no stay_id, no chart and no diagnosis -- /assess reports that honestly
# rather than inventing an assessment for a patient the warehouse has never seen.
ICU_STAY_REF_PREFIX = "ICUStay/"


def stay_id_from_patient_ref(patient_ref: str) -> int | None:
    if not patient_ref.startswith(ICU_STAY_REF_PREFIX):
        return None
    try:
        return int(patient_ref[len(ICU_STAY_REF_PREFIX) :])
    except ValueError:
        return None


def assessment_hour_for_stay(stay_id: int) -> tuple[int | None, str]:
    """Which hour to assess when the caller did not say. Returns (hour, basis).

    **The latest hour is the wrong default here, and using it was a real bug.**
    risk-engine's `GET /patients` uses "latest scored hour" for the ward view, and
    copying that convention into this endpoint produced assessments of the wrong
    moment: a patient alerting at hour 3 with NEWS2 11 was assessed at the stay's
    final hour, where NEWS2 was 6 and the policy did not escalate. The email then
    contradicted itself -- "HIGH alert, NEWS2 11" in the header, "tier low, did not
    escalate" in the body -- and the care plan was suppressed for a patient who
    genuinely was escalating.

    An assessment exists because an alert fired, so the hour that matters is the one
    that fired it. This prefers the latest hour whose NEWS2 row actually escalates
    (recomputed here through the same shared predicate every other consumer uses, so
    it cannot drift from what raised the alert), and falls back to the latest scored
    hour only when no hour escalates -- e.g. an alert raised by the live streaming
    path, which scores on wall-clock vitals and has no ICU hour at all.

    The caller passing an explicit `hour` always wins over both. That is the right
    answer whenever the caller has one, and `RaiseAlertRequest` now carries it.
    """
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    try:
        rows = conn.execute(
            "SELECT hour, tier_icu, max_component_nongcs, gcs_drop "
            "FROM capstone.news2 WHERE stay_id = ? ORDER BY hour DESC",
            [stay_id],
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None, "no scored hour"
    for hour, tier_icu, max_nongcs, gcs_drop in rows:
        if should_escalate(tier_icu, max_nongcs, bool(gcs_drop)):
            return int(hour), "latest escalating hour"
    return int(rows[0][0]), "latest scored hour (no hour escalates)"


class AssessRequest(BaseModel):
    """What alert-service knows when an alert fires: a patient reference, and
    sometimes an hour. Everything else is looked up here.

    `hour` is optional because the two alert-raising paths differ: event-studio and
    the batch replay know the ICU hour they scored, while the live streaming path
    works in wall-clock time and has none. When it is supplied it is authoritative;
    when it is not, `assessment_hour_for_stay` picks the escalating hour.
    """

    patient_ref: str
    hour: int | None = None


@app.post("/assess")
def assess(req: AssessRequest) -> dict:
    """Run the graph for an alerting patient and return an alert-ready assessment.

    This is the entry point alert-service calls when it raises a genuinely new
    high-severity alert, and it exists rather than reusing `/run` because the caller
    has a `patient_ref` and no `(stay_id, hour)` -- resolving that is this endpoint's
    job, not the caller's.

    **Why this is safe to call from the alerting path, when `/run` was not.**
    `services/stream-processor/escalation.py` deliberately does not call this graph
    per observation, and that reasoning still holds: the graph makes LLM calls and
    running one per streamed vital sign would be both slow and expensive. The
    difference is where it is called from. alert-service invokes this **once per
    genuinely new or freshly escalated alert** -- after its own 4-hourly dedup (R6,
    E16) has already collapsed a burst of repeated escalations into one alert. The
    dedup that protects notification-gateway from re-paging protects this from
    re-running.

    Never raises for an unassessable patient: a wearable subject with no ICU stay,
    or a stay with no scored hour, returns `assessable: false` with a reason. The
    alert still goes out -- it just goes out without an assessment attached, which
    is the correct degradation. An alerting pipeline must not lose the alert because
    the narrative layer could not produce narrative.
    """
    stay_id = stay_id_from_patient_ref(req.patient_ref)
    if stay_id is None:
        return {
            "assessable": False,
            "reason": (
                f"patient_ref {req.patient_ref!r} is not an ICU stay reference "
                f"({ICU_STAY_REF_PREFIX}<stay_id>); no chart, diagnosis or hourly "
                "grid exists for it"
            ),
            "patient_ref": req.patient_ref,
        }

    if req.hour is not None:
        hour, hour_basis = req.hour, "supplied by caller"
    else:
        hour, hour_basis = assessment_hour_for_stay(stay_id)
    if hour is None:
        return {
            "assessable": False,
            "reason": f"no scored hour in capstone.news2 for stay_id={stay_id}",
            "patient_ref": req.patient_ref,
        }

    deps = get_deps()
    graph = build_graph(deps)
    result = dict(
        graph.invoke(
            {
                "stay_id": stay_id,
                "hour": hour,
                "patient_ref": req.patient_ref,
                "audit_rows": [],
            }
        )
    )

    disease = result.get("disease_context") or {}
    risk = result.get("risk_score") or {}
    ml_risk = result.get("ml_risk") or {}
    return {
        "assessable": True,
        "patient_ref": req.patient_ref,
        "stay_id": stay_id,
        "hour": hour,
        # How this hour was chosen. Visible in the payload because assessing the
        # wrong hour is a silent failure otherwise -- the numbers all look valid,
        # they are just about a different moment than the one that alerted.
        "hour_basis": hour_basis,
        "condition": disease.get("dx_title"),
        "dx_chapter": disease.get("dx_chapter"),
        "comorbidities": disease.get("comorbidities", []),
        "charlson_comorbidity_index": disease.get("charlson_comorbidity_index"),
        "news2": risk.get("news2"),
        "news2_tier_icu": risk.get("news2_tier_icu"),
        "dx_group": risk.get("dx_group"),
        "threshold_is_disease_specific": risk.get("threshold_is_disease_specific", False),
        "escalate": result.get("escalate"),
        "escalation_reason": result.get("escalation_reason"),
        "severity": result.get("severity"),
        "severity_source": result.get("severity_source"),
        "ml_probability": ml_risk.get("probability"),
        "ml_in_validated_scope": ml_risk.get("in_validated_scope"),
        "ml_scope_note": ml_risk.get("scope_note"),
        "care_plan": result.get("care_plan"),
        "care_plan_skipped_reason": result.get("care_plan_skipped_reason"),
        "summary": result.get("summary"),
        "audit_rows": result.get("audit_rows", []),
    }


@app.get("/audit/verify")
def verify_audit() -> dict:
    ok, bad_seq = get_deps().audit_log.verify_chain()
    return {"intact": ok, "first_broken_seq": bad_seq}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008)
