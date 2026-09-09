"""event-studio: compose a synthetic patient event in a browser and drive the real pipeline.

Runs locally because it must reach `ingest-gateway` on localhost; a hosted page
cannot. The UI composes a complete multi-channel event (every channel the risk
model trains on), shows the NEWS2 the pipeline will compute *before* sending,
and posts it through the same `HTTPSink` contract the replay simulators use, so
this adds a front door rather than a second ingestion path.

**Sending an event now drives the real pipeline synchronously, not just the
async Kafka path.** Posting to `ingest-gateway` still happens, for the same
audit trail and Kafka-consumer path every other producer gets, but the HTTP
response you get back does not wait on that async path -- instead this calls
risk-engine, alert-service (which itself calls notification-gateway, which
sends email and/or SMS) and rag-service directly, in the same order the
streaming path uses, and returns what actually happened. See `_pipeline()`
below and `services/stream-processor/escalation.py`'s `EscalationLoop.run_now`,
which this reuses rather than reimplementing.

Neither email nor SMS is ever sent from a "Generate" click, only ever as a
real consequence of a real alert-service escalation reached via "Send to
pipeline" -- see `services/common/email.py` and `services/common/sms.py`'s
module docstrings for why those modules have exactly one real sender left
(notification-gateway) and this service only composes a preview of each.

**Optionally, also the on-demand agentic path -- for a real demo patient only.**
`_pipeline()`'s own docstring explains why the composed, synthetic vitals never
drive `agent-orchestrator`: its nodes read a real warehouse `(stay_id, hour)`
that a browser-composed patient does not have. That constraint is about the
*composed vitals*, not about whether this service can reach agent-orchestrator
at all -- `GET /patients` (proxying risk-engine, which already owns the
warehouse) lists the demo cohort's real stays, each with a real `(stay_id,
hour)`. When the operator picks one of those, "Send to pipeline" also calls
`agent-orchestrator POST /run` for that real stay, exactly the same call
`clinician-api` makes when a clinician opens that patient's chart -- see
`_agent_assessment()`. This runs *alongside*, not *instead of*, the fast
vitals path above: the two are independent questions ("does this composed
event escalate?" vs "what does this real patient's own chart say right now?")
and are reported separately so the two are never conflated as if the agent
had reasoned about the synthetic vitals.

    uvicorn app:app --app-dir services/event-studio --port 8009
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# stream-processor is a hyphenated directory (cannot be a dotted import path,
# same reason services/common/testing.py exists) -- this mirrors exactly how
# stream-processor's own app.py reaches its sibling modules: put the
# directory on sys.path, then a plain top-level import.
sys.path.insert(0, str(REPO_ROOT / "services" / "stream-processor"))

import duckdb  # noqa: E402
from escalation import EscalationLoop  # noqa: E402
from generator import generate  # noqa: E402
from services.common import email, sms  # noqa: E402
from services.contracts.observation import Observation, ObservationSource, QualityFlag  # noqa: E402
from simulators.sinks import DEFAULT_INGEST_API_KEY  # noqa: E402
from warehouse.news2 import load_thresholds, should_escalate, tier_for_score  # noqa: E402

WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"

GATEWAY_URL = os.environ.get("INGEST_GATEWAY_URL", "http://localhost:8000")
# Same constant the replay simulators authenticate with -- a private default
# here just produces a confusing 401 against a correctly-running gateway.
API_KEY = os.environ.get("INGEST_API_KEY", DEFAULT_INGEST_API_KEY)
# The same three real services EscalationLoop calls, plus rag-service for the
# context passages the agent's ContextRetriever would otherwise supply -- see
# _pipeline()'s docstring for why the full agent graph is not called for the
# composed vitals themselves.
RISK_ENGINE_URL = os.environ.get("RISK_ENGINE_URL", "http://localhost:8001")
ALERT_SERVICE_URL = os.environ.get("ALERT_SERVICE_URL", "http://localhost:8005")
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8004")
# Only reached for a real demo patient (see GET /patients, _agent_assessment) --
# never for the composed synthetic vitals. Same default port RUNBOOK.md and
# every other service use for agent-orchestrator.
AGENT_ORCHESTRATOR_URL = os.environ.get("AGENT_ORCHESTRATOR_URL", "http://localhost:8008")

app = FastAPI(title="event-studio")
STATIC = Path(__file__).resolve().parent / "static"

# Injectable for tests, same idiom as alert-service's set_store/set_notify_client
# and EscalationLoop's own `client` field: None means "real network client
# against RISK_ENGINE_URL/ALERT_SERVICE_URL/RAG_SERVICE_URL/GATEWAY_URL", which
# is what every real deployment gets. A test wires an httpx.MockTransport (or a
# TestClient mounted on another service's real ASGI app) through these instead
# of needing live infra or real ports.
_pipeline_client: httpx.Client | None = None
_rag_client: httpx.Client | None = None
_gateway_client: httpx.Client | None = None
_patients_client: httpx.Client | None = None
_agent_client: httpx.Client | None = None


def set_pipeline_client(client: httpx.Client | None) -> None:
    global _pipeline_client
    _pipeline_client = client


def set_rag_client(client: httpx.Client | None) -> None:
    global _rag_client
    _rag_client = client


def set_gateway_client(client: httpx.Client | None) -> None:
    global _gateway_client
    _gateway_client = client


def set_patients_client(client: httpx.Client | None) -> None:
    global _patients_client
    _patients_client = client


def set_agent_client(client: httpx.Client | None) -> None:
    global _agent_client
    _agent_client = client


_thresholds = None


def thresholds():
    """ICU-recalibrated cut-points, read from the warehouse rather than mirrored.

    `warehouse/news2.py` derives them per-build from the cohort's own
    percentiles, so a copy here would be a second source of truth that silently
    goes stale the first time the warehouse is rebuilt.
    """
    global _thresholds
    if _thresholds is None:
        conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
        try:
            _thresholds = load_thresholds(conn)
        finally:
            conn.close()
    return _thresholds


class EventRequest(BaseModel):
    severity: float = Field(0.5, ge=0.0, le=1.0)
    patient_ref: str = "Patient/10005866"
    seed: int | None = None
    send: bool = False
    # Set together, from GET /patients, to also drive agent-orchestrator's real
    # on-demand path for a real demo stay -- see _agent_assessment(). Left unset
    # (the default), behaviour is exactly what it always was: composed vitals
    # only, no agent-orchestrator call.
    stay_id: int | None = None
    hour: int | None = None


@app.get("/health")
def health() -> dict:
    th = thresholds()
    return {
        "status": "ok",
        "service": "event-studio",
        "sms_mode": os.environ.get("SMS_MODE", "dry_run"),
        "icu_medium": th.icu_medium,
        "icu_high": th.icu_high,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/patients")
def list_patients() -> list[dict]:
    """The demo cohort's real stays, each with a real `(stay_id, hour)` --
    proxies risk-engine's own `/patients` (the warehouse's owner) rather than
    reading the warehouse a second time here. Populates the UI's "real demo
    patient" picker; selecting one is what lets "Send to pipeline" also reach
    agent-orchestrator (see _agent_assessment()) instead of only the fast
    vitals path.
    """
    owns_client = _patients_client is None
    risk_client = _patients_client or httpx.Client(base_url=RISK_ENGINE_URL, timeout=10)
    try:
        r = risk_client.get("/patients")
        r.raise_for_status()
        return list(r.json())
    finally:
        if owns_client:
            risk_client.close()


def _observations(ev, patient_ref: str) -> list[Observation]:
    now = datetime.now(UTC)
    values = {**ev.values, **ev.extras}
    return [
        Observation.for_channel(
            channel=ch,
            patient_ref=patient_ref,
            device_id="event-studio",
            source=ObservationSource.manual,
            value=float(v),
            effective_time=now,
            # R7: composed, never measured. Every consumer can tell.
            quality_flags=[QualityFlag.synthetic],
        )
        for ch, v in values.items()
    ]


def _pipeline(patient_ref: str, vitals: dict, why: str) -> dict:
    """Drive the real pipeline synchronously and report what actually
    happened -- not the local preview's guess.

    Calls, in order: risk-engine (`EscalationLoop.run_now`, exactly the
    deterministic streaming path's own scoring + alert-raising code, reused
    rather than re-implemented), then alert-service (inside `run_now`), which
    itself calls notification-gateway (dashboard + push + a guarded SMS on
    high severity) and embeds that result -- and, only if the pipeline
    actually escalated, rag-service directly for context passages.

    **Why rag-service directly and not the full agent-orchestrator graph.**
    `agent-orchestrator`'s `VitalsMonitor`/`LabInterpreter`/`RiskScorer` nodes
    read `capstone.hourly_grid` and `labevents` by a real `(stay_id, hour)` --
    a browser-composed patient has neither. Inventing a fake `stay_id` would
    make those nodes silently return empty rows rather than an honest error,
    and `RiskScorer` would then relay risk-engine's *warehouse-backed* `/score`
    for a `stay_id` that does not exist -- a wrong, misleading result on
    exactly the demo path that most needs to be trustworthy. Calling
    rag-service's `/search` directly is the same retrieval call
    `ContextRetriever` makes, on the one thing this event actually has: an
    escalation reason to search on.
    """
    loop = EscalationLoop(
        risk_engine_url=RISK_ENGINE_URL,
        alert_service_url=ALERT_SERVICE_URL,
        client=_pipeline_client,
    )
    try:
        result = loop.run_now(patient_ref, vitals)
    finally:
        loop.close()

    if result.get("escalated"):
        owns_rag = _rag_client is None
        rag_client = _rag_client or httpx.Client(base_url=RAG_SERVICE_URL, timeout=10)
        try:
            r = rag_client.get("/search", params={"q": why, "k": 3})
            r.raise_for_status()
            result["context_passages"] = r.json()
        except httpx.HTTPError as exc:
            result["context_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if owns_rag:
                rag_client.close()

    return result


def _agent_assessment(stay_id: int, hour: int) -> dict:
    """Call agent-orchestrator's `/run` for a real demo stay -- the exact same
    call clinician-api makes for `POST /patients/{stay}/{hour}/assessment` when
    a clinician opens that patient's chart (RUNBOOK.md). This is deliberately a
    second, independent question from `_pipeline()`'s: it reasons over the
    *real* warehouse row at `(stay_id, hour)`, not over whatever vitals this
    event composed, and it runs regardless of whether the composed vitals
    escalated -- an on-demand assessment is not conditioned on this event's own
    alert, the same as the real system (workflow_simple.md's "on-demand, not in
    this path" note).
    """
    patient_ref = f"ICUStay/{stay_id}"
    owns_client = _agent_client is None
    agent_client = _agent_client or httpx.Client(base_url=AGENT_ORCHESTRATOR_URL, timeout=30)
    try:
        r = agent_client.post(
            "/run", json={"stay_id": stay_id, "hour": hour, "patient_ref": patient_ref}
        )
        r.raise_for_status()
        body = r.json()
        return {
            "stay_id": stay_id,
            "hour": hour,
            "patient_ref": patient_ref,
            "escalate": body.get("escalate"),
            "escalation_reason": body.get("escalation_reason"),
            "llm_advisory": body.get("llm_advisory"),
            "summary": body.get("summary"),
            "risk_score": body.get("risk_score"),
        }
    except httpx.HTTPError as exc:
        return {
            "stay_id": stay_id,
            "hour": hour,
            "patient_ref": patient_ref,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if owns_client:
            agent_client.close()


@app.post("/event")
def make_event(req: EventRequest) -> dict:
    ev = generate(req.severity, seed=req.seed)
    th = thresholds()
    tier = tier_for_score(ev.news2, th.icu_medium, th.icu_high)
    # GCS is excluded from the single-parameter limb (RCP 2017 + finding F1's
    # third limb handles falling GCS separately); a composed event has no
    # trajectory, so gcs_drop is False here by construction.
    max_nongcs = max(v for k, v in ev.subscores.items() if k != "gcs_total")
    escalates = should_escalate(tier, max_nongcs, False)
    why = (
        f"aggregate tier '{tier}'"
        if tier == "high"
        else (
            f"single red parameter (subscore 3) in "
            f"{[k for k, v in ev.subscores.items() if v >= 3 and k != 'gcs_total']}"
            if max_nongcs >= 3
            else "no limb triggered"
        )
    )

    result = {
        "severity": req.severity,
        "values": {**ev.values, **ev.extras},
        "subscores": ev.subscores,
        "news2": ev.news2,
        "tier_icu": tier,
        "max_component_nongcs": max_nongcs,
        "would_escalate": bool(escalates),
        "why": why,
        "sent": False,
        # Text only, zero network calls, no guard evaluated -- email.compose()/
        # sms.compose() cannot send anything. What the real pipeline would say
        # once escalated is `pipeline.alert.notification.{email,sms}` below,
        # populated only when this event is actually sent and actually
        # escalates for real.
        "email_preview": (
            email.compose(req.patient_ref, f"NEWS2 {ev.news2}", why).body if escalates else None
        ),
        "sms_preview": (
            sms.compose(req.patient_ref, f"NEWS2 {ev.news2}", why) if escalates else None
        ),
        "pipeline": None,
    }

    if not req.send:
        return result

    obs = _observations(ev, req.patient_ref)
    owns_gateway = _gateway_client is None
    gateway_client = _gateway_client or httpx.Client(
        base_url=GATEWAY_URL, headers={"X-API-Key": API_KEY}, timeout=10
    )
    try:
        r = gateway_client.post(
            "/observations/batch", json=[o.model_dump(mode="json") for o in obs]
        )
        result["sent"] = r.status_code in (200, 201, 202)
        result["gateway_status"] = r.status_code
    except httpx.HTTPError as exc:
        result["gateway_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if owns_gateway:
            gateway_client.close()

    # Realtime: score, alert and notify right now, synchronously, against the
    # same real services -- independent of whether the async Kafka path above
    # succeeded, since the vitals are already in hand either way.
    pipeline_result = _pipeline(req.patient_ref, {**ev.values, **ev.extras}, why)

    # A real demo patient was picked (GET /patients): also run the on-demand
    # agentic path for that stay's actual chart -- see _agent_assessment()'s
    # docstring for why this is independent of, not gated on, the line above.
    if req.stay_id is not None and req.hour is not None:
        pipeline_result["agent_assessment"] = _agent_assessment(req.stay_id, req.hour)

    result["pipeline"] = pipeline_result
    return result
