"""clinician-api: BFF for the UI, enforcing SMART-on-FHIR scopes.

PROJECT_PLAN.md section 10. Aggregates risk-engine, alert-service, and rag-service
behind endpoints a clinician UI calls directly, each gated by
services.common.auth.require_scope -- see that module's docstring for why JWT
verification + SMART scope matching is real today even without a live Keycloak.

Every PHI-shaped read and every alert acknowledgement is written to the audit log
(services.common.audit) -- Phase 8's compliance deliverable is this same table
under Postgres with the surrounding infra around it, not a different table.

Downstream services are called over real HTTP (httpx.AsyncClient), at URLs from
environment variables defaulting to each service's own Dockerfile port -- this is a
real BFF, not one that imports its dependencies' Python code directly. Async,
not sync, httpx: a sync client would block a worker thread for the entire
downstream round trip on every request, and -- found by actually trying it --
httpx's ASGITransport (used to test this against the real risk-engine/alert-service
app objects in-process, no live socket needed) only implements the async
`handle_async_request`, not the sync transport interface, so a sync Client raises
`AttributeError: 'ASGITransport' object has no attribute 'handle_request'` the
moment a test tries to use it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.common.audit import AuditLog  # noqa: E402
from services.common.auth import AuthContext, require_scope  # noqa: E402

RISK_ENGINE_URL = os.environ.get("RISK_ENGINE_URL", "http://localhost:8001")
ALERT_SERVICE_URL = os.environ.get("ALERT_SERVICE_URL", "http://localhost:8005")
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8004")
AGENT_ORCHESTRATOR_URL = os.environ.get("AGENT_ORCHESTRATOR_URL", "http://localhost:8008")

DEFAULT_AUDIT_DB_PATH = Path(__file__).resolve().parent / "clinician_api_audit.db"

app = FastAPI(title="clinician-api", version="0.1.0")

_clients: dict[str, httpx.AsyncClient] = {}
_audit_log: AuditLog | None = None


def configure_clients(
    risk_engine: httpx.AsyncClient | None = None,
    alert_service: httpx.AsyncClient | None = None,
    rag_service: httpx.AsyncClient | None = None,
    agent_orchestrator: httpx.AsyncClient | None = None,
) -> None:
    """Tests call this with httpx.AsyncClient(transport=httpx.ASGITransport(app=...))
    pointed at the real service app objects; production leaves it uncalled and
    get_client() lazily opens a real network client at the configured URL."""
    if risk_engine is not None:
        _clients["risk-engine"] = risk_engine
    if alert_service is not None:
        _clients["alert-service"] = alert_service
    if rag_service is not None:
        _clients["rag-service"] = rag_service
    if agent_orchestrator is not None:
        _clients["agent-orchestrator"] = agent_orchestrator


def get_client(name: str, base_url: str) -> httpx.AsyncClient:
    if name not in _clients:
        _clients[name] = httpx.AsyncClient(base_url=base_url)
    return _clients[name]


def get_audit_log() -> AuditLog:
    global _audit_log
    if _audit_log is None:
        _audit_log = AuditLog(DEFAULT_AUDIT_DB_PATH)
    return _audit_log


def set_audit_log(log: AuditLog) -> None:
    global _audit_log
    _audit_log = log


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "clinician-api"}


@app.get("/risk/{stay_id}/{hour}")
async def get_risk(
    stay_id: int,
    hour: int,
    patient_ref: str,
    ctx: AuthContext = Depends(require_scope("RiskAssessment", "read")),
) -> dict:
    """`patient_ref` is a query parameter, not a path segment: it is a full FHIR
    reference like "ICUStay/34547401" (services/contracts/observation.py's
    documented shape), and a literal "/" inside a plain Starlette path parameter
    does not match what its route declares -- found by getting a bare 404 (route
    not matched at all, before the scope check even ran) instead of the expected
    403/200 in this module's own tests.
    """
    client = get_client("risk-engine", RISK_ENGINE_URL)
    resp = await client.get(f"/score/{stay_id}/{hour}")
    if resp.status_code == 404:
        raise HTTPException(404, resp.json().get("detail", "not found"))
    resp.raise_for_status()
    get_audit_log().record(
        actor=f"clinician:{ctx.subject}",
        action="phi_read",
        subject_ref=patient_ref,
        payload={"resource": "RiskAssessment", "stay_id": stay_id, "hour": hour},
    )
    return resp.json()


@app.get("/patients")
async def list_patients(
    ctx: AuthContext = Depends(require_scope("RiskAssessment", "read")),
) -> list[dict]:
    """Ward view (PROJECT_PLAN.md section 12): every monitored patient ranked
    by current risk. Proxies risk-engine's /patients rather than reading the
    warehouse directly -- clinician-api stays a pure BFF.
    """
    client = get_client("risk-engine", RISK_ENGINE_URL)
    resp = await client.get("/patients")
    resp.raise_for_status()
    get_audit_log().record(
        actor=f"clinician:{ctx.subject}",
        action="phi_read",
        subject_ref=None,
        payload={"resource": "RiskAssessment", "op": "list_patients"},
    )
    return resp.json()


@app.get("/patients/{stay_id}/trace")
async def get_trace(
    stay_id: int, ctx: AuthContext = Depends(require_scope("Observation", "read"))
) -> list[dict]:
    """The NEWS2 trace for one patient's whole stay -- the chart on the
    dashboard's patient view."""
    client = get_client("risk-engine", RISK_ENGINE_URL)
    resp = await client.get(f"/trace/{stay_id}")
    if resp.status_code == 404:
        raise HTTPException(404, resp.json().get("detail", "not found"))
    resp.raise_for_status()
    get_audit_log().record(
        actor=f"clinician:{ctx.subject}",
        action="phi_read",
        subject_ref=f"ICUStay/{stay_id}",
        payload={"resource": "Observation", "op": "news2_trace"},
    )
    return resp.json()


@app.post("/risk/{stay_id}/{hour}/ml")
async def get_risk_ml(
    stay_id: int,
    hour: int,
    patient_ref: str,
    ctx: AuthContext = Depends(require_scope("RiskAssessment", "read")),
) -> dict:
    """Phase 5's learned model + SHAP attribution -- "contributing SHAP
    factors" on the patient view. A real 502 (not a fabricated score) if
    risk-engine itself has no model exported (it returns 503 in that case).
    """
    client = get_client("risk-engine", RISK_ENGINE_URL)
    resp = await client.post(f"/score/ml/{stay_id}/{hour}")
    if resp.status_code == 503:
        raise HTTPException(503, resp.json().get("detail", "no Phase 5 model exported"))
    if resp.status_code == 404:
        raise HTTPException(404, resp.json().get("detail", "not found"))
    resp.raise_for_status()
    get_audit_log().record(
        actor=f"clinician:{ctx.subject}",
        action="phi_read",
        subject_ref=patient_ref,
        payload={"resource": "RiskAssessment", "op": "ml_score", "stay_id": stay_id, "hour": hour},
    )
    return resp.json()


@app.post("/patients/{stay_id}/{hour}/assessment")
async def get_agent_assessment(
    stay_id: int,
    hour: int,
    patient_ref: str,
    ctx: AuthContext = Depends(require_scope("RiskAssessment", "read")),
) -> dict:
    """Runs the Phase 4 agent graph and returns its full state: risk score,
    retrieved note passages with fact-ledger citations, the escalation
    rationale, and the LLM summary -- the patient view's "agent's escalation
    rationale" and "retrieved note passages with ledger citations" panels in
    one call, since agent-orchestrator's /run already bundles exactly that.
    """
    client = get_client("agent-orchestrator", AGENT_ORCHESTRATOR_URL)
    resp = await client.post(
        "/run", json={"stay_id": stay_id, "hour": hour, "patient_ref": patient_ref}
    )
    resp.raise_for_status()
    get_audit_log().record(
        actor=f"clinician:{ctx.subject}",
        action="phi_read",
        subject_ref=patient_ref,
        payload={"resource": "RiskAssessment", "op": "agent_assessment", "stay_id": stay_id},
    )
    return resp.json()


@app.get("/alerts/active")
async def get_active_alerts(
    ctx: AuthContext = Depends(require_scope("Communication", "read")),
) -> list[dict]:
    """Ward-wide alert inbox (PROJECT_PLAN.md section 12), across every
    patient -- distinct from GET /alerts, which is scoped to one."""
    client = get_client("alert-service", ALERT_SERVICE_URL)
    resp = await client.get("/alerts/active")
    resp.raise_for_status()
    get_audit_log().record(
        actor=f"clinician:{ctx.subject}",
        action="phi_read",
        subject_ref=None,
        payload={"resource": "Communication", "op": "list_active_alerts"},
    )
    return resp.json()


@app.get("/alerts")
async def get_alerts(
    patient_ref: str, ctx: AuthContext = Depends(require_scope("Communication", "read"))
) -> list[dict]:
    client = get_client("alert-service", ALERT_SERVICE_URL)
    resp = await client.get("/alerts", params={"patient_ref": patient_ref})
    resp.raise_for_status()
    get_audit_log().record(
        actor=f"clinician:{ctx.subject}",
        action="phi_read",
        subject_ref=patient_ref,
        payload={"resource": "Communication", "op": "list_alerts"},
    )
    return resp.json()


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    patient_ref: str,
    ctx: AuthContext = Depends(require_scope("Communication", "write")),
) -> dict:
    client = get_client("alert-service", ALERT_SERVICE_URL)
    resp = await client.post(
        f"/alerts/{alert_id}/acknowledge", json={"acknowledged_by": ctx.subject}
    )
    if resp.status_code == 404:
        raise HTTPException(404, "alert not found")
    resp.raise_for_status()
    get_audit_log().record(
        actor=f"clinician:{ctx.subject}",
        action="alert_ack",
        subject_ref=patient_ref,
        payload={"alert_id": alert_id},
    )
    return resp.json()


@app.post("/alerts/{alert_id}/escalate")
async def escalate_alert(
    alert_id: int,
    patient_ref: str,
    ctx: AuthContext = Depends(require_scope("Communication", "write")),
) -> dict:
    client = get_client("alert-service", ALERT_SERVICE_URL)
    resp = await client.post(f"/alerts/{alert_id}/escalate")
    if resp.status_code == 404:
        raise HTTPException(404, "alert not found")
    resp.raise_for_status()
    get_audit_log().record(
        actor=f"clinician:{ctx.subject}",
        action="alert_escalate",
        subject_ref=patient_ref,
        payload={"alert_id": alert_id},
    )
    return resp.json()


@app.post("/alerts/{alert_id}/suppress")
async def suppress_alert(
    alert_id: int,
    patient_ref: str,
    ctx: AuthContext = Depends(require_scope("Communication", "write")),
) -> dict:
    client = get_client("alert-service", ALERT_SERVICE_URL)
    resp = await client.post(f"/alerts/{alert_id}/suppress")
    if resp.status_code == 404:
        raise HTTPException(404, "alert not found")
    resp.raise_for_status()
    get_audit_log().record(
        actor=f"clinician:{ctx.subject}",
        action="alert_suppress",
        subject_ref=patient_ref,
        payload={"alert_id": alert_id},
    )
    return resp.json()


@app.get("/search")
async def search_notes(
    q: str, k: int = 5, ctx: AuthContext = Depends(require_scope("DocumentReference", "read"))
) -> list[dict]:
    client = get_client("rag-service", RAG_SERVICE_URL)
    resp = await client.get("/search", params={"q": q, "k": k})
    resp.raise_for_status()
    get_audit_log().record(
        actor=f"clinician:{ctx.subject}",
        action="phi_read",
        subject_ref=None,
        payload={"resource": "DocumentReference", "op": "search", "query": q},
    )
    return resp.json()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8007)
