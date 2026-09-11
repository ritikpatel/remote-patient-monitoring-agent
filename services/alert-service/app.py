"""alert-service: raise/dedupe/suppress/escalate/acknowledge over HTTP.
See store.py for the real logic (R6: dedup aligned to the 4-hourly clock)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from services.common.observability import instrument_metrics, instrument_tracing  # noqa: E402
from store import Alert, AlertStore  # noqa: E402

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "alerts.db"
NOTIFICATION_GATEWAY_URL = os.environ.get("NOTIFICATION_GATEWAY_URL", "http://localhost:8006")
AGENT_ORCHESTRATOR_URL = os.environ.get("AGENT_ORCHESTRATOR_URL", "http://localhost:8008")
# The agent graph runs six-to-eight nodes, two of which may call an LLM, and
# ml/models/serving.score_one rebuilds the whole feature frame per call (a known,
# documented cost). 30s is generous for that and still bounded: if the assessment
# cannot be produced in time the alert goes out without it rather than being held up.
AGENT_ASSESS_TIMEOUT_S = 30.0

app = FastAPI(title="alert-service", version="0.1.0")
instrument_metrics(app)
instrument_tracing(app, "alert-service")
_store: AlertStore | None = None
_notify_client: httpx.Client | None = None
_agent_client: httpx.Client | None = None


def get_store() -> AlertStore:
    global _store
    if _store is None:
        _store = AlertStore(DEFAULT_DB_PATH)
    return _store


def set_store(store: AlertStore) -> None:
    global _store
    _store = store


def get_notify_client() -> httpx.Client:
    global _notify_client
    if _notify_client is None:
        _notify_client = httpx.Client(base_url=NOTIFICATION_GATEWAY_URL, timeout=2.0)
    return _notify_client


def set_notify_client(client: httpx.Client) -> None:
    global _notify_client
    _notify_client = client


def get_agent_client() -> httpx.Client:
    global _agent_client
    if _agent_client is None:
        _agent_client = httpx.Client(
            base_url=AGENT_ORCHESTRATOR_URL, timeout=AGENT_ASSESS_TIMEOUT_S
        )
    return _agent_client


def set_agent_client(client: httpx.Client | None) -> None:
    global _agent_client
    _agent_client = client


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "alert-service"}


class RaiseAlertRequest(BaseModel):
    patient_ref: str
    alert_type: str
    severity: str
    message: str
    # The ICU hour this alert was scored at, when the caller knows it. Passed
    # straight through to agent-orchestrator's /assess so the assessment is of the
    # hour that actually alerted rather than of whichever hour the stay happens to
    # end on -- assessing the wrong hour is a silent failure, since every number in
    # the resulting payload looks valid and is simply about a different moment.
    # The live streaming path works in wall-clock time and has no ICU hour, so this
    # is optional and /assess falls back to finding the escalating hour itself.
    hour: int | None = None


@app.post("/alerts")
def raise_alert(req: RaiseAlertRequest) -> dict:
    """Raise (or dedup) an alert, and on a genuinely new one run the agent
    assessment BEFORE notifying, so the escalation email carries it.

    The ordering is the whole point and it is not incidental: assessment first,
    notification second. notification-gateway is what actually pages a human, and an
    email that arrives saying "NEWS2 9, escalate" and is then followed minutes later
    by a separate message containing the clinical reasoning is worse than one message
    with both. So the assessment is fetched synchronously here and handed to
    notification-gateway as part of the same `/notify` call.

    Both external calls are best-effort in the same way and for the same reason: a
    clinician sees the alert via `GET /alerts/active` regardless, so neither
    agent-orchestrator nor notification-gateway being unreachable may fail alert
    *raising*. But neither may fail silently either -- an assessment that could not
    be produced is reported in the response as an error, not omitted.
    """
    alert, was_new = get_store().raise_alert(
        req.patient_ref, req.alert_type, req.severity, req.message
    )
    notification = None
    assessment = None
    regraded_from = None
    if was_new or alert.status == "escalated":
        assessment = _assess(alert, req.hour)
        graded = _graded_severity(alert, assessment)
        if graded != alert.severity:
            regraded_from = alert.severity
            alert = get_store().regrade(alert.id, graded)
        notification = _notify_dashboard(alert, assessment)
    return {
        "alert": vars(alert),
        "was_new": was_new,
        "assessment": assessment,
        "regraded_from": regraded_from,
        "notification": notification,
    }


def _graded_severity(alert: Alert, assessment: dict | None) -> str:
    """The severity this alert is actually notified at: the learned model's grade
    where it is trustworthy, the deterministic severity otherwise.

    **The division of authority, stated precisely.** `warehouse.news2.should_escalate`
    decides *whether* an alert exists -- that is unchanged, deterministic, and the
    learned model has no vote in it. This function decides only *how the existing
    alert is graded*, which in turn decides whether a human is paged and whether a
    care plan is generated. A model that can grade an alert cannot make one disappear:
    the alert is already in the store and already on `GET /alerts/active` before this
    is called, so the worst a mis-grade can do is route a real alert to the dashboard
    instead of to an inbox.

    **Every fallback direction is toward paging, not away from it.** The model
    downgrades an alert only when it is affirmatively confident and in-scope. It falls
    back to the deterministic severity when:

    * the assessment failed, was never attempted, or the patient is not assessable
      (a wearable subject has no chart to grade);
    * no promoted model is exported, so `severity` is None -- "not graded" is not
      "low", and collapsing the two would silently stop paging on a fresh checkout;
    * the score is **outside the model's validated scope**. `ml/models/serving.py`
      measures held-out AUPRC falling from 0.715 in the first 6 ICU hours to 0.074
      after, and unmeasurable past hour 24. Letting a number that weak downgrade a
      NEWS2 escalation would be trusting it exactly where it was shown not to work.

    So the only alert this ever silences is one where a promoted model, scoring inside
    the window where it was measured to discriminate well, says it is not severe.
    """
    if not assessment or not assessment.get("assessable"):
        return alert.severity
    severity = assessment.get("severity")
    if severity is None:
        return alert.severity
    if assessment.get("ml_in_validated_scope") is not True:
        return alert.severity
    return str(severity)


def _assess(alert: Alert, hour: int | None = None) -> dict | None:
    """Ask agent-orchestrator what is going on with this patient and what to do.

    **Why calling the agent graph from here is not the thing escalation.py refuses to
    do.** `services/stream-processor/escalation.py` deliberately does not invoke the
    agent, because doing so per streamed observation would run an LLM graph per vital
    sign. This call sits behind the store's 4-hourly dedup (R6, E16): it runs once per
    genuinely new alert, not once per observation that contributed to it. The same
    dedup that stops notification-gateway re-paging a recurring escalation stops this
    re-running the graph -- which is exactly why this call belongs here and nowhere
    upstream.

    Returns the assessment, an error dict, or None if never attempted. Never raises.
    """
    try:
        resp = get_agent_client().post(
            "/assess", json={"patient_ref": alert.patient_ref, "hour": hour}
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _notify_dashboard(alert: Alert, assessment: dict | None = None) -> dict | None:
    """Push to notification-gateway on a genuinely new alert or a fresh
    escalation -- never on a routine dedup repeat-count bump, which would
    otherwise re-broadcast (and, since notification-gateway also pages a
    high-severity alert by SMS, re-page) the same alert every time it recurs
    within its 4h bucket (PROJECT_PLAN.md section 12's dashboard needs a real
    live-update path, not just polling).

    **This is the one place in the whole pipeline that calls
    notification-gateway on a new alert.** It used to also happen a second
    time, independently, in `stream-processor`'s `EscalationLoop` -- every
    alert the streaming path raised was silently notifying twice. Harmless
    for a WebSocket toast; not harmless once notification-gateway also sends
    a real SMS on a high-severity notification, so this being the *only*
    caller is now load-bearing, not just tidy (see `escalation.py`'s module
    docstring for the fuller account).

    Returns notification-gateway's parsed response (channels/pushed/sms) --
    or an error dict, never raises -- so a caller of `POST /alerts` can see
    what actually happened rather than just that a request was attempted.
    Connection failures are reported, not swallowed: a clinician still sees
    the alert via `GET /alerts/active` on the next poll either way, so
    notification-gateway being briefly unreachable must not fail alert
    *raising* itself, but it must not go silently unrecorded either.
    """
    try:
        resp = get_notify_client().post(
            "/notify",
            json={
                "patient_ref": alert.patient_ref,
                "severity": alert.severity,
                "message": alert.message,
                # Only a usable assessment is forwarded. An error dict from a failed
                # /assess is kept in this service's own response for debugging but is
                # deliberately not sent onward: an escalation email is not the place
                # to render an httpx exception at a clinician.
                "assessment": (assessment if assessment and assessment.get("assessable") else None),
            },
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.post("/alerts/{alert_id}/suppress")
def suppress(alert_id: int) -> dict:
    try:
        return vars(get_store().suppress(alert_id))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/alerts/{alert_id}/escalate")
def escalate(alert_id: int) -> dict:
    try:
        return vars(get_store().escalate(alert_id))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


class AckRequest(BaseModel):
    acknowledged_by: str


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge(alert_id: int, req: AckRequest) -> dict:
    try:
        return vars(get_store().acknowledge(alert_id, req.acknowledged_by))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/alerts")
def active_for_patient(patient_ref: str) -> list[dict]:
    return [vars(a) for a in get_store().active_for_patient(patient_ref)]


@app.get("/alerts/active")
def all_active() -> list[dict]:
    """Ward-wide inbox for the Phase 6 dashboard -- every patient's active/
    escalated alerts, not just one (see AlertStore.all_active's docstring)."""
    return [vars(a) for a in get_store().all_active()]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)
