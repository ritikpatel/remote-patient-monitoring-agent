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

app = FastAPI(title="alert-service", version="0.1.0")
instrument_metrics(app)
instrument_tracing(app, "alert-service")
_store: AlertStore | None = None
_notify_client: httpx.Client | None = None


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "alert-service"}


class RaiseAlertRequest(BaseModel):
    patient_ref: str
    alert_type: str
    severity: str
    message: str


@app.post("/alerts")
def raise_alert(req: RaiseAlertRequest) -> dict:
    alert, was_new = get_store().raise_alert(
        req.patient_ref, req.alert_type, req.severity, req.message
    )
    notification = None
    if was_new or alert.status == "escalated":
        notification = _notify_dashboard(alert)
    return {"alert": vars(alert), "was_new": was_new, "notification": notification}


def _notify_dashboard(alert: Alert) -> dict | None:
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
