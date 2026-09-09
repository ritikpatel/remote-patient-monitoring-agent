"""notification-gateway: real WebSocket fan-out to the dashboard, real routing
logic, a pluggable push sender (NoopPushSender by default -- see push.py), and
the guarded SMS sender (services/common/sms.py) for a high-severity alert.

SMS is dispatched from here, not from wherever an alert originated, because
this is the one place every alert-raising path already converges: the
deterministic streaming path (stream-processor's EscalationLoop) and
event-studio's synchronous demo path (EscalationLoop.run_now) both call
alert-service, and alert-service's response is what triggers *this* /notify
call. Wiring SMS in here means "SMS was sent" always means "a real alert was
deemed to be triggered", never a second, disconnected decision -- see
services/common/sms.py's module docstring for the fuller reasoning and the
four guards that keep this safe by default (dry-run unless SMS_MODE=live)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from push import NoopPushSender, PushSender  # noqa: E402
from routing import CHANNEL_DASHBOARD, CHANNEL_PUSH, route_notification  # noqa: E402
from services.common import sms  # noqa: E402
from services.common.observability import instrument_metrics, instrument_tracing  # noqa: E402

# A high-severity notification is this project's definition of "page-worthy" --
# the same bar EscalationLoop already applies before an alert is even raised
# (every alert it raises is severity="high"; see escalation.py's ALERT_TYPE
# constant). Deliberately independent of route_notification()'s overnight-
# aware dashboard/push routing (PROJECT_PLAN.md section 10's three channels):
# SMS is this project's own addition on top of that plan, not an
# implementation of it, so it is decided here rather than folded into that
# function's tested behaviour.
SMS_WORTHY_SEVERITY = "high"


def send_escalation_sms(patient_ref: str, severity: str, message: str) -> dict:
    """The concrete "page a human" step for a high-severity alert, using the
    one guarded sender every paging path in this project shares. Safe to
    leave wired in every environment: SMS_MODE defaults to dry_run, so this
    composes and returns without ever dialling out unless an operator has
    explicitly opted in -- see services/common/sms.py's four guards."""
    text = sms.compose(patient_ref, f"{severity.upper()} alert", message)
    return sms.send(text).as_dict()


app = FastAPI(title="notification-gateway", version="0.1.0")
instrument_metrics(app)
instrument_tracing(app, "notification-gateway")
_push_sender: PushSender = NoopPushSender()


def set_push_sender(sender: PushSender) -> None:
    global _push_sender
    _push_sender = sender


class ConnectionManager:
    """Real WebSocket fan-out -- every connected dashboard client gets every
    broadcast message. No auth/room-scoping here; clinician-api's SMART-scope
    check gates who is allowed to open the connection in Phase 8's deployment.
    """

    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 -- a dead socket must not break the others
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "notification-gateway",
        "active_connections": len(manager.active),
    }


@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            # dashboard clients don't send anything meaningful; this just keeps the socket open
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


class NotifyRequest(BaseModel):
    patient_ref: str
    severity: str
    message: str
    device_token: str | None = None
    timestamp: datetime | None = None


@app.post("/notify")
async def notify(req: NotifyRequest) -> dict:
    ts = req.timestamp or datetime.now(UTC)
    channels = route_notification(req.severity, ts)

    if CHANNEL_DASHBOARD in channels:
        await manager.broadcast(
            {"patient_ref": req.patient_ref, "severity": req.severity, "message": req.message}
        )
    pushed = False
    if CHANNEL_PUSH in channels and req.device_token:
        pushed = _push_sender.send(req.device_token, f"{req.severity.upper()} alert", req.message)

    sms_result = None
    if req.severity == SMS_WORTHY_SEVERITY:
        sms_result = send_escalation_sms(req.patient_ref, req.severity, req.message)

    return {"channels": channels, "pushed": pushed, "sms": sms_result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)
