import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from push import NoopPushSender  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from services.common.testing import load_service_app  # noqa: E402

_module = load_service_app("notification-gateway", REPO_ROOT)
app, set_push_sender = _module.app, _module.set_push_sender

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_notify_broadcasts_over_a_real_websocket_connection():
    """Real end-to-end check: open a WebSocket, POST /notify, and confirm the
    connected client actually receives the broadcast message -- not just that
    manager.broadcast was called."""
    with client.websocket_connect("/ws/dashboard") as ws:
        resp = client.post(
            "/notify",
            json={
                "patient_ref": "ICUStay/1",
                "severity": "medium",
                "message": "NEWS2=6",
                "timestamp": "2110-01-01T12:00:00",
            },
        )
        assert resp.status_code == 200
        assert "dashboard" in resp.json()["channels"]

        received = ws.receive_json()
        assert received == {"patient_ref": "ICUStay/1", "severity": "medium", "message": "NEWS2=6"}


def test_notify_pushes_when_channel_includes_push():
    sender = NoopPushSender()
    set_push_sender(sender)
    resp = client.post(
        "/notify",
        json={
            "patient_ref": "ICUStay/1",
            "severity": "high",
            "message": "NEWS2=9",
            "device_token": "device-abc",
            "timestamp": "2110-01-01T12:00:00",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["pushed"] is True
    assert sender.sent == [("device-abc", "HIGH alert", "NEWS2=9")]


def test_notify_overnight_low_severity_still_pushes():
    sender = NoopPushSender()
    set_push_sender(sender)
    resp = client.post(
        "/notify",
        json={
            "patient_ref": "ICUStay/1",
            "severity": "low",
            "message": "minor change",
            "device_token": "device-abc",
            "timestamp": "2110-01-01T03:00:00",
        },
    )
    assert resp.json()["pushed"] is True


def test_notify_attempts_sms_only_for_high_severity(monkeypatch):
    """The new channel this project's own SMS sender added: 'high' triggers an
    attempt (dry-run by default), anything else does not attempt at all --
    distinct from route_notification()'s dashboard/push routing, which medium
    already reaches."""
    monkeypatch.delenv("SMS_MODE", raising=False)
    resp = client.post(
        "/notify",
        json={
            "patient_ref": "ICUStay/1",
            "severity": "medium",
            "message": "NEWS2=6",
            "timestamp": "2110-01-01T12:00:00",
        },
    )
    assert resp.json()["sms"] is None


def test_notify_sms_is_dry_run_by_default_for_a_high_severity_alert(monkeypatch):
    monkeypatch.delenv("SMS_MODE", raising=False)
    monkeypatch.setenv("CLINICIAN_PHONE", "+447700900123")
    resp = client.post(
        "/notify",
        json={
            "patient_ref": "ICUStay/1",
            "severity": "high",
            "message": "NEWS2=9: single red parameter",
            "timestamp": "2110-01-01T12:00:00",
        },
    )
    body = resp.json()
    assert body["sms"]["mode"] == "dry_run"
    assert body["sms"]["sent"] is False
    assert "SYNTHETIC DRILL" in body["sms"]["text"]
    assert "ICUStay/1" in body["sms"]["text"]


def test_notify_sends_a_real_sms_when_live_and_configured(monkeypatch):
    """Exercises the real request-building path via a mocked Twilio transport --
    this is the one place in the whole pipeline an alert can reach a phone, and
    it is reached by the same route a real EscalationLoop-raised alert takes."""
    import httpx

    sys.path.insert(0, str(REPO_ROOT))
    from services.common import sms as sms_module

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(201, json={"sid": "SM-fake"})

    monkeypatch.setenv("SMS_MODE", "live")
    monkeypatch.setenv("CLINICIAN_PHONE", "+447700900123")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC-test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550001111")

    real_send = sms_module.send
    fake_client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(sms_module, "send", lambda text, **kw: real_send(text, client=fake_client))
    # notification-gateway's app.py imported `sms` by reference at module load
    # time (`from services.common import sms`), so patching the module's own
    # `send` name -- not app.py's local binding -- is what both share.

    resp = client.post(
        "/notify",
        json={
            "patient_ref": "ICUStay/1",
            "severity": "high",
            "message": "NEWS2=9",
            "timestamp": "2110-01-01T12:00:00",
        },
    )
    body = resp.json()
    assert body["sms"]["sent"] is True
    assert body["sms"]["provider_sid"] == "SM-fake"
    # Form-encoded POST body -- "/" survives as either literal or %2F depending
    # on httpx's encoder, so check both rather than assume one.
    assert "ICUStay/1" in captured["body"] or "ICUStay%2F1" in captured["body"]


def test_dead_socket_does_not_break_broadcast_to_others():
    with (
        client.websocket_connect("/ws/dashboard") as ws1,
        client.websocket_connect("/ws/dashboard") as ws2,
    ):
        ws1.close()
        resp = client.post(
            "/notify",
            json={
                "patient_ref": "ICUStay/2",
                "severity": "low",
                "message": "m",
                "timestamp": "2110-01-01T12:00:00",
            },
        )
        assert resp.status_code == 200
        received = ws2.receive_json()
        assert received["patient_ref"] == "ICUStay/2"
