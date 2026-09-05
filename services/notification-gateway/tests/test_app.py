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
