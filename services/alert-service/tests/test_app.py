import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from store import AlertStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from services.common.testing import load_service_app  # noqa: E402

_module = load_service_app("alert-service", REPO_ROOT)
app, set_store, set_notify_client = _module.app, _module.set_store, _module.set_notify_client
notification_gateway_module = load_service_app("notification-gateway", REPO_ROOT)

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_store(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    set_store(store)
    # A real notification-gateway app, via Starlette's TestClient acting as a
    # genuine sync httpx.Client subclass (no live socket needed) -- so every
    # test exercises the real _notify_dashboard() call path instead of
    # silently swallowing it against an unreachable URL.
    set_notify_client(TestClient(notification_gateway_module.app, base_url="http://notify"))
    yield store
    store.close()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_raise_and_fetch_active():
    resp = client.post(
        "/alerts",
        json={
            "patient_ref": "ICUStay/1",
            "alert_type": "news2_high",
            "severity": "high",
            "message": "NEWS2=9",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["was_new"] is True

    resp2 = client.get("/alerts", params={"patient_ref": "ICUStay/1"})
    assert resp2.status_code == 200
    assert len(resp2.json()) == 1


def test_suppress_removes_from_active():
    resp = client.post(
        "/alerts",
        json={
            "patient_ref": "ICUStay/2",
            "alert_type": "news2_high",
            "severity": "high",
            "message": "m",
        },
    )
    alert_id = resp.json()["alert"]["id"]
    client.post(f"/alerts/{alert_id}/suppress")
    resp2 = client.get("/alerts", params={"patient_ref": "ICUStay/2"})
    assert resp2.json() == []


def test_acknowledge_requires_body():
    resp = client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/3", "alert_type": "t", "severity": "low", "message": "m"},
    )
    alert_id = resp.json()["alert"]["id"]
    resp2 = client.post(
        f"/alerts/{alert_id}/acknowledge", json={"acknowledged_by": "clinician:jdoe"}
    )
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "acknowledged"


def test_suppress_unknown_alert_404():
    resp = client.post("/alerts/999999/suppress")
    assert resp.status_code == 404


def test_raising_a_new_alert_broadcasts_to_a_real_connected_dashboard():
    """End-to-end: a real notification-gateway app, a real connected
    WebSocket client, and a real POST /alerts -- confirms the socket actually
    receives the broadcast, not just that some function was called."""
    gateway_client = TestClient(notification_gateway_module.app)
    with gateway_client.websocket_connect("/ws/dashboard") as ws:
        client.post(
            "/alerts",
            json={
                "patient_ref": "ICUStay/20",
                "alert_type": "news2_high",
                "severity": "high",
                "message": "NEWS2=9",
            },
        )
        received = ws.receive_json()
    assert received == {"patient_ref": "ICUStay/20", "severity": "high", "message": "NEWS2=9"}


def test_a_routine_dedup_repeat_does_not_rebroadcast():
    """Only a genuinely new alert (or a fresh escalation) should push to the
    dashboard -- a repeat within the same 4h dedup bucket must not re-fire a
    broadcast for something the clinician has already seen."""
    gateway_client = TestClient(notification_gateway_module.app)
    client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/21", "alert_type": "t", "severity": "low", "message": "1st"},
    )
    with gateway_client.websocket_connect("/ws/dashboard") as ws:
        client.post(
            "/alerts",
            json={
                "patient_ref": "ICUStay/21",
                "alert_type": "t",
                "severity": "low",
                "message": "2nd, same bucket",
            },
        )
        # Prove the socket is live (something else broadcasts) rather than
        # asserting an absence with no positive signal at all.
        client.post(
            "/alerts",
            json={
                "patient_ref": "ICUStay/22",
                "alert_type": "t",
                "severity": "low",
                "message": "m",
            },
        )
        received = ws.receive_json()
    assert received["patient_ref"] == "ICUStay/22"  # not the ICUStay/21 repeat


def test_alerts_active_spans_every_patient():
    client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/10", "alert_type": "a", "severity": "high", "message": "m"},
    )
    client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/11", "alert_type": "b", "severity": "low", "message": "m"},
    )
    resp = client.get("/alerts/active")
    assert resp.status_code == 200
    refs = {a["patient_ref"] for a in resp.json()}
    assert {"ICUStay/10", "ICUStay/11"}.issubset(refs)
