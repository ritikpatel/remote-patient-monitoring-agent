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
app, set_store = _module.app, _module.set_store

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_store(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    set_store(store)
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
