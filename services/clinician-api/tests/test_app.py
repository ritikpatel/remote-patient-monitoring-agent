"""clinician-api's own tests point it at the REAL risk-engine and alert-service
FastAPI app objects via httpx's ASGI transport -- a genuine in-process HTTP call
(status codes, JSON serialization, the lot), not a mock of the downstream response.
"""

import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.common.audit import AuditLog  # noqa: E402
from services.common.auth import issue_local_test_token  # noqa: E402
from services.common.testing import load_service_app  # noqa: E402

# Services live in hyphenated directories and every one's main file is named
# app.py -- load_service_app gives each a unique sys.modules name so risk-engine's
# `app`, alert-service's `app`, and clinician-api's own `app` never collide (a
# plain `import app` from each test file's own directory did collide, silently,
# the moment the whole repo's tests ran in one pytest process -- see
# services/common/testing.py's docstring).
risk_engine_module = load_service_app("risk-engine", REPO_ROOT)
alert_service_module = load_service_app("alert-service", REPO_ROOT)
clinician_api_module = load_service_app("clinician-api", REPO_ROOT)
app = clinician_api_module.app
configure_clients = clinician_api_module.configure_clients
set_audit_log = clinician_api_module.set_audit_log

pytestmark = pytest.mark.skipif(
    not risk_engine_module.DEFAULT_DB_PATH.exists(), reason="warehouse not built"
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def wired_dependencies(tmp_path):
    alert_service_module.set_store(alert_service_module.AlertStore(tmp_path / "alerts.db"))
    configure_clients(
        risk_engine=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=risk_engine_module.app), base_url="http://risk-engine"
        ),
        alert_service=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=alert_service_module.app),
            base_url="http://alert-service",
        ),
    )
    audit_log = AuditLog(tmp_path / "audit.db")
    set_audit_log(audit_log)
    yield audit_log
    audit_log.close()


def _auth_headers(scopes: list[str]) -> dict:
    token = issue_local_test_token("clinician:jdoe", scopes)
    return {"Authorization": f"Bearer {token}"}


def _known_stay_hour():
    import duckdb

    conn = duckdb.connect(str(risk_engine_module.DEFAULT_DB_PATH), read_only=True)
    row = conn.execute("SELECT stay_id, hour FROM capstone.news2 LIMIT 1").fetchone()
    conn.close()
    return row


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_get_risk_requires_scope():
    stay_id, hour = _known_stay_hour()
    resp = client.get(
        f"/risk/{stay_id}/{hour}",
        params={"patient_ref": f"ICUStay/{stay_id}"},
        headers=_auth_headers([]),
    )
    assert resp.status_code == 403


def test_get_risk_proxies_real_risk_engine_response(wired_dependencies):
    stay_id, hour = _known_stay_hour()
    resp = client.get(
        f"/risk/{stay_id}/{hour}",
        params={"patient_ref": f"ICUStay/{stay_id}"},
        headers=_auth_headers(["patient/RiskAssessment.read"]),
    )
    assert resp.status_code == 200
    assert resp.json()["stay_id"] == stay_id

    rows = wired_dependencies.all_rows()
    assert any(r.action == "phi_read" for r in rows)


def test_acknowledge_alert_proxies_and_audits(wired_dependencies):
    # Raise a real alert directly against the real alert-service app's own
    # TestClient (sync, and independent of clinician-api's async client wiring).
    alert_client = TestClient(alert_service_module.app)
    raise_resp = alert_client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/1", "alert_type": "t", "severity": "high", "message": "m"},
    )
    alert_id = raise_resp.json()["alert"]["id"]

    resp = client.post(
        f"/alerts/{alert_id}/acknowledge",
        params={"patient_ref": "ICUStay/1"},
        headers=_auth_headers(["patient/Communication.write"]),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "acknowledged"
    assert resp.json()["acknowledged_by"] == "clinician:jdoe"

    rows = wired_dependencies.all_rows()
    assert any(r.action == "alert_ack" for r in rows)


def test_acknowledge_unknown_alert_404():
    resp = client.post(
        "/alerts/999999/acknowledge",
        params={"patient_ref": "ICUStay/1"},
        headers=_auth_headers(["patient/Communication.write"]),
    )
    assert resp.status_code == 404
