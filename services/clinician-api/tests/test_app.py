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
agent_orchestrator_module = load_service_app("agent-orchestrator", REPO_ROOT)
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

    from starlette.testclient import TestClient as SyncASGIClient

    agent_orchestrator_module.set_deps(
        agent_orchestrator_module.Dependencies(
            db_path=risk_engine_module.DEFAULT_DB_PATH,
            risk_engine_client=SyncASGIClient(
                risk_engine_module.app, base_url="http://risk-engine"
            ),
            rag_client=SyncASGIClient(
                load_service_app("rag-service", REPO_ROOT).app, base_url="http://rag-service"
            ),
            audit_log=AuditLog(tmp_path / "agent_audit.db"),
            llm=None,
        )
    )
    configure_clients(
        risk_engine=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=risk_engine_module.app), base_url="http://risk-engine"
        ),
        alert_service=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=alert_service_module.app),
            base_url="http://alert-service",
        ),
        agent_orchestrator=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=agent_orchestrator_module.app),
            base_url="http://agent-orchestrator",
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


def test_list_patients_proxies_real_risk_engine_ward_view():
    resp = client.get("/patients", headers=_auth_headers(["patient/RiskAssessment.read"]))
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) > 0
    assert "news2" in body[0] and "patient_ref" in body[0]


def test_get_trace_proxies_real_news2_series():
    stay_id, _ = _known_stay_hour()
    resp = client.get(
        f"/patients/{stay_id}/trace", headers=_auth_headers(["patient/Observation.read"])
    )
    assert resp.status_code == 200
    assert len(resp.json()) > 0


def test_get_trace_404_for_unknown_stay():
    resp = client.get(
        "/patients/999999999/trace", headers=_auth_headers(["patient/Observation.read"])
    )
    assert resp.status_code == 404


def test_get_risk_ml_reflects_risk_engines_real_availability():
    from ml.models import serving

    stay_id, hour = _known_stay_hour()
    resp = client.post(
        f"/risk/{stay_id}/{hour}/ml",
        params={"patient_ref": f"ICUStay/{stay_id}"},
        headers=_auth_headers(["patient/RiskAssessment.read"]),
    )
    if serving.promoted_model_available():
        assert resp.status_code == 200
        assert 0.0 <= resp.json()["probability"] <= 1.0
    else:
        assert resp.status_code == 503


def test_get_agent_assessment_runs_the_real_graph():
    stay_id, hour = _known_stay_hour()
    resp = client.post(
        f"/patients/{stay_id}/{hour}/assessment",
        params={"patient_ref": f"ICUStay/{stay_id}"},
        headers=_auth_headers(["patient/RiskAssessment.read"]),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "escalate" in body
    assert "context_passages" in body
    assert len(body["audit_rows"]) == 6


def test_get_active_alerts_spans_every_patient(wired_dependencies):
    alert_client = TestClient(alert_service_module.app)
    alert_client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/5", "alert_type": "t", "severity": "high", "message": "m"},
    )
    resp = client.get("/alerts/active", headers=_auth_headers(["patient/Communication.read"]))
    assert resp.status_code == 200
    assert any(a["patient_ref"] == "ICUStay/5" for a in resp.json())


def test_escalate_and_suppress_alert_proxy_and_audit(wired_dependencies):
    alert_client = TestClient(alert_service_module.app)
    raise_resp = alert_client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/6", "alert_type": "t", "severity": "low", "message": "m"},
    )
    alert_id = raise_resp.json()["alert"]["id"]

    resp = client.post(
        f"/alerts/{alert_id}/escalate",
        params={"patient_ref": "ICUStay/6"},
        headers=_auth_headers(["patient/Communication.write"]),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "escalated"

    resp2 = client.post(
        f"/alerts/{alert_id}/suppress",
        params={"patient_ref": "ICUStay/6"},
        headers=_auth_headers(["patient/Communication.write"]),
    )
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "suppressed"

    rows = wired_dependencies.all_rows()
    assert any(r.action == "alert_escalate" for r in rows)
    assert any(r.action == "alert_suppress" for r in rows)
