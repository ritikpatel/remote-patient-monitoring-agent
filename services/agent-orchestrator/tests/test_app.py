import sys
from pathlib import Path

import duckdb
import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nodes import Dependencies  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from services.common.audit import AuditLog  # noqa: E402
from services.common.testing import load_service_app  # noqa: E402

_module = load_service_app("agent-orchestrator", REPO_ROOT)
DEFAULT_DB_PATH, app, set_deps = _module.DEFAULT_DB_PATH, _module.app, _module.set_deps

pytestmark = pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="warehouse not built")

client = TestClient(app)


@pytest.fixture(autouse=True)
def wired_deps(tmp_path):
    # Point at real, empty httpx clients against nothing -- these tests only
    # exercise the HTTP surface (routing, request/response shape), not the graph's
    # own logic (already proven for real in test_nodes.py / test_graph.py against
    # the actual risk-engine/rag-service apps).
    deps = Dependencies(
        db_path=DEFAULT_DB_PATH,
        risk_engine_client=httpx.Client(base_url="http://risk-engine.invalid"),
        rag_client=httpx.Client(base_url="http://rag-service.invalid"),
        audit_log=AuditLog(tmp_path / "audit.db"),
        llm=None,
    )
    set_deps(deps)
    yield deps


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["llm_configured"] is False


def test_audit_verify_on_empty_log():
    resp = client.get("/audit/verify")
    assert resp.status_code == 200
    assert resp.json()["intact"] is True


def test_run_fails_cleanly_when_risk_engine_unreachable():
    """With no real risk-engine wired, /run should fail loudly (a connection
    error surfaced as a 5xx), not silently fabricate a score -- the same
    constraint that keeps the LLM out of the scoring path applies to failures in
    the deterministic path too: no data means no answer, not a guess.

    TestClient's default `raise_server_exceptions=True` re-raises an unhandled
    route exception into the test itself (useful for debugging, but not what a
    real deployed client sees) -- raise_server_exceptions=False here reproduces
    what a real client actually receives: a 500 response, not a stack trace.
    """
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    stay_id, hour = conn.execute("SELECT stay_id, hour FROM capstone.news2 LIMIT 1").fetchone()
    conn.close()
    non_raising_client = TestClient(app, raise_server_exceptions=False)
    resp = non_raising_client.post(
        "/run", json={"stay_id": stay_id, "hour": hour, "patient_ref": f"ICUStay/{stay_id}"}
    )
    assert resp.status_code >= 500
