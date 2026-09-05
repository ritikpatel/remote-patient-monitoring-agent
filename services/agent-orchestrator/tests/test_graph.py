import sys
from pathlib import Path

import duckdb
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from services.common.testing import load_service_app  # noqa: E402

risk_engine_module = load_service_app("risk-engine", REPO_ROOT)
rag_service_module = load_service_app("rag-service", REPO_ROOT)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph import build_graph  # noqa: E402
from nodes import Dependencies  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from services.common.audit import AuditLog  # noqa: E402

DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
pytestmark = pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="warehouse not built")


def _sync_asgi_client(app, base_url: str):
    from starlette.testclient import TestClient

    return TestClient(app, base_url=base_url)


@pytest.fixture
def deps(tmp_path) -> Dependencies:
    return Dependencies(
        db_path=DEFAULT_DB_PATH,
        risk_engine_client=_sync_asgi_client(risk_engine_module.app, "http://risk-engine"),
        rag_client=_sync_asgi_client(rag_service_module.app, "http://rag-service"),
        audit_log=AuditLog(tmp_path / "audit.db"),
        llm=None,  # no network calls in this test -- see test_nodes.py for LLM-specific behaviour
    )


@pytest.fixture
def known_high_tier_stay():
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    row = conn.execute(
        "SELECT stay_id, hour FROM capstone.news2 WHERE tier_icu = 'high' LIMIT 1"
    ).fetchone()
    conn.close()
    return row


def test_full_graph_runs_all_six_nodes_in_order(deps, known_high_tier_stay):
    stay_id, hour = known_high_tier_stay
    graph = build_graph(deps)
    result = graph.invoke(
        {"stay_id": stay_id, "hour": hour, "patient_ref": f"ICUStay/{stay_id}", "audit_rows": []}
    )

    assert "vitals" in result
    assert "abnormal_labs" in result
    assert "risk_score" in result
    assert "context_passages" in result
    assert "escalate" in result
    assert "summary" in result

    # six nodes, six audit rows, one per node -- constraint 3, proven not asserted.
    assert len(result["audit_rows"]) == 6

    rows = deps.audit_log.all_rows()
    node_names = [r.actor.split(":")[1] for r in rows]
    assert node_names == [
        "VitalsMonitor",
        "LabInterpreter",
        "RiskScorer",
        "ContextRetriever",
        "EscalationDecider",
        "Summarizer",
    ]


def test_audit_chain_is_intact_after_a_full_run(deps, known_high_tier_stay):
    stay_id, hour = known_high_tier_stay
    graph = build_graph(deps)
    graph.invoke(
        {"stay_id": stay_id, "hour": hour, "patient_ref": f"ICUStay/{stay_id}", "audit_rows": []}
    )

    ok, bad_seq = deps.audit_log.verify_chain()
    assert ok is True
    assert bad_seq is None


def test_escalation_reaches_the_final_state_for_a_high_tier_stay(deps, known_high_tier_stay):
    stay_id, hour = known_high_tier_stay
    graph = build_graph(deps)
    result = graph.invoke(
        {"stay_id": stay_id, "hour": hour, "patient_ref": f"ICUStay/{stay_id}", "audit_rows": []}
    )
    assert result["escalate"] is True
    assert "high" in result["escalation_reason"]
