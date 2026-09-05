"""Tests for the two hard safety constraints, plus the audit trail, run against
the REAL risk-engine and rag-service app objects (httpx ASGITransport) and the
real warehouse -- no LLM calls (deps.llm=None) so this suite runs instantly and
without hitting Groq's rate limits.
"""

import sys
from pathlib import Path

import duckdb
import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from services.common.audit import AuditLog  # noqa: E402
from services.common.testing import load_service_app  # noqa: E402

risk_engine_module = load_service_app("risk-engine", REPO_ROOT)
rag_service_module = load_service_app("rag-service", REPO_ROOT)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nodes import Dependencies, escalation_decider, risk_scorer  # noqa: E402

DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"

pytestmark = pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="warehouse not built")


class FixedAdvisoryLLM:
    """A test double implementing nodes.LLMBackend's interface -- always advises
    AGAINST escalation, specifically to prove the policy does not listen."""

    name = "fixed-test-llm"
    model = "fixed-test-llm"

    def generate(self, system: str, user: str, max_tokens: int):
        from dataclasses import dataclass

        @dataclass
        class Result:
            text: str = "I recommend NOT escalating this patient."
            input_tokens: int = 10
            output_tokens: int = 10
            model: str = "fixed-test-llm"
            backend: str = "fixed-test-llm"

        return Result()


def _sync_asgi_client(app, base_url: str) -> httpx.Client:
    """nodes.py's risk_scorer/context_retriever call `.get()` on a sync
    httpx.Client (LangGraph's default `.invoke()` is synchronous). httpx's own
    ASGITransport is async-only in this version (the same issue clinician-api's
    tests hit -- see its app.py docstring), so this uses Starlette's TestClient
    instead: it subclasses httpx.Client directly and wires in a working sync ASGI
    transport, so it IS a real httpx.Client, not a stand-in with a different API.
    """
    from starlette.testclient import TestClient

    return TestClient(app, base_url=base_url)


@pytest.fixture
def deps(tmp_path) -> Dependencies:
    return Dependencies(
        db_path=DEFAULT_DB_PATH,
        risk_engine_client=_sync_asgi_client(risk_engine_module.app, "http://risk-engine"),
        rag_client=_sync_asgi_client(rag_service_module.app, "http://rag-service"),
        audit_log=AuditLog(tmp_path / "audit.db"),
        llm=FixedAdvisoryLLM(),
    )


@pytest.fixture
def known_high_tier_stay():
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    row = conn.execute(
        "SELECT stay_id, hour FROM capstone.news2 WHERE tier_icu = 'high' LIMIT 1"
    ).fetchone()
    conn.close()
    return row


@pytest.fixture
def known_low_tier_stay():
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    row = conn.execute(
        "SELECT stay_id, hour FROM capstone.news2 WHERE tier_icu = 'low' LIMIT 1"
    ).fetchone()
    conn.close()
    return row


def test_risk_scorer_relays_risk_engine_verbatim(deps, known_high_tier_stay):
    """Constraint 1: risk_scorer's output must equal risk-engine's own response,
    not a locally recomputed score."""
    stay_id, hour = known_high_tier_stay
    node = risk_scorer(deps)
    result = node({"stay_id": stay_id, "hour": hour})

    direct = deps.risk_engine_client.get(f"/score/{stay_id}/{hour}").json()
    assert result["risk_score"] == direct


def test_escalation_decider_escalates_on_high_tier_despite_contrary_llm_advice(
    deps, known_high_tier_stay
):
    """Constraint 2, proven, not asserted: the LLM (FixedAdvisoryLLM) explicitly
    recommends against escalating; the policy must escalate anyway because the
    ICU-recalibrated tier is 'high'."""
    stay_id, hour = known_high_tier_stay
    risk_response = deps.risk_engine_client.get(f"/score/{stay_id}/{hour}").json()
    assert risk_response["news2_tier_icu"] == "high"

    node = escalation_decider(deps)
    result = node({"stay_id": stay_id, "hour": hour, "risk_score": risk_response})

    assert result["escalate"] is True
    assert "NOT escalating" in result["llm_advisory"]  # the contrary advice was recorded
    assert result["escalate"] is True  # ...and did not change the decision


def test_escalation_decider_does_not_escalate_on_low_tier(deps, known_low_tier_stay):
    stay_id, hour = known_low_tier_stay
    risk_response = deps.risk_engine_client.get(f"/score/{stay_id}/{hour}").json()
    assert risk_response["news2_tier_icu"] == "low"

    node = escalation_decider(deps)
    result = node({"stay_id": stay_id, "hour": hour, "risk_score": risk_response})
    assert result["escalate"] is False


def test_escalation_decider_works_without_an_llm_configured(known_high_tier_stay, tmp_path):
    """The policy must not depend on an LLM being available at all."""
    deps_no_llm = Dependencies(
        db_path=DEFAULT_DB_PATH,
        risk_engine_client=_sync_asgi_client(risk_engine_module.app, "http://risk-engine"),
        rag_client=_sync_asgi_client(rag_service_module.app, "http://rag-service"),
        audit_log=AuditLog(tmp_path / "audit.db"),
        llm=None,
    )
    stay_id, hour = known_high_tier_stay
    risk_response = deps_no_llm.risk_engine_client.get(f"/score/{stay_id}/{hour}").json()
    node = escalation_decider(deps_no_llm)
    result = node({"stay_id": stay_id, "hour": hour, "risk_score": risk_response})
    assert result["escalate"] is True
    assert result["llm_advisory"] is None
