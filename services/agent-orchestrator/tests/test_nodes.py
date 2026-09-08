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
    """A genuinely non-escalating hour: low aggregate tier AND no red non-GCS
    parameter. Since finding F1 'low tier' alone is not enough -- a low-tier hour can
    still escalate on the single-parameter limb, which is the entire point of the fix.
    """
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    row = conn.execute(
        "SELECT stay_id, hour FROM capstone.news2 "
        "WHERE tier_icu = 'low' AND max_component_nongcs < 3 LIMIT 1"
    ).fetchone()
    conn.close()
    return row


@pytest.fixture
def known_red_parameter_stay():
    """A patient-hour that does NOT reach the high aggregate tier but has a single
    non-GCS parameter scoring 3 -- exactly the case the pre-F1 policy missed."""
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    row = conn.execute(
        "SELECT stay_id, hour FROM capstone.news2 "
        "WHERE tier_icu != 'high' AND max_component_nongcs >= 3 LIMIT 1"
    ).fetchone()
    conn.close()
    return row


@pytest.fixture
def known_red_gcs_only_stay():
    """A patient-hour whose ONLY red parameter is GCS, below the high tier. Must not
    escalate -- GCS 3 is routine under sedation (see warehouse/news2.py)."""
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    row = conn.execute(
        "SELECT stay_id, hour FROM capstone.news2 "
        "WHERE tier_icu != 'high' AND max_component >= 3 AND max_component_nongcs < 3 "
        "AND NOT gcs_drop LIMIT 1"
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


def test_escalation_decider_escalates_on_a_single_red_parameter_below_high_tier(
    deps, known_red_parameter_stay
):
    """Finding F1, the regression guard. NEWS2 (RCP 2017) escalates on a score of 3 in
    any single parameter, independently of the aggregate. The pre-fix policy tiered on
    the aggregate alone and returned escalate=False for hours exactly like this one --
    including a real patient with GCS 3 and SOFA 12 who then died.
    """
    stay_id, hour = known_red_parameter_stay
    risk_response = deps.risk_engine_client.get(f"/score/{stay_id}/{hour}").json()
    assert risk_response["news2_tier_icu"] != "high"  # aggregate limb does NOT fire
    assert risk_response["max_component_nongcs"] >= 3  # single-parameter limb does

    node = escalation_decider(deps)
    result = node({"stay_id": stay_id, "hour": hour, "risk_score": risk_response})

    assert result["escalate"] is True
    assert "single-parameter red flag" in result["escalation_reason"]


def test_escalation_decider_does_not_escalate_on_a_stable_red_gcs(deps, known_red_gcs_only_stay):
    """The deliberate exclusion, also pinned: a red GCS with no other red parameter
    must not escalate, and the reason must say so rather than staying silent."""
    if known_red_gcs_only_stay is None:
        pytest.skip("no GCS-only red hour in this warehouse")
    stay_id, hour = known_red_gcs_only_stay
    risk_response = deps.risk_engine_client.get(f"/score/{stay_id}/{hour}").json()

    node = escalation_decider(deps)
    result = node({"stay_id": stay_id, "hour": hour, "risk_score": risk_response})

    assert result["escalate"] is False
    assert "does not escalate on level alone" in result["escalation_reason"]


@pytest.fixture
def known_gcs_drop_stay():
    """A patient-hour escalating ONLY on the GCS-drop limb -- the class of case the
    motivating patient (34617352, hour 35) belongs to."""
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    row = conn.execute(
        "SELECT stay_id, hour FROM capstone.news2 "
        "WHERE tier_icu != 'high' AND max_component_nongcs < 3 AND gcs_drop LIMIT 1"
    ).fetchone()
    conn.close()
    return row


def test_escalation_decider_escalates_on_a_falling_gcs_off_sedation(deps, known_gcs_drop_stay):
    """Finding F1's third limb. Neither the aggregate tier nor any non-GCS parameter
    fires here; what escalates is the GCS trajectory. Without this limb the patient who
    exposed the bug still would not have been escalated."""
    if known_gcs_drop_stay is None:
        pytest.skip("no GCS-drop-only hour in this warehouse")
    stay_id, hour = known_gcs_drop_stay
    risk_response = deps.risk_engine_client.get(f"/score/{stay_id}/{hour}").json()
    assert risk_response["news2_tier_icu"] != "high"
    assert risk_response["max_component_nongcs"] < 3
    assert risk_response["gcs_drop"] is True

    node = escalation_decider(deps)
    result = node({"stay_id": stay_id, "hour": hour, "risk_score": risk_response})
    assert result["escalate"] is True
    assert "GCS fell" in result["escalation_reason"]


def test_the_motivating_case_now_escalates(deps):
    """Stay 34617352 hour 35 by name: GCS 7 -> 3 off sedation, aggregate tier 'medium'.
    The pre-F1 policy returned escalate=False for this patient, who then died."""
    risk_response = deps.risk_engine_client.get("/score/34617352/35").json()
    node = escalation_decider(deps)
    result = node({"stay_id": 34617352, "hour": 35, "risk_score": risk_response})
    assert result["escalate"] is True


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
