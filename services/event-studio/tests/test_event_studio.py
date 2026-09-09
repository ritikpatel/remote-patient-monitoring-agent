"""event-studio: the generator must agree with the scorer the pipeline uses,
and sending a real event must drive the real pipeline synchronously.

SMS guard tests moved to services/common/tests/test_sms.py when sms.py itself
moved there (it now has two real callers, not one).
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from services.common.testing import load_module, load_service_app  # noqa: E402

APP_DIR = REPO_ROOT / "services" / "event-studio"
WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"
sys.path.insert(0, str(APP_DIR))
generator = load_module(APP_DIR / "generator.py", "event_studio_generator")
app_mod = load_service_app("event-studio", REPO_ROOT)

# The ICU tier cut-points are read from the warehouse rather than mirrored, so
# the endpoint tests need it built; the pure-generator tests do not.
needs_warehouse = pytest.mark.skipif(not WAREHOUSE_DB.exists(), reason="warehouse not built")
client = TestClient(app_mod.app)


@pytest.fixture(autouse=True)
def _reset_injected_clients():
    """Every test starts from real-network-client defaults (None) and leaves
    them that way for the next one -- a test that injects a fake client must
    still see it cleared afterwards, or it would leak into an unrelated test
    that expected a real (or differently-mocked) one."""
    yield
    app_mod.set_pipeline_client(None)
    app_mod.set_rag_client(None)
    app_mod.set_gateway_client(None)


@pytest.mark.parametrize("severity", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_generated_values_score_what_the_generator_claims(severity):
    """The whole point of inverting news2.py's thresholds is that a requested
    severity and the score the pipeline computes cannot drift apart."""
    ev = generator.generate(severity, seed=7)
    for channel, value in ev.values.items():
        assert generator.SCORERS[channel](value) == ev.subscores[channel]
    assert ev.news2 == sum(ev.subscores.values())


def test_severity_is_monotonic_in_expectation():
    """Higher severity must not produce a systematically lower NEWS2."""
    lo = [generator.generate(0.15, seed=s).news2 for s in range(25)]
    hi = [generator.generate(0.85, seed=s).news2 for s in range(25)]
    assert sum(hi) / len(hi) > sum(lo) / len(lo)


@needs_warehouse
def test_a_healthy_event_does_not_escalate_and_a_critical_one_does():
    healthy = client.post("/event", json={"severity": 0.0, "seed": 1}).json()
    assert healthy["news2"] == 0
    assert healthy["would_escalate"] is False

    critical = client.post("/event", json={"severity": 1.0, "seed": 1}).json()
    assert critical["would_escalate"] is True


@needs_warehouse
def test_nothing_is_sent_unless_send_is_true():
    """A generate must never reach the gateway -- the UI previews before sending."""
    d = client.post("/event", json={"severity": 1.0, "seed": 2}).json()
    assert d["sent"] is False
    assert "gateway_status" not in d
    assert d["pipeline"] is None


@needs_warehouse
def test_sms_preview_is_text_only_and_never_a_real_send_attempt(monkeypatch):
    """The safety fix: a 'Generate' (send=False) must be incapable of paging a
    phone even if SMS_MODE=live happens to be set in the environment -- it was
    not, before this test existed, because the old code always called
    sms.send() regardless of `req.send`. sms_preview is composed text only."""
    monkeypatch.setenv("SMS_MODE", "live")
    monkeypatch.setenv("CLINICIAN_PHONE", "+447700900123")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC-should-never-be-used")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "should-never-be-used")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550001111")

    d = client.post("/event", json={"severity": 1.0, "seed": 3, "send": False}).json()
    assert d["would_escalate"] is True
    assert isinstance(d["sms_preview"], str)
    assert "SYNTHETIC DRILL" in d["sms_preview"]
    assert d["pipeline"] is None  # the real pipeline was never touched


def test_every_emitted_observation_is_watermarked_synthetic():
    ev = generator.generate(0.7, seed=4)
    obs = app_mod._observations(ev, "Patient/1")
    assert obs, "expected observations"
    for o in obs:
        assert "synthetic" in [f.value for f in o.quality_flags]


@needs_warehouse
def test_tiers_come_from_the_warehouse_not_a_local_copy():
    """A mirrored threshold silently goes stale when the warehouse is rebuilt."""
    th = app_mod.thresholds()
    health = client.get("/health").json()
    assert health["icu_medium"] == th.icu_medium
    assert health["icu_high"] == th.icu_high


# --- Realtime pipeline orchestration ----------------------------------------
# `send: true` now drives risk-engine, alert-service (which itself calls
# notification-gateway) and rag-service synchronously and reports what
# actually happened -- these tests inject fake transports rather than needing
# live infra, mirroring the pattern services/stream-processor/tests/
# test_escalation.py already established for EscalationLoop itself.


def _gateway_ok_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={"accepted": len(request.content)})

    return httpx.Client(
        base_url="http://ingest-gateway.test", transport=httpx.MockTransport(handler)
    )


@needs_warehouse
def test_send_with_low_severity_scores_but_does_not_touch_alerting(monkeypatch):
    calls = []

    def pipeline_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.url.path == "/score/live"
        return httpx.Response(200, json={"news2": 0, "escalation_recommended": False})

    app_mod.set_gateway_client(_gateway_ok_client())
    app_mod.set_pipeline_client(httpx.Client(transport=httpx.MockTransport(pipeline_handler)))

    d = client.post("/event", json={"severity": 0.0, "seed": 5, "send": True}).json()

    assert d["sent"] is True
    assert calls == ["/score/live"]
    assert d["pipeline"]["scored"] is True
    assert d["pipeline"]["escalated"] is False


@needs_warehouse
def test_send_with_high_severity_drives_score_alert_and_rag_in_order(monkeypatch):
    """The order the user asked for: risk-engine, alert-service (which pages
    notification-gateway internally), and -- only because this escalated --
    rag-service for context passages. Same real EscalationLoop.run_now code
    the streaming path uses, called synchronously instead of via Kafka."""
    calls = []

    def pipeline_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/score/live":
            return httpx.Response(
                200,
                json={
                    "news2": 13,
                    "escalation_recommended": True,
                    "escalation_reason": "aggregate tier 'high'",
                },
            )
        if request.url.path == "/alerts":
            return httpx.Response(
                201,
                json={
                    "was_new": True,
                    "alert": {"id": "a1", "severity": "high"},
                    "notification": {
                        "channels": ["dashboard", "push"],
                        "pushed": True,
                        "sms": {
                            "mode": "dry_run",
                            "sent": False,
                            "to": "<CLINICIAN_PHONE unset>",
                            "text": "[SYNTHETIC DRILL] ...",
                        },
                    },
                },
            )
        raise AssertionError(f"unexpected pipeline call: {request.url}")

    def rag_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.url.path == "/search"
        return httpx.Response(200, json=[{"text": "guideline passage", "fact_id": "F001"}])

    app_mod.set_gateway_client(_gateway_ok_client())
    app_mod.set_pipeline_client(httpx.Client(transport=httpx.MockTransport(pipeline_handler)))
    app_mod.set_rag_client(
        httpx.Client(base_url="http://rag-service.test", transport=httpx.MockTransport(rag_handler))
    )

    d = client.post("/event", json={"severity": 1.0, "seed": 6, "send": True}).json()

    assert calls == ["/score/live", "/alerts", "/search"]
    p = d["pipeline"]
    assert p["scored"] is True and p["escalated"] is True
    assert p["alert"]["was_new"] is True
    assert p["alert"]["notification"]["sms"]["mode"] == "dry_run"
    assert p["context_passages"] == [{"text": "guideline passage", "fact_id": "F001"}]


@needs_warehouse
def test_send_proceeds_to_the_pipeline_even_if_the_gateway_post_fails():
    """The gateway POST and the synchronous pipeline are independent -- a
    down ingest-gateway (the async Kafka path) must not silently suppress the
    real-time result the user is watching for."""

    def failing_gateway(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    def pipeline_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"news2": 0, "escalation_recommended": False})

    app_mod.set_gateway_client(httpx.Client(transport=httpx.MockTransport(failing_gateway)))
    app_mod.set_pipeline_client(httpx.Client(transport=httpx.MockTransport(pipeline_handler)))

    d = client.post("/event", json={"severity": 0.0, "seed": 7, "send": True}).json()

    assert d["sent"] is False
    assert "gateway_error" in d
    assert d["pipeline"]["scored"] is True  # still ran
