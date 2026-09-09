"""EscalationLoop's HTTP orchestration: score, then alert.

Previously exercised only live (RUNBOOK's demo sequence, VALIDATION_REPORT's
Appendix 2 run against a real broker) -- this pins the sequence and the
return shape with a fake transport, keyed by URL, so it runs with no infra.
Also covers ``run_now()``, the one-shot entry point ``event-studio`` uses,
and that this loop calls notification-gateway zero times itself: alert-service
owns that call now (see escalation.py's module docstring for the double-notify
bug that made this the correct design, not just a simplification).
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from escalation import EscalationLoop  # noqa: E402

RISK_URL = "http://risk-engine.test"
ALERT_URL = "http://alert-service.test"


def _loop(handler, **kwargs) -> EscalationLoop:
    return EscalationLoop(
        risk_engine_url=RISK_URL,
        alert_service_url=ALERT_URL,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def test_run_now_does_not_call_alert_service_when_not_escalated():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.url.path == "/score/live"
        return httpx.Response(200, json={"news2": 1, "escalation_recommended": False})

    loop = _loop(handler)
    result = loop.run_now("Patient/1", {"hr": 75})

    assert calls == ["/score/live"]
    assert result == {
        "scored": True,
        "escalated": False,
        "score": {"news2": 1, "escalation_recommended": False},
    }
    assert loop.scored == 1
    assert loop.escalated == 0


def test_run_now_scores_then_alerts_in_order_on_escalation():
    """The order that matters: risk-engine, then alert-service -- never any
    other order, never skipped. Only two calls: EscalationLoop does not also
    call notification-gateway itself (see module docstring)."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/score/live":
            return httpx.Response(
                200,
                json={
                    "news2": 9,
                    "escalation_recommended": True,
                    "escalation_reason": "single red parameter",
                },
            )
        if request.url.path == "/alerts":
            # alert-service's own real response shape: it already called
            # notification-gateway itself and embeds the result here.
            return httpx.Response(
                201,
                json={
                    "was_new": True,
                    "alert": {"id": "a1", "severity": "high", "message": "NEWS2 9"},
                    "notification": {
                        "channels": ["dashboard", "push"],
                        "pushed": True,
                        "sms": {"mode": "dry_run", "sent": False},
                    },
                },
            )
        raise AssertionError(f"unexpected call: {request.url}")

    loop = _loop(handler)
    result = loop.run_now("Patient/1", {"hr": 145, "spo2": 89})

    assert calls == ["/score/live", "/alerts"]  # exactly two -- no third /notify call
    assert result["scored"] is True
    assert result["escalated"] is True
    assert result["alert"]["was_new"] is True
    # Read back, not re-requested: alert-service's own embedded field.
    assert result["alert"]["notification"]["sms"]["mode"] == "dry_run"
    assert result["alert"]["notification"]["pushed"] is True
    assert loop.scored == 1
    assert loop.escalated == 1
    assert loop.alerts_raised == 1


def test_run_now_still_counts_alerts_raised_when_alert_service_dedupes():
    """was_new: False (the 4-hourly dedup, R6) means alert-service itself
    skipped notifying -- nothing here re-derives or re-requests it."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/score/live":
            return httpx.Response(
                200,
                json={"news2": 9, "escalation_recommended": True, "escalation_reason": "reason"},
            )
        if request.url.path == "/alerts":
            return httpx.Response(
                200, json={"was_new": False, "alert": {"id": "a1"}, "notification": None}
            )
        raise AssertionError(f"unexpected call: {request.url}")

    loop = _loop(handler)
    result = loop.run_now("Patient/1", {"hr": 145})

    assert calls == ["/score/live", "/alerts"]
    assert result["alert"]["was_new"] is False
    assert result["alert"]["notification"] is None
    assert loop.alerts_raised == 0


def test_run_now_has_no_stream_time_throttle():
    """The 15-stream-minute throttle exists for a continuous stream;
    run_now's caller already made one deliberate decision to submit."""
    hits = {"score": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/score/live":
            hits["score"] += 1
            return httpx.Response(200, json={"news2": 0, "escalation_recommended": False})
        raise AssertionError(f"unexpected call: {request.url}")

    loop = _loop(handler)
    loop.run_now("Patient/1", {"hr": 75})
    loop.run_now("Patient/1", {"hr": 75})  # immediately again -- on_observation's
    # `_due()` throttle would drop this; run_now must not.

    assert hits["score"] == 2


def test_run_now_reports_a_downstream_error_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    loop = _loop(handler)
    result = loop.run_now("Patient/1", {"hr": 75})

    assert result["scored"] is False
    assert "error" in result
    assert loop.errors == 1


@pytest.mark.parametrize("client_kw", [{}, {"client": None}])
def test_default_construction_still_makes_a_real_client(client_kw):
    """The new `client` field is additive -- every existing call site (real
    services, real deployments) that never passes it keeps working exactly as
    before this change."""
    loop = EscalationLoop(risk_engine_url=RISK_URL, alert_service_url=ALERT_URL, **client_kw)
    assert loop._client is not None
    loop.close()
