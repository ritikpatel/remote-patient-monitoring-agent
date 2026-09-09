"""services/common/sms.py's guards.

Moved here from services/event-studio/tests/test_event_studio.py when the
module itself moved (it now has two real callers, not one -- see sms.py's
module docstring). Every test below asserts a *refusal*. None of them sends
anything, and no test in this file is capable of contacting a provider: the
live-mode test injects a fake transport so a misconfigured environment
cannot page a person.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from services.common import sms  # noqa: E402


def test_dry_run_is_the_default_and_sends_nothing(monkeypatch):
    monkeypatch.delenv("SMS_MODE", raising=False)
    monkeypatch.setenv("CLINICIAN_PHONE", "+447700900123")
    r = sms.send(sms.compose("Patient/1", "NEWS2 12", "aggregate tier 'high'"))
    assert r.mode == "dry_run"
    assert r.sent is False


def test_a_message_without_the_drill_marker_is_blocked_in_every_mode(monkeypatch):
    """The guard that protects the person holding the phone."""
    monkeypatch.setenv("SMS_MODE", "live")
    monkeypatch.setenv("CLINICIAN_PHONE", "+447700900123")
    r = sms.send("Patient deteriorating, NEWS2 12, please attend")
    assert r.sent is False
    assert r.mode == "blocked"
    assert sms.DRILL_MARKER in (r.error or "")


@pytest.mark.parametrize("mode", ["", "true", "1", "LIVE", "yes", "dry_run"])
def test_only_the_exact_string_live_enables_sending(monkeypatch, mode):
    monkeypatch.setenv("SMS_MODE", mode)
    monkeypatch.setenv("CLINICIAN_PHONE", "+447700900123")
    r = sms.send(sms.compose("Patient/1", "NEWS2 12", "why"))
    assert r.mode == "dry_run", f"{mode!r} must not enable live sending"
    assert r.sent is False


def test_live_without_credentials_fails_closed_and_names_what_is_missing(monkeypatch):
    monkeypatch.setenv("SMS_MODE", "live")
    monkeypatch.setenv("CLINICIAN_PHONE", "+447700900123")
    for var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER"):
        monkeypatch.delenv(var, raising=False)
    r = sms.send(sms.compose("Patient/1", "NEWS2 12", "why"))
    assert r.sent is False
    assert "TWILIO_ACCOUNT_SID" in (r.error or "")


@pytest.mark.parametrize("number", ["", "07700900123", "+44 7700 900123", "notaphone"])
def test_a_non_e164_destination_is_refused(monkeypatch, number):
    monkeypatch.setenv("SMS_MODE", "live")
    monkeypatch.setenv("CLINICIAN_PHONE", number)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC-test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550001111")
    r = sms.send(sms.compose("Patient/1", "NEWS2 12", "why"))
    assert r.sent is False
    assert "CLINICIAN_PHONE" in (r.error or "")


def test_live_send_posts_the_drill_text_and_never_leaks_the_token(monkeypatch):
    """Exercises the real request-building path against a fake transport."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        captured["auth"] = request.headers.get("authorization", "")
        return httpx.Response(201, json={"sid": "SM-fake"})

    monkeypatch.setenv("SMS_MODE", "live")
    monkeypatch.setenv("CLINICIAN_PHONE", "+447700900123")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC-test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "super-secret-token")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550001111")

    with httpx.Client(transport=httpx.MockTransport(handler)) as fake:
        r = sms.send(sms.compose("Patient/1", "NEWS2 12", "aggregate tier 'high'"), client=fake)

    assert r.sent is True
    assert r.provider_sid == "SM-fake"
    assert "AC-test" in captured["url"]
    assert "SYNTHETIC+DRILL" in captured["body"] or "SYNTHETIC%20DRILL" in captured["body"]
    # The token rides in the Authorization header, and must never appear in the
    # result the API hands back to a browser.
    assert "super-secret-token" not in str(r.as_dict())


def test_provider_failure_is_reported_without_the_token(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="authenticate failed")

    monkeypatch.setenv("SMS_MODE", "live")
    monkeypatch.setenv("CLINICIAN_PHONE", "+447700900123")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC-test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "super-secret-token")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550001111")

    with httpx.Client(transport=httpx.MockTransport(handler)) as fake:
        r = sms.send(sms.compose("Patient/1", "NEWS2 12", "why"), client=fake)

    assert r.sent is False
    assert "401" in (r.error or "")
    assert "super-secret-token" not in (r.error or "")


def test_compose_is_generic_over_the_headline_both_real_callers_use():
    """event-studio passes 'NEWS2 {n}'; notification-gateway passes
    '{severity.upper()} alert'. Neither is special-cased."""
    a = sms.compose("Patient/1", "NEWS2 9", "single red parameter")
    b = sms.compose("ICUStay/2", "HIGH alert", "NEWS2 9: single red parameter")
    assert sms.DRILL_MARKER in a and sms.DRILL_MARKER in b
    assert "NEWS2 9" in a
    assert "HIGH alert" in b
