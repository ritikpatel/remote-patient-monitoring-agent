"""event-studio: the generator must agree with the scorer the pipeline uses."""

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


@needs_warehouse
def test_sms_is_dry_run_by_default_and_marks_itself_synthetic():
    d = client.post("/event", json={"severity": 1.0, "seed": 3}).json()
    assert d["would_escalate"] is True
    assert d["sms"]["mode"] == "dry_run"
    assert "SYNTHETIC DRILL" in d["sms"]["text"]


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


# --- SMS guards ------------------------------------------------------------
# Every one of these asserts a *refusal*. None of them sends anything, and no
# test in this file is capable of contacting a provider: the live-mode test
# injects a fake transport so a misconfigured environment cannot page a person.

sms = load_module(APP_DIR / "sms.py", "event_studio_sms")


def test_dry_run_is_the_default_and_sends_nothing(monkeypatch):
    monkeypatch.delenv("SMS_MODE", raising=False)
    monkeypatch.setenv("CLINICIAN_PHONE", "+447700900123")
    r = sms.send(sms.compose("Patient/1", 12, "aggregate tier 'high'"))
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
    r = sms.send(sms.compose("Patient/1", 12, "why"))
    assert r.mode == "dry_run", f"{mode!r} must not enable live sending"
    assert r.sent is False


def test_live_without_credentials_fails_closed_and_names_what_is_missing(monkeypatch):
    monkeypatch.setenv("SMS_MODE", "live")
    monkeypatch.setenv("CLINICIAN_PHONE", "+447700900123")
    for var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER"):
        monkeypatch.delenv(var, raising=False)
    r = sms.send(sms.compose("Patient/1", 12, "why"))
    assert r.sent is False
    assert "TWILIO_ACCOUNT_SID" in (r.error or "")


@pytest.mark.parametrize("number", ["", "07700900123", "+44 7700 900123", "notaphone"])
def test_a_non_e164_destination_is_refused(monkeypatch, number):
    monkeypatch.setenv("SMS_MODE", "live")
    monkeypatch.setenv("CLINICIAN_PHONE", number)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC-test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550001111")
    r = sms.send(sms.compose("Patient/1", 12, "why"))
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
        r = sms.send(sms.compose("Patient/1", 12, "aggregate tier 'high'"), client=fake)

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
        r = sms.send(sms.compose("Patient/1", 12, "why"), client=fake)

    assert r.sent is False
    assert "401" in (r.error or "")
    assert "super-secret-token" not in (r.error or "")
