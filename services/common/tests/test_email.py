"""services/common/email.py's guards -- the same shape as test_sms.py's,
deliberately: email and SMS share one safety design, not two independently
reasoned-about ones. Every test below asserts a *refusal* except the two
live-send tests, which inject a fake transport so the suite can never open a
real SMTP connection.
"""

from __future__ import annotations

import smtplib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from services.common import email  # noqa: E402


def test_dry_run_is_the_default_and_sends_nothing(monkeypatch):
    monkeypatch.delenv("EMAIL_MODE", raising=False)
    monkeypatch.setenv("CLINICIAN_EMAIL", "clinician@example.com")
    r = email.send(email.compose("Patient/1", "NEWS2 12", "aggregate tier 'high'"))
    assert r.mode == "dry_run"
    assert r.sent is False


def test_a_message_without_the_drill_marker_is_blocked_in_every_mode(monkeypatch):
    """The guard that protects the person reading the inbox."""
    monkeypatch.setenv("EMAIL_MODE", "live")
    monkeypatch.setenv("CLINICIAN_EMAIL", "clinician@example.com")
    content = email.EmailContent(subject="Patient deteriorating", body="NEWS2 12, please attend")
    r = email.send(content)
    assert r.sent is False
    assert r.mode == "blocked"
    assert email.DRILL_MARKER in (r.error or "")


@pytest.mark.parametrize("mode", ["", "true", "1", "LIVE", "yes", "dry_run"])
def test_only_the_exact_string_live_enables_sending(monkeypatch, mode):
    monkeypatch.setenv("EMAIL_MODE", mode)
    monkeypatch.setenv("CLINICIAN_EMAIL", "clinician@example.com")
    r = email.send(email.compose("Patient/1", "NEWS2 12", "why"))
    assert r.mode == "dry_run", f"{mode!r} must not enable live sending"
    assert r.sent is False


def test_live_without_credentials_fails_closed_and_names_what_is_missing(monkeypatch):
    monkeypatch.setenv("EMAIL_MODE", "live")
    monkeypatch.setenv("CLINICIAN_EMAIL", "clinician@example.com")
    for var in ("SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_ADDRESS"):
        monkeypatch.delenv(var, raising=False)
    r = email.send(email.compose("Patient/1", "NEWS2 12", "why"))
    assert r.sent is False
    assert "SMTP_HOST" in (r.error or "")


@pytest.mark.parametrize(
    "address", ["", "not-an-email", "missing-at-sign.com", "@no-local-part.com"]
)
def test_a_non_email_destination_is_refused(monkeypatch, address):
    monkeypatch.setenv("EMAIL_MODE", "live")
    monkeypatch.setenv("CLINICIAN_EMAIL", address)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("SMTP_FROM_ADDRESS", "bot@example.com")
    r = email.send(email.compose("Patient/1", "NEWS2 12", "why"))
    assert r.sent is False
    assert "CLINICIAN_EMAIL" in (r.error or "")


def test_a_non_integer_smtp_port_is_refused(monkeypatch):
    monkeypatch.setenv("EMAIL_MODE", "live")
    monkeypatch.setenv("CLINICIAN_EMAIL", "clinician@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "not-a-port")
    monkeypatch.setenv("SMTP_USERNAME", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("SMTP_FROM_ADDRESS", "bot@example.com")
    r = email.send(email.compose("Patient/1", "NEWS2 12", "why"))
    assert r.sent is False
    assert "SMTP_PORT" in (r.error or "")


def test_live_send_calls_the_transport_with_the_drill_text_and_never_leaks_the_password(
    monkeypatch,
):
    """Exercises the real message-building path against a fake transport --
    this suite can never open a real SMTP connection."""
    captured = {}

    def fake_transport(host, port, username, password, msg):
        captured["host"] = host
        captured["port"] = port
        captured["username"] = username
        captured["password"] = password
        captured["subject"] = msg["Subject"]
        captured["to"] = msg["To"]
        captured["body"] = msg.get_content()

    monkeypatch.setenv("EMAIL_MODE", "live")
    monkeypatch.setenv("CLINICIAN_EMAIL", "clinician@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "super-secret-app-password")
    monkeypatch.setenv("SMTP_FROM_ADDRESS", "bot@example.com")

    r = email.send(
        email.compose("Patient/1", "NEWS2 12", "aggregate tier 'high'"), transport=fake_transport
    )

    assert r.sent is True
    assert captured["host"] == "smtp.gmail.com"
    assert captured["port"] == 587
    assert captured["to"] == "clinician@example.com"
    assert email.DRILL_MARKER in captured["subject"]
    assert "Patient/1" in captured["body"]
    # The password reaches the transport (it has to, to authenticate) but must
    # never appear in the result handed back to a caller/browser.
    assert captured["password"] == "super-secret-app-password"
    assert "super-secret-app-password" not in str(r.as_dict())


def test_provider_failure_is_reported_without_the_password(monkeypatch):
    def failing_transport(host, port, username, password, msg):
        raise smtplib.SMTPAuthenticationError(535, b"Authentication failed")

    monkeypatch.setenv("EMAIL_MODE", "live")
    monkeypatch.setenv("CLINICIAN_EMAIL", "clinician@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "super-secret-app-password")
    monkeypatch.setenv("SMTP_FROM_ADDRESS", "bot@example.com")

    r = email.send(email.compose("Patient/1", "NEWS2 12", "why"), transport=failing_transport)

    assert r.sent is False
    assert "535" in (r.error or "") or "Authentication" in (r.error or "")
    assert "super-secret-app-password" not in (r.error or "")


def test_compose_is_generic_over_the_headline_both_real_callers_use():
    """event-studio passes 'NEWS2 {n}'; notification-gateway passes
    '{severity.upper()} alert' -- same as sms.compose, neither is special-cased."""
    a = email.compose("Patient/1", "NEWS2 9", "single red parameter")
    b = email.compose("ICUStay/2", "HIGH alert", "NEWS2 9: single red parameter")
    assert email.DRILL_MARKER in a.subject
    assert email.DRILL_MARKER in b.subject
    assert "NEWS2 9" in a.body
    assert "HIGH alert" in b.body


# ---------------------------------------------------------------------------
# The clinical assessment agent-orchestrator attaches to a high-severity alert
# ---------------------------------------------------------------------------


def _assessment(**overrides) -> dict:
    base = {
        "assessable": True,
        "condition": "Sepsis, unspecified organism",
        "dx_chapter": "Infectious",
        "comorbidities": ["renal_disease", "congestive_heart_failure"],
        "charlson_comorbidity_index": 7,
        "news2": 11,
        "news2_tier_icu": "high",
        "dx_group": "Circulatory",
        "threshold_is_disease_specific": True,
        "severity": "high",
        "ml_probability": 0.912,
        "ml_in_validated_scope": True,
        "care_plan": {
            "recommended_actions": "- Recheck lactate\n- Contact the on-call intensivist",
            "citations": [{"passage_id": "G003", "fact_ids": ["F001", "F002"]}],
            "generated": True,
        },
        "summary": "Patient deteriorating on a background of sepsis.",
    }
    return {**base, **overrides}


def test_an_assessment_puts_the_diagnosis_and_plan_in_the_body():
    """Before this, the entire email was "NEWS2 9: tier is high" -- true, and nearly
    useless at 3am because it says nothing about who the patient is or what to do."""
    content = email.compose("ICUStay/1", "HIGH alert", "NEWS2 11", assessment=_assessment())

    assert "Sepsis, unspecified organism" in content.body
    assert "renal_disease" in content.body
    assert "Recheck lactate" in content.body
    assert "G003" in content.body, "a recommendation must carry its grounding"
    assert "F001" in content.body, "fact-ledger ids make a note claim traceable"


def test_the_body_says_which_threshold_escalated_the_patient():
    """`tier_icu` is now fitted per diagnosis chapter where one was earned, so "tier
    is high" is no longer a single global statement. A clinician reading the page
    should be able to tell which threshold fired."""
    disease_specific = email.compose(
        "ICUStay/1", "HIGH alert", "NEWS2 11", assessment=_assessment()
    )
    pooled = email.compose(
        "ICUStay/1",
        "HIGH alert",
        "NEWS2 11",
        assessment=_assessment(threshold_is_disease_specific=False),
    )

    assert "disease-specific threshold (Circulatory)" in disease_specific.body
    assert "pooled ICU threshold" in pooled.body


def test_an_out_of_scope_model_score_carries_a_caution():
    """The model's validated scope is the first 6 ICU hours. A score from outside it
    must not reach a clinician looking like one from inside it."""
    content = email.compose(
        "ICUStay/1",
        "HIGH alert",
        "NEWS2 11",
        assessment=_assessment(
            ml_in_validated_scope=False, ml_scope_note="Outside the model's validated scope."
        ),
    )

    assert "CAUTION" in content.body


def test_a_deterministic_fallback_plan_is_labelled_as_not_generated():
    """A template must never be mistaken for clinical reasoning an LLM produced."""
    content = email.compose(
        "ICUStay/1",
        "HIGH alert",
        "NEWS2 11",
        assessment=_assessment(
            care_plan={"recommended_actions": "Recommend urgent review.", "generated": False}
        ),
    )

    assert "Deterministic fallback" in content.body


def test_a_skipped_care_plan_says_why_rather_than_going_quiet():
    content = email.compose(
        "ICUStay/1",
        "HIGH alert",
        "NEWS2 9",
        assessment=_assessment(
            care_plan=None, care_plan_skipped_reason="no care plan: severity is medium"
        ),
    )

    assert "severity is medium" in content.body


def test_an_unassessable_patient_composes_exactly_the_old_message():
    """A wearable volunteer has no chart. The email must degrade to what it always
    was, not to an empty template with blank headers."""
    with_none = email.compose("Subject/S05", "HIGH alert", "HR 148")
    unassessable = email.compose(
        "Subject/S05", "HIGH alert", "HR 148", assessment={"assessable": False}
    )

    assert with_none.body == unassessable.body
    assert "Clinical context" not in with_none.body


def test_the_drill_marker_survives_an_attached_assessment():
    """Guard 3 blocks any body missing the marker. A regression here would silently
    stop every high-severity page from sending."""
    content = email.compose("ICUStay/1", "HIGH alert", "NEWS2 11", assessment=_assessment())

    assert email.DRILL_MARKER in content.body
    assert email.DRILL_MARKER in content.subject
