"""Send an escalation email, with the guards a drill needs to be safe to run.

Sibling to `services/common/sms.py`, same shape and the same four guard
classes, deliberately: this project's alerting now has two independently
configurable "page a human" channels rather than one, and a clinician
escalation reaching an inbox is exactly as real a thing to get wrong as one
reaching a phone.

**Why email is the default channel and SMS is not.** SMS needs a paid Twilio
account (a real phone number, real billing) before it can send anything at
all. Email needs an SMTP account -- for the common case, a Gmail address and
an app password, which is free and something a clinician demoing this
project is more likely to already have. Both channels keep the identical
safety posture (dry-run by default, guarded, drill-marked); which one an
operator actually turns on is a credentials question, not a code path this
module decides. `notification-gateway` attempts both independently on a
high-severity notification -- see its module docstring.

Configuration (all from the environment; nothing is committed):

    EMAIL_MODE=dry_run | live      default dry_run -- composes and returns, sends nothing
    CLINICIAN_EMAIL=name@example.com   destination
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587                  587 (STARTTLS) is what Gmail and most providers expect
    SMTP_USERNAME=you@gmail.com
    SMTP_PASSWORD=...              a Gmail *app password*, not your account password --
                                    Google requires one once 2-Step Verification is on
    SMTP_FROM_ADDRESS=you@gmail.com

Four guards, each of which fails closed:

1. **`EMAIL_MODE` must be exactly `live`.** Anything else -- unset, typo'd,
   `true`, `1` -- is treated as dry run.
2. **Every credential must be present.** A half-configured provider returns a
   named error instead of a partially-formed connection attempt.
3. **The body must carry the synthetic-drill marker.** Same reasoning as
   `sms.py`: this project has no live clinical deployment, so every message
   it is capable of sending is a drill, and there is deliberately no code
   path that composes one without the marker.
4. **The destination must look like an email address.** A malformed address
   is a configuration error worth surfacing, not something to discover from
   an SMTP bounce.

The SMTP password is never logged or returned. Errors carry the SMTP
server's response, which does not include the password, so nothing further
is stripped from it (contrast `sms.py`, which truncates a provider body that
could theoretically echo back request data).
"""

from __future__ import annotations

import os
import re
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol

E164_LIKE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Same marker sms.py uses -- one drill vocabulary across every channel this
# project can page a human on, not a per-channel convention to keep in sync.
DRILL_MARKER = "[SYNTHETIC DRILL]"
REQUEST_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class EmailContent:
    subject: str
    body: str


@dataclass(frozen=True)
class EmailResult:
    mode: str
    to: str
    subject: str
    body: str
    sent: bool = False
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "to": self.to,
            "subject": self.subject,
            "body": self.body,
            "sent": self.sent,
            "error": self.error,
        }


def compose(
    patient_ref: str,
    headline: str,
    detail: str,
    assessment: dict | None = None,
) -> EmailContent:
    """Same first three arguments `sms.compose` takes, for the same reason: both
    channels are fed by the same callers (notification-gateway's real send,
    event-studio's zero-network preview) with the same information -- a short
    escalation label and the reason.

    `assessment` is agent-orchestrator's `/assess` payload, and it is what turns
    this from a number into something a clinician can act on. Before it, the entire
    email was "NEWS2 9: ICU-recalibrated NEWS2 tier is 'high'" -- true, and nearly
    useless at 3am, because it says nothing about *who* this patient is or *what to
    do*. With it the body carries the diagnosis, the chronic comorbidities, which
    threshold escalated (disease-specific or pooled), the learned model's severity
    grade, and the grounded care plan.

    Email is deliberately the channel that carries the full plan; `sms.compose`
    stays short. A care plan is several hundred characters of clinical prose --
    right for an inbox, wrong for a text message that gets truncated mid-sentence.

    Kept a pure function of its arguments: no network, no environment reads. That is
    what lets event-studio render an exact preview of a real escalation email
    without sending one.
    """
    subject = f"{DRILL_MARKER} Deterioration alert -- {patient_ref}"
    lines = [
        DRILL_MARKER,
        "",
        f"Patient: {patient_ref}",
        f"Escalation: {headline}",
        f"Reason: {detail}",
    ]

    if assessment and assessment.get("assessable"):
        lines += ["", "-- Clinical context " + "-" * 40]
        condition = assessment.get("condition")
        if condition:
            lines.append(f"Condition: {condition}")
        if assessment.get("dx_chapter"):
            lines.append(f"Diagnosis group: {assessment['dx_chapter']}")
        comorbidities = assessment.get("comorbidities") or []
        if comorbidities:
            lines.append(f"Chronic comorbidities: {', '.join(comorbidities)}")
        if assessment.get("charlson_comorbidity_index") is not None:
            lines.append(f"Charlson index: {assessment['charlson_comorbidity_index']}")

        lines += ["", "-- Risk " + "-" * 52]
        if assessment.get("news2") is not None:
            tier_note = (
                f"disease-specific threshold ({assessment.get('dx_group')})"
                if assessment.get("threshold_is_disease_specific")
                else "pooled ICU threshold"
            )
            lines.append(
                f"NEWS2 {assessment['news2']} -- tier "
                f"'{assessment.get('news2_tier_icu')}' on the {tier_note}"
            )
        if assessment.get("severity"):
            prob = assessment.get("ml_probability")
            prob_str = f" (p={prob:.3f})" if isinstance(prob, int | float) else ""
            lines.append(f"Model severity: {str(assessment['severity']).upper()}{prob_str}")
        # The model's validated scope is only the first 6 ICU hours. A score from
        # outside it must not reach a clinician looking like one from inside it --
        # ml/models/serving.py carries the measurement this warning is drawn from.
        if assessment.get("ml_in_validated_scope") is False and assessment.get("ml_scope_note"):
            lines.append(f"CAUTION: {assessment['ml_scope_note']}")

        care_plan = assessment.get("care_plan")
        if care_plan:
            lines += ["", "-- Suggested actions " + "-" * 39]
            lines.append(str(care_plan.get("recommended_actions", "")).strip())
            if not care_plan.get("generated"):
                lines.append(
                    "\n(Deterministic fallback -- no LLM was configured; "
                    "no clinical reasoning was generated.)"
                )
            citations = care_plan.get("citations") or []
            if citations:
                fact_ids = sorted({f for c in citations for f in (c.get("fact_ids") or [])})
                sources = ", ".join(
                    str(c.get("passage_id")) for c in citations if c.get("passage_id")
                )
                lines.append(f"\nGrounded in: {sources}")
                if fact_ids:
                    lines.append(f"Fact ledger ids: {', '.join(fact_ids)}")
        elif assessment.get("care_plan_skipped_reason"):
            lines += ["", str(assessment["care_plan_skipped_reason"])]

        if assessment.get("summary"):
            lines += ["", "-- Summary " + "-" * 49, str(assessment["summary"]).strip()]

    lines += [
        "",
        "This message describes no real patient. Not a real clinical alert.",
    ]
    return EmailContent(subject=subject, body="\n".join(lines))


class Transport(Protocol):
    def __call__(
        self, host: str, port: int, username: str, password: str, msg: EmailMessage
    ) -> None: ...


def _smtp_transport(host: str, port: int, username: str, password: str, msg: EmailMessage) -> None:
    """The real transport: STARTTLS on the common submission port (587).
    Kept as a plain function, not a method, so a test can substitute a fake
    with the same signature instead of needing a live SMTP server -- the
    same seam `sms.py` gets from accepting an `httpx.Client`."""
    with smtplib.SMTP(host, port, timeout=REQUEST_TIMEOUT_S) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)


def send(content: EmailContent, *, transport: Transport | None = None) -> EmailResult:
    """Send `content`, or explain precisely why it was not sent."""
    mode = os.environ.get("EMAIL_MODE", "dry_run")
    to = os.environ.get("CLINICIAN_EMAIL", "")

    if DRILL_MARKER not in content.subject and DRILL_MARKER not in content.body:
        # Guard 3 applies in every mode, same reasoning as sms.py: a message
        # missing the marker is a bug in the caller, and reporting it in dry
        # run is how it gets found before live sending is ever switched on.
        return EmailResult(
            "blocked",
            to,
            content.subject,
            content.body,
            error=f"neither subject nor body carries {DRILL_MARKER}",
        )

    if mode != "live":
        return EmailResult(
            "dry_run", to or "<CLINICIAN_EMAIL unset>", content.subject, content.body
        )

    if not to:
        return EmailResult(
            "live", "", content.subject, content.body, error="CLINICIAN_EMAIL is not set"
        )
    if not E164_LIKE_EMAIL.match(to):
        return EmailResult(
            "live",
            to,
            content.subject,
            content.body,
            error=f"CLINICIAN_EMAIL {to!r} does not look like an email address",
        )

    host = os.environ.get("SMTP_HOST", "")
    port_raw = os.environ.get("SMTP_PORT", "")
    username = os.environ.get("SMTP_USERNAME", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("SMTP_FROM_ADDRESS", "")
    missing = [
        name
        for name, value in (
            ("SMTP_HOST", host),
            ("SMTP_PORT", port_raw),
            ("SMTP_USERNAME", username),
            ("SMTP_PASSWORD", password),
            ("SMTP_FROM_ADDRESS", sender),
        )
        if not value
    ]
    if missing:
        return EmailResult(
            "live",
            to,
            content.subject,
            content.body,
            error=f"missing provider config: {', '.join(missing)}",
        )

    try:
        port = int(port_raw)
    except ValueError:
        return EmailResult(
            "live",
            to,
            content.subject,
            content.body,
            error=f"SMTP_PORT {port_raw!r} is not an integer",
        )

    msg = EmailMessage()
    msg["Subject"] = content.subject
    msg["From"] = sender
    msg["To"] = to
    msg.set_content(content.body)

    transport = transport or _smtp_transport
    try:
        transport(host, port, username, password, msg)
        return EmailResult("live", to, content.subject, content.body, sent=True)
    except (smtplib.SMTPException, OSError) as exc:
        return EmailResult(
            "live", to, content.subject, content.body, error=f"{type(exc).__name__}: {exc}"
        )
