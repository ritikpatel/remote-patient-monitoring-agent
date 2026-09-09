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


def compose(patient_ref: str, headline: str, detail: str) -> EmailContent:
    """Same two arguments `sms.compose` takes, for the same reason: both
    channels are fed by the same two callers (notification-gateway's real
    send, event-studio's zero-network preview) with the same two pieces of
    information -- a short escalation label and the reason."""
    subject = f"{DRILL_MARKER} Deterioration alert -- {patient_ref}"
    body = (
        f"{DRILL_MARKER}\n\n"
        f"Patient: {patient_ref}\n"
        f"Escalation: {headline}\n"
        f"Reason: {detail}\n\n"
        "This message describes no real patient. Not a real clinical alert."
    )
    return EmailContent(subject=subject, body=body)


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
