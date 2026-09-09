"""Send an escalation SMS, with the guards a drill needs to be safe to run.

Paging a real phone is the one thing in this repo that reaches a human being
directly, so the defaults are deliberately timid and every unsafe combination
fails closed rather than silently sending.

**Lives in `services/common/` because it has two callers, not one.**
`notification-gateway`'s `/notify` handler calls `send()` for real -- SMS is
now genuinely part of "the notification sent after an alert is deemed to be
triggered" (a high-severity `/notify` call), the same way WebSocket broadcast
and FCM push already are. `event-studio` calls only `compose()`, never
`send()`, to show what the message *would* say the instant severity is
composed, with zero network calls and no guard evaluation -- so a "Generate"
click, which posts nothing anywhere, can never accidentally page a phone. The
real send only happens downstream of a real alert-service escalation, via
notification-gateway, exactly like every other alert this project raises.

Configuration (all from the environment; nothing is committed):

    SMS_MODE=dry_run | live      default dry_run -- composes and returns, sends nothing
    CLINICIAN_PHONE=+4477...     destination, E.164
    TWILIO_ACCOUNT_SID=AC...     provider credentials
    TWILIO_AUTH_TOKEN=...
    TWILIO_FROM_NUMBER=+1555...

Four guards, each of which fails closed:

1. **`SMS_MODE` must be exactly `live`.** Anything else -- unset, typo'd,
   `true`, `1` -- is treated as dry run.
2. **Every credential must be present.** A half-configured provider returns a
   named error instead of a partially-formed request.
3. **The body must carry the synthetic-drill marker.** This module refuses to
   transmit a message that could be mistaken for a real clinical alert, which
   is the failure that would actually matter to a person receiving it. This
   project has no live clinical deployment, so every SMS it is capable of
   sending is a drill -- there is deliberately no code path that composes a
   message without the marker.
4. **The destination must be E.164.** A malformed number is a configuration
   error worth surfacing, not something to discover in a provider's logs.

The auth token is never logged or returned. Errors carry the provider's status
and a truncated body, which is enough to debug without leaking the credential.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
E164 = re.compile(r"^\+[1-9]\d{7,14}$")
# Every message this project sends must be identifiable as a drill by the person
# holding the phone, in the first few words, before any clinical content.
DRILL_MARKER = "[SYNTHETIC DRILL]"
REQUEST_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class SmsResult:
    mode: str
    to: str
    text: str
    sent: bool = False
    error: str | None = None
    provider_sid: str | None = None

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "to": self.to,
            "text": self.text,
            "sent": self.sent,
            "error": self.error,
            "provider_sid": self.provider_sid,
        }


def compose(patient_ref: str, headline: str, detail: str) -> str:
    """`headline` is the short escalation label -- event-studio passes
    `"NEWS2 {n}"`, notification-gateway passes `"{severity.upper()} alert"` --
    and `detail` is the reason. Kept generic rather than NEWS2-specific
    because both real callers now share this one function."""
    return (
        f"{DRILL_MARKER} {patient_ref}: {headline}, escalation - {detail}. " "Not a real patient."
    )


def send(text: str, *, client: httpx.Client | None = None) -> SmsResult:
    """Send `text`, or explain precisely why it was not sent."""
    mode = os.environ.get("SMS_MODE", "dry_run")
    to = os.environ.get("CLINICIAN_PHONE", "")

    if DRILL_MARKER not in text:
        # Guard 3 applies in every mode: a message without the marker is a bug
        # in the caller, and reporting it in dry run is how it gets found before
        # anyone turns live sending on.
        return SmsResult("blocked", to, text, error=f"body is missing {DRILL_MARKER}")

    if mode != "live":
        return SmsResult("dry_run", to or "<CLINICIAN_PHONE unset>", text)

    if not to:
        return SmsResult("live", "", text, error="CLINICIAN_PHONE is not set")
    if not E164.match(to):
        return SmsResult("live", to, text, error=f"CLINICIAN_PHONE {to!r} is not E.164")

    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    sender = os.environ.get("TWILIO_FROM_NUMBER", "")
    missing = [
        name
        for name, value in (
            ("TWILIO_ACCOUNT_SID", sid),
            ("TWILIO_AUTH_TOKEN", token),
            ("TWILIO_FROM_NUMBER", sender),
        )
        if not value
    ]
    if missing:
        return SmsResult("live", to, text, error=f"missing provider config: {', '.join(missing)}")

    owns_client = client is None
    client = client or httpx.Client(timeout=REQUEST_TIMEOUT_S)
    try:
        resp = client.post(
            TWILIO_API.format(sid=sid),
            data={"To": to, "From": sender, "Body": text},
            auth=(sid, token),
        )
        if resp.status_code in (200, 201):
            return SmsResult("live", to, text, sent=True, provider_sid=resp.json().get("sid"))
        # Status plus a truncated body: enough to debug, and the auth token is
        # never part of either.
        return SmsResult(
            "live", to, text, error=f"provider HTTP {resp.status_code}: {resp.text[:200]}"
        )
    except httpx.HTTPError as exc:
        return SmsResult("live", to, text, error=f"{type(exc).__name__}: {exc}")
    finally:
        if owns_client:
            client.close()
