"""Which channels a notification goes out on, including the overnight path.

PROJECT_PLAN.md section 10: "WebSocket to dashboard, FCM push to clinician phones,
overnight escalation path." E16: "The 05:00 lab round is the worst case -- it
triggers when staffing is thinnest, so escalation policy needs an explicit
overnight path." This module is that policy: pure, deterministic, and independent
of whether a WebSocket or FCM connection actually exists to carry the result.
"""

from __future__ import annotations

from datetime import datetime

OVERNIGHT_START_HOUR = 22  # 22:00
OVERNIGHT_END_HOUR = 7  # 07:00, exclusive

CHANNEL_DASHBOARD = "dashboard"
CHANNEL_PUSH = "push"
CHANNEL_ONCALL_ESCALATION = "oncall_escalation"


def is_overnight(ts: datetime) -> bool:
    return ts.hour >= OVERNIGHT_START_HOUR or ts.hour < OVERNIGHT_END_HOUR


def route_notification(severity: str, ts: datetime) -> list[str]:
    """Daytime: dashboard always; push added for medium/high. Overnight: the
    dashboard alone is not enough (E16's thin-staffing risk) -- every severity gets
    a push, and high severity additionally pages the on-call escalation contact
    rather than waiting for someone to notice the dashboard or a routine push.
    """
    overnight = is_overnight(ts)
    channels = [CHANNEL_DASHBOARD]

    if severity in ("medium", "high") or overnight:
        channels.append(CHANNEL_PUSH)

    if severity == "high" and overnight:
        channels.append(CHANNEL_ONCALL_ESCALATION)

    return channels
