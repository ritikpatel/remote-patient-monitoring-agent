import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from push import FCMPushSender, NoopPushSender  # noqa: E402
from routing import (  # noqa: E402
    CHANNEL_DASHBOARD,
    CHANNEL_ONCALL_ESCALATION,
    CHANNEL_PUSH,
    is_overnight,
    route_notification,
)


def test_is_overnight_boundaries():
    assert is_overnight(datetime(2110, 1, 1, 22, 0)) is True
    assert is_overnight(datetime(2110, 1, 1, 3, 0)) is True
    assert is_overnight(datetime(2110, 1, 1, 6, 59)) is True
    assert is_overnight(datetime(2110, 1, 1, 7, 0)) is False
    assert is_overnight(datetime(2110, 1, 1, 12, 0)) is False


def test_daytime_low_severity_is_dashboard_only():
    channels = route_notification("low", datetime(2110, 1, 1, 12, 0))
    assert channels == [CHANNEL_DASHBOARD]


def test_daytime_medium_severity_adds_push():
    channels = route_notification("medium", datetime(2110, 1, 1, 12, 0))
    assert CHANNEL_DASHBOARD in channels
    assert CHANNEL_PUSH in channels
    assert CHANNEL_ONCALL_ESCALATION not in channels


def test_overnight_low_severity_still_gets_push():
    """E16: overnight staffing is thin -- the dashboard alone is not enough even
    for a low-severity alert overnight."""
    channels = route_notification("low", datetime(2110, 1, 1, 5, 0))
    assert CHANNEL_PUSH in channels


def test_overnight_high_severity_pages_oncall():
    channels = route_notification("high", datetime(2110, 1, 1, 5, 0))
    assert CHANNEL_ONCALL_ESCALATION in channels


def test_daytime_high_severity_does_not_page_oncall():
    channels = route_notification("high", datetime(2110, 1, 1, 12, 0))
    assert CHANNEL_ONCALL_ESCALATION not in channels


def test_noop_push_sender_records_sends():
    sender = NoopPushSender()
    assert sender.send("token123", "Alert", "NEWS2 high") is True
    assert sender.sent == [("token123", "Alert", "NEWS2 high")]


def test_fcm_push_sender_fails_loudly_without_credentials(monkeypatch):
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)
    sender = FCMPushSender()
    with pytest.raises(RuntimeError, match="FIREBASE_PROJECT_ID"):
        sender.send("token", "t", "b")
