"""FCM push, behind an interface.

PROJECT_PLAN.md section 10: "FCM push to clinician phones." Real HTTP call to
Firebase Cloud Messaging's HTTP v1 API -- never exercised end-to-end in this
environment, since it needs a Firebase service-account credential this project
does not have. `NoopPushSender` is what every test in this service actually runs
against; `FCMPushSender.send` fails loudly (not silently) if invoked without
credentials, rather than pretending to have sent anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


class PushSender(Protocol):
    def send(self, device_token: str, title: str, body: str) -> bool: ...


@dataclass
class NoopPushSender:
    """Default, and what every test uses: records what would have been sent
    without any network call or credential."""

    sent: list[tuple[str, str, str]] = field(default_factory=list)

    def send(self, device_token: str, title: str, body: str) -> bool:
        self.sent.append((device_token, title, body))
        return True


@dataclass
class FCMPushSender:
    """Real FCM HTTP v1 API integration. `FIREBASE_PROJECT_ID` and a service-
    account access token (`FCM_ACCESS_TOKEN` -- a real deployment mints this from
    the service account JSON via google-auth, not a static env var, but that
    minting step needs credentials this project doesn't have to demonstrate)."""

    project_id: str = field(default_factory=lambda: os.environ.get("FIREBASE_PROJECT_ID", ""))

    def send(self, device_token: str, title: str, body: str) -> bool:
        if not self.project_id:
            raise RuntimeError(
                "FIREBASE_PROJECT_ID not set -- FCMPushSender needs real Firebase "
                "credentials this environment does not have. Use NoopPushSender for "
                "local/dev, or configure Firebase in Phase 8."
            )
        access_token = os.environ.get("FCM_ACCESS_TOKEN")
        if not access_token:
            raise RuntimeError("FCM_ACCESS_TOKEN not set -- cannot authenticate to FCM.")

        import httpx

        url = f"https://fcm.googleapis.com/v1/projects/{self.project_id}/messages:send"
        payload = {
            "message": {
                "token": device_token,
                "notification": {"title": title, "body": body},
            }
        }
        resp = httpx.post(
            url, json=payload, headers={"Authorization": f"Bearer {access_token}"}, timeout=10.0
        )
        return resp.status_code == 200
