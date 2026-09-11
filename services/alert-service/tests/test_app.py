import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from store import AlertStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from services.common.testing import load_service_app  # noqa: E402

_module = load_service_app("alert-service", REPO_ROOT)
app, set_store, set_notify_client = _module.app, _module.set_store, _module.set_notify_client
set_agent_client = _module.set_agent_client
notification_gateway_module = load_service_app("notification-gateway", REPO_ROOT)


def agent_client_returning(payload: dict) -> httpx.Client:
    """A stand-in agent-orchestrator that answers /assess with `payload`.

    Every test needs one: alert-service now calls agent-orchestrator on a genuinely
    new alert, and without an injected client that call goes to a real socket on
    localhost:8008 -- which is slow when nothing is listening and, worse, would make
    the suite's behaviour depend on whether a dev happens to have the service running.
    """
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
        base_url="http://agent",
    )


client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_store(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    set_store(store)
    # A real notification-gateway app, via Starlette's TestClient acting as a
    # genuine sync httpx.Client subclass (no live socket needed) -- so every
    # test exercises the real _notify_dashboard() call path instead of
    # silently swallowing it against an unreachable URL.
    set_notify_client(TestClient(notification_gateway_module.app, base_url="http://notify"))
    # Default: a patient the agent cannot assess. Keeps every pre-existing test on
    # the path it was written for (deterministic severity, no assessment attached),
    # and the grading tests below inject their own.
    set_agent_client(
        agent_client_returning({"assessable": False, "reason": "stub agent for tests"})
    )
    yield store
    store.close()
    set_agent_client(None)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_raise_and_fetch_active():
    resp = client.post(
        "/alerts",
        json={
            "patient_ref": "ICUStay/1",
            "alert_type": "news2_high",
            "severity": "high",
            "message": "NEWS2=9",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["was_new"] is True
    # The one place a new alert triggers notification-gateway -- embedded here,
    # not requested a second time by whoever called POST /alerts. Membership,
    # not equality: _notify_dashboard sends no explicit timestamp, so
    # route_notification's overnight-only oncall_escalation channel may or may
    # not also be present depending on the real wall-clock time the suite runs.
    assert {"dashboard", "push"}.issubset(resp.json()["notification"]["channels"])

    resp2 = client.get("/alerts", params={"patient_ref": "ICUStay/1"})
    assert resp2.status_code == 200
    assert len(resp2.json()) == 1


def test_suppress_removes_from_active():
    resp = client.post(
        "/alerts",
        json={
            "patient_ref": "ICUStay/2",
            "alert_type": "news2_high",
            "severity": "high",
            "message": "m",
        },
    )
    alert_id = resp.json()["alert"]["id"]
    client.post(f"/alerts/{alert_id}/suppress")
    resp2 = client.get("/alerts", params={"patient_ref": "ICUStay/2"})
    assert resp2.json() == []


def test_acknowledge_requires_body():
    resp = client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/3", "alert_type": "t", "severity": "low", "message": "m"},
    )
    alert_id = resp.json()["alert"]["id"]
    resp2 = client.post(
        f"/alerts/{alert_id}/acknowledge", json={"acknowledged_by": "clinician:jdoe"}
    )
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "acknowledged"


def test_suppress_unknown_alert_404():
    resp = client.post("/alerts/999999/suppress")
    assert resp.status_code == 404


def test_raising_a_new_alert_broadcasts_to_a_real_connected_dashboard():
    """End-to-end: a real notification-gateway app, a real connected
    WebSocket client, and a real POST /alerts -- confirms the socket actually
    receives the broadcast, not just that some function was called."""
    gateway_client = TestClient(notification_gateway_module.app)
    with gateway_client.websocket_connect("/ws/dashboard") as ws:
        client.post(
            "/alerts",
            json={
                "patient_ref": "ICUStay/20",
                "alert_type": "news2_high",
                "severity": "high",
                "message": "NEWS2=9",
            },
        )
        received = ws.receive_json()
    assert received == {"patient_ref": "ICUStay/20", "severity": "high", "message": "NEWS2=9"}


def test_a_routine_dedup_repeat_does_not_rebroadcast():
    """Only a genuinely new alert (or a fresh escalation) should push to the
    dashboard -- a repeat within the same 4h dedup bucket must not re-fire a
    broadcast for something the clinician has already seen."""
    gateway_client = TestClient(notification_gateway_module.app)
    client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/21", "alert_type": "t", "severity": "low", "message": "1st"},
    )
    with gateway_client.websocket_connect("/ws/dashboard") as ws:
        client.post(
            "/alerts",
            json={
                "patient_ref": "ICUStay/21",
                "alert_type": "t",
                "severity": "low",
                "message": "2nd, same bucket",
            },
        )
        # Prove the socket is live (something else broadcasts) rather than
        # asserting an absence with no positive signal at all.
        client.post(
            "/alerts",
            json={
                "patient_ref": "ICUStay/22",
                "alert_type": "t",
                "severity": "low",
                "message": "m",
            },
        )
        received = ws.receive_json()
    assert received["patient_ref"] == "ICUStay/22"  # not the ICUStay/21 repeat


def test_a_new_high_severity_alert_attempts_sms_dry_run_by_default(monkeypatch):
    """SMS is dry-run unless an operator explicitly opts in (SMS_MODE=live) --
    safe to leave wired through this path in every environment."""
    monkeypatch.delenv("SMS_MODE", raising=False)
    resp = client.post(
        "/alerts",
        json={
            "patient_ref": "ICUStay/30",
            "alert_type": "news2_high",
            "severity": "high",
            "message": "NEWS2=9: single red parameter",
        },
    )
    sms = resp.json()["notification"]["sms"]
    assert sms["mode"] == "dry_run"
    assert sms["sent"] is False
    assert "ICUStay/30" in sms["text"]


def test_a_new_high_severity_alert_attempts_email_dry_run_by_default(monkeypatch):
    """Email shares SMS's trigger and default -- both attempted, both
    dry-run unless their own *_MODE=live is set."""
    monkeypatch.delenv("EMAIL_MODE", raising=False)
    resp = client.post(
        "/alerts",
        json={
            "patient_ref": "ICUStay/33",
            "alert_type": "news2_high",
            "severity": "high",
            "message": "NEWS2=9: single red parameter",
        },
    )
    mail = resp.json()["notification"]["email"]
    assert mail["mode"] == "dry_run"
    assert mail["sent"] is False
    assert "ICUStay/33" in mail["body"]


def test_a_new_low_severity_alert_does_not_attempt_email_or_sms():
    resp = client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/31", "alert_type": "t", "severity": "low", "message": "m"},
    )
    notification = resp.json()["notification"]
    assert notification["sms"] is None
    assert notification["email"] is None


def test_a_dedup_repeat_does_not_attempt_a_second_sms():
    """was_new: False must not re-trigger _notify_dashboard at all -- this is
    the same guarantee test_a_routine_dedup_repeat_does_not_rebroadcast proves
    for the WebSocket, extended to the channel that reaches a phone."""
    first = client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/32", "alert_type": "t", "severity": "high", "message": "1st"},
    )
    assert first.json()["notification"]["sms"] is not None

    second = client.post(
        "/alerts",
        json={
            "patient_ref": "ICUStay/32",
            "alert_type": "t",
            "severity": "high",
            "message": "2nd, same bucket",
        },
    )
    assert second.json()["was_new"] is False
    assert second.json()["notification"] is None


def test_alerts_active_spans_every_patient():
    client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/10", "alert_type": "a", "severity": "high", "message": "m"},
    )
    client.post(
        "/alerts",
        json={"patient_ref": "ICUStay/11", "alert_type": "b", "severity": "low", "message": "m"},
    )
    resp = client.get("/alerts/active")
    assert resp.status_code == 200
    refs = {a["patient_ref"] for a in resp.json()}
    assert {"ICUStay/10", "ICUStay/11"}.issubset(refs)


# ---------------------------------------------------------------------------
# Severity grading: NEWS2 gates, the learned model grades
# ---------------------------------------------------------------------------


class TestGradedSeverity:
    """`_graded_severity` decides whether a human is paged. Every one of its
    fallbacks must point toward paging, never away from it -- these tests are the
    proof of that, not the docstring.
    """

    @staticmethod
    def _alert(severity: str = "high"):
        from store import Alert

        return Alert(
            id=1,
            patient_ref="ICUStay/1",
            alert_type="news2_escalation",
            severity=severity,
            message="NEWS2=9",
            dedup_key="k",
            raised_at="2026-01-01T00:00:00+00:00",
            last_seen_at="2026-01-01T00:00:00+00:00",
            repeat_count=1,
            status="active",
            acknowledged_by=None,
            acknowledged_at=None,
        )

    def test_an_in_scope_grade_is_used(self):
        graded = _module._graded_severity(
            self._alert("high"),
            {"assessable": True, "severity": "medium", "ml_in_validated_scope": True},
        )
        assert graded == "medium"

    def test_an_out_of_scope_grade_is_ignored(self):
        """The model's held-out AUPRC falls from 0.715 in the first 6 ICU hours to
        0.074 after. Letting a number that weak downgrade a NEWS2 escalation would be
        trusting it exactly where it was measured not to work."""
        graded = _module._graded_severity(
            self._alert("high"),
            {"assessable": True, "severity": "low", "ml_in_validated_scope": False},
        )
        assert graded == "high"

    def test_an_ungraded_alert_keeps_its_deterministic_severity(self):
        """severity=None means no promoted model exists. Treating that as "low" would
        silently stop paging on a fresh checkout."""
        graded = _module._graded_severity(
            self._alert("high"),
            {"assessable": True, "severity": None, "ml_in_validated_scope": True},
        )
        assert graded == "high"

    def test_an_unassessable_patient_keeps_its_deterministic_severity(self):
        graded = _module._graded_severity(
            self._alert("high"), {"assessable": False, "reason": "no ICU stay"}
        )
        assert graded == "high"

    def test_a_failed_assessment_keeps_its_deterministic_severity(self):
        graded = _module._graded_severity(
            self._alert("high"), {"error": "ConnectError: nothing listening"}
        )
        assert graded == "high"

    def test_no_assessment_at_all_keeps_its_deterministic_severity(self):
        assert _module._graded_severity(self._alert("high"), None) == "high"


class TestAssessmentReachesTheAlert:
    def test_a_high_severity_alert_carries_its_assessment_into_the_email(self):
        """The end of the chain the user asked for: composite event -> agent ->
        care plan -> email. Asserted on the real notification-gateway app, so the
        care plan really does reach a composed (dry-run) email body."""
        set_agent_client(
            agent_client_returning(
                {
                    "assessable": True,
                    "severity": "high",
                    "ml_in_validated_scope": True,
                    "ml_probability": 0.91,
                    "condition": "Sepsis, unspecified organism",
                    "dx_chapter": "Infectious",
                    "comorbidities": ["renal_disease"],
                    "news2": 11,
                    "news2_tier_icu": "high",
                    "dx_group": "Infectious",
                    "threshold_is_disease_specific": False,
                    "care_plan": {
                        "condition": "Sepsis, unspecified organism",
                        "recommended_actions": "- Recheck lactate\n- Contact the on-call team",
                        "citations": [{"passage_id": "G003", "fact_ids": ["F001"]}],
                        "generated": True,
                    },
                }
            )
        )

        resp = client.post(
            "/alerts",
            json={
                "patient_ref": "ICUStay/77",
                "alert_type": "news2_escalation",
                "severity": "high",
                "message": "NEWS2 11: tier is high",
            },
        )

        body = resp.json()
        assert body["assessment"]["assessable"] is True
        email_body = body["notification"]["email"]["body"]
        assert "Sepsis, unspecified organism" in email_body
        assert "Recheck lactate" in email_body
        assert "renal_disease" in email_body
        assert "G003" in email_body, "the care plan must carry its grounding"

    def test_a_model_downgrade_regrades_the_alert_and_stops_the_page(self):
        """NEWS2 raised it; the model graded it medium. The alert still exists and is
        still on the dashboard -- but notification-gateway only pages on high, so no
        email is composed. That is the whole 'NEWS2 gates, model grades' design in
        one assertion."""
        set_agent_client(
            agent_client_returning(
                {
                    "assessable": True,
                    "severity": "medium",
                    "ml_in_validated_scope": True,
                    "care_plan": None,
                    "care_plan_skipped_reason": "severity is medium",
                }
            )
        )

        resp = client.post(
            "/alerts",
            json={
                "patient_ref": "ICUStay/78",
                "alert_type": "news2_escalation",
                "severity": "high",
                "message": "NEWS2 9",
            },
        )

        body = resp.json()
        assert body["regraded_from"] == "high"
        assert body["alert"]["severity"] == "medium"
        assert body["notification"]["email"] is None, "medium severity must not page"
        # The alert itself is untouched by the downgrade -- it is still live.
        assert client.get("/alerts/active").json()[0]["patient_ref"] == "ICUStay/78"
