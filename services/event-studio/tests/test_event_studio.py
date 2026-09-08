"""event-studio: the generator must agree with the scorer the pipeline uses."""

from __future__ import annotations

import sys
from pathlib import Path

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
