from __future__ import annotations

from pathlib import Path

import pytest

from reports.post_discharge_digest import N_DEMO_DAYS, build_post_discharge_digest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEARABLE_ROOT = (
    REPO_ROOT
    / "data"
    / "raw"
    / "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"
)

pytestmark = pytest.mark.skipif(not WEARABLE_ROOT.exists(), reason="wearable dataset not present")


def test_build_post_discharge_digest_produces_seven_daily_buckets() -> None:
    result = build_post_discharge_digest(
        "STRESS", "S01", llm=None, start_score=0, end_score=6, duration_s=600
    )
    assert len(result.daily_buckets) == N_DEMO_DAYS
    assert [b.day_index for b in result.daily_buckets] == list(range(1, N_DEMO_DAYS + 1))


def test_build_post_discharge_digest_shows_the_configured_deterioration_direction() -> None:
    """HR should trend up and SpO2 down across the week -- the whole point of
    the morphing config -- not merely be present."""
    result = build_post_discharge_digest(
        "STRESS", "S01", llm=None, start_score=0, end_score=6, duration_s=600
    )
    first, last = result.daily_buckets[0], result.daily_buckets[-1]
    assert last.mean_hr > first.mean_hr
    assert last.mean_spo2 < first.mean_spo2


def test_build_post_discharge_digest_narrative_names_it_as_morphed_when_llm_absent() -> None:
    result = build_post_discharge_digest("STRESS", "S01", llm=None, duration_s=600)
    assert result.narrative.startswith("[no LLM configured]")
    assert "morphed" in result.narrative
