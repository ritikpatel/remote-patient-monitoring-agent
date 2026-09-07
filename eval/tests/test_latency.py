from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from eval.latency import (
    LATENCY_BAR_MS,
    LatencyResult,
    events_per_sec_for_device_count,
    k6_available,
    peak_events_per_patient_per_hour,
    run_k6_tier,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_peak_events_per_patient_per_hour_matches_e12_within_rounding() -> None:
    # E12: "peak 33.7 ev/pt/hr" -- computed here from the same fitted
    # arrival_models.json, not hard-coded, so this pins it to roughly that
    # figure without assuming the exact fitted value never moves.
    peak = peak_events_per_patient_per_hour()
    assert 30 < peak < 40


def test_events_per_sec_scales_linearly_with_device_count() -> None:
    rate_10 = events_per_sec_for_device_count(10)
    rate_100 = events_per_sec_for_device_count(100)
    assert rate_100 == pytest.approx(rate_10 * 10)


def test_latency_result_meets_p95_bar_is_none_when_no_data() -> None:
    result = LatencyResult(10, 1.0, None, None, None, 0)
    assert result.meets_p95_bar is None
    assert result.all_checks_passed is False  # zero checks recorded at all


def test_latency_result_flags_a_bar_violation() -> None:
    result = LatencyResult(1000, 9.4, 100, LATENCY_BAR_MS + 1, LATENCY_BAR_MS + 500, 100, 300, 0)
    assert result.meets_p95_bar is False


def test_latency_result_all_checks_passed_is_false_on_any_failure() -> None:
    result = LatencyResult(10, 1.0, 50, 60, 70, 10, checks_passed=29, checks_failed=1)
    assert result.all_checks_passed is False


@pytest.mark.skipif(not k6_available(), reason="k6 not installed")
def test_run_k6_tier_returns_a_clear_error_when_services_are_unreachable() -> None:
    # No services running at these ports (deliberately unlikely to be in use) --
    # k6 itself still runs and produces a summary; every check fails, which
    # must be visible in the result, not silently swallowed as "0 failures".
    result = run_k6_tier(
        10,
        duration="3s",
        base_urls={
            "ingest": "http://localhost:19999",
            "stream": "http://localhost:19999",
            "risk": "http://localhost:19999",
            "alert": "http://localhost:19999",
            "notify": "http://localhost:19999",
        },
    )
    assert result.error is None  # k6 ran fine; the *requests* failed, not k6 itself
    assert result.checks_failed > 0
    assert result.all_checks_passed is False


def _services_reachable() -> bool:
    try:
        return httpx.get("http://localhost:8001/health", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not k6_available(), reason="k6 not installed")
@pytest.mark.skipif(
    not _services_reachable(),
    reason="pipeline services not running locally -- see eval/README.md to start them",
)
def test_run_k6_tier_against_real_running_services() -> None:
    result = run_k6_tier(10, duration="5s")
    assert result.error is None
    assert result.n_iterations > 0
    assert result.all_checks_passed
    assert result.p95_ms is not None
