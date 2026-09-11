import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from windowing import (  # noqa: E402
    hrv_rmssd,
    load_stationary_rate,
    normalized_event_rate,
    rolling_stats,
    trend_slope,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ARRIVAL_MODELS = REPO_ROOT / "simulators" / "arrival_models.json"


def test_rolling_stats_basic():
    stats = rolling_stats([1.0, 2.0, 3.0, 4.0])
    assert stats.mean == 2.5
    assert stats.n == 4
    assert stats.std == pytest.approx(1.1180339887, rel=1e-6)


def test_trend_slope_perfectly_linear():
    slope = trend_slope([0, 1, 2, 3], [10, 12, 14, 16])
    assert slope == pytest.approx(2.0)


def test_trend_slope_flat_is_zero():
    assert trend_slope([0, 1, 2], [50, 50, 50]) == pytest.approx(0.0)


def test_trend_slope_single_point_returns_zero():
    assert trend_slope([0], [50]) == 0.0


def test_hrv_rmssd_constant_ibi_is_zero():
    """No successive-difference variability -> RMSSD 0, the floor a maximally
    HRV-suppressed (morphed) segment approaches."""
    assert hrv_rmssd([0.8, 0.8, 0.8, 0.8]) == pytest.approx(0.0)


def test_hrv_rmssd_needs_at_least_two_values():
    assert hrv_rmssd([0.8]) == 0.0
    assert hrv_rmssd([]) == 0.0


def test_hrv_rmssd_matches_the_hand_computed_value():
    """Pins the metric's definition against arithmetic that can be checked by eye.

    This replaces a test that read a real IBI recording out of the PhysioNet
    volunteer wearable dataset and asserted only `0 < rmssd < 500`. That dataset has
    been removed from the project, and the assertion it supported was weak enough
    that a unit error would have passed it. IBI [0.80, 0.85, 0.80, 0.90] s gives
    successive differences of [50, -50, 100] ms, mean square 5000, so RMSSD is
    sqrt(5000) = 70.71 ms -- which also fixes the seconds-to-milliseconds conversion
    that the range check could not see.
    """
    assert hrv_rmssd([0.80, 0.85, 0.80, 0.90]) == pytest.approx(70.7106781, rel=1e-6)


def test_hrv_rmssd_is_reported_in_milliseconds_not_seconds():
    """The unit is the easiest thing to get wrong here and the hardest to notice: a
    plausible-looking 0.07 would flow all the way to a clinician's digest."""
    assert hrv_rmssd([1.0, 1.1]) == pytest.approx(100.0)


@pytest.mark.skipif(
    not ARRIVAL_MODELS.exists(),
    reason="arrival_models.json not built (run simulators/arrival_models.py)",
)
def test_load_stationary_rate_matches_the_fitted_model_file():
    import json

    models = json.loads(ARRIVAL_MODELS.read_text())
    assert (
        load_stationary_rate("ICU monitoring", ARRIVAL_MODELS)
        == models["ICU monitoring"]["stationary_rate"]
    )


@pytest.mark.skipif(not ARRIVAL_MODELS.exists(), reason="arrival_models.json not built")
def test_normalized_event_rate_of_exactly_the_baseline_is_one():
    baseline = load_stationary_rate("ICU monitoring", ARRIVAL_MODELS)
    rate = normalized_event_rate(round(baseline * 2), 2.0, "ICU monitoring", ARRIVAL_MODELS)
    assert rate == pytest.approx(1.0, rel=0.05)


def test_normalized_event_rate_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        normalized_event_rate(10, 0, "ICU monitoring")
