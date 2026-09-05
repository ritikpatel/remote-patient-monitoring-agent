import numpy as np
import pytest
from services.contracts.observation import QualityFlag
from warehouse.news2 import hr_score, spo2_score

from simulators.morphing import (
    HR_BREAKPOINTS,
    SPO2_BREAKPOINTS,
    MorphConfig,
    morph_session,
    to_observations,
)
from simulators.wearable_replay import DEFAULT_ROOT

pytestmark = pytest.mark.skipif(
    not DEFAULT_ROOT.exists(), reason="wearable dataset not found under data/raw/"
)


def test_breakpoints_round_trip_through_real_news2_thresholds():
    for subscore, hr in HR_BREAKPOINTS.items():
        assert hr_score(hr) == subscore
    for subscore, spo2 in SPO2_BREAKPOINTS.items():
        assert spo2_score(spo2) == subscore


def test_trajectory_is_monotonic_between_start_and_end_score():
    config = MorphConfig(start_score=0, end_score=6, gamma=1.0)
    frac = np.linspace(0, 1, 20)
    traj = config.trajectory(frac)
    assert np.all(np.diff(traj) >= 0)
    assert traj[0] == pytest.approx(0)
    assert traj[-1] == pytest.approx(6)


def test_hr_rises_and_spo2_falls_over_the_replayed_window():
    config = MorphConfig(start_score=0, end_score=8)
    channels = morph_session("STRESS", "S01", config, duration_s=30.0)
    hr = channels["hr"].values
    spo2 = channels["spo2"].values
    # allow for real-recording noise: compare early vs late thirds, not point-to-point
    third = len(hr) // 3
    assert hr[-third:].mean() > hr[:third].mean()
    third_s = len(spo2) // 3
    assert spo2[-third_s:].mean() < spo2[:third_s].mean()


def test_hrv_is_suppressed_by_the_end_of_the_window():
    config = MorphConfig(
        start_score=0, end_score=8, hrv_suppression_start=1.0, hrv_suppression_end=0.2
    )
    channels = morph_session("STRESS", "S01", config, duration_s=120.0)
    ibi = channels["ibi"].values
    half = len(ibi) // 2
    early_std = np.std(ibi[:half])
    late_std = np.std(ibi[half:])
    assert late_std < early_std


def test_every_observation_is_watermarked_synthetic():
    config = MorphConfig(start_score=0, end_score=6)
    channels = morph_session("STRESS", "S01", config, duration_s=10.0)
    obs = to_observations(channels, "S01")
    assert obs, "expected at least one observation"
    assert all(QualityFlag.synthetic in o.quality_flags for o in obs)
    assert all(o.device_id == "morph-sim" for o in obs)


def test_window_is_anchored_to_the_replayed_slice_not_the_whole_recording():
    """A short --duration-s window should reach close to end_score by its own end,
    not still be near start_score because the trajectory spans the full ~30min
    source recording.
    """
    config = MorphConfig(start_score=0, end_score=8)
    channels = morph_session("STRESS", "S01", config, duration_s=10.0)
    # at end_score=8, HR target -> ~136 (interpolated between breakpoints 2->120, 3->145)
    assert channels["hr"].values[-5:].mean() > 100
