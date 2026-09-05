import pytest

from simulators.wearable_replay import (
    DEFAULT_ROOT,
    build_session_index,
    find_session,
    load_session,
    to_observations,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_ROOT.exists(), reason="wearable dataset not found under data/raw/"
)


def test_session_index_finds_all_three_split_sessions():
    sessions = build_session_index()
    splits = {(s.activity, s.participant) for s in sessions if s.is_split}
    assert splits == {("STRESS", "f14"), ("AEROBIC", "S11"), ("ANAEROBIC", "S16")}


def test_session_index_finds_both_fault_fixtures():
    sessions = build_session_index()
    faults = {(s.activity, s.participant) for s in sessions if s.is_fault_fixture}
    assert faults == {("STRESS", "S02"), ("STRESS", "f07")}


def test_f07_flags_bvp_and_temp_but_not_eda():
    session = find_session(DEFAULT_ROOT, "STRESS", "f07")
    channels = load_session(session, DEFAULT_ROOT, ["bvp", "temp_c", "eda"])
    assert all("device_fault" in [f.value for f in fl] for fl in channels["bvp"].quality_flags)
    assert all("device_fault" in [f.value for f in fl] for fl in channels["temp_c"].quality_flags)
    assert not any(fl for fl in channels["eda"].quality_flags)


def test_s02_duplicate_flag_starts_at_documented_row():
    session = find_session(DEFAULT_ROOT, "STRESS", "S02")
    channels = load_session(session, DEFAULT_ROOT, ["eda"])
    flags = channels["eda"].quality_flags
    # documented line 6,195 minus the 2 header lines -> 0-indexed sample 6193
    assert flags[6192] == []
    assert flags[6193] == ["duplicate"] or flags[6193][0].value == "duplicate"


def test_split_session_concatenates_in_order():
    session = find_session(DEFAULT_ROOT, "STRESS", "f14")
    channels = load_session(session, DEFAULT_ROOT, ["hr"])
    times = channels["hr"].times
    assert (times[:-1] <= times[1:]).all(), "concatenated session must be time-ordered"


def test_to_observations_respects_duration_window():
    session = find_session(DEFAULT_ROOT, "STRESS", "S01")
    channels = load_session(session, DEFAULT_ROOT, ["bvp", "hr"])
    obs = to_observations(channels, "S01", start_offset_s=0, duration_s=2.0)
    span = (obs[-1].effective_time - obs[0].effective_time).total_seconds()
    assert span <= 2.01


def test_wearable_observations_use_subject_not_patient_reference():
    session = find_session(DEFAULT_ROOT, "STRESS", "S01")
    channels = load_session(session, DEFAULT_ROOT, ["hr"])
    obs = to_observations(channels, "S01", duration_s=1.0)
    assert all(o.patient_ref.startswith("Subject/") for o in obs)
