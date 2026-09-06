"""Tests for ml/features/ecg.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import ecg


def test_paired_mean_diff_ms_computes_duration_in_milliseconds() -> None:
    onsets = np.array([100.0, 600.0, 1100.0])
    offsets = np.array([180.0, 680.0, 1180.0])  # 80 samples every time
    fs = 500.0
    result = ecg._paired_mean_diff_ms(onsets, offsets, fs)
    assert result == pytest.approx(80.0 / 500.0 * 1000.0)  # 160 ms


def test_paired_mean_diff_ms_returns_none_when_no_pairs() -> None:
    assert ecg._paired_mean_diff_ms(np.array([]), np.array([]), 500.0) is None


def test_attach_nearest_ecg_only_matches_backward_within_tolerance() -> None:
    grid = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "hour": [0, 1, 2],
            "subject_id": [10, 10, 10],
            "row_abs_time": pd.to_datetime(
                ["2100-01-01 00:00", "2100-01-01 01:00", "2100-01-01 10:00"]
            ),
        }
    )
    ecg_features = pd.DataFrame(
        {
            "subject_id": [10],
            "ecg_time": pd.to_datetime(["2100-01-01 00:30"]),
            "ecg_hr_mean": [90.0],
        }
    )
    out = ecg.attach_nearest_ecg(grid, ecg_features, max_lookback_h=2.0)
    # hour 0 (00:00) is BEFORE the ECG (00:30) -- backward-only, no match.
    assert pd.isna(out[out.hour == 0].iloc[0].ecg_hr_mean)
    # hour 1 (01:00) is 0.5h after -- within 2h tolerance, matches.
    assert out[out.hour == 1].iloc[0].ecg_hr_mean == pytest.approx(90.0)
    # hour 2 (10:00) is 9.5h after -- outside the 2h tolerance, no match.
    assert pd.isna(out[out.hour == 2].iloc[0].ecg_hr_mean)


def test_attach_nearest_ecg_excludes_the_join_key_from_feature_columns() -> None:
    """ecg_time itself starts with 'ecg_' but is a timestamp, not a feature --
    it must never end up in the returned frame (a downstream model would
    choke trying to promote a datetime64 column into a numeric matrix)."""
    grid = pd.DataFrame(
        {
            "stay_id": [1],
            "hour": [0],
            "subject_id": [10],
            "row_abs_time": pd.to_datetime(["2100-01-01 01:00"]),
        }
    )
    ecg_features = pd.DataFrame(
        {
            "subject_id": [10],
            "ecg_time": pd.to_datetime(["2100-01-01 00:00"]),
            "ecg_hr_mean": [80.0],
        }
    )
    out = ecg.attach_nearest_ecg(grid, ecg_features, max_lookback_h=24.0)
    assert "ecg_time" not in out.columns
    assert "ecg_hr_mean" in out.columns


def test_attach_nearest_ecg_tolerates_mismatched_subject_id_dtypes() -> None:
    grid = pd.DataFrame(
        {
            "stay_id": [1],
            "hour": [0],
            "subject_id": pd.array([10], dtype="int32"),
            "row_abs_time": pd.to_datetime(["2100-01-01 01:00"]),
        }
    )
    ecg_features = pd.DataFrame(
        {
            "subject_id": pd.array([10], dtype="int64"),
            "ecg_time": pd.to_datetime(["2100-01-01 00:00"]),
            "ecg_hr_mean": [80.0],
        }
    )
    out = ecg.attach_nearest_ecg(grid, ecg_features, max_lookback_h=24.0)
    assert out.iloc[0].ecg_hr_mean == pytest.approx(80.0)


ECG_ROOT = ecg.DEFAULT_ECG_ROOT


@pytest.mark.skipif(not ECG_ROOT.exists(), reason="ECG waveform dataset not present")
def test_extract_features_for_record_on_a_real_waveform() -> None:
    record_list = ecg.load_record_list()
    features = ecg.extract_features_for_record(ECG_ROOT / record_list.iloc[0].path)
    assert features is not None
    assert 20 < features["ecg_hr_mean"] < 220  # a physiologically plausible rate
    assert features["ecg_qrs_ms"] > 0
