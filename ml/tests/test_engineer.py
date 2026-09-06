"""Tests for ml/features/engineer.py."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from ml.features import engineer


def test_rolling_features_mean_std_slope_are_causal_and_per_stay() -> None:
    grid = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 2, 2],
            "hour": [0, 1, 2, 3, 0, 1],
            **{v: [0.0] * 6 for v in engineer.CORE_VITALS},
        }
    )
    grid["hr"] = [60.0, 70.0, 80.0, 90.0, 100.0, 100.0]
    out = engineer.add_rolling_features(grid, windows=(2,))

    # Stay 1, hour 1: trailing 2h window = hours [0, 1] = [60, 70].
    row = out[(out.stay_id == 1) & (out.hour == 1)].iloc[0]
    assert row["hr_2h_mean"] == pytest.approx(65.0)
    assert row["hr_2h_slope"] == pytest.approx(10.0)  # rising 10 bpm/hour

    # Stay 2 must never see stay 1's values.
    row2 = out[(out.stay_id == 2) & (out.hour == 0)].iloc[0]
    assert row2["hr_2h_mean"] == pytest.approx(100.0)
    assert row2["hr_2h_std"] == pytest.approx(0.0)


def _synthetic_conn_for_static_features() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute("create schema mimiciv_derived")
    conn.execute("create schema mimiciv_icu")
    conn.execute(
        """
        create table mimiciv_derived.icustay_detail as select * from (values
            (1, 65, 'M', timestamp '2100-01-01 00:00:00')
        ) as t(stay_id, admission_age, gender, icu_intime)
        """
    )
    conn.execute(
        """
        create table mimiciv_icu.icustays as select * from (values
            (1, 'Medical Intensive Care Unit')
        ) as t(stay_id, first_careunit)
        """
    )
    conn.execute(
        """
        create table mimiciv_derived.invasive_line as select * from (values
            (1, 'Arterial', timestamp '2100-01-01 02:00:00', timestamp '2100-01-01 08:00:00')
        ) as t(stay_id, line_type, starttime, endtime)
        """
    )
    return conn


def test_arterial_line_feature_is_hour_resolved_not_a_stay_level_flag() -> None:
    conn = _synthetic_conn_for_static_features()
    grid = pd.DataFrame({"stay_id": [1] * 10, "hour": range(10)})
    out = engineer.add_arterial_line_feature(grid, conn)
    # Line active hour 2..8 inclusive (start_h=2, end_h=8).
    assert out[out.hour == 1].iloc[0].has_arterial_line == 0
    assert out[out.hour == 2].iloc[0].has_arterial_line == 1
    assert out[out.hour == 8].iloc[0].has_arterial_line == 1
    assert out[out.hour == 9].iloc[0].has_arterial_line == 0


def test_static_features_join_age_gender_careunit() -> None:
    conn = _synthetic_conn_for_static_features()
    grid = pd.DataFrame({"stay_id": [1, 1], "hour": [0, 1]})
    out = engineer.add_static_features(grid, conn)
    assert (out.admission_age == 65).all()
    assert (out.gender == "M").all()
    assert (out.first_careunit == "Medical Intensive Care Unit").all()


@pytest.mark.skipif(
    not (
        __import__("pathlib").Path(__file__).resolve().parent.parent.parent
        / "warehouse"
        / "mimic4_demo.db"
    ).exists(),
    reason="real warehouse db not built",
)
def test_build_feature_frame_against_real_warehouse_has_no_infinite_values() -> None:
    from pathlib import Path

    db_path = Path(__file__).resolve().parent.parent.parent / "warehouse" / "mimic4_demo.db"
    conn = duckdb.connect(str(db_path), read_only=True)
    features = engineer.build_feature_frame(conn)
    assert len(features) == 12_004
    numeric = features.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()
