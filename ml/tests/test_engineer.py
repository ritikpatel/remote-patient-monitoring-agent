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
    conn.execute("""
        create table mimiciv_derived.icustay_detail as select * from (values
            (1, 900, 65, 'M', timestamp '2100-01-01 00:00:00'),
            (2, 900, 65, 'M', timestamp '2100-02-01 00:00:00')
        ) as t(stay_id, subject_id, admission_age, gender, icu_intime)
        """)
    conn.execute("""
        create table mimiciv_icu.icustays as select * from (values
            (1, 'Medical Intensive Care Unit'),
            (2, 'Medical Intensive Care Unit')
        ) as t(stay_id, first_careunit)
        """)
    conn.execute("""
        create table mimiciv_derived.invasive_line as select * from (values
            (1, 'Arterial', timestamp '2100-01-01 02:00:00', timestamp '2100-01-01 08:00:00')
        ) as t(stay_id, line_type, starttime, endtime)
        """)
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


def _feature_frame(stay_ids: list[int], subject_ids: list[int]) -> pd.DataFrame:
    """A feature frame with every column `feature_matrix_for_training` selects.

    Built from the module's own column lists rather than by running the real
    builders, so the test exercises the grouping contract without needing a
    warehouse -- and stays correct if a feature is added, because the column set
    is read from `engineer`, not restated here.
    """
    n = len(stay_ids)
    frame = pd.DataFrame(
        {
            "stay_id": stay_ids,
            "subject_id": subject_ids,
            "hour": list(range(n)),
            "gender": ["M"] * n,
            "first_careunit": ["Medical Intensive Care Unit"] * n,
        }
    )
    for col in (
        list(engineer.FEATURE_COLUMNS_BASE)
        + engineer.rolling_feature_columns()
        # Disease context (warehouse/disease.py) joined the selected column set when
        # the platform became disease-aware. Read from `engineer` for the same reason
        # as the other two lists: so this helper keeps working when the default
        # disease feature set changes, rather than pinning one of them here.
        + engineer.disease_feature_columns()
    ):
        if col not in frame.columns:
            frame[col] = 1.0
    return frame


def test_feature_matrix_groups_by_subject_not_stay() -> None:
    """The grouping unit CV splits on. Two stays of one patient must return the
    SAME group, or that patient lands in train and test at once.

    This is a regression test for a leak that was live for the whole project:
    `groups` used to be `stay_id`, while `models/splits.py` documented
    subject-level grouping and offered a `group_key()` helper that nothing ever
    called with a subject id. In the demo cohort 21 of 93 at-risk subjects have
    more than one stay, carrying 44% of the positives.
    """
    features = _feature_frame(stay_ids=[1, 1, 2, 2], subject_ids=[900, 900, 900, 900])
    labels_df = pd.DataFrame(
        {"stay_id": [1, 1, 2, 2], "hour": [0, 1, 0, 1], "label_6h": [0, 1, 0, 1]}
    )

    _x, _y, groups = engineer.feature_matrix_for_training(features, labels_df, "label_6h")

    assert groups.nunique() == 1, "stays 1 and 2 belong to subject 900 and must share a group"
    assert set(groups) == {900}


def test_subject_id_is_a_grouping_key_not_a_feature() -> None:
    """Carrying subject_id through the feature frame must not let it reach the
    model -- a patient identifier is a perfect in-sample predictor and pure
    leakage if it is ever fitted on.
    """
    features = _feature_frame(stay_ids=[1, 1], subject_ids=[900, 900])
    labels_df = pd.DataFrame({"stay_id": [1, 1], "hour": [0, 1], "label_6h": [0, 1]})

    x, _y, _groups = engineer.feature_matrix_for_training(features, labels_df, "label_6h")
    assert "subject_id" not in x.columns
    assert "stay_id" not in x.columns


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
