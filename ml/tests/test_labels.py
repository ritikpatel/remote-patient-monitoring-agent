"""Tests for ml/features/labels.py.

Two layers, matching the pattern used for the warehouse concepts in Phase 1:
synthetic-schema unit tests that pin down each event type's exact semantics
in isolation, and one real-warehouse regression test that pins down the
actual numbers this cohort produces (so a future change to the composite
definition is forced to explain itself against a known baseline, the same
role ``hourly_grid.py``'s ``EXPECTED_ROWS`` plays).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from ml.features import labels

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"


def _synthetic_conn() -> duckdb.DuckDBPyConnection:
    """An in-memory warehouse with just enough schema for labels.py -- two
    hospitalisations exercising the death-attribution edge case, one
    vasopressor stay, one ventilation stay, and one same-hadm ICU bounce-back.
    """
    conn = duckdb.connect(":memory:")
    conn.execute("create schema mimiciv_derived")
    conn.execute("create schema mimiciv_hosp")

    # hadm 1: two ICU stays, patient dies during the SECOND one. Stay 101's
    # own transfer-out was a live discharge to the ward, not death.
    # hadm 2: a single stay, no death.
    # hadm 3: two ICU stays 45h apart -- inside the 72h readmission window.
    conn.execute(
        """
        create table mimiciv_derived.icustay_detail as select * from (values
            (1, 100, 101, 1, timestamp '2100-01-01 00:00:00', timestamp '2100-01-02 00:00:00', 1),
            (1, 100, 102, 2, timestamp '2100-01-10 00:00:00', timestamp '2100-01-12 00:00:00', 1),
            (2, 200, 201, 1, timestamp '2100-02-01 00:00:00', timestamp '2100-02-04 00:00:00', 0),
            (3, 300, 301, 1, timestamp '2100-03-01 00:00:00', timestamp '2100-03-02 00:00:00', 0),
            (3, 300, 302, 2, timestamp '2100-03-04 00:00:00', timestamp '2100-03-06 00:00:00', 0)
        ) as t(subject_id, hadm_id, stay_id, icustay_seq, icu_intime, icu_outtime,
               hospital_expire_flag)
        """
    )
    conn.execute(
        """
        create table mimiciv_hosp.admissions as select * from (values
            (100, timestamp '2100-01-11 12:00:00'),
            (200, cast(null as timestamp)),
            (300, cast(null as timestamp))
        ) as t(hadm_id, deathtime)
        """
    )
    conn.execute(
        """
        create table mimiciv_derived.vasoactive_agent as select * from (values
            (201, timestamp '2100-02-01 05:00:00', timestamp '2100-02-01 08:00:00', 1.0),
            (201, timestamp '2100-02-01 08:00:00', timestamp '2100-02-01 12:00:00', 1.0)
        ) as t(stay_id, starttime, endtime, norepinephrine)
        """
    )
    conn.execute(
        """
        create table mimiciv_derived.ventilation as select * from (values
            (301, timestamp '2100-03-01 10:00:00', timestamp '2100-03-01 20:00:00',
             'InvasiveVent'),
            (301, timestamp '2100-03-01 02:00:00', timestamp '2100-03-01 04:00:00',
             'SupplementalOxygen')
        ) as t(stay_id, starttime, endtime, ventilation_status)
        """
    )
    return conn


def test_death_attributed_only_to_the_stay_in_progress_at_death() -> None:
    conn = _synthetic_conn()
    events = labels.death_events(conn).frame
    assert list(events.stay_id) == [102]
    assert events.iloc[0].event_time == pd.Timestamp("2100-01-11 12:00:00")


def test_vasopressor_events_take_the_first_starttime() -> None:
    conn = _synthetic_conn()
    events = labels.vasopressor_events(conn).frame
    row = events[events.stay_id == 201].iloc[0]
    assert row.event_time == pd.Timestamp("2100-02-01 05:00:00")


def test_ventilation_events_ignore_non_invasive_status() -> None:
    conn = _synthetic_conn()
    events = labels.ventilation_events(conn).frame
    row = events[events.stay_id == 301].iloc[0]
    # 10:00, not the 02:00 SupplementalOxygen row.
    assert row.event_time == pd.Timestamp("2100-03-01 10:00:00")


def test_icu_readmission_within_window_and_not_beyond() -> None:
    conn = _synthetic_conn()
    events = labels.icu_readmission_events(conn, window_hours=72).frame
    assert list(events.stay_id) == [301]  # 301 -> 302 gap is 48h, inside 72h
    events_tight = labels.icu_readmission_events(conn, window_hours=24).frame
    assert events_tight.empty  # 48h gap now excluded


def test_build_labels_censors_rows_at_and_after_the_event() -> None:
    conn = _synthetic_conn()
    # Stay 201's vasopressor starts 5h after intime -- hourly_grid rows exist
    # for hours 0..9.
    grid = pd.DataFrame({"stay_id": [201] * 10, "hour": range(10)})
    result = labels.build_labels(conn, grid, horizons=(6, 12))

    # Hours 5..9 are at/after the event and must be dropped entirely.
    assert set(result.hour) == set(range(5))
    # Hour 0 is 5h before the event -> inside the 6h window, outside no window
    # shorter than 5h.
    row0 = result[result.hour == 0].iloc[0]
    assert row0.label_6h == 1
    assert row0.label_12h == 1
    # Hour 4 is 1h before the event -> also inside both windows.
    row4 = result[result.hour == 4].iloc[0]
    assert row4.label_6h == 1


def test_build_labels_leaves_eventless_stays_fully_labelled_negative() -> None:
    conn = _synthetic_conn()
    # Stay 101 has no vasopressor/ventilation/readmission of its own, and its
    # hadm's death event belongs to the *later* stay (102) -- 101 itself must
    # come through with no event and an all-negative label.
    grid = pd.DataFrame({"stay_id": [101] * 5, "hour": range(5)})
    result = labels.build_labels(conn, grid, horizons=(6, 12))
    assert len(result) == 5
    assert (result.label_6h == 0).all()
    assert (result.label_12h == 0).all()
    assert result.composite_event_type.isna().all()


@pytest.mark.skipif(not WAREHOUSE_DB.exists(), reason="real warehouse db not built")
def test_build_labels_against_real_warehouse_matches_known_counts() -> None:
    """Regression pin: if the composite definition changes, this must change
    with it, deliberately -- not drift silently. See ml/README.md for how
    these numbers were derived and why the raw 12,004-row hourly grid shrinks
    to this size once R1 censoring is applied.
    """
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    grid = conn.execute("select stay_id, hour from capstone.hourly_grid").fetchdf()
    result = labels.build_labels(conn, grid)
    summary = labels.label_summary(result)

    assert len(result) == 2979
    six_h = summary[summary.horizon_h == 6].iloc[0]
    twelve_h = summary[summary.horizon_h == 12].iloc[0]
    assert six_h.positive_hours == 120
    assert twelve_h.positive_hours == 161
    assert six_h.positive_stays == 60
