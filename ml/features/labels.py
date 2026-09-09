"""Composite hourly deterioration outcome (PROJECT_PLAN.md section 11).

Primary task: predict, for every row of ``capstone.hourly_grid``, whether a
composite deterioration event -- death, vasopressor initiation, invasive
ventilation initiation, or unplanned ICU readmission -- occurs within the next
6h or 12h. This is the only adequately powered task in this cohort
(**E2**: ~12,004 patient-hours vs 140 stays; the whole-stay outcomes have only
20 ICU deaths and 53 readmissions -- see ``ml/README.md``).

Design decisions, and why (each is a place a naive implementation goes wrong):

* **Death is attributed to the LAST ICU stay of the hospitalisation, not every
  stay under that admission.** ``mimiciv_derived.icustay_detail`` propagates
  ``hospital_expire_flag`` from the admission to every ICU stay under it. A
  patient readmitted to the ICU within one hospitalisation before dying has
  *every* one of those stays flagged, but only the stay actually in progress
  at ``deathtime`` should carry the death event -- the earlier stay(s) ended in
  a live transfer. Concretely: subject 10006053 (hadm 22942076) has two ICU
  stays; the first ends hours before the second even begins, yet both carry
  the flag. Attributing the event to the first stay would label it "about to
  die" hours before its own (live) transfer out. We attribute to
  ``max(icustay_seq)`` per ``hadm_id``, which collapses the naive 20
  flag-carrying stays (E4/E6's headline count) to 15 distinct hospitalisations
  -- the number of *events*, not the number of *rows the flag touches*.
* **Vasopressor / ventilation are "initiation" events**, i.e. the first
  ``starttime`` per stay in ``mimiciv_derived.vasoactive_agent`` /
  ``mimiciv_derived.ventilation``. Ventilation is restricted to
  ``ventilation_status = 'InvasiveVent'`` -- the table also carries
  SupplementalOxygen/HFNC/NonInvasiveVent, which are much lower-acuity and
  would dilute "ventilation" into "any oxygen device."
* **Unplanned ICU readmission** is a same-``hadm_id`` return to the ICU within
  72h of the outgoing stay's ``icu_outtime`` (a standard ICU-quality-metric
  window; see e.g. Society of Critical Care Medicine readmission audits). This
  is a genuinely different concept from the EDA's "53 readmissions" figure
  (E6), which counts 30-day *hospital* readmission at the whole-admission
  level -- the composite task needs an ICU-to-ICU bounce-back with a
  timestamp, not a hospital-level flag. In this 140-stay cohort, 9 outgoing
  stays qualify.
* **R1 -- windows are anchored to hours-since-ICU-admission** (``hourly_grid``'s
  own ``hour`` column is already exactly that), never to a stay-relative
  position, and **"evaluated only on windows ending strictly before the
  outcome"**: once a stay's first composite event fires at hour ``h_event``,
  every row at ``hour >= h_event`` is dropped from that stay, not labelled 0.
  A row at or after the event is not a fresh "will this deteriorate" question
  -- it is inside (or past) the deterioration already, and keeping it as a
  labelled non-event would silently teach a model that unfolding organ
  failure looks like health.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

READMISSION_WINDOW_HOURS = 72.0
HORIZONS_HOURS = (6, 12)


@dataclass(frozen=True)
class EventTable:
    """One row per stay_id that ever has this component event, with its
    first occurrence time. A stay absent from the table never has it.
    """

    name: str
    frame: pd.DataFrame  # columns: stay_id, event_time


def death_events(conn: duckdb.DuckDBPyConnection) -> EventTable:
    df = conn.execute("""
        with flagged as (
            select d.hadm_id, d.stay_id, d.icustay_seq, a.deathtime,
                   row_number() over (
                       partition by d.hadm_id order by d.icustay_seq desc
                   ) as rn
            from mimiciv_derived.icustay_detail d
            join mimiciv_hosp.admissions a using (hadm_id)
            where d.hospital_expire_flag = 1
        )
        select stay_id, deathtime as event_time
        from flagged
        where rn = 1
        """).fetchdf()
    return EventTable("death", df)


def vasopressor_events(conn: duckdb.DuckDBPyConnection) -> EventTable:
    df = conn.execute("""
        select stay_id, min(starttime) as event_time
        from mimiciv_derived.vasoactive_agent
        group by stay_id
        """).fetchdf()
    return EventTable("vasopressor", df)


def ventilation_events(conn: duckdb.DuckDBPyConnection) -> EventTable:
    df = conn.execute("""
        select stay_id, min(starttime) as event_time
        from mimiciv_derived.ventilation
        where ventilation_status = 'InvasiveVent'
        group by stay_id
        """).fetchdf()
    return EventTable("ventilation", df)


def icu_readmission_events(
    conn: duckdb.DuckDBPyConnection, window_hours: float = READMISSION_WINDOW_HOURS
) -> EventTable:
    df = conn.execute(
        """
        with seq as (
            select hadm_id, stay_id, icustay_seq, icu_intime, icu_outtime
            from mimiciv_derived.icustay_detail
        )
        select a.stay_id as stay_id, b.icu_intime as event_time
        from seq a
        join seq b on a.hadm_id = b.hadm_id and b.icustay_seq = a.icustay_seq + 1
        where date_diff('minute', a.icu_outtime, b.icu_intime) / 60.0 <= ?
        """,
        [window_hours],
    ).fetchdf()
    return EventTable("icu_readmission", df)


def all_event_tables(conn: duckdb.DuckDBPyConnection) -> list[EventTable]:
    return [
        death_events(conn),
        vasopressor_events(conn),
        ventilation_events(conn),
        icu_readmission_events(conn),
    ]


def first_composite_event_per_stay(tables: list[EventTable]) -> pd.DataFrame:
    """Collapse the per-type event tables to one row per stay: the earliest
    event of any type, and which type it was. A stay with no row in any table
    never composite-deteriorates in this cohort.
    """
    frames = []
    for t in tables:
        if t.frame.empty:
            continue
        f = t.frame.copy()
        f["event_type"] = t.name
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["stay_id", "event_time", "event_type"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["stay_id", "event_time"])
    first = combined.groupby("stay_id", as_index=False).first()
    return first[["stay_id", "event_time", "event_type"]]


def build_labels(
    conn: duckdb.DuckDBPyConnection,
    hourly_grid: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS_HOURS,
) -> pd.DataFrame:
    """Return ``hourly_grid`` restricted to (stay_id, hour) rows still at risk
    (R1 censoring applied), with one ``label_{h}h`` column per horizon plus
    ``composite_event_type`` for the rows that are themselves inside the
    lookback of an eventual event (NaN for rows with no event ahead at all).
    """
    intime = conn.execute(
        "select stay_id, icu_intime from mimiciv_derived.icustay_detail"
    ).fetchdf()
    first_event = first_composite_event_per_stay(all_event_tables(conn))

    grid = hourly_grid[["stay_id", "hour"]].merge(intime, on="stay_id", how="left")
    grid = grid.merge(first_event, on="stay_id", how="left")
    grid["row_start"] = grid["icu_intime"] + pd.to_timedelta(grid["hour"], unit="h")

    has_event = grid["event_time"].notna()
    seconds_to_event = (grid["event_time"] - grid["row_start"]).dt.total_seconds()
    # R1: censor rows at/after the event -- they are no longer a valid
    # "will this deteriorate" question for this stay.
    at_risk = ~has_event | (seconds_to_event > 0)
    grid = grid[at_risk].copy()
    seconds_to_event = seconds_to_event[at_risk]
    has_event = has_event[at_risk]

    for h in horizons:
        within_horizon = has_event & (seconds_to_event <= h * 3600)
        grid[f"label_{h}h"] = within_horizon.astype(int)

    return grid[
        ["stay_id", "hour", "event_time", "event_type", *[f"label_{h}h" for h in horizons]]
    ].rename(columns={"event_time": "composite_event_time", "event_type": "composite_event_type"})


def label_summary(labels: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS_HOURS) -> pd.DataFrame:
    """One row per horizon: positives, total at-risk rows, prevalence -- the
    numbers that justify (or refute) 'adequately powered' for that horizon.
    """
    rows = []
    for h in horizons:
        col = f"label_{h}h"
        positives = int(labels[col].sum())
        total = len(labels)
        rows.append(
            {
                "horizon_h": h,
                "positive_hours": positives,
                "total_hours": total,
                "prevalence": positives / total if total else 0.0,
                "positive_stays": labels.loc[labels[col] == 1, "stay_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)
