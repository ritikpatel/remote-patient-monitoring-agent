"""Daily patient summary: trajectory, interventions, and outstanding risks
(PROJECT_PLAN.md section 12).

"Day" here is admission-relative (hours-since-admission 0-23 = day 0, 24-47 =
day 1, ...), not wall-clock -- R1: "anchor all prediction windows to hours-
since-ICU-admission, never to stay-relative position" is written about
prediction windows specifically, but the same reasoning applies to how a day
boundary is defined for a report: a wall-clock day would mean stays that
happen to start at different hours of the day get different-length first
"days," which is exactly the stay-relative leakage R1 warns about, just in a
reporting context instead of a modelling one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.features import labels  # noqa: E402

from reports.narrative import (  # noqa: E402
    ANTI_FABRICATION_INSTRUCTION,
    LLMBackend,
    generate_narrative,
)

HOURS_PER_DAY = 24

DAILY_ANTI_FABRICATION_SYSTEM = (
    "You write a daily patient summary for the clinical record from the "
    "structured facts given, covering one admission-relative day of one "
    "ICU stay. " + ANTI_FABRICATION_INSTRUCTION
)


@dataclass
class DailySummary:
    stay_id: int
    patient_ref: str
    day_index: int
    hour_range: tuple[int, int]
    news2_trajectory: list[int]
    tier_trajectory: list[str]
    vasopressor_started_today: bool
    ventilation_started_today: bool
    abnormal_lab_count: int
    outstanding_tier: str
    narrative: str


def build_daily_summary(
    conn: duckdb.DuckDBPyConnection, stay_id: int, day_index: int, llm: LLMBackend | None
) -> DailySummary:
    hour_lo, hour_hi = day_index * HOURS_PER_DAY, (day_index + 1) * HOURS_PER_DAY - 1

    news2 = conn.execute(
        "select hour, news2, tier_icu from capstone.news2 "
        "where stay_id = ? and hour between ? and ? order by hour",
        [stay_id, hour_lo, hour_hi],
    ).fetchdf()
    if news2.empty:
        raise ValueError(f"no news2 rows for stay_id={stay_id} on day {day_index}")

    hadm_row = conn.execute(
        "select hadm_id from mimiciv_icu.icustays where stay_id = ?", [stay_id]
    ).fetchone()
    hadm_id = hadm_row[0] if hadm_row else None
    abnormal_labs = 0
    if hadm_id is not None:
        labs_row = conn.execute(
            "select count(*) from mimiciv_hosp.labevents where hadm_id = ? and flag = 'abnormal'",
            [hadm_id],
        ).fetchone()
        assert labs_row is not None  # COUNT(*) always returns exactly one row
        (abnormal_labs,) = labs_row

    intime_row = conn.execute(
        "select icu_intime from mimiciv_derived.icustay_detail where stay_id = ?", [stay_id]
    ).fetchone()
    assert intime_row is not None  # news2 non-empty above implies this stay_id exists
    icu_intime = intime_row[0]
    day_start = icu_intime + pd.Timedelta(hours=hour_lo)
    day_end = icu_intime + pd.Timedelta(hours=hour_hi + 1)

    vaso = labels.vasopressor_events(conn).frame
    vent = labels.ventilation_events(conn).frame
    vaso_today = bool(
        (vaso.stay_id == stay_id).any()
        and day_start <= vaso.loc[vaso.stay_id == stay_id, "event_time"].iloc[0] < day_end
    )
    vent_today = bool(
        (vent.stay_id == stay_id).any()
        and day_start <= vent.loc[vent.stay_id == stay_id, "event_time"].iloc[0] < day_end
    )

    user = (
        f"Stay: ICUStay/{stay_id}, admission day {day_index} (hours {hour_lo}-{hour_hi})\n"
        f"NEWS2 trajectory: {list(news2.news2)}\n"
        f"ICU-recalibrated tier trajectory: {list(news2.tier_icu)}\n"
        f"Vasopressor started today: {vaso_today}\n"
        f"Ventilation started today: {vent_today}\n"
        f"Abnormal labs this admission (cumulative): {abnormal_labs}\n"
        f"Outstanding risk tier as of end of day: {news2.tier_icu.iloc[-1]}"
    )
    narrative = generate_narrative(llm, DAILY_ANTI_FABRICATION_SYSTEM, user)

    return DailySummary(
        stay_id=stay_id,
        patient_ref=f"ICUStay/{stay_id}",
        day_index=day_index,
        hour_range=(hour_lo, hour_hi),
        news2_trajectory=list(news2.news2),
        tier_trajectory=list(news2.tier_icu),
        vasopressor_started_today=vaso_today,
        ventilation_started_today=vent_today,
        abnormal_lab_count=int(abnormal_labs),
        outstanding_tier=news2.tier_icu.iloc[-1],
        narrative=narrative,
    )
