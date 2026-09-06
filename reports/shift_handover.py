"""Shift handover summary per ward, on the real 4-hourly boundary
(PROJECT_PLAN.md section 12, E16: "48.4% of events land on 00/04/08/12/16/20
vs 25% uniform... alerts cluster into six daily bursts").

**What "per ward, on the boundary" means in this dataset, precisely stated
because it is not obvious:** MIMIC-IV's de-identification shifts each
patient's calendar dates independently (by a random per-patient offset) but
preserves each event's real hour-of-day and day-of-week -- E16's periodicity
finding depends on exactly that preserved structure. It also means there is
no shared wall-clock "now" across different patients' de-identified
timelines to synchronise a ward-wide snapshot against. This report therefore
covers, per ward, **each of that ward's currently-monitored patients' own
most recent real 4-hourly wall-clock block** (their own last
00/04/08/12/16/20-aligned window, using their own -- de-identified but
internally consistent -- calendar), combined into one ward-level handover.
This is the same "latest available data stands in for now" convention
risk-engine's own `/patients` ward view uses (E1: care here is hourly-
charted, not streamed, so there is no better notion of "now" to begin with).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.features import labels  # noqa: E402
from services.common.testing import load_module  # noqa: E402

from reports.narrative import (  # noqa: E402
    ANTI_FABRICATION_INSTRUCTION,
    LLMBackend,
    generate_narrative,
)

_alert_store_module = load_module(
    REPO_ROOT / "services" / "alert-service" / "store.py", "reports_alert_service_store"
)
dedup_bucket_start = _alert_store_module.dedup_bucket_start
AlertStore = _alert_store_module.AlertStore
DEFAULT_ALERT_DB_PATH = REPO_ROOT / "services" / "alert-service" / "alerts.db"

WARD_ANTI_FABRICATION_SYSTEM = (
    "You write a shift handover summary for ICU nursing and physician staff "
    "from the structured per-patient facts given, covering one ward. "
    + ANTI_FABRICATION_INSTRUCTION
)


@dataclass
class PatientShiftEntry:
    stay_id: int
    patient_ref: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    news2_start: int
    news2_end: int
    tier_end: str
    new_vasopressor: bool
    new_ventilation: bool
    active_alerts: int


@dataclass
class WardShiftHandover:
    ward: str
    patients: list[PatientShiftEntry]
    narrative: str


def _last_4h_window(stay_rows: pd.DataFrame) -> pd.DataFrame:
    """`stay_rows` has one row per hourly_grid hour for a single stay, with an
    `abs_time` column already attached. Returns just the rows inside that
    stay's own most recent real 00/04/08/12/16/20-aligned 4h block.
    """
    latest = stay_rows["abs_time"].max()
    bucket_start = dedup_bucket_start(latest.to_pydatetime())
    bucket_end = bucket_start + timedelta(hours=4)
    return stay_rows[(stay_rows.abs_time >= bucket_start) & (stay_rows.abs_time < bucket_end)]


def build_ward_shift_handover(
    conn: duckdb.DuckDBPyConnection,
    ward: str,
    llm: LLMBackend | None,
    alert_store: Any | None = None,
) -> WardShiftHandover:
    store = alert_store or AlertStore(DEFAULT_ALERT_DB_PATH)

    stays = conn.execute(
        """
        select d.stay_id, d.icu_intime
        from mimiciv_derived.icustay_detail d
        join mimiciv_icu.icustays i using (stay_id)
        where i.first_careunit = ?
        """,
        [ward],
    ).fetchdf()

    news2 = conn.execute("select stay_id, hour, news2, tier_icu from capstone.news2").fetchdf()
    vaso_events = {
        e.stay_id: e.event_time for e in labels.vasopressor_events(conn).frame.itertuples()
    }
    vent_events = {
        e.stay_id: e.event_time for e in labels.ventilation_events(conn).frame.itertuples()
    }

    entries: list[PatientShiftEntry] = []
    for stay in stays.itertuples():
        stay_id = int(stay.stay_id)  # type: ignore[arg-type]
        icu_intime = pd.Timestamp(stay.icu_intime)  # type: ignore[arg-type]
        stay_news2 = news2[news2.stay_id == stay_id].copy()
        if stay_news2.empty:
            continue
        stay_news2["abs_time"] = icu_intime + pd.to_timedelta(stay_news2.hour, unit="h")
        window = _last_4h_window(stay_news2)
        if window.empty:
            continue
        window = window.sort_values("hour")

        window_start, window_end = window.abs_time.iloc[0], window.abs_time.iloc[-1]
        new_vaso = stay_id in vaso_events and (window_start <= vaso_events[stay_id] <= window_end)
        new_vent = stay_id in vent_events and (window_start <= vent_events[stay_id] <= window_end)
        active_alerts = len(store.active_for_patient(f"ICUStay/{stay_id}"))

        entries.append(
            PatientShiftEntry(
                stay_id=stay_id,
                patient_ref=f"ICUStay/{stay_id}",
                window_start=window_start,
                window_end=window_end,
                news2_start=int(window.news2.iloc[0]),
                news2_end=int(window.news2.iloc[-1]),
                tier_end=window.tier_icu.iloc[-1],
                new_vasopressor=new_vaso,
                new_ventilation=new_vent,
                active_alerts=active_alerts,
            )
        )

    entries.sort(key=lambda e: e.news2_end, reverse=True)

    facts = "\n".join(
        f"- {e.patient_ref}: NEWS2 {e.news2_start}->{e.news2_end} ({e.tier_end} tier), "
        f"{'new vasopressor start; ' if e.new_vasopressor else ''}"
        f"{'new ventilation start; ' if e.new_ventilation else ''}"
        f"{e.active_alerts} active alert(s), window {e.window_start} to {e.window_end}"
        for e in entries
    )
    user = f"Ward: {ward}\nPatients this shift:\n{facts or '(no monitored patients this shift)'}"
    narrative = generate_narrative(llm, WARD_ANTI_FABRICATION_SYSTEM, user)

    return WardShiftHandover(ward=ward, patients=entries, narrative=narrative)
