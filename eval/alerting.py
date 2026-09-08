"""Axis 2 -- Alerting (PROJECT_PLAN.md section 13): "sensitivity at fixed
alert budget, alerts per patient-day, median lead time to event (the metric
that matters clinically), false-alarm rate stratified by hour of day (E16)."

The alert history this axis measures against is a **real replay of
alert-service's actual production code** (``services/alert-service/store.py``'s
``AlertStore.raise_alert``, loaded directly -- not a second, parallel
reimplementation of the dedup/escalation rule) over every high-ICU-tier hour
in the warehouse, in chronological order. This is the same rule
agent-orchestrator's EscalationDecider uses (``ESCALATE_ON_TIER = "high"``),
so the simulated alert stream is what this system would actually have raised
across the whole cohort, not a synthetic approximation of it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.features import labels  # noqa: E402
from services.common.testing import load_module  # noqa: E402
from warehouse.news2 import escalation_reason, should_escalate  # noqa: E402

_alert_store_module = load_module(
    REPO_ROOT / "services" / "alert-service" / "store.py", "eval_alert_service_store"
)
AlertStore = _alert_store_module.AlertStore

# Covers both of NEWS2's escalation limbs since finding F1, not just the aggregate
# tier -- named for the decision, not for one of its two causes.
ALERT_TYPE = "news2_escalation"
FALSE_ALARM_HORIZON_H = 12  # matches ml/features/labels.py's secondary horizon


def simulate_alert_history(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Replays every ESCALATING hour across the whole cohort through the real
    AlertStore, in chronological (wall-clock) order, and returns one row per RAISED
    alert record (i.e. already deduped/escalated by the real production logic) with
    its real timestamp and patient_ref.

    "Escalating" is ``warehouse.news2.should_escalate`` -- the same predicate
    agent-orchestrator's EscalationDecider applies, imported rather than restated, so
    this replay cannot measure a rule the running system does not use. Before finding
    F1 this selected ``tier_icu = 'high'`` alone, which measured only one of NEWS2's
    two escalation triggers.
    """
    rows = conn.execute(
        """
        select n.stay_id, n.hour, n.tier_icu, n.max_component_nongcs, n.red_params,
               n.gcs_drop, d.icu_intime
        from capstone.news2 n
        join mimiciv_derived.icustay_detail d using (stay_id)
        """
    ).fetchdf()
    rows = rows[
        [
            should_escalate(t, m, d)
            for t, m, d in zip(rows.tier_icu, rows.max_component_nongcs, rows.gcs_drop, strict=True)
        ]
    ]
    rows["abs_time"] = rows.icu_intime + pd.to_timedelta(rows.hour, unit="h")
    rows = rows.sort_values("abs_time")
    # Built here from typed Series rather than inside the loop: itertuples erases
    # column dtypes, so per-row extraction cannot be typed without casting each field.
    rows["reason"] = [
        escalation_reason(t, m, r, d)
        for t, m, r, d in zip(
            rows.tier_icu, rows.max_component_nongcs, rows.red_params, rows.gcs_drop, strict=True
        )
    ]

    store = AlertStore(":memory:")
    history = []
    for row in rows.itertuples(index=False):
        patient_ref = f"ICUStay/{row.stay_id}"
        alert, was_new = store.raise_alert(
            patient_ref,
            ALERT_TYPE,
            "high",
            row.reason,
            row.abs_time,
        )
        if was_new:
            # A repeat within the same 4h bucket bumps repeat_count/escalates
            # this same alert.id (was_new=False) -- recorded once, here, at
            # the time it was first actually raised.
            history.append(
                {
                    "alert_id": alert.id,
                    "stay_id": row.stay_id,
                    "patient_ref": patient_ref,
                    "raised_at": row.abs_time,
                }
            )
    store.close()

    return pd.DataFrame(history).sort_values("raised_at").reset_index(drop=True)


def alerts_per_patient_day(alert_history: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> float:
    row = conn.execute("select count(*) from capstone.hourly_grid").fetchone()
    assert row is not None
    total_patient_days = row[0] / 24
    return len(alert_history) / total_patient_days


@dataclass
class LeadTimeResult:
    median_lead_time_h: float | None
    n_events_with_a_preceding_alert: int
    n_events_total: int
    coverage: float


def median_lead_time_to_event(
    alert_history: pd.DataFrame, conn: duckdb.DuckDBPyConnection
) -> LeadTimeResult:
    """For every stay's real first composite deterioration event
    (``ml/features/labels.py``, the same R1-censored event definition Phase 5
    predicts), find that patient's last alert raised strictly before the
    event, and report the median hours of lead time -- "the metric that
    matters clinically" per the plan, over just reporting whether an alert
    ever fired.
    """
    intime = conn.execute(
        "select stay_id, icu_intime from mimiciv_derived.icustay_detail"
    ).fetchdf()
    events = labels.first_composite_event_per_stay(labels.all_event_tables(conn))
    events = events.merge(intime, on="stay_id")

    lead_times = []
    for event in events.itertuples(index=False):
        patient_alerts = alert_history[alert_history.stay_id == event.stay_id]
        preceding = patient_alerts[patient_alerts.raised_at < event.event_time]
        if preceding.empty:
            continue
        last_alert = preceding.raised_at.max()
        lead_h = (event.event_time - last_alert).total_seconds() / 3600
        lead_times.append(lead_h)

    n_total = len(events)
    n_covered = len(lead_times)
    return LeadTimeResult(
        median_lead_time_h=float(np.median(lead_times)) if lead_times else None,
        n_events_with_a_preceding_alert=n_covered,
        n_events_total=n_total,
        coverage=n_covered / n_total if n_total else 0.0,
    )


def false_alarm_rate_by_hour_of_day(
    alert_history: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
    horizon_h: int = FALSE_ALARM_HORIZON_H,
) -> pd.DataFrame:
    """An alert is a "true alarm" if that patient's first composite event
    falls within `horizon_h` hours after it; otherwise a false alarm. Grouped
    by the alert's real hour-of-day (E16: care runs on a 4-hourly clock, so a
    flat false-alarm rate across hours would itself be a notable finding).
    """
    events = labels.first_composite_event_per_stay(labels.all_event_tables(conn))
    event_time_by_stay = dict(zip(events.stay_id, events.event_time, strict=True))

    history = alert_history.copy()
    history["hour_of_day"] = history.raised_at.dt.hour

    def _is_true_alarm(row) -> bool:  # noqa: ANN001
        event_time = event_time_by_stay.get(row.stay_id)
        if event_time is None:
            return False
        delta_h = (event_time - row.raised_at).total_seconds() / 3600
        return 0 <= delta_h <= horizon_h

    history["true_alarm"] = history.apply(_is_true_alarm, axis=1)

    grouped = history.groupby("hour_of_day").agg(
        n_alerts=("alert_id", "count"), n_true=("true_alarm", "sum")
    )
    grouped["false_alarm_rate"] = 1 - grouped["n_true"] / grouped["n_alerts"]
    return grouped.reset_index()


def sensitivity_at_alert_budget(
    y_true: np.ndarray, y_score: np.ndarray, budget_fraction: float
) -> float:
    """ "Sensitivity at fixed alert budget": if the system may only alert on
    the top `budget_fraction` of patient-hours by predicted risk (a realistic
    clinical/staffing constraint -- a ward cannot triage an alert on every
    hour), what fraction of true composite events are caught? Ties broken
    towards flagging (>= the cutoff score, not >) so the reported budget is
    a floor, not silently underspent when there are score ties.
    """
    n_alert = max(1, int(round(len(y_score) * budget_fraction)))
    cutoff = np.sort(y_score)[::-1][n_alert - 1]
    flagged = y_score >= cutoff
    true_positives = int(np.sum(flagged & (y_true == 1)))
    total_positives = int(np.sum(y_true == 1))
    return true_positives / total_positives if total_positives else 0.0
