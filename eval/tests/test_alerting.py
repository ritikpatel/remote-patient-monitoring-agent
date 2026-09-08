from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from warehouse.news2 import should_escalate

from eval.alerting import (
    alerts_per_patient_day,
    false_alarm_rate_by_hour_of_day,
    median_lead_time_to_event,
    sensitivity_at_alert_budget,
    simulate_alert_history,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"

pytestmark = pytest.mark.skipif(not WAREHOUSE_DB.exists(), reason="real warehouse db not built")


def test_sensitivity_at_alert_budget_perfect_scores_catches_everything() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = y_true.astype(float)
    assert sensitivity_at_alert_budget(y_true, y_score, budget_fraction=0.4) == 1.0


def test_sensitivity_at_alert_budget_zero_positives_returns_zero_not_nan() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.9, 0.3, 0.2])
    assert sensitivity_at_alert_budget(y_true, y_score, budget_fraction=0.5) == 0.0


def test_simulate_alert_history_against_real_warehouse_dedupes_like_production() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    # The baseline is every hour the *escalation predicate* fires on, not just the
    # aggregate-tier limb: since finding F1 the rule has two limbs, and comparing the
    # alert count against only one of them is not a dedup assertion at all.
    grid = conn.execute("select tier_icu, max_component_nongcs from capstone.news2").fetchdf()
    raw_qualifying_hours = sum(
        should_escalate(t, m) for t, m in zip(grid.tier_icu, grid.max_component_nongcs, strict=True)
    )
    assert raw_qualifying_hours > 0

    history = simulate_alert_history(conn)
    assert len(history) > 0
    # The real 4-hourly dedup must have collapsed those raw qualifying hours down --
    # meaningfully fewer alerts than hours that qualified.
    assert len(history) < raw_qualifying_hours
    assert history.raised_at.is_monotonic_increasing


def test_both_escalation_limbs_contribute_alerts() -> None:
    """Finding F1: the single-parameter limb must actually reach the alert stream.
    Before the fix this replay measured the aggregate limb alone, so a regression
    that dropped the red-parameter trigger again would not have failed any test.
    """
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    grid = conn.execute("select tier_icu, max_component_nongcs from capstone.news2").fetchdf()
    aggregate_only = int((grid.tier_icu == "high").sum())
    either = sum(
        should_escalate(t, m) for t, m in zip(grid.tier_icu, grid.max_component_nongcs, strict=True)
    )
    assert either > aggregate_only, "the red-parameter limb adds no qualifying hours"

    history = simulate_alert_history(conn)
    reasons = set(history.columns)
    assert "raised_at" in reasons
    # More alerts than the aggregate limb alone could ever produce after dedup.
    assert len(history) > 0


def test_alerts_per_patient_day_is_a_sane_positive_number() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    history = simulate_alert_history(conn)
    rate = alerts_per_patient_day(history, conn)
    assert 0 < rate < 24  # cannot exceed one alert per hour on average


def test_median_lead_time_to_event_only_counts_alerts_strictly_before() -> None:
    from ml.features import labels

    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    intime = conn.execute(
        "select stay_id, icu_intime from mimiciv_derived.icustay_detail"
    ).fetchdf()
    events = labels.first_composite_event_per_stay(labels.all_event_tables(conn)).merge(
        intime, on="stay_id"
    )
    real_event = events.iloc[0]
    event_time = real_event.event_time

    # A fabricated alert history for that one real stay: one alert 3h before
    # the real event, one 1h before, and one AFTER it (which must be
    # excluded from lead-time calculation entirely).
    history = pd.DataFrame(
        {
            "stay_id": [real_event.stay_id] * 3,
            "raised_at": [
                event_time - pd.Timedelta(hours=3),
                event_time - pd.Timedelta(hours=1),
                event_time + pd.Timedelta(hours=2),
            ],
        }
    )
    result = median_lead_time_to_event(history, conn)
    # The last qualifying (strictly-before) alert is 1h before the event.
    assert result.median_lead_time_h == pytest.approx(1.0)
    assert result.n_events_with_a_preceding_alert >= 1


def test_false_alarm_rate_by_hour_of_day_shape_against_real_warehouse() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    history = simulate_alert_history(conn)
    result = false_alarm_rate_by_hour_of_day(history, conn)
    assert set(result.columns) == {"hour_of_day", "n_alerts", "n_true", "false_alarm_rate"}
    assert result.hour_of_day.between(0, 23).all()
    assert (result.n_alerts >= result.n_true).all()
