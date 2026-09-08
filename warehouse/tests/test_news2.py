"""NEWS2's two escalation limbs (review finding F1).

The pre-F1 code computed every component subscore and then tiered on the aggregate
alone, silently dropping RCP 2017's "score of 3 in any single parameter" trigger.
Nothing failed, because nothing tested the component scores individually. These tests
exist so that regression cannot happen quietly again.
"""

from __future__ import annotations

import pandas as pd
import pytest
from warehouse.news2 import (
    ESCALATION_COMPONENTS,
    RED_COMPONENT_SCORE,
    component_scores,
    escalation_reason,
    red_flags,
    should_escalate,
)


def _row(**kw) -> pd.Series:
    base = {
        "hr": 80.0,
        "rr": 16.0,
        "spo2": 98.0,
        "sbp": 120.0,
        "temp_c": 37.0,
        "gcs_total": 15.0,
        "fio2": 21.0,
    }
    base.update(kw)
    return pd.Series(base)


def test_a_perfectly_well_patient_scores_zero_everywhere() -> None:
    flags = red_flags(_row())
    assert flags.max_component == 0
    assert flags.max_component_nongcs == 0
    assert flags.red_params == ""
    assert should_escalate("low", flags.max_component_nongcs) is False


def test_a_stable_or_sedated_red_gcs_does_not_escalate_on_level_alone() -> None:
    """GCS 3 is routine in a sedated ICU patient -- escalating on the *level* fires on
    71.5% of all patient-hours. The red flag is still recorded so the decision stays
    auditable, and the reason says so rather than staying silent.
    """
    flags = red_flags(_row(gcs_total=3.0))
    assert flags.max_component == RED_COMPONENT_SCORE  # it IS a red parameter
    assert flags.max_component_nongcs == 0  # but not an escalating one
    assert "gcs_total" in flags.red_params
    assert should_escalate("medium", flags.max_component_nongcs, gcs_drop=False) is False

    reason = escalation_reason("medium", flags.max_component_nongcs, flags.red_params, False)
    assert "does not escalate on level alone" in reason


def test_the_case_that_found_the_bug_escalates_on_the_gcs_drop_limb() -> None:
    """Stay 34617352 hour 35 -- the patient who exposed F1.

    Aggregate tier 'medium', and GCS is the only red parameter, so neither the
    aggregate limb nor the non-GCS single-parameter limb fires. What makes this an
    emergency rather than sedation is the *trajectory*: GCS 7 for six hours, then 3,
    with no sedative running. The third limb exists for exactly this patient, who died
    two days later.
    """
    flags = red_flags(_row(gcs_total=3.0))
    assert should_escalate("medium", flags.max_component_nongcs, gcs_drop=True) is True

    reason = escalation_reason("medium", flags.max_component_nongcs, flags.red_params, True)
    assert "GCS fell" in reason
    assert "no sedative running" in reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rr", 4.0),  # <=8  -> 3
        ("rr", 30.0),  # >=25 -> 3
        ("spo2", 88.0),  # <=91 -> 3
        ("sbp", 85.0),  # <=90 -> 3
        ("hr", 35.0),  # <=40 -> 3
        ("hr", 140.0),  # >=131 -> 3
        ("temp_c", 34.0),  # <=35 -> 3
    ],
)
def test_any_single_red_non_gcs_parameter_escalates_regardless_of_aggregate(
    field: str, value: float
) -> None:
    """The whole point of F1: these escalate even when the aggregate tier is 'low'."""
    flags = red_flags(_row(**{field: value}))
    assert flags.max_component_nongcs >= RED_COMPONENT_SCORE
    assert should_escalate("low", flags.max_component_nongcs) is True
    assert field in flags.red_params


def test_aggregate_limb_still_escalates_on_its_own() -> None:
    """E5's ICU-recalibrated tier is untouched by the F1 fix -- it is the other limb,
    not a replaced one."""
    assert should_escalate("high", 0) is True
    assert "tier is 'high'" in escalation_reason("high", 0, "")


def test_escalation_reason_names_both_limbs_when_both_fire() -> None:
    reason = escalation_reason("high", 3, "rr,spo2")
    assert "tier is 'high'" in reason
    assert "single-parameter red flag" in reason
    assert "AND" in reason


def test_missing_vitals_do_not_invent_a_red_flag() -> None:
    """A patient-hour with nothing charted must not escalate. Absent is not zero and
    it is certainly not critical (R2/R3)."""
    row = pd.Series(
        {k: float("nan") for k in ["hr", "rr", "spo2", "sbp", "temp_c", "gcs_total", "fio2"]}
    )
    flags = red_flags(row)
    assert component_scores(row) == {}
    assert flags.max_component == 0
    assert should_escalate("low", flags.max_component_nongcs) is False


def test_gcs_is_the_only_component_excluded_from_escalation() -> None:
    """Guards the constant itself: if someone adds a component to the scorer they must
    make a deliberate choice about whether it escalates."""
    scored = set(component_scores(_row()))
    assert set(ESCALATION_COMPONENTS) == scored - {"gcs_total"}


def test_gcs_drop_is_computed_from_trajectory_and_suppressed_by_sedation() -> None:
    """add_gcs_drop's two halves: a fall below the recent best escalates, but only when
    no sedative covers that hour."""
    import pandas as pd
    from warehouse.news2 import add_gcs_drop

    hours = pd.DataFrame(
        {
            "stay_id": [1] * 5,
            "hour": [0, 1, 2, 3, 4],
            "gcs_total": [15.0, 15.0, 15.0, 7.0, 7.0],
            "abs_time": pd.date_range("2110-01-01", periods=5, freq="h"),
        }
    )
    no_sedation = pd.DataFrame(columns=["stay_id", "starttime", "endtime"])
    out = add_gcs_drop(hours, no_sedation)
    # Hours 3 AND 4 both fire: the lookback window still holds the pre-drop baseline of
    # 15, so the patient stays flagged as depressed relative to their recent best until
    # the window rolls past it. That is deliberate -- it is why the real motivating case
    # (GCS 7 -> 3 at hour 32) was still escalating at hour 35 -- and alert-service's
    # 4-hourly dedup collapses the repeats into one alert anyway.
    assert out.gcs_drop.tolist() == [False, False, False, True, True]

    sedated = pd.DataFrame(
        {
            "stay_id": [1],
            "starttime": [pd.Timestamp("2110-01-01 02:30")],
            "endtime": [pd.Timestamp("2110-01-01 05:00")],
        }
    )
    out_sedated = add_gcs_drop(hours, sedated)
    assert (
        out_sedated.gcs_drop.tolist() == [False] * 5
    ), "a GCS fall while a sedative is running must be attributed to the drug"
