"""Tests for the F4 fairness audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.evaluation import fairness


def test_age_band_covers_the_range_and_labels_the_edges():
    assert fairness.age_band(49) == "<50"
    assert fairness.age_band(50) == "50-64"
    assert fairness.age_band(64) == "50-64"
    assert fairness.age_band(65) == "65-79"
    assert fairness.age_band(91) == "80+"


def test_drop_demographics_removes_gender_but_keeps_clinical_context():
    x = pd.DataFrame({"gender": ["M"], "admission_age": [70], "first_careunit": ["MICU"]})
    out = fairness.drop_demographics(x)
    assert "gender" not in out.columns
    # Age and care unit are a validated severity covariate and clinical context --
    # not protected attributes, and deliberately retained.
    assert "admission_age" in out.columns
    assert "first_careunit" in out.columns


def test_tiny_subgroups_report_counts_but_withhold_metrics():
    """A metric computed on a handful of patients is worse than no metric: it invites
    exactly the overinterpretation the audit exists to prevent."""
    n = 20
    y = np.zeros(n)
    y[0] = 1
    score = np.linspace(0, 1, n)
    sub = pd.DataFrame({"sex": ["F"] * n})
    table = fairness.subgroup_metrics(y, score, sub, alert_threshold=0.5)
    row = table.iloc[0]
    assert row.uninformative
    assert np.isnan(row.auroc) and np.isnan(row.auprc)
    # Counts stay visible, so the suppression is legible rather than a silent gap.
    assert row.n_rows == n and row.n_positives == 1


def test_a_wide_confidence_interval_withholds_the_point_estimate(monkeypatch):
    """The regression this module exists for.

    Trauma SICU had 453 rows and 17 positives -- comfortably past any count-based gate
    -- and a patient-grouped AUROC CI of 0.13-0.91. The count gate let it through and
    it was written up as "worse than chance", a claim the data never supported. The
    gate is now the interval width itself, so a subgroup is published only when its
    estimate is precise enough to mean something.

    The threshold is tightened here rather than trying to synthesise a specific
    bootstrap width: what needs pinning is that width, not counts, decides.
    """
    rng = np.random.default_rng(0)
    n, n_patients = 400, 20
    groups = np.repeat(np.arange(n_patients), n // n_patients)
    y = np.zeros(n)
    y[rng.choice(n, 30, replace=False)] = 1
    score = rng.random(n)
    sub = pd.DataFrame({"care_unit": ["TSICU"] * n})

    wide_open = fairness.subgroup_metrics(y, score, sub, 0.5, groups=groups).iloc[0]
    assert not wide_open.uninformative and not np.isnan(wide_open.auroc)

    # Same data, same counts -- only the precision demanded changes.
    monkeypatch.setattr(fairness, "MAX_INFORMATIVE_CI_WIDTH", wide_open.ci_width / 2)
    gated = fairness.subgroup_metrics(y, score, sub, 0.5, groups=groups).iloc[0]
    assert gated.n_rows >= fairness.MIN_SUBGROUP_ROWS
    assert gated.n_positives >= fairness.MIN_SUBGROUP_POSITIVES
    assert gated.uninformative, "counts passed, so only the CI width can have gated it"
    assert np.isnan(gated.auroc) and np.isnan(gated.auprc)
    # The interval itself stays visible -- suppression must be legible, not silent.
    assert not np.isnan(gated.auroc_lo) and not np.isnan(gated.auroc_hi)


def test_alert_rate_uses_one_shared_threshold_across_subgroups():
    """The point of the alert-rate column: the same threshold landing differently on
    different populations is the finding, not a bug to normalise away."""
    y = np.zeros(400)
    y[:20] = 1
    score = np.concatenate([np.full(200, 0.9), np.full(200, 0.1)])
    sub = pd.DataFrame({"sex": ["M"] * 200 + ["F"] * 200})
    table = fairness.subgroup_metrics(y, score, sub, alert_threshold=0.5)
    rates = dict(zip(table.subgroup, table.alert_rate, strict=True))
    assert rates["M"] == 1.0
    assert rates["F"] == 0.0


def test_ablation_requires_a_clear_majority_not_a_coin_flip():
    """13/20 is what `gender` actually scored. A feature has to win convincingly to
    justify carrying a protected attribute, not just win slightly more than half."""
    coin_flip = fairness.AblationResult(0.51, 0.50, 0.01, 13, 20)
    assert not coin_flip.demographics_earn_their_place
    convincing = fairness.AblationResult(0.60, 0.50, 0.10, 18, 20)
    assert convincing.demographics_earn_their_place


def test_subgroups_of_concern_tests_the_ci_upper_bound_not_the_point_estimate():
    """A low point estimate with a high upper bound means "not measured", not "bad".
    Only a subgroup whose whole plausible range is poor belongs on the concern list.
    """
    table = pd.DataFrame(
        {
            "dimension": ["care_unit"] * 3,
            "subgroup": ["wide_ci", "confidently_poor", "good"],
            "n_rows": [453, 600, 305],
            "n_positives": [17, 40, 23],
            "auroc": [0.469, 0.55, 0.99],
            "auroc_lo": [0.13, 0.45, 0.97],
            "auroc_hi": [0.91, 0.64, 1.00],
            "ci_width": [0.78, 0.19, 0.03],
            "uninformative": [True, False, False],
        }
    )
    concerns = fairness.subgroups_of_concern(table)
    # wide_ci is excluded twice over: uninformative, and its upper bound is 0.91.
    assert list(concerns.subgroup) == ["confidently_poor"]


def test_time_band_is_a_subgroup_dimension_when_hours_are_supplied():
    """Time since admission is the adequately-powered audit axis, so it has to be
    available to the audit at all."""
    x = pd.DataFrame({"gender": ["M", "F", "M"], "admission_age": [70, 60, 50]})
    out = fairness.subgroup_frame(x, hours=np.array([0, 10, 30]))
    assert list(out["time_in_stay"]) == ["hour 0-5", "hour 6-23", "hour 24+"]
    assert "time_in_stay" not in fairness.subgroup_frame(x).columns
