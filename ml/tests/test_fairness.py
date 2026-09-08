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


def test_underpowered_subgroups_report_counts_but_withhold_metrics():
    """A metric computed on four patients is worse than no metric: it invites exactly
    the overinterpretation the audit exists to prevent."""
    n = 20
    y = np.zeros(n)
    y[0] = 1
    score = np.linspace(0, 1, n)
    sub = pd.DataFrame({"sex": ["F"] * n})
    table = fairness.subgroup_metrics(y, score, sub, alert_threshold=0.5)
    row = table.iloc[0]
    assert row.underpowered
    assert np.isnan(row.auroc) and np.isnan(row.auprc)
    # Counts are still shown, so the suppression is visible rather than a silent gap.
    assert row.n_rows == n and row.n_positives == 1


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


def test_subgroups_of_concern_surfaces_below_chance_ranking_only_when_powered():
    table = pd.DataFrame(
        {
            "dimension": ["care_unit", "care_unit", "care_unit"],
            "subgroup": ["TSICU", "CVICU", "tiny"],
            "n_rows": [453, 305, 19],
            "n_positives": [17, 23, 0],
            "auroc": [0.469, 0.99, np.nan],
            "auprc": [0.066, 0.89, np.nan],
            "underpowered": [False, False, True],
        }
    )
    concerns = fairness.subgroups_of_concern(table)
    assert list(concerns.subgroup) == ["TSICU"]
