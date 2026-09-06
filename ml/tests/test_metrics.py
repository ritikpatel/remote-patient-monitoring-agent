"""Tests for ml/evaluation/metrics.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.evaluation import metrics


def test_perfect_separation_gives_auroc_and_auprc_of_one() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    groups = np.array([1, 2, 3, 4, 5, 6])
    result = metrics.auroc_auprc_with_ci(y_true, y_score, groups, n_boot=200)
    assert result["auroc"].point == 1.0
    assert result["auprc"].point == 1.0
    assert 0.0 <= result["auroc"].lo <= result["auroc"].hi <= 1.0


def test_bootstrap_ci_resamples_whole_groups_not_individual_rows() -> None:
    # Two patients, one all-positive-high-score, one all-negative-low-score.
    # Any bootstrap resample can only ever draw whole patients, so the metric
    # can only ever be exactly 0.0 or 1.0 across resamples -- never
    # in between, which is the tell that rows (not groups) leaked through.
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    groups = np.array([1, 1, 1, 2, 2, 2])
    result = metrics.bootstrap_ci_grouped(y_true, y_score, groups, metrics._safe_auroc, n_boot=500)
    assert result.point == 1.0


def test_calibration_curve_bins_and_reports_observed_rate() -> None:
    rng = np.random.default_rng(0)
    y_score = rng.uniform(0, 1, size=200)
    y_true = (rng.uniform(0, 1, size=200) < y_score).astype(int)
    curve = metrics.calibration_curve(y_true, y_score, n_bins=5)
    assert "mean_predicted" in curve.columns
    assert "observed_rate" in curve.columns
    assert curve["n"].sum() == 200


def test_beats_baseline_in_n_of_k_repeats_counts_correctly() -> None:
    fold_results = pd.DataFrame(
        [
            {"repeat": 0, "model": "gbm", "auprc": 0.6},
            {"repeat": 0, "model": "gbm", "auprc": 0.4},
            {"repeat": 0, "model": "news2", "auprc": 0.3},
            {"repeat": 0, "model": "news2", "auprc": 0.5},
            {"repeat": 1, "model": "gbm", "auprc": 0.2},
            {"repeat": 1, "model": "news2", "auprc": 0.9},
        ]
    )
    wins, total = metrics.beats_baseline_in_n_of_k_repeats(fold_results, "gbm", "news2")
    # repeat 0: gbm mean 0.5 > news2 mean 0.4 -> win. repeat 1: gbm 0.2 < news2 0.9 -> loss.
    assert wins == 1
    assert total == 2
