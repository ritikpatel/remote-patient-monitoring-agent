"""Evaluation methodology (PROJECT_PLAN.md section 11): "Given n=140,
methodology carries the credibility." AUPRC is the headline metric (base
rates are low, ~4-5%, see ml/features/labels.py); AUROC, calibration, and
Brier score are reported alongside it, never instead of it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


@dataclass
class PointWithCI:
    point: float
    lo: float
    hi: float


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def bootstrap_ci_grouped(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> PointWithCI:
    """Bootstrap 95% CI resampling whole patients (groups), not individual
    patient-hours -- resampling hours directly would treat correlated
    within-patient rows as independent evidence and understate the interval.
    """
    rng = np.random.default_rng(random_state)
    unique_groups = np.unique(groups)
    point = metric_fn(y_true, y_score)

    boot_values = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled_groups])
        value = metric_fn(y_true[idx], y_score[idx])
        if not np.isnan(value):
            boot_values.append(value)

    if not boot_values:
        return PointWithCI(point=point, lo=float("nan"), hi=float("nan"))
    lo, hi = np.percentile(boot_values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return PointWithCI(point=point, lo=float(lo), hi=float(hi))


def auroc_auprc_with_ci(
    y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, n_boot: int = 1000
) -> dict[str, PointWithCI]:
    return {
        "auroc": bootstrap_ci_grouped(y_true, y_score, groups, _safe_auroc, n_boot=n_boot),
        "auprc": bootstrap_ci_grouped(y_true, y_score, groups, _safe_auprc, n_boot=n_boot),
    }


def calibration_curve(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """ "A miscalibrated alerting model is worse than none" -- bucket
    predicted probability into deciles and compare to observed frequency.
    """
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    df["bin"] = pd.qcut(df["y_score"], q=min(n_bins, df["y_score"].nunique()), duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(
        mean_predicted=("y_score", "mean"), observed_rate=("y_true", "mean"), n=("y_true", "size")
    )
    return grouped.reset_index(drop=True)


def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_score))


def per_repeat_fold_scores(fold_results: pd.DataFrame, metric: str = "auprc") -> pd.DataFrame:
    """``fold_results`` has one row per (repeat, fold, model) with an
    ``auroc``/``auprc`` column already computed on that fold's held-out set.
    Collapses to one row per (repeat, model): the mean across that repeat's
    folds -- the unit the plan's ">=15 of 20 repeats" comparison is made in.
    """
    return fold_results.groupby(["repeat", "model"], as_index=False).agg(
        **{f"mean_{metric}": (metric, "mean")}
    )


def beats_baseline_in_n_of_k_repeats(
    fold_results: pd.DataFrame,
    model_name: str,
    baseline_name: str,
    metric: str = "auprc",
) -> tuple[int, int]:
    """Returns (wins, total_repeats): how many repeats ``model_name``'s
    mean-across-folds metric exceeds ``baseline_name``'s, on the SAME repeats
    (same fold splits, since both were evaluated inside the same CV loop).
    """
    per_repeat = per_repeat_fold_scores(fold_results, metric=metric)
    wide = per_repeat.pivot(index="repeat", columns="model", values=f"mean_{metric}")
    wins = int((wide[model_name] > wide[baseline_name]).sum())
    return wins, len(wide)
