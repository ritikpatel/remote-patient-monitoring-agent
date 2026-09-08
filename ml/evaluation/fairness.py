"""Subgroup performance and the demographic-feature question (review finding F4).

F4: `gender` ranked third by mean |SHAP| in the promoted model, above most vitals,
and nothing anywhere in `ml/` or `eval/` measured subgroup performance. For a clinical
model that is both a credibility gap and the first question any reviewer asks.

The association is real *in this cohort* -- 66.2% of male ICU stays reach a composite
deterioration event against 42.9% of female (Fisher OR 2.62, p=0.007) -- but this is a
100-patient convenience sample, and an effect that size in 140 stays is exactly what
sampling noise looks like. "It is statistically significant here" is not a reason to
ship a demographic feature; whether it *earns its place* is an empirical question, and
this module answers it two ways:

1. **Ablation.** Refit the promoted model with and without the demographic columns
   under identical grouped CV, and compare AUPRC per repeat. If dropping them costs
   nothing measurable, drop them -- the simplest defensible answer.
2. **Subgroup metrics.** Whatever the ablation says, report per-subgroup AUROC/AUPRC,
   positive rate and alert rate. A model can be equally accurate overall and still
   distribute its errors unequally, which is what fairness auditing is actually for.

Subgroups here are sex, age band, first ICU care unit, and **time since ICU admission**.
Race is deliberately NOT included: MIMIC-IV records it, but at n=100 most categories
hold single-digit patient counts, and a subgroup metric computed on four patients
invites exactly the overinterpretation this module exists to prevent. Stated rather
than silently omitted.

Why suppression is by confidence-interval width, not by row and positive counts
-------------------------------------------------------------------------------
The first version of this module gated on ``n_rows >= 200 and n_positives >= 10``, and
that gate was not strict enough to do its job. Trauma SICU passed it -- 453 rows, 17
positives -- and reported AUROC 0.469, which the review then wrote up as "worse than
chance: the model cannot rank this population at all."

Bootstrapping that estimate, grouped by patient, gives a 95% CI of **0.13-0.91**. An
interval that wide is consistent with a model that is useless *and* with one that is
excellent; it supports no claim in either direction. The count-based gate let a number
through that looked like a finding and was actually noise, which is the precise failure
this module exists to prevent -- so the gate now measures the thing that matters
directly. A subgroup is reported when its CI is narrow enough to mean something, and
``subgroups_of_concern`` flags a subgroup only when the CI's *upper* bound is poor,
i.e. when the data can actually support "this is bad" rather than "this is unmeasured".

By that standard the well-measured subgroups here are CVICU (0.99, CI width 0.03) and
SICU (0.98, width 0.05), and the real, adequately-powered finding is not a care unit at
all -- it is time since admission (see ``TIME_BANDS``), where the early and late
intervals do not overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.evaluation import metrics

DEMOGRAPHIC_COLUMNS = ("gender",)
# Floors, not the real gate: below these a bootstrap is not worth running at all.
MIN_SUBGROUP_ROWS = 100
MIN_SUBGROUP_POSITIVES = 5
# The real gate. A 95% CI wider than this cannot distinguish a useless model from a
# good one, so the point estimate is withheld rather than published as a finding.
MAX_INFORMATIVE_CI_WIDTH = 0.40
N_BOOTSTRAP = 600
# Fraction of CV repeats a demographic feature must win to be carried.
EARN_THEIR_PLACE_FRACTION = 0.75
AGE_BANDS = [(0, 50, "<50"), (50, 65, "50-64"), (65, 80, "65-79"), (80, 200, "80+")]
# Time since ICU admission. The adequately-powered axis: the model is trained mostly on
# early rows (57% of positives fall in the first two hours) and degrades after them.
TIME_BANDS = [(0, 6, "hour 0-5"), (6, 24, "hour 6-23"), (24, 10**6, "hour 24+")]


@dataclass(frozen=True)
class AblationResult:
    with_demographics_auprc: float
    without_demographics_auprc: float
    delta: float
    repeats_where_with_is_better: int
    n_repeats: int

    @property
    def demographics_earn_their_place(self) -> bool:
        """A demographic feature is kept only if it wins on a clear majority of
        repeats. A coin-flip result means the model does just as well without it, and
        the cheaper thing to defend is the model that never saw it.

        Known weakness, recorded because this criterion has already flipped once:
        it is a sign test on the win count and ignores effect size entirely. When
        the CV grouping was corrected from stay to subject, `gender` moved from
        13/20 to 15/20 -- across this threshold -- on a delta of +0.0103 AUPRC
        that sits far inside a bootstrap CI many times wider. Nothing about the
        feature changed. Treat a result near the bar as the weak evidence it is,
        and read it next to the subgroup table rather than instead of it.
        """
        return self.repeats_where_with_is_better >= EARN_THEIR_PLACE_FRACTION * self.n_repeats


def age_band(age: float) -> str:
    for lo, hi, label in AGE_BANDS:
        if lo <= age < hi:
            return label
    return "unknown"


def time_band(hour: float) -> str:
    for lo, hi, label in TIME_BANDS:
        if lo <= hour < hi:
            return label
    return "unknown"


def subgroup_frame(x: pd.DataFrame, hours: np.ndarray | None = None) -> pd.DataFrame:
    """The subgroup labels for each scored row, derived from features already present
    in X so no extra warehouse round trip is needed."""
    out = pd.DataFrame(index=x.index)
    if "gender" in x:
        out["sex"] = x["gender"].astype(str)
    if "admission_age" in x:
        out["age_band"] = x["admission_age"].astype(float).map(age_band)
    if "first_careunit" in x:
        out["care_unit"] = x["first_careunit"].astype(str)
    if hours is not None:
        out["time_in_stay"] = pd.Series(np.asarray(hours), index=x.index).map(time_band)
    return out


def subgroup_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    subgroups: pd.DataFrame,
    alert_threshold: float,
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-subgroup AUROC/AUPRC with bootstrap CIs, event rate and alert rate.

    ``alert_threshold`` is applied identically to every subgroup -- the point of the
    alert-rate column is to show how one shared threshold lands on different
    populations, which is where a model that is "equally accurate" can still be
    unequally useful.

    ``groups`` (patient ids) makes the bootstrap resample *patients*, not rows. Rows
    from one patient are not independent, and a row-level bootstrap would report a
    CI far narrower than the data supports -- which is how an uninformative number
    gets mistaken for a finding.
    """
    rows = []
    ok = ~np.isnan(y_true) & ~np.isnan(y_score)
    for dimension in subgroups.columns:
        values = subgroups[dimension].to_numpy()
        for level in pd.unique(values[ok]):
            mask = ok & (values == level)
            n = int(mask.sum())
            n_pos = int(y_true[mask].sum())
            too_small = n < MIN_SUBGROUP_ROWS or n_pos < MIN_SUBGROUP_POSITIVES
            auroc = auprc = lo = hi = width = np.nan
            if not too_small:
                g = groups[mask] if groups is not None else np.arange(n)
                r = metrics.bootstrap_ci_grouped(
                    y_true[mask], y_score[mask], g, metrics._safe_auroc, n_boot=N_BOOTSTRAP
                )
                lo, hi, width = r.lo, r.hi, r.hi - r.lo
                if width <= MAX_INFORMATIVE_CI_WIDTH:
                    auroc = r.point
                    auprc = metrics._safe_auprc(y_true[mask], y_score[mask])
            rows.append(
                {
                    "dimension": dimension,
                    "subgroup": str(level),
                    "n_rows": n,
                    "n_positives": n_pos,
                    "event_rate": round(n_pos / n, 4) if n else np.nan,
                    "alert_rate": (
                        round(float((y_score[mask] >= alert_threshold).mean()), 4) if n else np.nan
                    ),
                    # Withheld unless the interval is narrow enough to mean something.
                    "auroc": auroc,
                    "auprc": auprc,
                    "auroc_lo": lo,
                    "auroc_hi": hi,
                    "ci_width": width,
                    "uninformative": bool(too_small or not (width <= MAX_INFORMATIVE_CI_WIDTH)),
                }
            )
    return pd.DataFrame(rows).sort_values(["dimension", "subgroup"]).reset_index(drop=True)


CONCERN_AUROC = 0.70


def subgroups_of_concern(table: pd.DataFrame, threshold: float = CONCERN_AUROC) -> pd.DataFrame:
    """Subgroups the data can actually show are poor.

    The test is on the CI's **upper** bound, not the point estimate. A point estimate
    below threshold with an upper bound above it means "not measured", not "bad" --
    Trauma SICU scored 0.469 with an upper bound of 0.91, and reporting that as a
    concern is what produced a finding the data never supported.
    """
    ok = table[~table.uninformative].copy()
    return ok[ok.auroc_hi < threshold].sort_values("auroc").reset_index(drop=True)


def drop_demographics(x: pd.DataFrame) -> pd.DataFrame:
    return x.drop(columns=[c for c in DEMOGRAPHIC_COLUMNS if c in x.columns])


def run_ablation(
    fold_results_with: pd.DataFrame, fold_results_without: pd.DataFrame
) -> AblationResult:
    """Compares two already-run CV results per repeat, using the same
    ``per_repeat_fold_scores`` aggregation Phase 5 uses for its own beats-baseline
    claim, so the comparison is like-for-like with the headline number.
    """
    with_by_repeat = metrics.per_repeat_fold_scores(fold_results_with, "auprc")
    without_by_repeat = metrics.per_repeat_fold_scores(fold_results_without, "auprc")
    merged = with_by_repeat.merge(without_by_repeat, on="repeat", suffixes=("_with", "_without"))
    wins = int((merged["mean_auprc_with"] > merged["mean_auprc_without"]).sum())
    return AblationResult(
        with_demographics_auprc=float(merged["mean_auprc_with"].mean()),
        without_demographics_auprc=float(merged["mean_auprc_without"].mean()),
        delta=float(merged["mean_auprc_with"].mean() - merged["mean_auprc_without"].mean()),
        repeats_where_with_is_better=wins,
        n_repeats=len(merged),
    )
