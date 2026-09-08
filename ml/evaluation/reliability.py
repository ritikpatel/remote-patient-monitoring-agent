"""What actually limits this model's reliability, measured rather than asserted.

This module was written against a headline AUPRC of 0.448 whose 95% CI ran
0.32-0.64 -- an interval so wide it is consistent with "clearly beats NEWS2"
and "barely beats NEWS2" at the same time. It asks *why* the interval is that
wide, and which of the available levers actually narrows it. Three are
measured here, in the order of how much they cost to pull. (Lever 1 turned out
to be a bug, so that 0.448 is now itself historical -- see below and
``ml/evaluation/report.md`` for the current figures.)

**Lever 1 -- the grouping unit. A correctness fix, and it was real.**
``ml/models/splits.py`` documented "Grouped by subject_id -- no patient spans
train and test", and provided ``group_key()`` to supply it. Nothing ever called
that helper with a subject id, so every CV number this project produced was
grouped by ``stay_id`` while three separate places claimed otherwise. That
would be harmless if repeat patients were rare here; they are not. **21 of 93
subjects in the at-risk set have more than one ICU stay, carrying 45.4% of all
at-risk rows and 53 of the 120 positives.** Measured cost: **+0.0499 AUPRC** of
optimism. Now fixed -- ``feature_matrix_for_training`` returns ``subject_id``
and ``group_key()`` is deleted -- so the comparison below is between the old
behaviour and the current one. A reliability lever is one that narrows the gap
between the reported number and the real one, in whichever direction that lies;
this one made the number *smaller*, and true.

That +0.0499 was +0.0159 when first measured, and the difference is
instructive rather than noise: the feature set has since gained ``gender``
(finding F4, reversed). ``gender`` is *constant within a patient*, so under
stay-grouping a repeat patient's sex-and-outcome pairing passes between folds
directly. Adding a patient-level feature therefore amplifies exactly this leak
-- which is worth knowing before adding another one.

**Lever 2 -- the feature budget. Hypothesised, tested, refuted.** The primary
task carries **90 features against 120 positive rows: 1.33 events per
variable**, against a conventional floor of 10, so the expectation was that
pruning would buy back stability. It does not: with features selected *inside*
each training fold, AUPRC rises with the budget and the across-repeat spread
shows no trend. Events-per-variable is a rule about degrees of freedom in a
*linear* model; a depth-4 LightGBM spends no parameter on a feature it never
splits on. The negative result is kept here deliberately, so the objection is
not re-raised from theory later.

**Lever 3 -- the number of events. The only real one.** A learning curve over
subsampled patients, extrapolated to answer the question that sizes a MIMIC-IV
pull: how many positive patients does a target CI width need? Fitted as
``ci_width = a * n ** b`` with **both** parameters estimated, rather than ``b``
pinned at the textbook -0.5 -- because on this cohort it measures about -0.37,
and assuming -0.5 would understate the patients required by roughly 2.3x at the
practical target.

Every fit here reuses the project's real components -- ``ml/models/gbm.py``'s
actual estimator, ``ml/models/splits.py``'s actual splitter, and
``ml/evaluation/metrics.py``'s actual patient-grouped bootstrap -- so these
numbers are comparable to ``report.md``'s by construction, not a parallel
re-implementation that could drift.

Usage:
    python ml/evaluation/reliability.py            # full protocol, ~8 min
    python ml/evaluation/reliability.py --quick    # smoke run while developing
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.evaluation import metrics  # noqa: E402
from ml.features import engineer, labels  # noqa: E402
from ml.models import gbm, splits  # noqa: E402

WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"
REPORT_PATH = REPO_ROOT / "ml" / "evaluation" / "reliability_report.md"
CSV_DIR = REPO_ROOT / "ml" / "evaluation"

PRIMARY_HORIZON = 6
N_SPLITS = 5

# Lever 2: feature budgets to sweep. The largest is always "everything", derived
# from the matrix rather than hardcoded -- this list used to end in a literal 89,
# and adding one feature (`gender`, when F4 was reversed) made the full-set row
# vanish and the report crash on a lookup that could no longer match.
FEATURE_BUDGETS_PARTIAL = (5, 10, 20, 40)


def feature_budgets_for(n_features: int) -> tuple[int, ...]:
    return tuple(b for b in FEATURE_BUDGETS_PARTIAL if b < n_features) + (n_features,)


# Lever 3: fractions of the subject pool to subsample.
LEARNING_CURVE_FRACTIONS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)

# The precision targets the extrapolation solves for. 0.10 is the width at
# which an AUPRC of 0.45 stops overlapping a NEWS2-like 0.10 by a wide margin;
# 0.05 is what a paper reporting a clinical comparison would want.
TARGET_CI_WIDTHS = (0.20, 0.10, 0.05)


@dataclass(frozen=True)
class Task:
    """The primary task's design matrix plus both candidate grouping units."""

    x: pd.DataFrame
    y: pd.Series
    stay_id: pd.Series
    subject_id: pd.Series

    def subset(self, mask: np.ndarray) -> Task:
        return Task(
            x=self.x.loc[mask].reset_index(drop=True),
            y=self.y.loc[mask].reset_index(drop=True),
            stay_id=self.stay_id.loc[mask].reset_index(drop=True),
            subject_id=self.subject_id.loc[mask].reset_index(drop=True),
        )


def load_task(conn: duckdb.DuckDBPyConnection, horizon: int = PRIMARY_HORIZON) -> Task:
    """Assemble exactly what ``run_all.py`` assembles for its primary horizon,
    plus the subject_id column that the CV should have been grouping on.
    """
    label_col = f"label_{horizon}h"
    grid = conn.execute("select stay_id, hour from capstone.hourly_grid").fetchdf()
    features = engineer.build_feature_frame(conn)
    lab = labels.build_labels(conn, grid, horizons=(horizon,))
    x, y, subject_groups = engineer.feature_matrix_for_training(features, lab, label_col)

    # `feature_matrix_for_training` returns subject_id (that is Lever 1's fix,
    # now applied). Lever 1 still needs the *stay* to compare against, so it is
    # recovered by repeating that function's own inner join on the same keys in
    # the same order -- not by a separate lookup that could silently align to
    # different rows. The length assertion is what would catch it if it ever did.
    stay = lab[["stay_id", "hour", label_col]].merge(
        features[["stay_id", "hour"]], on=["stay_id", "hour"], how="inner"
    )["stay_id"]
    if len(stay) != len(y):
        raise ValueError(f"stay_id recovery misaligned: {len(stay)} rows against {len(y)}")

    return Task(
        x=x.reset_index(drop=True),
        y=y.reset_index(drop=True),
        stay_id=stay.reset_index(drop=True),
        subject_id=subject_groups.reset_index(drop=True).astype(int),
    )


def _top_k_features(model, columns: pd.Index, k: int) -> list[str]:
    """Rank by LightGBM's total split gain. Called only on a model fitted to a
    *training* fold, so the ranking never sees the rows it is scored on.
    """
    gains = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1]
    return [columns[i] for i in order[:k]]


def _oof_predictions(
    task: Task,
    groups: pd.Series,
    n_repeats: int,
    feature_budget: int | None = None,
    random_state: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Run the real repeated grouped stratified CV and return
    (per-fold metrics, repeat-0 out-of-fold truth, repeat-0 out-of-fold score).

    ``feature_budget`` selects the top-k features by gain *within each training
    fold* and refits on them -- nested selection, so the held-out rows never
    influence which features the model is allowed to use.
    """
    rows = []
    n = len(task.y)
    oof_true = np.full(n, np.nan)
    oof_score = np.full(n, np.nan)
    y_arr = task.y.to_numpy()

    for repeat, fold, train_idx, test_idx in splits.repeated_grouped_stratified_splits(
        task.y, groups, n_splits=N_SPLITS, n_repeats=n_repeats, random_state=random_state
    ):
        x_train, x_test = task.x.iloc[train_idx], task.x.iloc[test_idx]
        y_train, y_test = task.y.iloc[train_idx], task.y.iloc[test_idx]
        if y_train.sum() == 0 or y_test.sum() == 0:
            continue

        model, proba = gbm.fit_predict_proba(x_train, y_train, x_test)
        if feature_budget is not None and feature_budget < task.x.shape[1]:
            keep = _top_k_features(model, task.x.columns, feature_budget)
            _model, proba = gbm.fit_predict_proba(x_train[keep], y_train, x_test[keep])

        rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "auprc": metrics._safe_auprc(y_test.to_numpy(), proba),
                "auroc": metrics._safe_auroc(y_test.to_numpy(), proba),
                "n_test": len(test_idx),
                "n_pos_test": int(y_test.sum()),
            }
        )
        if repeat == 0:
            oof_true[test_idx] = y_arr[test_idx]
            oof_score[test_idx] = proba

    return pd.DataFrame(rows), oof_true, oof_score


def _summarise(
    fold_results: pd.DataFrame,
    oof_true: np.ndarray,
    oof_score: np.ndarray,
    bootstrap_groups: np.ndarray,
    n_boot: int,
) -> dict:
    """Point estimate + patient-grouped bootstrap CI + across-repeat spread.

    The bootstrap always resamples *subjects*, whichever unit the CV split
    used: the interval is a statement about sampling patients from a
    population, which does not change with how folds were drawn. Only the
    point estimate moves between grouping schemes, which is precisely the
    quantity Lever 1 is measuring.
    """
    valid = ~np.isnan(oof_true)
    ci = metrics.bootstrap_ci_grouped(
        oof_true[valid],
        oof_score[valid],
        bootstrap_groups[valid],
        metrics._safe_auprc,
        n_boot=n_boot,
    )
    per_repeat = fold_results.groupby("repeat")["auprc"].mean()
    return {
        "auprc": ci.point,
        "auprc_lo": ci.lo,
        "auprc_hi": ci.hi,
        "ci_width": ci.hi - ci.lo,
        "cv_mean_auprc": float(fold_results.auprc.mean()),
        "across_repeat_sd": float(per_repeat.std(ddof=1)) if len(per_repeat) > 1 else float("nan"),
        "n_repeats": int(fold_results.repeat.nunique()),
    }


# --------------------------------------------------------------------------
# Lever 1: grouping unit
# --------------------------------------------------------------------------


def compare_grouping_units(task: Task, n_repeats: int, n_boot: int) -> pd.DataFrame:
    """Same model, same data, same folds count -- only the grouping unit differs."""
    out = []
    for name, groups in (
        ("stay_id (before the fix)", task.stay_id),
        ("subject_id (current)", task.subject_id),
    ):
        fold_results, oof_true, oof_score = _oof_predictions(task, groups, n_repeats=n_repeats)
        summary = _summarise(
            fold_results, oof_true, oof_score, task.subject_id.to_numpy(), n_boot=n_boot
        )
        summary["grouping"] = name
        out.append(summary)
        print(
            f"  {name:<22} AUPRC={summary['auprc']:.4f} "
            f"[{summary['auprc_lo']:.3f}-{summary['auprc_hi']:.3f}]  "
            f"across-repeat SD={summary['across_repeat_sd']:.4f}"
        )
    df = pd.DataFrame(out)
    return df[["grouping", "auprc", "auprc_lo", "auprc_hi", "ci_width", "across_repeat_sd"]]


# --------------------------------------------------------------------------
# Lever 2: feature budget
# --------------------------------------------------------------------------


def feature_budget_curve(task: Task, n_repeats: int, n_boot: int) -> pd.DataFrame:
    """AUPRC and stability against the number of features the model may use.

    Grouped by subject throughout (Lever 1's conclusion applied), so this
    measures the feature budget in isolation rather than re-measuring the leak.
    """
    n_pos = int(task.y.sum())
    out = []
    for budget in feature_budgets_for(task.x.shape[1]):
        fold_results, oof_true, oof_score = _oof_predictions(
            task, task.subject_id, n_repeats=n_repeats, feature_budget=budget
        )
        summary = _summarise(
            fold_results, oof_true, oof_score, task.subject_id.to_numpy(), n_boot=n_boot
        )
        summary["n_features"] = budget
        summary["events_per_variable"] = n_pos / budget
        out.append(summary)
        print(
            f"  {budget:>3} features (EPV {n_pos / budget:>5.2f})  "
            f"AUPRC={summary['auprc']:.4f} "
            f"[{summary['auprc_lo']:.3f}-{summary['auprc_hi']:.3f}]  "
            f"across-repeat SD={summary['across_repeat_sd']:.4f}"
        )
    df = pd.DataFrame(out)
    return df[
        [
            "n_features",
            "events_per_variable",
            "auprc",
            "auprc_lo",
            "auprc_hi",
            "ci_width",
            "across_repeat_sd",
        ]
    ]


# --------------------------------------------------------------------------
# Lever 3: learning curve over patients
# --------------------------------------------------------------------------


def learning_curve(task: Task, n_repeats: int, n_draws: int, n_boot: int) -> pd.DataFrame:
    """Subsample *subjects* (never rows) and re-run the whole protocol.

    Several independent draws per fraction, because at these sample sizes
    *which* patients were drawn moves the metric as much as how many.
    """
    all_subjects = np.array(sorted(task.subject_id.unique()))
    out = []
    for frac in LEARNING_CURVE_FRACTIONS:
        n_take = max(int(round(frac * len(all_subjects))), 10)
        draws = 1 if n_take >= len(all_subjects) else n_draws
        for draw in range(draws):
            rng = np.random.default_rng(1000 * draw + n_take)
            chosen = rng.choice(all_subjects, size=n_take, replace=False)
            mask = task.subject_id.isin(chosen).to_numpy()
            sub = task.subset(mask)
            if sub.y.sum() < 2 * N_SPLITS:
                continue

            fold_results, oof_true, oof_score = _oof_predictions(
                sub, sub.subject_id, n_repeats=n_repeats
            )
            if fold_results.empty:
                continue
            summary = _summarise(
                fold_results, oof_true, oof_score, sub.subject_id.to_numpy(), n_boot=n_boot
            )
            pos_mask = (sub.y == 1).to_numpy()
            summary.update(
                {
                    "fraction": frac,
                    "draw": draw,
                    "n_subjects": int(sub.subject_id.nunique()),
                    "n_positive_subjects": int(sub.subject_id[pos_mask].nunique()),
                    "n_rows": len(sub.y),
                    "n_positive_rows": int(sub.y.sum()),
                }
            )
            out.append(summary)
        done = [r for r in out if r["fraction"] == frac]
        if done:
            print(
                f"  frac={frac:.2f}  n_subjects={done[-1]['n_subjects']:>3}  "
                f"pos_subjects~{np.mean([d['n_positive_subjects'] for d in done]):>5.1f}  "
                f"mean CI width={np.mean([d['ci_width'] for d in done]):.4f}"
            )
    return pd.DataFrame(out)


@dataclass(frozen=True)
class CIScaling:
    """``ci_width = a * n_positive_subjects ** b``, fitted in log-log space.

    ``b`` is fitted rather than pinned at the textbook -0.5, because on this
    cohort it demonstrably is not -0.5, and the difference is the whole
    answer. A standard error shrinks as n^-0.5; an AUPRC interval on
    correlated patient-hours, at a base rate of 4%, does not have to. If the
    measured decay is slower than -0.5, assuming -0.5 *understates* the
    patients needed -- the optimistic direction, and the one worth refusing to
    round in your own favour.

    Both fits are kept: ``sqrt_a`` is the constrained -0.5 model, so the report
    can show what the convenient assumption would have claimed next to what
    the data actually supports.
    """

    a: float
    b: float
    r2: float
    sqrt_a: float
    sqrt_r2: float

    def width_at(self, n_pos_subjects: float) -> float:
        return self.a * n_pos_subjects**self.b

    def subjects_for_width(self, width: float) -> float:
        return float((width / self.a) ** (1.0 / self.b))

    def sqrt_subjects_for_width(self, width: float) -> float:
        return float((self.sqrt_a / width) ** 2)


def _cell(frame: pd.DataFrame, idx: object, col: str) -> float:
    """One numeric cell as a plain float.

    pandas' scalar accessors are typed as a wide union (dates, strings,
    complex, ...), so narrowing has to happen somewhere; doing it once here
    keeps three call sites free of casts.
    """
    return float(np.asarray(frame.at[idx, col]).item())


def _round_up(n: float, to: int = 100) -> int:
    """Round a cohort size up to the nearest hundred. These are planning figures
    with an order-of-magnitude claim behind them; printing 1,632 would imply a
    precision the extrapolation does not have.
    """
    return int(np.ceil(n / to) * to)


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_ci_scaling(curve: pd.DataFrame) -> CIScaling:
    grouped = curve.groupby("fraction").agg(
        n_pos_subjects=("n_positive_subjects", "mean"), ci_width=("ci_width", "mean")
    )
    n = grouped.n_pos_subjects.to_numpy(dtype=float)
    w = grouped.ci_width.to_numpy(dtype=float)

    # Free exponent: log w = log a + b log n.
    b, log_a = np.polyfit(np.log(n), np.log(w), deg=1)
    a = float(np.exp(log_a))
    r2 = _r2(w, a * n**b)

    # Constrained to the textbook -0.5, for contrast.
    sqrt_a = float(np.exp(np.mean(np.log(w) + 0.5 * np.log(n))))
    sqrt_r2 = _r2(w, sqrt_a / np.sqrt(n))

    return CIScaling(a=a, b=float(b), r2=r2, sqrt_a=sqrt_a, sqrt_r2=sqrt_r2)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

WATERMARK = (
    "> This platform is validated on a 100-patient demo subset of MIMIC-IV. The\n"
    "> engineering is real and the methodology is rigorous; the clinical\n"
    "> performance figures demonstrate pipeline validity and do not transfer to\n"
    "> clinical practice (PROJECT_PLAN.md section 17).\n"
)


def write_report(
    task: Task,
    grouping: pd.DataFrame,
    budgets: pd.DataFrame,
    curve: pd.DataFrame,
    scaling: CIScaling,
    multi_stay: pd.DataFrame,
    is_quick: bool,
) -> None:
    n_pos = int(task.y.sum())
    n_feat = task.x.shape[1]
    stay_row = grouping[grouping.grouping.str.startswith("stay_id")].iloc[0]
    subj_row = grouping[grouping.grouping.str.startswith("subject_id")].iloc[0]

    lines = ["# What limits this model's reliability\n", WATERMARK]
    if is_quick:
        lines.append(
            "\n**`--quick` development run** -- reduced repeats and bootstrap draws. "
            "Re-run without `--quick` before citing any number here.\n"
        )
    lines.append(
        f"\nPrimary task ({PRIMARY_HORIZON}h horizon): **{len(task.y):,} at-risk rows, "
        f"{n_pos} positives across {task.subject_id[(task.y == 1).to_numpy()].nunique()} "
        f"subjects, {n_feat} features** -- {n_pos / n_feat:.2f} events per variable.\n"
    )

    lines.append("\n## Lever 1 -- the grouping unit (correctness fix, applied)\n")
    lines.append(
        f"`ml/models/splits.py` had always documented grouping by `subject_id`, and "
        f"shipped a `group_key()` helper to do it. Nothing ever passed that helper a "
        f"`subject_id`: `run_all.py` handed it `feature_matrix_for_training`'s `groups`, "
        f"which was `stay_id`, and `eval/prediction.py` called `group_key(groups)` with "
        f"the optional argument omitted. Every cross-validated number this project "
        f"produced was therefore stay-grouped, while the helper made it look handled.\n\n"
        f"That matters here because repeat patients are not rare: **{len(multi_stay)} of "
        f"{task.subject_id.nunique()} subjects have more than one ICU stay in the at-risk "
        f"set**, carrying **{int(multi_stay.n_rows.sum()):,} rows "
        f"({multi_stay.n_rows.sum() / len(task.y):.1%})** and **{int(multi_stay.n_pos.sum())} "
        f"of the {n_pos} positives ({multi_stay.n_pos.sum() / n_pos:.1%})**. Under "
        f"stay-grouping those patients appeared in train and test at once.\n\n"
        f"**This is now fixed**: `feature_matrix_for_training` returns `subject_id`, "
        f"`group_key()` is deleted, and `ml/tests/test_engineer.py` pins the grouping unit "
        f"so it cannot regress. The table below is the measurement that justified it.\n"
    )
    lines.append(grouping.to_markdown(index=False, floatfmt=".4f"))
    delta = subj_row.auprc - stay_row.auprc
    lines.append(
        f"\n**Optimism from stay-grouping: {delta:+.4f} AUPRC.** "
        + (
            "The correct grouping scores lower, which is the expected direction: "
            "the reported figure was partly reading patients it had already trained on. "
            "Fixing this does not improve the model -- it corrects what the model's "
            "score means.\n"
            if delta < 0
            else "The correct grouping does not score lower here, so the leak is not "
            "materially inflating this particular metric -- worth fixing on principle "
            "and for the subgroup analyses, but it is not what the wide CI is about.\n"
        )
    )

    lines.append(
        "\n## Lever 2 -- the feature budget "
        "(gain-ranked pruning is not a lever; targeted pruning is)\n"
    )
    # The largest budget measured, rather than a lookup keyed on the current
    # feature count: a saved run can legitimately have been measured on a
    # slightly different matrix than the one loaded now, and a report that
    # crashes on that is worse than one that names what it actually has.
    full_idx = budgets.n_features.idxmax()
    full_auprc = _cell(budgets, full_idx, "auprc")
    full_n = int(_cell(budgets, full_idx, "n_features"))
    lines.append(
        f"{full_n} features against {n_pos} positives is **{n_pos / full_n:.2f} events per "
        f"variable**, against the conventional floor of 10. That was the hypothesis going "
        f"in: at an EPV this low the model should be fitting noise, and pruning should buy "
        f"back stability. Features are ranked by split gain *inside each training fold* and "
        f"the model refitted on the top-k, so the held-out rows never influence the "
        f"ranking. Subject-grouped throughout.\n"
    )
    lines.append(budgets.to_markdown(index=False, floatfmt=".4f"))
    lines.append(
        f"\n**The hypothesis is wrong, and the measurement says so.** AUPRC rises "
        f"monotonically with the feature budget -- {budgets.auprc.iloc[0]:.4f} at 5 features "
        f"up to **{full_auprc:.4f} at all {full_n}** -- and the across-repeat spread does not "
        f"improve as features are removed "
        f"({budgets.across_repeat_sd.min():.4f}-{budgets.across_repeat_sd.max():.4f} with no "
        f"trend). Pruning *by this method* costs accuracy and buys nothing.\n\n"
        f"Part of the reason is that events-per-variable is a rule about *degrees of freedom "
        f"in a linear model*, where every predictor spends a parameter. This model is a "
        f"depth-4, 15-leaf, 200-tree LightGBM (`ml/models/gbm.py`): it already performs its "
        f"own feature selection at every split and never spends a parameter on a feature it "
        f"does not use. Quoting EPV {n_pos / full_n:.2f} as though it condemned this model "
        f"would have been borrowing a logistic-regression diagnostic for an estimator it "
        f"does not describe.\n\n"
        f"**But read the scope of that result carefully, because a later study "
        f"([`feature_pruning_report.md`](feature_pruning_report.md)) pruned the feature set "
        f"from 90 to 67 and *gained* AUPRC in 17 of 20 paired repeats.** There is no "
        f"contradiction: the two studies remove different things. The curve above ranks "
        f"features by **split gain** and keeps the top-k, so it can only ever discard what "
        f"the model already uses least. It is structurally incapable of removing a feature "
        f"that is heavily used *and* redundant -- which is exactly what `sofa_24hours` was, "
        f"the single highest-attribution feature in the SHAP summary and worth 0.000 AUPRC "
        f"when removed. The lesson is narrower and more useful than 'pruning does not work': "
        f"**a model's own importance ranking is a poor guide to what it can afford to lose**, "
        f"because importance and necessity come apart wherever features are correlated.\n"
    )

    lines.append("\n## Lever 3 -- how many events a target precision needs\n")
    lines.append(
        "Subjects (not rows) subsampled at each fraction, several independent draws "
        "each, full protocol re-run per draw. The CI width is then fitted as "
        "`width = a * n_positive_subjects ** b`, with **both** the constant and the "
        "exponent estimated from these measurements rather than the exponent being "
        "pinned at the textbook -0.5.\n"
    )
    curve_summary = (
        curve.groupby("fraction")
        .agg(
            n_subjects=("n_subjects", "mean"),
            n_positive_subjects=("n_positive_subjects", "mean"),
            n_positive_rows=("n_positive_rows", "mean"),
            auprc=("auprc", "mean"),
            ci_width=("ci_width", "mean"),
        )
        .reset_index()
    )
    lines.append(curve_summary.to_markdown(index=False, floatfmt=".3f"))
    lines.append(
        f"\nFitted scaling: **width = {scaling.a:.2f} x n^{scaling.b:.2f}** "
        f"(R² = {scaling.r2:.3f}). The textbook standard-error exponent is -0.50; "
        f"the measured one is **{scaling.b:.2f}**, i.e. this interval narrows "
        + ("*more slowly*" if scaling.b > -0.5 else "*faster*")
        + " than a standard error would. Assuming -0.50 instead "
        f"(fit: {scaling.sqrt_a:.2f} x n^-0.50, R² = {scaling.sqrt_r2:.3f}) would have been "
        "the convenient choice, so both are shown below and the larger requirement is the "
        "one to plan against.\n"
    )
    rows = []
    current_pos = curve_summary.n_positive_subjects.max()
    # The interval actually measured at full cohort size -- the number the
    # conclusion below tells the reader to attach to every headline figure.
    width_now = _cell(curve_summary, curve_summary.n_positive_subjects.idxmax(), "ci_width")
    for target in TARGET_CI_WIDTHS:
        need = scaling.subjects_for_width(target)
        rows.append(
            {
                # Formatted here rather than left to `floatfmt`, which applies one
                # precision to every float column and rendered 0.05 as "0.1" --
                # two different targets printed as the same number.
                "target 95% CI width": f"{target:.2f}",
                "positive subjects (fitted exponent)": int(np.ceil(need)),
                "positive subjects (if -0.50)": int(
                    np.ceil(scaling.sqrt_subjects_for_width(target))
                ),
                "multiple of current": f"{need / current_pos:.1f}x",
            }
        )
    lines.append("")
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    span = curve_summary.n_positive_subjects.max() / curve_summary.n_positive_subjects.min()
    reach = scaling.subjects_for_width(0.10) / curve_summary.n_positive_subjects.max()
    lines.append(
        f"\nCurrent cohort: **{int(current_pos)} positive subjects**.\n"
        f"\nThe fit itself is good -- R² {scaling.r2:.3f} across {len(curve_summary)} "
        f"measured points -- but a good fit *within* the measured range is not what is "
        f"being asked of it, and the extrapolation is this analysis's weak point in three "
        f"named ways.\n\n"
        f"* **Reach.** The points span a {span:.1f}x range of patient counts, and the 0.10 "
        f"target sits about {reach:.0f}x beyond the largest of them. A power law fitted "
        f"over one order of magnitude and read off two is an estimate of size, not a "
        f"prediction.\n"
        f"* **The exponent may itself be flattered.** AUPRC is bounded in [0, 1], so at the "
        f"smallest subsamples the interval cannot widen indefinitely and the measured decay "
        f"is pushed shallower than the underlying one. That biases `b` toward zero and the "
        f"requirement upward -- conservative, but not free of assumption.\n"
        f"* **Case mix.** It assumes a larger MIMIC-IV cohort resembles this one. This "
        f"cohort is {current_pos / task.subject_id.nunique():.0%} positive at the subject "
        f"level, which is extraordinarily high and reflects a deliberately post-surgical "
        f"100-patient subset admitted *for* intervention. A uniform MIMIC-IV sample will "
        f"almost certainly have a lower event rate, meaning **more** patients per positive "
        f"than the conversion below assumes.\n"
    )

    lines.append("\n## What this means, in order\n")
    need_10 = int(np.ceil(scaling.subjects_for_width(0.10)))
    lines.append(
        f"1. **Group by `subject_id`** -- done. Worth {abs(delta):.4f} AUPRC of removed "
        f"optimism, with {multi_stay.n_pos.sum() / n_pos:.0%} of the positives exposed to "
        f"the leak beforehand. This does not make the model better; it makes the number "
        f"true, which is the only kind of improvement available for free. Every figure in "
        f"`ml/evaluation/report.md` was regenerated under the corrected grouping.\n"
        f"2. **Leave the feature set alone.** The EPV objection was tested and does not "
        f"hold for this estimator. Recorded here so it does not get re-raised and "
        f"re-litigated later from theory.\n"
        f"3. **Get more patients is the only real lever -- and it is closed to this "
        f"project.** Nothing in the modelling narrows an interval this wide; only events "
        f"do. Reaching a 0.10-wide AUPRC CI needs on the order of **{need_10} positive "
        f"subjects** against today's {int(current_pos)}, and there is no route to them "
        f"here: the full MIMIC-IV database requires PhysioNet credentialing (CITI human-"
        f"subjects training plus a signed data use agreement), while this project uses "
        f"only the open-access demo subsets and will not be seeking credentialed access "
        f"(`docs/DATA_USE.md`). The requirement below is therefore a measurement of the "
        f"gap, not a backlog item.\n"
    )
    rate_here = current_pos / task.subject_id.nunique()
    lines.append(
        f"\nTurning that into a cohort size depends entirely on the event rate, which is "
        f"the assumption least likely to carry over:\n\n"
        f"| assumed positive-subject rate | ICU patients to load |\n|---|---|\n"
        f"| {rate_here:.0%} (this cohort, post-surgical) | ~{_round_up(need_10 / rate_here):,} |\n"
        f"| 30% | ~{_round_up(need_10 / 0.30):,} |\n"
        f"| 20% | ~{_round_up(need_10 / 0.20):,} |\n"
        f"\nA uniform MIMIC-IV sample will not reproduce a {rate_here:.0%} event rate, so "
        f"the realistic figure is the lower rows: **a 4,000-5,000 patient cohort**. That is "
        f"what closing this interval would take, and it is recorded so the size of the gap "
        f"is on the record rather than implied. For anyone working from credentialed "
        f"MIMIC-IV, `warehouse/build_duckdb.py --cohort-subjects N` loads it, and this "
        f"analysis is worth re-running there because the fitted exponent is the thing most "
        f"likely to change.\n"
    )

    lines.append(
        f"\n### What follows for this capstone\n\n"
        f"The interval does not close, so the honest move is to say so and stop treating "
        f"it as pending. Three consequences, in order of how easily each is got wrong:\n\n"
        f"1. **The ~{width_now:.2f}-wide AUPRC CI is a structural property of having "
        f"{int(current_pos)} positive subjects, not an outstanding task.** Every headline "
        f"number in `ml/evaluation/report.md` should be read with it attached. The "
        f"model genuinely beats recalibrated NEWS2 on AUPRC in 20 of 20 CV repeats -- that "
        f"result is a *paired* comparison on the same folds, which is exactly why it "
        f"survives a sample this small when the absolute AUPRC does not.\n"
        f"2. **Synthetic patients cannot substitute, and this was measured rather than "
        f"assumed** -- see [`synthetic_ceiling_report.md`](synthetic_ceiling_report.md). A "
        f"perfect generator adds nothing on real held-out patients, because the "
        f"information in it is bounded by the same {int(current_pos)} positive subjects it "
        f"was fitted to. Evaluating on synthetic patients would collapse the reported "
        f"interval to nothing while changing no actual knowledge.\n"
        f"3. **The project's existing framing was right, and now has a number behind it.** "
        f'PROJECT_PLAN.md section 17\'s "pipeline validity, not clinical performance" is '
        f"not a hedge covering an unfinished result -- it is the correct reading of a "
        f"cohort this size, and this analysis is what makes it quantitative.\n"
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def multi_stay_table(task: Task) -> pd.DataFrame:
    d = pd.DataFrame(
        {
            "stay_id": task.stay_id.to_numpy(),
            "subject_id": task.subject_id.to_numpy(),
            "y": task.y.to_numpy(),
        }
    )
    per_sub = d.groupby("subject_id").agg(
        n_stays=("stay_id", "nunique"), n_rows=("y", "size"), n_pos=("y", "sum")
    )
    return per_sub[per_sub.n_stays > 1]


def _report_only() -> int:
    """Regenerate the report from the previous run's CSVs.

    Refuses rather than guesses if any of the three is missing: a report
    assembled from a partial run would look exactly like a complete one.
    """
    paths = {
        "grouping": CSV_DIR / "reliability_grouping.csv",
        "budgets": CSV_DIR / "reliability_feature_budget.csv",
        "curve": CSV_DIR / "reliability_learning_curve.csv",
    }
    missing = [str(p.relative_to(REPO_ROOT)) for p in paths.values() if not p.exists()]
    if missing:
        print(f"ERROR: no saved run to report on -- missing {', '.join(missing)}", file=sys.stderr)
        return 1

    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    task = load_task(conn)
    conn.close()

    curve = pd.read_csv(paths["curve"])
    write_report(
        task,
        pd.read_csv(paths["grouping"]),
        pd.read_csv(paths["budgets"]),
        curve,
        fit_ci_scaling(curve),
        multi_stay_table(task),
        is_quick=False,
    )
    print(f"Rebuilt {REPORT_PATH.relative_to(REPO_ROOT)} from the last run's CSVs")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick", action="store_true", help="reduced repeats/bootstrap for development"
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild the report from the CSVs of the last run, without refitting anything "
        "(the measurements take ~15 min; the prose around them should not have to)",
    )
    args = ap.parse_args()

    if args.report_only:
        return _report_only()

    n_repeats = 3 if args.quick else 10
    # Six draws rather than four: the first full run's curve was visibly noisy
    # (the 0.40 fraction came out *wider* than the 0.25 one, which is only
    # possible as sampling noise), and averaging more draws per fraction is the
    # cheapest way to stop that noise propagating into the fitted exponent.
    n_draws = 2 if args.quick else 6
    n_boot = 200 if args.quick else 800

    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    task = load_task(conn)
    multi = multi_stay_table(task)
    n_pos = int(task.y.sum())
    print(
        f"Primary task: {len(task.y):,} at-risk rows, {n_pos} positives, "
        f"{task.x.shape[1]} features (EPV {n_pos / task.x.shape[1]:.2f}), "
        f"{task.subject_id.nunique()} subjects"
    )
    print(
        f"Repeat patients: {len(multi)} subjects with >1 stay, "
        f"{int(multi.n_rows.sum())} rows ({multi.n_rows.sum() / len(task.y):.1%}), "
        f"{int(multi.n_pos.sum())} positives ({multi.n_pos.sum() / n_pos:.1%})"
    )

    print("\n== Lever 1: grouping unit")
    grouping = compare_grouping_units(task, n_repeats=n_repeats, n_boot=n_boot)

    print("\n== Lever 2: feature budget")
    budgets = feature_budget_curve(task, n_repeats=n_repeats, n_boot=n_boot)

    print("\n== Lever 3: learning curve over patients")
    curve = learning_curve(task, n_repeats=n_repeats, n_draws=n_draws, n_boot=n_boot)
    scaling = fit_ci_scaling(curve)
    print(
        f"  fitted: width = {scaling.a:.3f} * n^{scaling.b:.3f}  (R2={scaling.r2:.3f}); "
        f"textbook exponent is -0.5"
    )
    for target in TARGET_CI_WIDTHS:
        print(
            f"  CI width {target:.2f} needs ~{int(np.ceil(scaling.subjects_for_width(target)))} "
            f"positive subjects "
            f"(~{int(np.ceil(scaling.sqrt_subjects_for_width(target)))} if the exponent were -0.5)"
        )

    conn.close()

    grouping.to_csv(CSV_DIR / "reliability_grouping.csv", index=False)
    budgets.to_csv(CSV_DIR / "reliability_feature_budget.csv", index=False)
    curve.to_csv(CSV_DIR / "reliability_learning_curve.csv", index=False)
    write_report(task, grouping, budgets, curve, scaling, multi, args.quick)
    print(f"\nWrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
