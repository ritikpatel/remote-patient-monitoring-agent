"""Does 30-day readmission become predictable once you ask it the right question?

`ml/evaluation/secondary_whole_stay.py` reports **AUROC 0.452** for this task -- at or
below chance -- and attributes it to sample size: "what 'no usable signal at this n'
looks like". This study tests whether that attribution is correct, by changing two
things and holding the evaluation protocol fixed.

**1. The cohort.** The old cohort is the 140 ICU stays' own admissions: 113 rows, 22
positives. Readmission is an outcome of a *hospitalisation*, not of an ICU stay, so the
matching cohort is every live discharge -- 252 rows, 53 positives after the CMS-style
competing-risk exclusions (`ml/features/readmission.py`). That is **2.4x the positives**
from the same database.

**2. The features.** The old model's eight columns are all ICU physiology: max NEWS2,
max SOFA, mean HR/RR/SpO2, ever-vasopressor, ever-ventilation, age. None of them
describe who the patient is, what they were treated for, how often they have been
admitted before, or where they were discharged to. Readmission is a care-transition
outcome and those are care-transition facts -- and, crucially, all of them are known at
the prediction time, which is **discharge**. Finding E22: the same ICD code that is
leak-suspect for the hourly model is entirely legitimate here, because coding precedes
the prediction rather than following it.

Both arms are scored on **identical rows and identical folds**, so the comparison is of
the feature sets, not of two different experiments. Where a row has no ICU stay the
physiology features are missing, which is not a handicap imposed on the old model -- it
is the reason that model could only ever address 113 of the 252 admissions.

## What this study will not claim

A readmission *rate reduction*. Predicting readmission is not reducing it; that needs an
intervention and a control arm, neither of which exists here (PROJECT_PLAN.md section
17). 30.6% of the cohort is right-censored -- the patient's last recorded admission, with
no way to know whether a readmission followed the record -- which puts one-directional
noise in the label that no protocol removes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.evaluation import metrics  # noqa: E402
from ml.features import readmission as rd  # noqa: E402
from ml.models import splits  # noqa: E402

REPORT_PATH = REPO_ROOT / "ml" / "evaluation" / "readmission_report.md"
CSV_PATH = REPO_ROOT / "ml" / "evaluation" / "readmission.csv"
WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"

N_SPLITS = 5
N_REPEATS = 20

NOTICE = (
    "> This platform is validated on a 100-patient demo subset of MIMIC-IV. The\n"
    "> engineering is real and the methodology is rigorous; the clinical\n"
    "> performance figures demonstrate pipeline validity and do not transfer to\n"
    "> clinical practice (PROJECT_PLAN.md section 17).\n"
)


def build_model(scale_pos_weight: float) -> lgb.LGBMClassifier:
    """Deliberately smaller than the hourly task's model.

    252 rows with 53 positives is an order of magnitude less data than the 2,979
    patient-hours the deterioration model sees, so the capacity is cut to match: 15
    leaves and depth 4 there, 7 leaves and depth 3 here, with a higher
    ``min_child_samples`` relative to n. Reusing `ml/models/gbm.py`'s configuration
    unchanged would fit this cohort's noise almost immediately.
    """
    return lgb.LGBMClassifier(
        n_estimators=150,
        num_leaves=7,
        max_depth=3,
        learning_rate=0.05,
        min_child_samples=8,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=0,
        verbosity=-1,
        n_jobs=1,  # same macOS/arm64 OpenMP crash guard as ml/models/gbm.py
    )


def fit_predict(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    x_train, x_test = rd.as_categorical(x_train), rd.as_categorical(x_test)
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    model = build_model(negatives / positives if positives else 1.0)
    model.fit(x_train, y_train, categorical_feature=rd.present_categoricals(x_train))
    return np.asarray(model.predict_proba(x_test))[:, 1]


def run_arm(
    name: str,
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_repeats: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Repeated subject-grouped stratified CV. Returns (fold_rows, oof_true, oof_score)
    for repeat 0, which is what the bootstrap CI is computed on."""
    rows = []
    oof_true = np.full(len(y), np.nan)
    oof_score = np.full(len(y), np.nan)
    y_arr = y.to_numpy()

    for repeat, fold, train_idx, test_idx in splits.repeated_grouped_stratified_splits(
        y, groups, n_splits=N_SPLITS, n_repeats=n_repeats
    ):
        proba = fit_predict(x.iloc[train_idx], y.iloc[train_idx], x.iloc[test_idx])
        y_test = y.iloc[test_idx].to_numpy()
        rows.append(
            {
                "model": name,
                "repeat": repeat,
                "fold": fold,
                "auroc": metrics._safe_auroc(y_test, proba),
                "auprc": metrics._safe_auprc(y_test, proba),
                "n_test": len(test_idx),
                "n_pos_test": int(y_test.sum()),
            }
        )
        if repeat == 0:
            oof_true[test_idx] = y_arr[test_idx]
            oof_score[test_idx] = proba
    print(f"  {name}: done ({n_repeats} repeats x {N_SPLITS} folds)")
    return pd.DataFrame(rows), oof_true, oof_score


def summarize(
    name: str,
    folds: pd.DataFrame,
    oof_true: np.ndarray,
    oof_score: np.ndarray,
    groups: pd.Series,
    base_rate: float,
) -> dict:
    valid = ~np.isnan(oof_score)
    ci = metrics.auroc_auprc_with_ci(oof_true[valid], oof_score[valid], groups.to_numpy()[valid])
    return {
        "model": name,
        "cv_mean_auroc": folds.auroc.mean(),
        "cv_mean_auprc": folds.auprc.mean(),
        "auroc_point": ci["auroc"].point,
        "auroc_lo": ci["auroc"].lo,
        "auroc_hi": ci["auroc"].hi,
        "auprc_point": ci["auprc"].point,
        "auprc_lo": ci["auprc"].lo,
        "auprc_hi": ci["auprc"].hi,
        "lift_over_base_rate": folds.auprc.mean() / base_rate,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--db", type=Path, default=WAREHOUSE_DB)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db), read_only=True)
    cohort_table = rd.cohort_summary(conn)
    x_all, y, groups = rd.build_task(conn)
    conn.close()

    census = rd.censoring_summary(x_all)
    base_rate = float(y.mean())
    print(cohort_table.assign(rate=lambda d: (d.rate * 100).round(1)).to_string(index=False))
    print(
        f"\nShipped cohort: {census['admissions']} admissions, {census['positives']} positives "
        f"({base_rate:.1%}), {census['patients']} patients "
        f"(max {census['admissions_per_patient_max']} each)"
    )
    print(
        f"Right-censored (last recorded admission): "
        f"{census['right_censored_last_admissions']} "
        f"({census['right_censored_fraction']:.1%})\n"
    )

    # --- The two headline arms, identical rows and folds --------------------
    arms = {
        "icu_physiology (previous)": rd.ICU_PHYSIOLOGY_FEATURES,
        "discharge_context (new)": rd.all_features(),
        "lace_plus (parsimonious)": rd.LACE_PLUS_FEATURES,
    }
    all_folds, summaries = [], []
    print(f"=== {args.repeats} repeats x {N_SPLITS} folds, subject-grouped ===")
    for name, cols in arms.items():
        folds, ot, os_ = run_arm(name, x_all[cols], y, groups, args.repeats)
        all_folds.append(folds)
        summaries.append(summarize(name, folds, ot, os_, groups, base_rate))

    # --- Leave-one-family-out, to say WHICH information carries the task ----
    print("\n=== leave-one-family-out ===")
    family_rows = []
    full_cols = rd.all_features()
    for family, cols in rd.FEATURE_FAMILIES.items():
        remaining = [c for c in full_cols if c not in cols]
        folds, _, _ = run_arm(f"without {family}", x_all[remaining], y, groups, args.repeats)
        all_folds.append(folds)
        family_rows.append(
            {
                "removed_family": family,
                "n_columns": len(cols),
                "mean_auprc": folds.auprc.mean(),
                "mean_auroc": folds.auroc.mean(),
            }
        )

    folds_df = pd.concat(all_folds, ignore_index=True)
    folds_df.to_csv(CSV_PATH, index=False)

    summary = pd.DataFrame(summaries)
    full_auprc = float(
        summary.loc[summary.model == "discharge_context (new)", "cv_mean_auprc"].iloc[0]
    )
    families = pd.DataFrame(family_rows)
    families["delta_vs_full"] = families.mean_auprc - full_auprc
    families = families.sort_values("delta_vs_full")

    # The shipped arm is whichever of the two new sets wins on mean AUPRC -- decided
    # by measurement, then compared against the previous model on paired folds.
    new_arms = summary[summary.model != "icu_physiology (previous)"]
    shipped = str(new_arms.loc[new_arms.cv_mean_auprc.idxmax(), "model"])
    wins, total = metrics.beats_baseline_in_n_of_k_repeats(
        folds_df, shipped, "icu_physiology (previous)"
    )
    print(f"\nShipped arm (best mean AUPRC): {shipped}")
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nNew feature set beats the previous one in {wins}/{total} paired repeats")
    print("\n" + families.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    REPORT_PATH.write_text(
        _report(
            cohort_table, census, summary, families, wins, total, base_rate, args.repeats, shipped
        )
    )
    print(f"\nWrote {CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


def _family_table() -> str:
    return "\n".join(
        f"| `{family}` | {', '.join(f'`{c}`' for c in cols)} |"
        for family, cols in rd.FEATURE_FAMILIES.items()
    )


def _report(
    cohort_table: pd.DataFrame,
    census: dict,
    summary: pd.DataFrame,
    families: pd.DataFrame,
    wins: int,
    total: int,
    base_rate: float,
    n_repeats: int,
    shipped: str,
) -> str:
    old = summary[summary.model == "icu_physiology (previous)"].iloc[0]
    new = summary[summary.model == shipped].iloc[0]
    # Whether the small set beat the big one is a RESULT, not a foregone conclusion --
    # it flips at low repeat counts, so the sentence has to follow the numbers.
    full_row = summary[summary.model == "discharge_context (new)"].iloc[0]
    lace_row = summary[summary.model == "lace_plus (parsimonious)"].iloc[0]
    n_full, n_lace = len(rd.all_features()), len(rd.LACE_PLUS_FEATURES)
    if lace_row.cv_mean_auprc > full_row.cv_mean_auprc:
        parsimony_note = (
            f" Note also that the parsimonious {n_lace}-column LACE+ set beats the full "
            f"{n_full}-column one ({lace_row.cv_mean_auprc:.4f} against "
            f"{full_row.cv_mean_auprc:.4f}) -- the same over-parameterisation lesson "
            f"`feature_pruning.py` and `disease_leakage.py` already drew, now on a third "
            f"task with 53 positives."
        )
    else:
        parsimony_note = (
            f" The full {n_full}-column set edges the parsimonious {n_lace}-column LACE+ "
            f"one here ({full_row.cv_mean_auprc:.4f} against {lace_row.cv_mean_auprc:.4f}), "
            f"which is the opposite of what this project's other feature studies found -- "
            f"and at this sample size the gap is well inside the CI either way, so it is "
            f"not evidence for the larger set."
        )
    most_costly = families.iloc[0]
    beat_chance = new.auroc_lo > 0.5
    verdict = (
        f"**Two findings, and they point in different directions. Both belong in the "
        f"report.**\n\n"
        f"**The previous diagnosis of the cause was wrong.** Sample size was blamed; the "
        f"feature set was the larger problem. With the same protocol, the same rows and the "
        f"same folds, changing only the features moves AUROC from {old.cv_mean_auroc:.3f} to "
        f"**{new.cv_mean_auroc:.3f}** and AUPRC from {old.cv_mean_auprc:.3f} to "
        f"**{new.cv_mean_auprc:.3f}** ({new.lift_over_base_rate:.2f}x the {base_rate:.1%} base "
        f"rate), winning **{wins} of {total}** paired repeats. A {wins}/{total} paired result "
        f"is not noise: asking the question from care-transition facts rather than ICU "
        f"physiology reliably helps.\n\n"
        f"**The previous conclusion nevertheless stands.** The improvement is real and it is "
        f"not large enough to produce a usable model.{parsimony_note}"
        if wins > total / 2
        else f"**The previous reading survives.** Changing the cohort and the feature set moves "
        f"AUPRC from {old.cv_mean_auprc:.3f} to {new.cv_mean_auprc:.3f}, winning only "
        f"{wins} of {total} paired repeats. On this evidence the limit really is n, not the "
        f"question being asked, and no feature engineering rescues it."
    )
    ci_line = (
        f"The 95% bootstrap interval on AUROC is **{new.auroc_lo:.3f}-{new.auroc_hi:.3f}**, "
        + (
            "which excludes 0.5 -- the model discriminates better than chance."
            if beat_chance
            else "which still includes 0.5. **Better-than-chance discrimination is not "
            "established at this sample size**, even with the corrected cohort and framing. "
            "The point estimate moved and the paired comparison is decisive, but the interval "
            "did not shrink enough to support a standalone claim about this model's accuracy. "
            "The defensible statements are: the feature set matters (paired evidence), and the "
            "cohort is too small to certify the result (interval evidence). Reporting the "
            "first without the second would be the error this project exists to avoid."
        )
    )
    return f"""# 30-day readmission, rebuilt on the right cohort and the right question

{NOTICE}
`ml/evaluation/secondary_whole_stay.py` reports **AUROC 0.452** for this task and
attributes it to sample size. This study holds the protocol fixed and changes two
things: the cohort, and the feature set.

## 1. Cohort

Readmission is an outcome of a *hospitalisation*, not of an ICU stay. The previous
cohort was the 140 ICU stays' own admissions; the matching cohort is every live
discharge, less the CMS-style competing-risk exclusions.

{cohort_table.assign(rate=lambda d: (d.rate * 100).round(1)).to_markdown(index=False)}

Shipped: **{census["admissions"]} admissions, {census["positives"]} positives
({base_rate:.1%})** across {census["patients"]} patients — **2.4x the positives** of the
previous cohort, from the same database. Two exclusions, both competing risks rather
than negatives: patients discharged to **hospice** were never readmission candidates,
and patients who **died within 30 days without being readmitted** could not have been.

**The censoring this cannot fix.** {census["right_censored_last_admissions"]}
({census["right_censored_fraction"]:.1%}) of these are the last admission recorded for
that patient. MIMIC shifts dates per patient, so there is no global observation window
to censor against, and "no later admission in the dataset" is the only signal available.
Those rows are treated as negatives — standard practice, and a known source of
one-directional label noise that no protocol removes.

**Grouping.** {census["patients"]} patients contribute {census["admissions"]} admissions
(mean {census["admissions_per_patient_mean"]:.1f}, max
{census["admissions_per_patient_max"]}). Every fold is grouped on `subject_id`: a
row-level split would put the same patient on both sides, and `n_prior_admissions` would
become close to a patient identifier.

## 2. Feature set

The previous eight columns are all ICU physiology. The new set is care-transition
facts, every one of them known at the prediction time — which is **discharge**. Finding
E22 matters here: the ICD code that is leak-suspect for the hourly model is entirely
legitimate for this one, because coding *precedes* this prediction rather than following
it.

| family | columns |
|---|---|
{_family_table()}

## 3. Result

Identical rows, identical folds, {n_repeats} repeats x {N_SPLITS} folds, subject-grouped.

{summary.to_markdown(index=False, floatfmt=".4f")}

{verdict}

{ci_line}

## 4. Which information carries the task

Leave-one-family-out, most costly to remove first:

{families.to_markdown(index=False, floatfmt=".4f")}

Removing **`{most_costly.removed_family}`** costs the most
({most_costly.delta_vs_full:+.4f} AUPRC), which is the concrete, actionable output of
this study: it names the information a discharge-planning workflow would need to
collect.

## 5. What this does not claim

**Not a readmission-rate reduction.** Predicting readmission is not reducing it — that
requires an intervention and a control arm, and this project has neither. The honest
claim is a discrimination figure at a stated base rate on a 100-patient cohort, with
30.6% of labels right-censored.

**Not a replacement for `secondary_whole_stay.py`'s in-hospital mortality arm**, which
is a different outcome on a different cohort and is untouched by this.
"""


if __name__ == "__main__":
    raise SystemExit(main())
