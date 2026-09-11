"""Does making the model disease-aware help, or does it just leak the outcome?

"Add the diagnosis as a feature" sounds like free signal. In MIMIC it is not, and
the reason is administrative rather than clinical: `diagnoses_icd` is populated by
**billing coders after the patient is discharged**. A primary diagnosis is therefore
a summary of how the admission turned out, not a fact available at the bedside in
hour 3 -- and this project's label is a composite of death, vasopressor initiation,
invasive ventilation initiation, and ICU bounce-back. "Acute respiratory failure
with hypoxia" as a discharge-coded primary diagnosis is uncomfortably close to a
restatement of the ventilation component of the label.

Charlson's comorbidity flags do not have that problem by construction. The index
exists to score *pre-existing chronic* burden for 10-year mortality: diabetes, CHF,
COPD, renal disease, prior MI. Those describe the patient who walked in.

So "disease-aware" splits into two claims with very different evidential standing,
and this module measures them separately rather than shipping both and hoping:

1. **Does each disease feature set beat the disease-blind model?** Paired repeated
   grouped CV, same folds for every arm, judged on the paired per-repeat win count
   against `none` -- the same standard `feature_pruning.py` and finding F4 used, and
   for the same reason: with 49 positive subjects the mean AUPRC difference between
   two arms is routinely smaller than the CI on either one, while the paired count
   at least controls for which folds were drawn.

2. **Is `dx_chapter` leaking?** Two probes that a merely-predictive feature passes
   and a leaking one fails:

   * **Chapter alone.** Fit on nothing but the discharge-coded chapter -- no vitals,
     no trend, no age. A patient-constant column with no physiology in it should be
     near the base rate. Materially above it means the chapter is carrying outcome
     information that a bedside model would not have.
   * **Chapter against each label component.** The composite has four components,
     and a leak is usually component-specific: if the Respiratory chapter's lift is
     concentrated in the ventilation component, that is the coder describing the
     intubation, not the lungs predicting it.

Neither probe is decisive alone -- a genuinely predictive case-mix feature also
raises both, because sick case mixes really do deteriorate more. What makes the
combination readable is the *shape*: broad, modest lift across components is case
mix; a spike in one component that matches the chapter's own organ system is a leak.

Output: `disease_leakage.csv` and `disease_leakage_report.md`. Nothing here changes
a default; `ml/features/engineer.DEFAULT_DISEASE_FEATURES` is set by hand from what
this reports, so the decision stays a stated judgement rather than a silent one.
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.evaluation import metrics  # noqa: E402
from ml.evaluation.run_all import (  # noqa: E402
    WAREHOUSE_DB,
    gbm_fit_predict,
    run_cv_for_model,
)
from ml.features import engineer, labels  # noqa: E402

REPORT_PATH = REPO_ROOT / "ml" / "evaluation" / "disease_leakage_report.md"
CSV_PATH = REPO_ROOT / "ml" / "evaluation" / "disease_leakage.csv"

HORIZON = 6
BASELINE_ARM = "none"
# Ordered from disease-blind outward, so the report reads as a progression.
ARMS = ["none", "chronic_index_only", "chronic", "coded_only", "all"]


def build_arm(features: pd.DataFrame, lab: pd.DataFrame, arm: str):
    return engineer.feature_matrix_for_training(
        features, lab, f"label_{HORIZON}h", disease_features=arm
    )


def chapter_alone_probe(
    features: pd.DataFrame, lab: pd.DataFrame, n_repeats: int
) -> tuple[float, float]:
    """AUPRC of a model given ONLY ``dx_chapter``, against the label's base rate.

    No vitals, no trend, no age -- if a single patient-constant billing code beats
    the base rate by a wide margin, it is carrying outcome information.
    """
    merged = lab[["stay_id", "hour", f"label_{HORIZON}h"]].merge(
        features, on=["stay_id", "hour"], how="inner"
    )
    x = merged[["dx_chapter"]].copy()
    y = merged[f"label_{HORIZON}h"]
    groups = merged["subject_id"]
    fold_results, _, _, _ = run_cv_for_model(
        "chapter_alone", gbm_fit_predict, x, y, groups, n_repeats=n_repeats
    )
    return float(fold_results.auprc.mean()), float(y.mean())


def per_component_lift(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """For each chapter x label component: share of that chapter's stays that ever
    have that component, against the cohort-wide share.

    This is the shape test. Case mix raises everything a bit; a leak spikes the one
    component that matches the chapter's own organ system.
    """
    disease = conn.execute("select stay_id, dx_chapter from capstone.disease_context").fetchdf()
    tables = labels.all_event_tables(conn)
    n_stays = disease.stay_id.nunique()

    rows = []
    for table in tables:
        has_event = set(table.frame.stay_id)
        overall = len(has_event & set(disease.stay_id)) / n_stays
        for chapter, g in disease.groupby("dx_chapter"):
            stays = set(g.stay_id)
            if len(stays) < 5:  # a rate over four patients is not a rate
                continue
            share = len(stays & has_event) / len(stays)
            rows.append(
                {
                    "dx_chapter": chapter,
                    "component": table.name,
                    "stays": len(stays),
                    "share_with_event": share,
                    "cohort_share": overall,
                    "lift": share / overall if overall else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--db", type=Path, default=WAREHOUSE_DB)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db), read_only=True)
    grid = conn.execute("select stay_id, hour from capstone.hourly_grid").fetchdf()
    lab = labels.build_labels(conn, grid, horizons=(HORIZON,))
    features = engineer.build_feature_frame(conn)

    print(f"=== Disease feature arms ({args.repeats} repeats x 5 folds, horizon {HORIZON}h) ===")
    all_folds = []
    summary_rows = []
    for arm in ARMS:
        x, y, groups = build_arm(features, lab, arm)
        fold_results, _, _, _ = run_cv_for_model(
            arm, gbm_fit_predict, x, y, groups, n_repeats=args.repeats
        )
        all_folds.append(fold_results)
        summary_rows.append(
            {
                "arm": arm,
                "n_features": x.shape[1],
                "disease_columns": len(engineer.disease_feature_columns(arm)),
                "mean_auprc": float(fold_results.auprc.mean()),
                "mean_auroc": float(fold_results.auroc.mean()),
            }
        )

    folds = pd.concat(all_folds, ignore_index=True)
    for row in summary_rows:
        if row["arm"] == BASELINE_ARM:
            row["wins_vs_none"], row["repeats"] = None, args.repeats
            continue
        wins, total = metrics.beats_baseline_in_n_of_k_repeats(folds, row["arm"], BASELINE_ARM)
        row["wins_vs_none"], row["repeats"] = wins, total

    summary = pd.DataFrame(summary_rows)
    base_auprc = float(summary.loc[summary.arm == BASELINE_ARM, "mean_auprc"].iloc[0])
    summary["delta_auprc"] = summary.mean_auprc - base_auprc
    summary.to_csv(CSV_PATH, index=False)
    print("\n" + summary.to_string(index=False))

    print("\n=== Probe 1: dx_chapter alone ===")
    chapter_auprc, base_rate = chapter_alone_probe(features, lab, args.repeats)
    print(f"  AUPRC from dx_chapter alone: {chapter_auprc:.4f} (base rate {base_rate:.4f})")

    print("\n=== Probe 2: per-component lift ===")
    lift = per_component_lift(conn)
    pivot = lift.pivot(index="dx_chapter", columns="component", values="lift").round(2)
    print(pivot.to_string())
    conn.close()

    REPORT_PATH.write_text(_report(summary, chapter_auprc, base_rate, pivot, lift, args.repeats))
    print(f"\nWrote {CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


def _report(
    summary: pd.DataFrame,
    chapter_auprc: float,
    base_rate: float,
    pivot: pd.DataFrame,
    lift: pd.DataFrame,
    n_repeats: int,
) -> str:
    best = summary.loc[summary.mean_auprc.idxmax()]
    ratio = chapter_auprc / base_rate if base_rate else float("nan")
    worst = lift.loc[lift.lift.idxmax()]
    return f"""# Is the disease-aware model learning physiology or reading the coder's notes?

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice (PROJECT_PLAN.md section 17).

`diagnoses_icd` is assigned by billing coders **after discharge**, so a primary
diagnosis summarises how an admission turned out rather than what was knowable at
the bedside. Charlson's comorbidity flags are pre-existing chronic conditions by
construction. This study measures the two separately -- see the module docstring in
`ml/evaluation/disease_leakage.py` for the full argument.

Protocol: {n_repeats} repeats x 5 folds, subject-grouped stratified CV, identical
folds across arms, LightGBM throughout. Judged on the **paired per-repeat win count**
against the disease-blind arm, not on mean AUPRC alone -- with 49 positive subjects
the difference between two arms is routinely smaller than the CI on either.

## Arms

{summary.to_markdown(index=False)}

Best mean AUPRC: **`{best.arm}`** at {best.mean_auprc:.4f}
({best.delta_auprc:+.4f} against disease-blind).

## Probe 1 -- `dx_chapter` alone

A model given nothing but the discharge-coded chapter -- no vitals, no trend, no age:

| | AUPRC |
|---|---|
| `dx_chapter` alone | {chapter_auprc:.4f} |
| Label base rate | {base_rate:.4f} |
| Ratio | **{ratio:.2f}x** |

A patient-constant billing code contains no physiology. Whatever it scores above the
base rate is either genuine case mix (sicker diagnoses really do deteriorate more) or
outcome information written into the chart after the fact. Probe 2 separates them.

## Probe 2 -- lift by label component

Share of each chapter's stays that ever have each composite component, divided by the
cohort-wide share. Chapters with fewer than 5 stays are omitted (a rate over four
patients is not a rate).

{pivot.to_markdown()}

The shape is what matters. Broad, modest lift across all four components is case mix.
A spike in the single component matching the chapter's own organ system is the coder
describing the event. The largest single cell here is
**{worst.dx_chapter} x {worst.component} at {worst.lift:.2f}x**
({worst.stays} stays, {worst.share_with_event:.0%} against a cohort {worst.cohort_share:.0%}).

## What ships

`ml/features/engineer.DEFAULT_DISEASE_FEATURES` is set **by hand** from this report,
not automatically from the winning arm. An arm can win on AUPRC precisely *because*
it leaks, so promoting the top row mechanically would be the exact error this study
exists to prevent.
"""


if __name__ == "__main__":
    raise SystemExit(main())
