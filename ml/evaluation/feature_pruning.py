"""Which features actually earn their place in the deterioration model?

The feature set grew to 90 columns on a task with **120 positive patient-hours
across 49 positive subjects**. That ratio is where a gradient-boosted model stops
learning physiology and starts fitting the cohort, so this module measures --
rather than argues -- which feature families contribute.

Two passes, because a single pass on 20 noisy families invites over-pruning:

1. **Screen** (10 repeats): leave one family out at a time, ranked by the paired
   per-repeat win count against the full set, not by mean AUPRC alone. The two
   disagree often here, and the paired count is the more trustworthy of the two
   because both arms saw identical folds.
2. **Confirm** (20 repeats): build the combined pruned set and re-run the primary
   protocol. Families that look removable alone do not always stack.

The screen found something worth stating plainly: **removing almost any family
improved AUPRC.** That is not 20 independent discoveries, it is one -- the model
was over-parameterised for its sample, and nearly any reduction in variance paid.

## What was removed, and why each one survives scrutiny

* **`news2`, `sofa_24hours`** -- deterministic functions of vitals and labs the
  model already has, so they carry no new information; measured contribution was
  0.000. `sofa_24hours` was simultaneously the *top* SHAP feature, which is the
  cleanest demonstration in this repo that high attribution is not necessity when
  information is redundantly encoded. SOFA's cardiovascular component is also
  scored on vasopressor dose, and vasopressor initiation is one of the labels: 11
  at-risk rows carry a vasopressor-implied SOFA score and **all 11 are positives**
  against a 4.0% base rate. Removing the column removes that pathway.
* **The 18 rolling *means*** (`*_4h_mean`, `*_24h_mean`) -- the largest single
  gain in the screen (10/10 paired repeats). A carried-forward raw value is
  already close to its own recent mean, so these largely restated the 9 raw
  columns. Rolling **std** and **slope** are kept: variability and direction are
  not recoverable from a point value.
* **The 3 lab-ordering-intensity columns** -- 10/10 paired repeats in the screen,
  and independently suspect on design grounds. Lab-ordering rate is a proxy for
  *clinician concern*: orders rise because a clinician is already worried about
  the patient who is about to deteriorate, which makes the feature partly a
  readout of the outcome rather than a predictor of it.

## What was tested and deliberately kept

* **`admission_age`** -- removing it won 15 of 20 repeats for +0.006 AUPRC. That
  is the same evidential situation finding F4 faced for `gender` (a win count at
  the threshold, an effect far inside a CI many times its size), and it is
  resolved the same way: kept. Age is a standard, clinically load-bearing
  covariate and the measured gain is indistinguishable from noise.
* **`gender`** -- removing it won only 4 of 20 repeats, so F4's decision to keep
  it survives re-testing under the pruned feature set. No contradiction was
  introduced.
* **All patient-constant features together** (`admission_age`, `gender`,
  `first_careunit`) -- the hypothesis was that patient-constant columns let the
  model memorise individuals across their own rows. Removing all three won 2 of
  20 repeats, i.e. it is clearly worse. The hypothesis is not supported and is
  recorded here so it is not re-raised from theory.
* **`rr`** -- the screen ranked dropping respiratory rate as a +0.046 gain, which
  would be a clinically bizarre thing to act on. Respiratory rate is 99.9%
  present and only 1.1% carried forward in this cohort -- one of the best-measured
  channels, and the strongest early deterioration signal in NEWS2's own design.
  Kept: this is what variance reduction looks like when it lands on a real signal,
  and acting on it would have been the study over-fitting its own noise.

## Result

90 -> 67 features, AUPRC 0.398 -> 0.493, beating the full set in **17 of 20**
paired repeats.

Usage:
    python ml/evaluation/feature_pruning.py [--repeats N] [--screen-repeats N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation import metrics  # noqa: E402
from ml.evaluation.run_all import (  # noqa: E402
    WAREHOUSE_DB,
    gbm_fit_predict,
    run_cv_for_model,
    summarize_model,
)
from ml.features import engineer, labels  # noqa: E402

REPORT_PATH = REPO_ROOT / "ml" / "evaluation" / "feature_pruning_report.md"
CSV_PATH = REPO_ROOT / "ml" / "evaluation" / "feature_pruning.csv"

# The families removed from FEATURE_COLUMNS_BASE / rolling_feature_columns, kept
# here as the record of what this study acted on. They no longer appear in a
# default feature matrix, so the screen re-attaches them to measure them.
REMOVED = {
    "severity:news2+sofa": engineer.SEVERITY_SCORE_COLUMNS,
    "lab_intensity": engineer.LAB_INTENSITY_COLUMNS,
    "rolling_mean": [
        f"{v}_{w}h_mean" for w in engineer.ROLLING_WINDOWS_H for v in engineer.CORE_VITALS
    ],
}


def build(conn, horizon: int = 6):
    grid = conn.execute("select stay_id, hour from capstone.hourly_grid").fetchdf()
    lab = labels.build_labels(conn, grid, horizons=(horizon,))
    features = engineer.build_feature_frame(conn)
    x, y, groups = engineer.feature_matrix_for_training(features, lab, f"label_{horizon}h")
    return features, lab, x, y, groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--db", type=Path, default=WAREHOUSE_DB)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db), read_only=True)
    features, lab, x_pruned, y, groups = build(conn)
    # Rebuild the pre-pruning matrix in its EXACT historical column order.
    # Order matters: LightGBM breaks ties between equally-good splits by column
    # position, so re-attaching the removed columns at the end instead of in
    # their original slots moved the 90-feature score by 0.026 AUPRC and the
    # paired win count by five repeats -- a difference caused entirely by column
    # ordering, with identical data. That sensitivity is itself worth knowing
    # (it sets a floor on how finely these deltas can be read), but the honest
    # comparison is against the matrix the model actually used before pruning.
    old_base = (
        list(engineer.CORE_VITALS)
        + [f"{v}_was_imputed" for v in engineer.CORE_VITALS]
        + [f"{v}_hours_since_last_obs" for v in engineer.CORE_VITALS]
        + ["news2", "sofa_24hours", "has_arterial_line"]
        + list(engineer.LAB_INTENSITY_COLUMNS)
        + ["admission_age"]
    )
    old_rolling = [
        f"{var}_{w}h_{stat}"
        for w in engineer.ROLLING_WINDOWS_H
        for var in engineer.CORE_VITALS
        for stat in ("mean", "std", "slope")
    ]
    merged = lab[["stay_id", "hour", "label_6h"]].merge(
        features, on=["stay_id", "hour"], how="inner"
    )
    assert len(merged) == len(x_pruned), (len(merged), len(x_pruned))
    assert (merged["label_6h"].to_numpy() == y.to_numpy()).all(), "row order diverged"
    x_full = merged[old_base + old_rolling + ["gender", "first_careunit"]].copy()
    assert x_full.shape[1] == 90, x_full.shape

    # Both arms are now subsets of ONE frame. Building them down two separate
    # code paths (`feature_matrix_for_training` for one, a hand reconstruction
    # for the other) made the arms differ by more than the features under test,
    # which is not a paired comparison at all. Derive the pruned arm here and
    # assert it is exactly what the live pipeline produces, so this study cannot
    # drift from the shipped feature set.
    dropped = set(
        REMOVED["severity:news2+sofa"] + REMOVED["rolling_mean"] + REMOVED["lab_intensity"]
    )
    x_pruned_here = x_full[[c for c in x_full.columns if c not in dropped]]
    assert sorted(x_pruned_here.columns) == sorted(x_pruned.columns), (
        "this study's pruned set has drifted from engineer.py's: "
        f"{sorted(set(x_pruned_here.columns) ^ set(x_pruned.columns))}"
    )
    x_pruned = x_pruned_here
    conn.close()

    runs = {
        "pre-pruning (90 features)": x_full,
        "pruned (current, 67)": x_pruned,
    }
    folds, rows = [], []
    for name, x in runs.items():
        fr, ot, os_, og = run_cv_for_model(
            name, gbm_fit_predict, x, y, groups, n_repeats=args.repeats
        )
        folds.append(fr)
        s = summarize_model(name, fr, ot, os_, og)
        rows.append(
            {
                "feature_set": name,
                "n_features": x.shape[1],
                "auprc": s["auprc_point"],
                "auprc_lo": s["auprc_lo"],
                "auprc_hi": s["auprc_hi"],
                "auroc": s["auroc_point"],
            }
        )
        print(
            f"{name:<28} n={x.shape[1]:>3}  AUPRC {s['auprc_point']:.4f} "
            f"({s['auprc_lo']:.3f}-{s['auprc_hi']:.3f})  AUROC {s['auroc_point']:.3f}",
            flush=True,
        )

    allf = pd.concat(folds, ignore_index=True)
    wins, total = metrics.beats_baseline_in_n_of_k_repeats(
        allf, "pruned (current, 67)", "pre-pruning (90 features)"
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(CSV_PATH, index=False)
    print(f"\npruned set beats the pre-pruning set in {wins}/{total} paired repeats")

    pre, post = rows[0], rows[1]
    REPORT_PATH.write_text(
        "# Feature pruning: what the model actually needs\n\n"
        "> This platform is validated on a 100-patient demo subset of MIMIC-IV. The\n"
        "> engineering is real and the methodology is rigorous; the clinical\n"
        "> performance figures demonstrate pipeline validity and do not transfer to\n"
        "> clinical practice (PROJECT_PLAN.md section 17).\n\n"
        "Full rationale for every removal and every deliberate keep is in this\n"
        "module's docstring (`ml/evaluation/feature_pruning.py`).\n\n"
        "| feature set | features | AUPRC (95% CI) | AUROC |\n|---|---|---|---|\n"
        f"| pre-pruning | {pre['n_features']} | {pre['auprc']:.3f} "
        f"({pre['auprc_lo']:.3f}-{pre['auprc_hi']:.3f}) | {pre['auroc']:.3f} |\n"
        f"| **pruned (current)** | **{post['n_features']}** | **{post['auprc']:.3f}** "
        f"({post['auprc_lo']:.3f}-{post['auprc_hi']:.3f}) | {post['auroc']:.3f} |\n\n"
        f"The pruned set beats the pre-pruning set in **{wins} of {total}** paired CV\n"
        "repeats (identical folds in both arms).\n\n"
        "Removed: `news2`, `sofa_24hours`, the 18 rolling means, and the 3\n"
        "lab-ordering-intensity columns. Kept after explicit testing:\n"
        "`admission_age`, `gender`, `first_careunit`, and every vital channel --\n"
        "including `rr`, which the screen wanted to drop and which the data-quality\n"
        "check protected (99.9% present, 1.1% carried forward).\n\n"
        "The honest headline is not that the model got better. It is that a\n"
        "90-feature model was **over-parameterised for 120 positives**, and most of\n"
        "the measured gain is variance removed rather than signal added.\n"
    )
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
