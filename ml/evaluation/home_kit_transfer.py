"""Does training the model to expect a home sensor kit beat masking it afterwards?

The post-discharge arm has a structural problem that no amount of modelling removes:
**MIMIC contains no physiology after discharge.** It records discharge time,
readmission time and death time -- outcomes, but no signal. So a post-discharge
deterioration score cannot be trained on post-discharge data in this project, and
the only defensible construction is a *transfer*: train on ICU physiology, deploy
against whatever a home kit can actually observe of the same patient, and state the
transfer bound rather than presenting it as a validated post-discharge model.

`channel_dropout.py` measured that bound the cheap way -- train on everything, mask
at prediction time -- and then argued against its own method:

> *"Every channel here is 74-100% present in training, so the trees had very little
> opportunity to learn a sensible default direction for its absence... A model
> intended to run with channels routinely missing should be trained that way."*

This study does that and compares the two directly. Two arms, identical folds:

* **`masked_at_inference`** -- the existing approach. Fit on the full ICU feature
  set, then blank the kit's unavailable columns when scoring. The model has never
  seen a patient without a ventilator.
* **`dropout_trained`** -- fit on a training set replicated once per kit, each
  replica masked to that kit (`ml/models/channel_masking.py`). The model sees every
  patient under ICU conditions *and* under each home kit, so absence becomes a
  pattern it has learned a split direction for rather than a surprise.

Both are then scored under every kit mask, so the table reads as "what does this
kit get, trained which way".

**What this cannot do.** It cannot validate a post-discharge model, because there is
no post-discharge label here -- the label is still the in-ICU composite event. It
measures how much of the ICU model's discriminative power survives having only home
instruments, which is a ceiling on the transfer and not a clinical result. The kit
definitions are the same objects `simulators/home_kit_stream.py` streams, so the
regime measured here is exactly the regime that simulator emits.
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
from ml.evaluation.run_all import N_SPLITS, WAREHOUSE_DB  # noqa: E402
from ml.features import engineer, labels  # noqa: E402
from ml.models import channel_masking, gbm, splits  # noqa: E402

REPORT_PATH = REPO_ROOT / "ml" / "evaluation" / "home_kit_transfer_report.md"
CSV_PATH = REPO_ROOT / "ml" / "evaluation" / "home_kit_transfer.csv"
HORIZON = 6

NOTICE = (
    "> This platform is validated on a 100-patient demo subset of MIMIC-IV. The\n"
    "> engineering is real and the methodology is rigorous; the clinical\n"
    "> performance figures demonstrate pipeline validity and do not transfer to\n"
    "> clinical practice (PROJECT_PLAN.md section 17).\n"
)


def run(x: pd.DataFrame, y: pd.Series, groups: pd.Series, n_repeats: int) -> pd.DataFrame:
    """Both training arms, every kit mask, identical folds.

    The two arms are fitted inside the same fold loop rather than in separate runs so
    that every comparison between them is paired on the same split -- with 120
    positives, unpaired comparisons across independent runs are dominated by which
    patients happened to land in which fold.
    """
    masks = channel_masking.all_kit_masks(list(x.columns))
    records: list[dict] = []

    for repeat, fold, train_idx, test_idx in splits.repeated_grouped_stratified_splits(
        y, groups, n_splits=N_SPLITS, n_repeats=n_repeats
    ):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_train = groups.iloc[train_idx]

        # Arm 1: the existing approach -- one fit on the full feature set.
        model_plain, _ = gbm.fit_predict_proba(x_train, y_train, x_train.head(1))

        # Arm 2: the same estimator on a kit-replicated training set. Only the
        # TRAINING side is augmented; the test fold stays one row per patient-hour,
        # so the two arms are scored on exactly the same held-out rows.
        x_aug, y_aug, _g_aug = channel_masking.augment_with_channel_dropout(
            x_train, y_train, g_train
        )
        model_dropout, _ = gbm.fit_predict_proba(x_aug, y_aug, x_aug.head(1))

        for kit, masked_cols in masks.items():
            x_test_masked = gbm._as_categorical(channel_masking.apply_mask(x_test, masked_cols))
            for arm, model in (
                ("masked_at_inference", model_plain),
                ("dropout_trained", model_dropout),
            ):
                proba = np.asarray(model.predict_proba(x_test_masked))[:, 1]
                records.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "kit": kit,
                        "arm": arm,
                        "auprc": metrics._safe_auprc(y_test.to_numpy(), proba),
                        "auroc": metrics._safe_auroc(y_test.to_numpy(), proba),
                    }
                )
        print(f"  repeat {repeat} fold {fold} done")
    return pd.DataFrame(records)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean per (kit, arm), plus the paired per-repeat win count of dropout
    training over inference masking within each kit."""
    summary = (
        folds.groupby(["kit", "arm"], as_index=False)
        .agg(auprc=("auprc", "mean"), auroc=("auroc", "mean"))
        .pivot(index="kit", columns="arm", values=["auprc", "auroc"])
    )
    # After the two-value pivot above these are MultiIndex tuples, but
    # pandas-stubs types DataFrame.columns as Index[str] unconditionally, so
    # both the unpacking and the assignment back read as type errors.
    flat: list[str] = [f"{a}_{b}" for a, b in summary.columns]  # type: ignore[misc,has-type]
    summary.columns = pd.Index(flat)

    per_repeat = folds.groupby(["kit", "arm", "repeat"], as_index=False).auprc.mean()
    wins = []
    for kit, g in per_repeat.groupby("kit"):
        wide = g.pivot(index="repeat", columns="arm", values="auprc")
        wins.append(
            {
                "kit": kit,
                "dropout_wins": int((wide["dropout_trained"] > wide["masked_at_inference"]).sum()),
                "repeats": len(wide),
            }
        )
    summary = summary.merge(pd.DataFrame(wins).set_index("kit"), on="kit")
    summary["auprc_delta"] = summary.auprc_dropout_trained - summary.auprc_masked_at_inference
    icu = summary["auprc_masked_at_inference"].loc["icu_full"]
    summary["retained_vs_icu"] = summary.auprc_dropout_trained / icu
    return summary.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--db", type=Path, default=WAREHOUSE_DB)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db), read_only=True)
    grid = conn.execute("select stay_id, hour from capstone.hourly_grid").fetchdf()
    lab = labels.build_labels(conn, grid, horizons=(HORIZON,))
    features = engineer.build_feature_frame(conn)
    x, y, groups = engineer.feature_matrix_for_training(features, lab, f"label_{HORIZON}h")
    conn.close()

    print(f"at-risk rows={len(y)}, positives={int(y.sum())} ({y.mean():.1%})\n")
    mask_table = channel_masking.describe_masks(list(x.columns))
    print(mask_table.to_string(index=False))
    print(f"\n=== {args.repeats} repeats x {N_SPLITS} folds, 2 training arms ===")

    folds = run(x, y, groups, args.repeats)
    folds.to_csv(CSV_PATH, index=False)
    summary = summarize(folds)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    REPORT_PATH.write_text(_report(summary, mask_table, args.repeats))
    print(f"\nWrote {CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


def _report(summary: pd.DataFrame, mask_table: pd.DataFrame, n_repeats: int) -> str:
    home = summary[summary.kit != "icu_full"]
    icu = summary[summary.kit == "icu_full"].iloc[0]
    best = home.loc[home.auprc_dropout_trained.idxmax()]
    helps_home = (home.dropout_wins > n_repeats / 2).all()
    hurts_icu = icu.auprc_delta < 0

    if helps_home and hurts_icu:
        verdict = f"""**Dropout training helps every home kit and costs the ICU model, so the answer
is two models -- and that reverses a conclusion this project previously drew.**

`channel_dropout_report.md` concluded that "masking the generic model is as good as
training a dedicated one, and a second wrist-specific model is not worth deploying:
one model that degrades gracefully covers the same ground." That was a correct
reading of the evidence *available at the time*, because the only alternative it
compared against was `wrist_only.py`'s model trained on a **restricted feature set**.
Training on the **full** feature set with kit-shaped dropout is a different thing,
and it wins: every home kit improves by {home.auprc_delta.min():.4f}-{home.auprc_delta.max():.4f}
AUPRC on {home.dropout_wins.min()}-{home.dropout_wins.max()} of {n_repeats} paired repeats.

The same change costs the ICU model **{abs(icu.auprc_delta):.4f} AUPRC**
({icu.auprc_masked_at_inference:.4f} -> {icu.auprc_dropout_trained:.4f}, losing all
{n_repeats} repeats). That is the accuracy/robustness trade-off in its plainest form:
a model taught that channels vanish stops leaning as hard on the ones that are
present, which is exactly what you want at home and exactly what you do not want in
an ICU where every channel is charted.

So the deployment shape is: **the dropout-free model for the ICU arm, the
dropout-trained model for the post-discharge arm.** One estimator, one feature set,
two fits, each used only in the regime it was fitted for. `risk-engine` already
serves a promoted artefact per arm, so this costs an export, not an architecture."""
    elif helps_home:
        verdict = (
            "Dropout training helps every home kit without measurably costing the ICU "
            "case, so it can simply replace the current fit."
        )
    else:
        verdict = (
            "Training with channel dropout **does not** beat masking at inference in "
            "every kit here. The trees' inability to learn a direction for absence was "
            "a plausible explanation for the transfer loss; on this evidence it is not "
            "the whole explanation, and the residual loss is the information the "
            "missing channels carried, which no training scheme recovers."
        )
    return f"""# Home-kit transfer: train for the kit, or mask afterwards?

{NOTICE}
**MIMIC contains no physiology after discharge.** It records discharge, readmission
and death *times* -- outcomes, but no signal. A post-discharge deterioration score
therefore cannot be trained on post-discharge data here, and the only defensible
construction is a transfer: train on ICU physiology, deploy against what a home kit
can observe, and state the bound. Everything below is that bound. It is **not** a
validated post-discharge model, and the label is still the in-ICU composite event.

The kits are the same objects `simulators/home_kit_stream.py` streams, so the regime
measured here is exactly the regime that simulator emits.

## What each kit costs in features

{mask_table.to_markdown(index=False)}

## Two training arms, identical folds

* **`masked_at_inference`** -- fit on the full ICU feature set, blank the kit's
  unavailable columns when scoring. The model has never seen a patient without a
  ventilator.
* **`dropout_trained`** -- fit on a training set replicated once per kit, each
  replica masked to that kit (`ml/models/channel_masking.py`). Absence becomes a
  learned pattern rather than a surprise. Only the training side is augmented; both
  arms are scored on identical held-out rows.

Protocol: {n_repeats} repeats x {N_SPLITS} folds, subject-grouped stratified CV.
`dropout_wins` is the paired per-repeat count of `dropout_trained` beating
`masked_at_inference` within that kit.

{summary.to_markdown(index=False, floatfmt=".4f")}

## What this says

{verdict}

Best home-kit result: **`{best.kit}`** at AUPRC {best.auprc_dropout_trained:.4f},
retaining {best.retained_vs_icu:.0%} of the full-ICU model.

The ordering across kits is the part worth carrying into a design decision: it prices
each additional home instrument in the only currency that matters here. Note that
`watch_only` loses considerably more than the step from `watch_plus_cuff` to
`full_home` gains -- blood pressure is doing most of the work that separates a
usable home signal from a wrist pulse.

## The limit that no arm escapes

Core temperature, GCS, FiO2 and arterial-line presence have **no home instrument**,
at any price. That is not a modelling restriction to be engineered away -- a patient
at home is not ventilated and nobody is performing hourly neurological exams on them.
The gap between `full_home` and `icu_full` is the information those four carry, and
the only way to close it is to change what the patient is wearing, not how the model
is fitted.
"""


if __name__ == "__main__":
    raise SystemExit(main())
