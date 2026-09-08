"""What happens to the deployed model when a sensor is missing?

The model is trained where blood pressure is present 99.7% of the time. A
patient at home has no arterial line, usually no cuff reading at that minute,
no core thermometer and no GCS assessment. So the question that actually
matters for the post-discharge arm is not "how would a smaller model score"
but **"how does *this* model score when channels go missing?"**

That is a different experiment from `wrist_only.py`, and a more faithful one.
`wrist_only.py` trains a *separate* model on a restricted feature set; this
module takes the model that is actually promoted, trains it once per fold on
everything, and then **masks channels at prediction time only**. LightGBM
routes NaN down a learned default direction, so the model still produces a
score -- the question is how much worse it gets, which nothing in the repo
measured before.

Read the result as an upper bound on graceful degradation, for one specific
reason: a channel that is almost always present in training gives the trees
almost no opportunity to learn a sensible default for its absence. Wide
degradation here is evidence the model needs missingness-aware training (or
per-setting models), not evidence the channel is unimportant.

Masks, and why each one:

* **one channel family at a time** -- the marginal cost of losing each sensor.
* **`post_discharge_realistic`** -- core temperature, GCS, FiO2 and
  arterial-line presence together. This is not a hypothetical: none of the four
  has a home sensor (a wearable measures *skin* temperature, which this project
  deliberately refuses to map onto core `temp_c`), so it is the actual channel
  set a discharged patient presents.
* **`wrist_only_channels`** -- everything except HR and SpO2, the comparison
  point against `wrist_only.py`'s separately-trained model. If masking beats
  retraining, one generic model with missing inputs is the better architecture;
  if it loses badly, the restricted model is doing real work.

Usage:
    python ml/evaluation/channel_dropout.py [--repeats N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation import metrics  # noqa: E402
from ml.evaluation.run_all import N_REPEATS, N_SPLITS, WAREHOUSE_DB  # noqa: E402
from ml.features import engineer, labels  # noqa: E402
from ml.models import gbm, splits  # noqa: E402

REPORT_PATH = REPO_ROOT / "ml" / "evaluation" / "channel_dropout_report.md"
CSV_PATH = REPO_ROOT / "ml" / "evaluation" / "channel_dropout.csv"
HORIZON = 6

NOTICE = (
    "> This platform is validated on a 100-patient demo subset of MIMIC-IV. The\n"
    "> engineering is real and the methodology is rigorous; the clinical\n"
    "> performance figures demonstrate pipeline validity and do not transfer to\n"
    "> clinical practice (PROJECT_PLAN.md section 17).\n"
)

# Channels with no home sensor. `temp_c` is core temperature: a wrist device
# reports skin temperature, which reads 31.5-33.9 C on a healthy person and is
# a different measurement -- conflating them is a bug this project already
# fixed once (services/contracts/observation.py, `temp_skin`).
NO_HOME_SENSOR = ["temp_c", "gcs_total", "fio2"]
WRIST_CHANNELS = ["hr", "spo2"]


def channel_columns(var: str, cols: list[str]) -> list[str]:
    return [c for c in cols if c == var or c.startswith(f"{var}_")]


def build_masks(cols: list[str]) -> dict[str, list[str]]:
    masks: dict[str, list[str]] = {"none (baseline)": []}
    for var in engineer.CORE_VITALS:
        masks[f"drop {var}"] = channel_columns(var, cols)
    realistic = [c for v in NO_HOME_SENSOR for c in channel_columns(v, cols)]
    realistic += [c for c in cols if c == "has_arterial_line"]
    masks["post_discharge_realistic"] = realistic
    masks["wrist_only_channels"] = [
        c for v in engineer.CORE_VITALS if v not in WRIST_CHANNELS for c in channel_columns(v, cols)
    ] + [c for c in cols if c == "has_arterial_line"]
    return masks


def run(x: pd.DataFrame, y: pd.Series, groups: pd.Series, n_repeats: int) -> pd.DataFrame:
    """Fit once per fold on the full feature set; predict once per mask.

    Refitting per mask would measure a different model each time, which is the
    thing this experiment exists to avoid -- and it would cost ~13x the compute
    for a less faithful answer.
    """
    masks = build_masks(list(x.columns))
    records: list[dict] = []
    for repeat, fold, train_idx, test_idx in splits.repeated_grouped_stratified_splits(
        y, groups, n_splits=N_SPLITS, n_repeats=n_repeats
    ):
        x_tr, y_tr = x.iloc[train_idx], y.iloc[train_idx]
        x_te, y_te = x.iloc[test_idx], y.iloc[test_idx]
        model, _ = gbm.fit_predict_proba(x_tr, y_tr, x_te)
        for name, cols in masks.items():
            x_masked = x_te.copy()
            for c in cols:
                # float("nan") rather than pd.NA: LightGBM's native missing
                # handling keys on NaN, and a pandas NA in an object column
                # would be rejected outright.
                x_masked[c] = float("nan")
            proba = np.asarray(model.predict_proba(gbm._as_categorical(x_masked)))[:, 1]
            records.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": name,
                    "n_masked": len(cols),
                    "auprc": metrics._safe_auprc(y_te.to_numpy(), proba),
                    "auroc": metrics._safe_auroc(y_te.to_numpy(), proba),
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--db", type=Path, default=WAREHOUSE_DB)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db), read_only=True)
    grid = conn.execute("select stay_id, hour from capstone.hourly_grid").fetchdf()
    lab = labels.build_labels(conn, grid, horizons=(HORIZON,))
    features = engineer.build_feature_frame(conn)
    x, y, groups = engineer.feature_matrix_for_training(features, lab, f"label_{HORIZON}h")
    conn.close()

    print(
        f"{len(x):,} rows, {int(y.sum())} positives, {x.shape[1]} features, "
        f"{args.repeats} repeats"
    )
    folds = run(x, y, groups, args.repeats)
    folds.to_csv(CSV_PATH, index=False)

    summary = (
        folds.groupby("model", as_index=False)
        .agg(n_masked=("n_masked", "first"), auprc=("auprc", "mean"), auroc=("auroc", "mean"))
        .sort_values("auprc", ascending=False)
    )
    base = float(summary.loc[summary.model == "none (baseline)", "auprc"].iloc[0])
    summary["delta"] = summary.auprc - base
    summary["retained_pct"] = 100 * summary.auprc / base
    # Paired per-repeat: how often does the baseline beat this mask? 20/20 means
    # the loss is consistent rather than an artefact of a few folds.
    summary["baseline_wins"] = [
        (
            metrics.beats_baseline_in_n_of_k_repeats(folds, "none (baseline)", m)[0]
            if m != "none (baseline)"
            else None
        )
        for m in summary.model
    ]
    print(summary.to_string(index=False))
    write_report(summary, base, args.repeats)
    print(f"\nWrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


def _wrist_comparison(masked_auprc: float) -> str:
    """Compare masking against `wrist_only.py`'s separately-trained model.

    Read from that study's own CSV rather than quoted, so the two cannot drift
    apart when either is re-run.
    """
    csv = REPO_ROOT / "ml" / "evaluation" / "wrist_only.csv"
    if not csv.exists():
        return ". Run `wrist_only.py` to compare against a separately-trained wrist model."
    rows = pd.read_csv(csv)
    match = rows.loc[rows.model == "wrist_consumer", "auprc_point"]
    if match.empty:
        return ""
    trained = float(match.iloc[0])
    verdict = (
        "so masking the generic model is **as good as training a dedicated one**, and a "
        "second wrist-specific model is not worth deploying: one model that degrades "
        "gracefully covers the same ground. The wrist study's role is to characterise the "
        "loss, not to ship an artefact."
        if masked_auprc >= trained - 0.01
        else "so the separately-trained wrist model is doing real work that masking does not "
        "recover, and a per-setting model is justified."
    )
    return (
        f". A model *trained* on only those channels (`wrist_only_report.md`) scores "
        f"{trained:.3f} against this {masked_auprc:.3f} -- {verdict}"
    )


def write_report(summary: pd.DataFrame, base: float, repeats: int) -> None:
    rows = ""
    for _, r in summary.iterrows():
        wins = "--" if r.model == "none (baseline)" else f"{int(r.baseline_wins)}/{repeats}"
        rows += (
            f"| {r.model} | {int(r.n_masked)} | {r.auprc:.3f} | {r.delta:+.3f} | "
            f"{r.retained_pct:.0f}% | {wins} |\n"
        )
    # Single-channel masks only -- `summary` is sorted ascending by AUPRC, so its
    # last row is the combined post-discharge/wrist mask, which is not a channel.
    single = summary[summary.model.str.startswith("drop ")]
    worst = single.iloc[-1]
    realistic = summary.loc[summary.model == "post_discharge_realistic"].iloc[0]
    wrist = summary.loc[summary.model == "wrist_only_channels"].iloc[0]

    REPORT_PATH.write_text(
        "# Channel dropout: how this model degrades when sensors go missing\n\n"
        + NOTICE
        + "\nThe promoted model, trained once per fold on the full feature set, then "
        "scored with channels masked **at prediction time only**. Full rationale, and "
        "why this differs from `wrist_only_report.md`, is in "
        "`ml/evaluation/channel_dropout.py`'s docstring.\n\n"
        "| masked | columns | AUPRC | delta | retained | baseline wins |\n"
        "|---|---|---|---|---|---|\n" + rows + "\n"
        f"Baseline (nothing masked): **{base:.3f}** AUPRC over {repeats} repeats.\n\n"
        "## What this says\n\n"
        f"**The realistic post-discharge case retains {realistic.retained_pct:.0f}%** of the "
        f"model's AUPRC ({realistic.auprc:.3f} against {base:.3f}). That is the honest number "
        "to quote for a discharged patient wearing a full home sensor suite, because core "
        "temperature, GCS and FiO2 have no home sensor and arterial-line presence is always "
        "false once the patient is home.\n\n"
        f"**Masking every channel but HR and SpO2 retains {wrist.retained_pct:.0f}%** "
        f"({wrist.auprc:.3f})" + _wrist_comparison(float(wrist.auprc)) + "\n\n"
        f"**The single most costly channel to lose is `{worst.model.replace('drop ', '')}`** "
        f"({worst.auprc:.3f}, {worst.delta:+.3f}).\n\n"
        "## The caveat that limits all of it\n\n"
        "Every channel here is 74-100% present in training, so the trees had very little "
        "opportunity to learn a sensible default direction for its absence. These numbers "
        "therefore measure *an artefact of training-time availability* as much as the "
        "clinical value of each signal. A model intended to run with channels routinely "
        "missing should be trained that way -- with dropout applied during training, not "
        "only measured at inference. That is the natural next step this report argues for, "
        "and it is not done here.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
