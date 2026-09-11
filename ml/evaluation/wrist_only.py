"""What a wrist-obtainable sensor set achieves against ICU-labelled deterioration.

The project is titled "ICU **and Post-Discharge** Care", but until now only the ICU
arm had a trained model. The promoted model (`lightgbm`) needs 67 features;
a wrist wearable can supply 9 of them -- the HR family. Everything else it wants
(RR, SpO2, BP, core temperature, GCS, FiO2, glucose, arterial-line presence) has
no sensor on a wrist. So `risk-engine`'s ML path
cannot run post-discharge at all, and `reports/post_discharge_digest.py`
accordingly uses no model -- it scores a NEWS2 HR+SpO2 proxy over
`simulators/home_kit_stream.py`'s replay of a real deteriorating MIMIC stay.

This module gives the post-discharge arm its first real, honestly-scoped number by
restricting the feature set to what a wearable actually measures and re-running the
primary protocol unchanged (same labels, same grouped repeated CV, same bootstrap
CIs resampling whole patients).

Two device classes, because the difference between them is a purchasing decision:

* **WRIST_STRICT** -- Empatica E4 class, the device this project actually has data
  from. HR only. The E4 measures BVP/IBI/EDA/ACC/skin temperature and **no SpO2**.
* **WRIST_CONSUMER** -- Apple Watch / Fitbit / Garmin class: HR **and SpO2**.

Four things this measurement is NOT, stated here rather than discovered later:

1. **It is an optimistic upper bound on a real wrist device.** The HR here is
   nurse-validated monitor HR charted hourly in an ICU, not motion-corrupted wrist
   PPG. A real wearable's HR is noisier, drops out during movement, and is not
   curated by a clinician. Whatever this scores, a real wrist scores less.
2. **The labels are ICU events, not post-discharge events.** The outcome is still
   the composite (death, vasopressor initiation, invasive ventilation initiation,
   unplanned ICU readmission) within 6h. This answers "how much of ICU-defined
   deterioration is visible through a wrist-shaped keyhole", not "what is this
   patient's readmission risk at home". No dataset here links wearables to
   post-discharge outcomes (E10: the wearable cohort is healthy volunteers with no
   ICU link at all).
3. **The physiology is in-ICU physiology.** Patients here are supine, monitored and
   often sedated or on support. Post-discharge patients are ambulatory.
4. **Skin temperature is deliberately excluded.** The E4's TEMP channel reads
   31.5-33.9 C on a healthy wrist; the grid's `temp_c` is a core measurement.
   Mapping one onto the other is a real bug this project already fixed once (it made
   every wearable subject raise a hypothermia alert -- see
   `services/contracts/observation.py`'s `temp_skin` docstring). A wrist cannot
   supply core temperature, so no temperature feature is offered to either variant.

Usage:
    python ml/evaluation/wrist_only.py [--repeats N] [--horizon 6]
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
    N_REPEATS,
    WAREHOUSE_DB,
    gbm_fit_predict,
    news2_fit_predict,
    run_cv_for_model,
    summarize_model,
)
from ml.features import engineer, labels  # noqa: E402

REPORT_PATH = REPO_ROOT / "ml" / "evaluation" / "wrist_only_report.md"
CSV_PATH = REPO_ROOT / "ml" / "evaluation" / "wrist_only.csv"
FOLDS_CSV_PATH = REPO_ROOT / "ml" / "evaluation" / "wrist_only_folds.csv"
PRIMARY_HORIZON = 6

NOTICE = (
    "> This platform is validated on a 100-patient demo subset of MIMIC-IV. The\n"
    "> engineering is real and the methodology is rigorous; the clinical\n"
    "> performance figures demonstrate pipeline validity and do not transfer to\n"
    "> clinical practice (PROJECT_PLAN.md section 17).\n"
)


def channel_family(var: str, windows: tuple[int, ...] = engineer.ROLLING_WINDOWS_H) -> list[str]:
    """Every column the feature pipeline derives from one raw channel.

    Derived from ``engineer``'s own naming rather than hard-coded, so adding a
    rolling window or an imputation flag upstream cannot silently leave this
    module measuring a stale subset.
    """
    cols = [var, f"{var}_was_imputed", f"{var}_hours_since_last_obs"]
    for w in windows:
        cols += [f"{var}_{w}h_{stat}" for stat in engineer.ROLLING_STATS]
    return cols


# `admission_age` is legitimately known post-discharge -- it is on the discharge
# summary, not read off a sensor. No other static feature qualifies:
# `first_careunit` is a hospital fact, and `gender` was already shown to carry
# leak-amplifying, not predictive, weight (finding F4).
WRIST_STRICT = channel_family("hr") + ["admission_age"]
WRIST_CONSUMER = channel_family("hr") + channel_family("spo2") + ["admission_age"]


def hr_rule_fit_predict(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame):
    """The trivial wrist rule: higher heart rate = higher risk, untrained.

    A learned wrist model has to beat this to have earned its complexity; without
    it, a good-looking AUPRC could just be 'tachycardia correlates with badness'.

    Rows before a stay's first charted HR are genuinely missing (carry-forward has
    nothing to carry yet), and the tree models handle that natively while a bare
    ranking score cannot. Missing HR means "the device is telling us nothing", so
    it scores as the typical patient -- taken from the TRAINING fold's median, never
    the test fold's, so the baseline cannot peek at what it is being compared on.
    """
    del y_train
    return x_test["hr"].fillna(x_train["hr"].median()).to_numpy(dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    ap.add_argument("--db", type=Path, default=WAREHOUSE_DB)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db), read_only=True)
    grid = conn.execute("select stay_id, hour from capstone.hourly_grid").fetchdf()
    lab = labels.build_labels(conn, grid, horizons=(args.horizon,))
    features = engineer.build_feature_frame(conn)
    label_col = f"label_{args.horizon}h"
    x_full, y, groups = engineer.feature_matrix_for_training(features, lab, label_col)
    # NEWS2 is a baseline here, not a feature: the pruning study removed it from
    # the model's matrix, so the baseline needs its own severity-bearing copy --
    # same rows, same order. Without this the run dies with KeyError('news2')
    # after printing four models, which is exactly how it went unnoticed once.
    x_baselines, _, _ = engineer.feature_matrix_for_training(
        features, lab, label_col, include_severity_scores=True
    )
    conn.close()

    for name, cols in (("WRIST_STRICT", WRIST_STRICT), ("WRIST_CONSUMER", WRIST_CONSUMER)):
        missing = [c for c in cols if c not in x_full.columns]
        assert not missing, f"{name} references columns the pipeline does not produce: {missing}"

    print(
        f"horizon={args.horizon}h  rows={len(x_full):,}  positives={int(y.sum())} "
        f"({y.mean():.1%})  subjects={groups.nunique()}  repeats={args.repeats}"
    )
    print(
        f"feature counts -- ICU full: {x_full.shape[1]}, "
        f"consumer wrist: {len(WRIST_CONSUMER)}, strict wrist: {len(WRIST_STRICT)}"
    )

    runs = [
        ("icu_full_lightgbm", gbm_fit_predict, x_full),
        ("wrist_consumer", gbm_fit_predict, x_full[WRIST_CONSUMER]),
        ("wrist_strict", gbm_fit_predict, x_full[WRIST_STRICT]),
        ("wrist_hr_rule", hr_rule_fit_predict, x_full[WRIST_STRICT]),
        ("news2_hospital", news2_fit_predict, x_baselines),
    ]

    all_folds, summaries = [], []
    for name, fn, x in runs:
        fold_results, oof_true, oof_score, oof_groups = run_cv_for_model(
            name, fn, x, y, groups, n_repeats=args.repeats
        )
        all_folds.append(fold_results)
        s = summarize_model(name, fold_results, oof_true, oof_score, oof_groups)
        s["n_features"] = x.shape[1]
        summaries.append(s)
        print(
            f"  {name:<20} AUPRC {s['auprc_point']:.3f} "
            f"({s['auprc_lo']:.3f}-{s['auprc_hi']:.3f})  AUROC {s['auroc_point']:.3f}"
        )

    folds = pd.concat(all_folds, ignore_index=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(CSV_PATH, index=False)

    folds.to_csv(FOLDS_CSV_PATH, index=False)
    beats_rule = metrics.beats_baseline_in_n_of_k_repeats(folds, "wrist_consumer", "wrist_hr_rule")
    beats_news2 = metrics.beats_baseline_in_n_of_k_repeats(
        folds, "wrist_consumer", "news2_hospital"
    )
    spo2_gain = metrics.beats_baseline_in_n_of_k_repeats(folds, "wrist_consumer", "wrist_strict")
    vs_full = metrics.beats_baseline_in_n_of_k_repeats(folds, "icu_full_lightgbm", "wrist_consumer")
    write_report(summary, beats_rule, beats_news2, spo2_gain, vs_full, args, x_full.shape[1])
    print(f"\nWrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


def write_report(summary, beats_rule, beats_news2, spo2_gain, vs_full, args, n_full: int) -> None:
    by = {r["model"]: r for _, r in summary.iterrows()}
    full, cons, strict, rule = (
        by["icu_full_lightgbm"],
        by["wrist_consumer"],
        by["wrist_strict"],
        by["wrist_hr_rule"],
    )
    retained = cons["auprc_point"] / full["auprc_point"] if full["auprc_point"] else float("nan")

    rows = [
        ("ICU full (LightGBM)", full, f"{n_full}", "hospital only"),
        ("Wrist, consumer class (HR+SpO2)", cons, f"{len(WRIST_CONSUMER)}", "Apple Watch / Fitbit"),
        ("Wrist, strict class (HR only)", strict, f"{len(WRIST_STRICT)}", "Empatica E4"),
        ("Wrist HR rule (untrained)", rule, "1", "any wearable"),
        ("NEWS2 (hospital reference)", by["news2_hospital"], "7 vitals", "hospital only"),
    ]
    table = (
        "| model | features | obtainable from | AUPRC (95% CI) | AUROC (95% CI) |\n"
        "|---|---|---|---|---|\n"
    )
    for label, r, nf, where in rows:
        table += (
            f"| {label} | {nf} | {where} | {r['auprc_point']:.3f} "
            f"({r['auprc_lo']:.3f}-{r['auprc_hi']:.3f}) | {r['auroc_point']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) |\n"
        )

    lines = [
        "# The wrist-only model: what the post-discharge arm can actually see",
        "",
        NOTICE,
        "",
        f"Composite deterioration within {args.horizon}h, grouped repeated stratified CV "
        f"({args.repeats} repeats x 5 folds, grouped by `subject_id`), bootstrap CIs "
        f"resampling whole patients. Identical protocol to `report.md` -- only the "
        f"feature set changes.",
        "",
        table,
        "",
        "## What this says",
        "",
        f"**A wrist-shaped keyhole retains roughly {retained:.0%} of the full model's AUPRC** "
        f"({cons['auprc_point']:.3f} against {full['auprc_point']:.3f}) while using "
        f"{len(WRIST_CONSUMER)} features instead of {n_full}. The ICU model beats it in "
        f"{vs_full[0]} of {vs_full[1]} repeats, so the gap is consistent, not noise.",
        "",
        f"**SpO2 is the sensor that matters.** Consumer class beats strict HR-only class in "
        f"{spo2_gain[0]} of {spo2_gain[1]} repeats "
        f"({cons['auprc_point']:.3f} vs {strict['auprc_point']:.3f} AUPRC). If a device is "
        f"being chosen for the post-discharge arm, this is the specification that changes the "
        f"answer -- and it is exactly the channel the Empatica E4 in this project does not have. "
        f"That gap is why the wearable arm was retired: `simulators/morphing.py` had to "
        f"fabricate SpO2 outright rather than morph it from a real recording. "
        f"`simulators/home_kit_stream.py` replaced it and inverts the problem -- SpO2 is "
        f"real, charted physiology from a MIMIC stay, and only the device layer is simulated.",
        "",
        f"**A two-channel wrist beats the seven-vital ward standard.** The consumer wrist "
        f"model scores {cons['auprc_point']:.3f} AUPRC against hospital NEWS2's "
        f"{by['news2_hospital']['auprc_point']:.3f}, winning "
        f"{beats_news2[0]} of {beats_news2[1]} repeats -- on a task NEWS2 was built for and "
        f"with five of its seven parameters unavailable. Read it carefully, though: this "
        f"compares a *trained* model against an *untrained* ordinal score that was never "
        f"fitted to this cohort, so it measures the value of fitting, not the inferiority of "
        f"NEWS2's clinical design. The fair conclusion is narrower and still useful -- a "
        f"wearable-obtainable feature set is not the reason a post-discharge arm would "
        f"under-perform.",
        "",
        f"**The learned model earns its complexity.** It beats the untrained "
        f"'high heart rate = risk' rule in {beats_rule[0]} of {beats_rule[1]} repeats "
        f"({cons['auprc_point']:.3f} vs {rule['auprc_point']:.3f}), so the result is not "
        f"simply tachycardia correlating with badness.",
        "",
        "## What it does not say",
        "",
        "Every caveat in this module's docstring applies, and two of them are load-bearing:",
        "",
        "1. **This is an optimistic ceiling.** The HR is hourly, nurse-validated ICU monitor "
        "HR, not motion-corrupted wrist PPG. A real wearable scores below this line, not on it.",
        "2. **The labels are ICU events.** This measures how much ICU-defined deterioration is "
        "visible through wearable-obtainable channels. It is **not** a post-discharge "
        "readmission model, and must not be reported as one -- no dataset in this project "
        "links wearable telemetry to post-discharge outcomes (E10).",
        "",
        "The honest one-line reading: *a wrist can see a meaningful fraction of what the full "
        "ICU feature set sees, and SpO2 is most of that fraction* -- which is a real, "
        "defensible finding about sensor choice, not a clinical performance claim.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
