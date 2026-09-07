"""Phase 5 end-to-end run (PROJECT_PLAN.md section 11 and section 15's
verification step: ``pytest ml/ -v && python ml/evaluation/run_all.py``,
asserting the deterioration model beats recalibrated NEWS2 on AUPRC in >=15
of 20 CV repeats).

Pipeline: composite labels (``ml/features/labels.py``) -> feature frame
(``ml/features/engineer.py``) -> optional ECG fusion (``ml/features/ecg.py``)
-> baselines and models under repeated grouped stratified CV
(``ml/models/*``) -> bootstrap CIs / calibration / Brier
(``ml/evaluation/metrics.py``) -> SHAP attribution -> MLflow logging and
model-registry promotion -> ``ml/evaluation/report.md``.

Run with ``--quick`` for a fast (few-repeat) smoke run while developing; the
real verification run uses the full protocol and takes on the order of
10-15 minutes on a laptop CPU, almost all of it the GRU's training loops.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# See ml/__init__.py: LightGBM followed by PyTorch in one macOS process is a
# reproduced SIGSEGV without this. Set again here (belt-and-braces) since
# this file is normally invoked directly (`python ml/evaluation/run_all.py`),
# which still imports the `ml` package first, but redundancy costs nothing.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `python ml/evaluation/run_all.py` finds the `ml` package

import duckdb  # noqa: E402
import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from ml.evaluation import metrics  # noqa: E402
from ml.features import ecg, engineer, labels  # noqa: E402
from ml.models import baselines, gbm, gru, logistic, splits  # noqa: E402

WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"
ECG_FEATURES_CACHE = REPO_ROOT / "data" / "processed" / "ecg_features.parquet"
PROMOTED_MODEL_DIR = REPO_ROOT / "ml" / "models" / "promoted"
REPORT_PATH = REPO_ROOT / "ml" / "evaluation" / "report.md"
# A local sqlite file by default (no infra required); Phase 8's real dockerized
# MLflow tracking server (infra/compose/docker-compose.yml) is used instead the
# moment MLFLOW_TRACKING_URI is set, with no code change -- the same
# infra-presence-is-the-only-switch pattern as every other not-yet-deployed
# dependency in this project (services/common/publisher.py's KafkaPublisher, etc.).
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlruns.db'}")
MLFLOW_EXPERIMENT = "phase5-deterioration"

N_SPLITS = 5
N_REPEATS = 20
HORIZONS = (6, 12)
PRIMARY_HORIZON = 6

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------


def get_or_build_ecg_features(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    if ECG_FEATURES_CACHE.exists():
        return pd.read_parquet(ECG_FEATURES_CACHE)
    subs = set(
        conn.execute("select distinct subject_id from mimiciv_derived.icustay_detail")
        .fetchdf()
        .subject_id
    )
    record_list = ecg.load_record_list(cohort_subject_ids=subs)
    feats, n_failed = ecg.build_ecg_feature_table(record_list)
    print(
        f"ECG feature extraction: {len(feats)} succeeded, {n_failed} failed of {len(record_list)}"
    )
    ECG_FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(ECG_FEATURES_CACHE, index=False)
    return feats


def attach_ecg(
    keyed_rows: pd.DataFrame, ecg_features: pd.DataFrame, conn: duckdb.DuckDBPyConnection
) -> pd.DataFrame:
    intime = conn.execute(
        "select stay_id, subject_id, icu_intime from mimiciv_derived.icustay_detail"
    ).fetchdf()
    merged = keyed_rows[["stay_id", "hour"]].merge(intime, on="stay_id")
    merged["row_abs_time"] = merged.icu_intime + pd.to_timedelta(merged.hour, unit="h")
    return ecg.attach_nearest_ecg(
        merged[["stay_id", "hour", "subject_id", "row_abs_time"]], ecg_features
    )


# --------------------------------------------------------------------------
# CV runner
# --------------------------------------------------------------------------


def run_cv_for_model(
    name: str,
    fit_predict_fn,
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    random_state: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (fold_results, repeat0_true, repeat0_score, repeat0_groups) --
    the repeat-0 out-of-fold predictions are what calibration/Brier/bootstrap
    CI are computed on, since those need one coherent set of held-out
    predictions rather than a metric-per-fold summary.
    """
    rows = []
    n = len(y)
    oof_true = np.full(n, np.nan)
    oof_score = np.full(n, np.nan)
    y_arr = y.to_numpy()
    groups_arr = groups.to_numpy()

    for repeat, fold, train_idx, test_idx in splits.repeated_grouped_stratified_splits(
        y, groups, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    ):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        proba = fit_predict_fn(x_train, y_train, x_test)
        auroc = metrics._safe_auroc(y_test.to_numpy(), proba)
        auprc = metrics._safe_auprc(y_test.to_numpy(), proba)
        rows.append(
            {
                "model": name,
                "repeat": repeat,
                "fold": fold,
                "auroc": auroc,
                "auprc": auprc,
                "n_test": len(test_idx),
                "n_pos_test": int(y_test.sum()),
            }
        )
        if repeat == 0:
            oof_true[test_idx] = y_arr[test_idx]
            oof_score[test_idx] = proba

    print(f"  {name}: done ({n_repeats} repeats x {n_splits} folds)")
    return pd.DataFrame(rows), oof_true, oof_score, groups_arr


def news2_fit_predict(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame
) -> np.ndarray:
    del x_train, y_train
    return baselines.news2_score(x_test)


def sofa_fit_predict(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    del x_train, y_train
    return baselines.sofa_score(x_test)


def age_vitals_fit_predict(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame
) -> np.ndarray:
    pipeline = baselines.build_age_vitals_logistic_regression()
    pipeline.fit(baselines.age_vitals_subset(x_train), y_train)
    return pipeline.predict_proba(baselines.age_vitals_subset(x_test))[:, 1]


def logistic_fit_predict(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame
) -> np.ndarray:
    return logistic.fit_predict_proba(x_train, y_train, x_test)[1]


def gbm_fit_predict(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    return gbm.fit_predict_proba(x_train, y_train, x_test)[1]


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------


# NEWS2 and SOFA are ordinal severity scores, not calibrated probabilities --
# AUROC/AUPRC are rank-based and still meaningful, but Brier score and a
# calibration curve are only defined for a value in [0, 1] representing an
# actual probability, which these deliberately are not.
PROBABILISTIC_MODELS = {"age_vitals_lr", "logistic_full", "lightgbm", "lightgbm_ecg", "gru"}


def summarize_model(
    name: str,
    fold_results: pd.DataFrame,
    oof_true: np.ndarray,
    oof_score: np.ndarray,
    groups: np.ndarray,
) -> dict:
    valid = ~np.isnan(oof_true)
    ci = metrics.auroc_auprc_with_ci(oof_true[valid], oof_score[valid], groups[valid])
    brier = (
        metrics.brier(oof_true[valid], oof_score[valid])
        if name in PROBABILISTIC_MODELS
        else float("nan")
    )
    return {
        "model": name,
        "cv_mean_auroc": fold_results.auroc.mean(),
        "cv_mean_auprc": fold_results.auprc.mean(),
        "auroc_point": ci["auroc"].point,
        "auroc_lo": ci["auroc"].lo,
        "auroc_hi": ci["auroc"].hi,
        "auprc_point": ci["auprc"].point,
        "auprc_lo": ci["auprc"].lo,
        "auprc_hi": ci["auprc"].hi,
        "brier": brier,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Few-repeat smoke run for development")
    args = parser.parse_args()

    n_repeats = 2 if args.quick else N_REPEATS
    gru_n_repeats = 2 if args.quick else gru.GRU_N_REPEATS

    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    grid = conn.execute("select stay_id, hour from capstone.hourly_grid").fetchdf()
    features = engineer.build_feature_frame(conn)
    ecg_features = get_or_build_ecg_features(conn)
    raw_grid = engineer.load_hourly_grid_raw(conn)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    all_summaries = []
    all_fold_results = []
    report_sections = []
    promoted_candidate = None  # (name, horizon, model_obj, feature_columns, auprc_point)

    for horizon in HORIZONS:
        label_col = f"label_{horizon}h"
        print(f"\n=== Horizon: {horizon}h ===")
        lab = labels.build_labels(conn, grid, horizons=(horizon,))
        summary_row = labels.label_summary(lab, horizons=(horizon,)).iloc[0]
        print(
            f"  at-risk rows={summary_row.total_hours}, positives={summary_row.positive_hours} "
            f"({summary_row.prevalence:.1%}), positive stays={summary_row.positive_stays}"
        )

        ecg_attached = attach_ecg(features, ecg_features, conn)
        x_no_ecg, y, groups = engineer.feature_matrix_for_training(features, lab, label_col)
        x_ecg, _, _ = engineer.feature_matrix_for_training(
            features, lab, label_col, include_ecg=ecg_attached
        )

        results_this_horizon: dict[
            str, tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame | None]
        ] = {}
        for name, fit_fn, x in [
            ("news2", news2_fit_predict, x_no_ecg),
            ("sofa", sofa_fit_predict, x_no_ecg),
            ("age_vitals_lr", age_vitals_fit_predict, x_no_ecg),
            ("logistic_full", logistic_fit_predict, x_no_ecg),
            ("lightgbm", gbm_fit_predict, x_no_ecg),
            ("lightgbm_ecg", gbm_fit_predict, x_ecg),
        ]:
            fold_results, oof_true, oof_score, oof_groups = run_cv_for_model(
                name, fit_fn, x, y, groups, n_repeats=n_repeats
            )
            fold_results["horizon"] = horizon
            all_fold_results.append(fold_results)
            summary = summarize_model(name, fold_results, oof_true, oof_score, oof_groups)
            summary["horizon"] = horizon
            all_summaries.append(summary)
            results_this_horizon[name] = (fold_results, oof_true, oof_score, oof_groups, x)

            with mlflow.start_run(run_name=f"{name}_{horizon}h"):
                mlflow.log_params({"model": name, "horizon_h": horizon, "n_features": x.shape[1]})
                mlflow.log_metrics(
                    {
                        "auroc": summary["auroc_point"],
                        "auroc_lo": summary["auroc_lo"],
                        "auroc_hi": summary["auroc_hi"],
                        "auprc": summary["auprc_point"],
                        "auprc_lo": summary["auprc_lo"],
                        "auprc_hi": summary["auprc_hi"],
                        "brier": summary["brier"],
                    }
                )

        # GRU only at the primary horizon (compute-budget note in gru.py).
        if horizon == PRIMARY_HORIZON:
            print("  Running GRU (reduced repeat budget, see ml/models/gru.py) ...")
            batch = gru.build_sequences(raw_grid, lab, label_col)
            gru_rows = []
            oof_true_gru = np.full(len(batch.y), np.nan)
            oof_score_gru = np.full(len(batch.y), np.nan)
            for repeat, fold, train_idx, test_idx in splits.repeated_grouped_stratified_splits(
                pd.Series(batch.y),
                pd.Series(batch.stay_ids),
                n_splits=gru.GRU_N_SPLITS,
                n_repeats=gru_n_repeats,
            ):
                _model, proba = gru.fit_predict_proba(
                    batch.x[train_idx],
                    batch.lengths[train_idx],
                    batch.y[train_idx],
                    batch.x[test_idx],
                    batch.lengths[test_idx],
                    seed=repeat,
                )
                y_test = batch.y[test_idx]
                gru_rows.append(
                    {
                        "model": "gru",
                        "repeat": repeat,
                        "fold": fold,
                        "auroc": metrics._safe_auroc(y_test, proba),
                        "auprc": metrics._safe_auprc(y_test, proba),
                        "n_test": len(test_idx),
                        "n_pos_test": int(y_test.sum()),
                        "horizon": horizon,
                    }
                )
                if repeat == 0:
                    oof_true_gru[test_idx] = y_test
                    oof_score_gru[test_idx] = proba
            gru_fold_results = pd.DataFrame(gru_rows)
            all_fold_results.append(gru_fold_results)
            gru_summary = summarize_model(
                "gru", gru_fold_results, oof_true_gru, oof_score_gru, batch.stay_ids
            )
            gru_summary["horizon"] = horizon
            all_summaries.append(gru_summary)
            results_this_horizon["gru"] = (
                gru_fold_results,
                oof_true_gru,
                oof_score_gru,
                batch.stay_ids,
                None,
            )
            with mlflow.start_run(run_name=f"gru_{horizon}h"):
                mlflow.log_params(
                    {
                        "model": "gru",
                        "horizon_h": horizon,
                        "n_repeats": gru_n_repeats,
                        "n_splits": gru.GRU_N_SPLITS,
                    }
                )
                mlflow.log_metrics(
                    {
                        "auroc": gru_summary["auroc_point"],
                        "auprc": gru_summary["auprc_point"],
                        "brier": gru_summary["brier"],
                    }
                )

        # >=15/20 comparison (section 15's literal verification step).
        combined_fold_results = pd.concat(
            [results_this_horizon[m][0] for m in results_this_horizon], ignore_index=True
        )
        for model_name in ["logistic_full", "lightgbm", "lightgbm_ecg"]:
            wins, total = metrics.beats_baseline_in_n_of_k_repeats(
                combined_fold_results, model_name, "news2", metric="auprc"
            )
            print(f"  {model_name} beats NEWS2 on AUPRC in {wins}/{total} repeats")
            if horizon == PRIMARY_HORIZON and model_name == "lightgbm":
                primary_wins, primary_total = wins, total

        # ECG delta.
        def _auprc_for(model_name: str, horizon: int = horizon) -> float:
            return next(
                s for s in all_summaries if s["model"] == model_name and s["horizon"] == horizon
            )["auprc_point"]

        ecg_delta_auprc = _auprc_for("lightgbm_ecg") - _auprc_for("lightgbm")
        print(f"  ECG fusion delta on AUPRC: {ecg_delta_auprc:+.4f}")

        report_sections.append(
            {
                "horizon": horizon,
                "label_summary": summary_row,
                "ecg_delta_auprc": ecg_delta_auprc,
                "results": results_this_horizon,
            }
        )

        if horizon == PRIMARY_HORIZON:
            best_variant = "lightgbm_ecg" if ecg_delta_auprc > 0 else "lightgbm"
            best_summary = next(
                s for s in all_summaries if s["model"] == best_variant and s["horizon"] == horizon
            )
            # Only best_variant and its AUPRC are actually needed below -- the
            # horizon is always PRIMARY_HORIZON by construction (this branch
            # only runs `if horizon == PRIMARY_HORIZON`), and the fold's own
            # feature matrix is never reused: the promoted model gets a fresh
            # final refit (x_final, below) instead. Found while auditing for
            # dead code -- this tuple used to carry both, unused.
            promoted_candidate = (best_variant, best_summary["auprc_point"])

    # ----------------------------------------------------------------
    # SHAP attribution + model promotion (primary horizon's best model)
    # ----------------------------------------------------------------
    assert promoted_candidate is not None
    best_name, best_auprc = promoted_candidate
    lab_primary = labels.build_labels(conn, grid, horizons=(PRIMARY_HORIZON,))
    ecg_attached = attach_ecg(features, ecg_features, conn)
    include_ecg = ecg_attached if "ecg" in best_name else None
    # No CV here -- this is the one final refit on everything, so the group
    # labels feature_matrix_for_training returns (for grouped splitting) have
    # nothing to do.
    x_final, y_final, _groups_final = engineer.feature_matrix_for_training(
        features, lab_primary, f"label_{PRIMARY_HORIZON}h", include_ecg=include_ecg
    )
    final_model, _ = gbm.fit_predict_proba(x_final, y_final, x_final)
    x_final_cat = gbm._as_categorical(x_final)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(x_final_cat)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=x_final_cat.columns
    ).sort_values(ascending=False)
    top_shap = mean_abs_shap.head(15)
    print("\nTop 15 SHAP features (promoted model, mean |SHAP|):")
    print(top_shap.to_string())

    PROMOTED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, PROMOTED_MODEL_DIR / "deterioration_model.joblib")
    (PROMOTED_MODEL_DIR / "feature_manifest.json").write_text(
        json.dumps(
            {
                "model_name": best_name,
                "horizon_h": PRIMARY_HORIZON,
                "feature_columns": list(x_final_cat.columns),
                "categorical_columns": gbm.CATEGORICAL_COLUMNS,
                "cv_auprc_point_estimate": best_auprc,
                "top_shap_features": top_shap.to_dict(),
            },
            indent=2,
        )
    )
    top_shap.to_csv(REPO_ROOT / "ml" / "evaluation" / "shap_summary.csv")

    with mlflow.start_run(run_name=f"promoted_{best_name}_{PRIMARY_HORIZON}h"):
        mlflow.log_params({"model": best_name, "horizon_h": PRIMARY_HORIZON})
        mlflow.log_metric("cv_auprc_point_estimate", best_auprc)
        mlflow.lightgbm.log_model(
            final_model,
            name="model",
            registered_model_name="deterioration-risk-model",
        )
    print(f"\nPromoted '{best_name}' (horizon={PRIMARY_HORIZON}h) to the MLflow model registry")
    print(f"Also exported to {PROMOTED_MODEL_DIR} for risk-engine to load directly")

    write_report(all_summaries, report_sections, primary_wins, primary_total, top_shap, args.quick)
    print(f"\nWrote {REPORT_PATH.relative_to(REPO_ROOT)}")

    if args.quick:
        print(
            f"\n--quick run: lightgbm beat recalibrated NEWS2 on AUPRC in "
            f"{primary_wins}/{primary_total} repeats at the primary {PRIMARY_HORIZON}h horizon. "
            f"The plan's >=15/20 bar is defined for the full {N_REPEATS}-repeat protocol -- "
            f"re-run without --quick for the real verification."
        )
    else:
        print(
            f"\nVERIFICATION (PROJECT_PLAN.md section 15): lightgbm beats recalibrated NEWS2 "
            f"on AUPRC in {primary_wins}/{primary_total} repeats at the primary {PRIMARY_HORIZON}h "
            f"horizon -- {'PASS' if primary_wins >= 15 else 'DOES NOT MEET'} the >=15/20 bar."
        )
    return 0


def write_report(
    all_summaries: list[dict],
    report_sections: list[dict],
    primary_wins: int,
    primary_total: int,
    top_shap: pd.Series,
    is_quick: bool,
) -> None:
    summary_df = pd.DataFrame(all_summaries)
    lines = ["# Phase 5 -- predictive models: results\n"]
    lines.append(
        "> This platform is validated on a 100-patient demo subset of MIMIC-IV. "
        "Clinical narrative is LLM-generated from structured data. Wearable "
        "deterioration signals are synthetically morphed from healthy-volunteer "
        "recordings. The engineering is real and the methodology is rigorous, "
        "and the clinical performance figures below demonstrate pipeline "
        "validity -- they do not transfer to clinical practice (PROJECT_PLAN.md "
        "section 17).\n"
    )
    for section in report_sections:
        h = section["horizon"]
        s = section["label_summary"]
        lines.append(f"\n## Horizon: {h}h\n")
        lines.append(
            f"- At-risk patient-hours (after R1 censoring at first event): "
            f"{int(s.total_hours)}\n"
            f"- Positives: {int(s.positive_hours)} ({s.prevalence:.1%}), across "
            f"{int(s.positive_stays)} distinct stays\n"
            f"- ECG-fusion delta on AUPRC (with ECG - without): "
            f"{section['ecg_delta_auprc']:+.4f}\n"
        )
        table = summary_df[summary_df.horizon == h].drop(columns=["horizon"])
        lines.append(table.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("\n## Verification (PROJECT_PLAN.md section 15)\n")
    if is_quick:
        lines.append(
            f"**This was a `--quick` development run ({primary_total} repeats, not the full "
            f"{N_REPEATS}).** LightGBM beat recalibrated NEWS2 on AUPRC in "
            f"**{primary_wins}/{primary_total}** of those repeats at the primary "
            f"{PRIMARY_HORIZON}h horizon. The plan's >=15/20 bar applies to the full-protocol "
            "run; re-run without `--quick` before citing this number as the verification result."
        )
    else:
        lines.append(
            f"LightGBM beats recalibrated NEWS2 on AUPRC in **{primary_wins}/{primary_total}** "
            f"repeats at the primary {PRIMARY_HORIZON}h horizon "
            f"({'meets' if primary_wins >= 15 else 'does not meet'} the >=15/20 bar)."
        )

    lines.append("\n## Top SHAP features (promoted model)\n")
    lines.append(top_shap.to_frame("mean_abs_shap").to_markdown(floatfmt=".4f"))

    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
