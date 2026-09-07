"""Axis 1 -- Prediction (PROJECT_PLAN.md section 13): "AUROC/AUPRC with CIs,
calibration, decision-curve analysis, against all three baselines."

AUROC/AUPRC-with-CI and calibration reuse Phase 5's own tested machinery
(``ml/evaluation/metrics.py``) rather than re-deriving them -- this axis's
genuinely new contribution is **decision-curve analysis**, which Phase 5's
own report did not compute. Both need a real (y_true, y_score) array, not
just the CV summary statistics ``ml/evaluation/report.md`` already states, so
this module runs one real grouped holdout split (not another full 20-repeat
CV -- Phase 5 already did that; repeating it here would just cost the same
minutes for a number this axis doesn't need at that precision) to obtain one
honest set of held-out predictions per model.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import ml  # noqa: E402,F401 -- sets KMP_DUPLICATE_LIB_OK/OMP_NUM_THREADS as a side effect
from ml.evaluation import metrics  # noqa: E402
from ml.features import ecg, engineer, labels  # noqa: E402
from ml.models import baselines, gbm, logistic  # noqa: E402
from ml.models.splits import group_key  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

PRIMARY_HORIZON = 6
HOLDOUT_SPLITS = 5  # one fold held out, four trained on -- an 80/20-ish split
HOLDOUT_RANDOM_STATE = 0


@dataclass
class ModelPredictions:
    name: str
    y_true: np.ndarray
    y_score: np.ndarray
    groups: np.ndarray


def _one_holdout_split(y: pd.Series, groups: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=HOLDOUT_RANDOM_STATE
    )
    train_idx, test_idx = next(iter(splitter.split(np.zeros(len(y)), y, groups)))
    return train_idx, test_idx


def collect_holdout_predictions(
    conn: duckdb.DuckDBPyConnection, horizon: int = PRIMARY_HORIZON
) -> dict[str, ModelPredictions]:
    grid = conn.execute("select stay_id, hour from capstone.hourly_grid").fetchdf()
    lab = labels.build_labels(conn, grid, horizons=(horizon,))
    features = engineer.build_feature_frame(conn)

    subs = {
        int(s)
        for s in conn.execute(
            "select distinct subject_id from mimiciv_derived.icustay_detail"
        ).fetchdf()["subject_id"]
    }
    ecg_cache = REPO_ROOT / "data" / "processed" / "ecg_features.parquet"
    if ecg_cache.exists():
        ecg_features = pd.read_parquet(ecg_cache)
    else:
        record_list = ecg.load_record_list(cohort_subject_ids=subs)
        ecg_features, _ = ecg.build_ecg_feature_table(record_list)
    intime = conn.execute(
        "select stay_id, subject_id, icu_intime from mimiciv_derived.icustay_detail"
    ).fetchdf()
    keyed = features[["stay_id", "hour"]].merge(intime, on="stay_id")
    keyed["row_abs_time"] = keyed.icu_intime + pd.to_timedelta(keyed.hour, unit="h")
    ecg_attached = ecg.attach_nearest_ecg(
        keyed[["stay_id", "hour", "subject_id", "row_abs_time"]], ecg_features
    )

    label_col = f"label_{horizon}h"
    x, y, groups = engineer.feature_matrix_for_training(features, lab, label_col)
    x_ecg, _, _ = engineer.feature_matrix_for_training(
        features, lab, label_col, include_ecg=ecg_attached
    )

    train_idx, test_idx = _one_holdout_split(y, group_key(groups))
    y_test = y.iloc[test_idx].to_numpy()
    groups_test = groups.iloc[test_idx].to_numpy()

    results: dict[str, ModelPredictions] = {}

    results["news2"] = ModelPredictions(
        "news2", y_test, baselines.news2_score(x.iloc[test_idx]), groups_test
    )
    results["sofa"] = ModelPredictions(
        "sofa", y_test, baselines.sofa_score(x.iloc[test_idx]), groups_test
    )

    age_vitals_pipe = baselines.build_age_vitals_logistic_regression()
    age_vitals_pipe.fit(baselines.age_vitals_subset(x.iloc[train_idx]), y.iloc[train_idx])
    results["age_vitals_lr"] = ModelPredictions(
        "age_vitals_lr",
        y_test,
        age_vitals_pipe.predict_proba(baselines.age_vitals_subset(x.iloc[test_idx]))[:, 1],
        groups_test,
    )

    _model, proba = logistic.fit_predict_proba(
        x.iloc[train_idx], y.iloc[train_idx], x.iloc[test_idx]
    )
    results["logistic_full"] = ModelPredictions("logistic_full", y_test, proba, groups_test)

    _model, proba = gbm.fit_predict_proba(x.iloc[train_idx], y.iloc[train_idx], x.iloc[test_idx])
    results["lightgbm"] = ModelPredictions("lightgbm", y_test, proba, groups_test)

    _model, proba = gbm.fit_predict_proba(
        x_ecg.iloc[train_idx], y.iloc[train_idx], x_ecg.iloc[test_idx]
    )
    results["lightgbm_ecg"] = ModelPredictions("lightgbm_ecg", y_test, proba, groups_test)

    return results


def decision_curve(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """Standard decision-curve analysis (Vickers & Elkin 2006): net benefit of
    acting on the model at each threshold probability `pt`, compared against
    "treat everyone" and "treat no one". Net benefit at threshold pt:

        NB = (TP/n) - (FP/n) * (pt / (1 - pt))

    which converts a false positive's cost into the same units as a true
    positive's benefit, weighted by the odds of the threshold itself -- a
    model is only useful over a range of pt where its net benefit exceeds
    both baseline strategies.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.5, 50)
    n = len(y_true)
    prevalence = float(np.mean(y_true))

    rows = []
    for pt in thresholds:
        predicted_positive = y_score >= pt
        tp = int(np.sum(predicted_positive & (y_true == 1)))
        fp = int(np.sum(predicted_positive & (y_true == 0)))
        net_benefit_model = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append(
            {
                "threshold": pt,
                "net_benefit_model": net_benefit_model,
                "net_benefit_treat_all": net_benefit_all,
                "net_benefit_treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_prediction_axis(
    predictions: dict[str, ModelPredictions],
) -> dict[str, dict]:
    summary = {}
    for name, pred in predictions.items():
        ci = metrics.auroc_auprc_with_ci(pred.y_true, pred.y_score, pred.groups, n_boot=500)
        calibration = metrics.calibration_curve(pred.y_true, pred.y_score, n_bins=8)
        entry = {
            "auroc": ci["auroc"],
            "auprc": ci["auprc"],
            "calibration": calibration,
        }
        if name in ("age_vitals_lr", "logistic_full", "lightgbm", "lightgbm_ecg"):
            entry["brier"] = metrics.brier(pred.y_true, pred.y_score)
            entry["decision_curve"] = decision_curve(pred.y_true, pred.y_score)
        summary[name] = entry
    return summary
