"""Serving the promoted Phase 5 model outside the training pipeline
(PROJECT_PLAN.md section 10: "``risk-engine`` ... serves Phase 5 models").

Loads the joblib export + feature manifest that ``ml/evaluation/run_all.py``
writes to ``ml/models/promoted/`` (the local stand-in for pulling the
registered model from the MLflow Model Registry -- see that script's own
promotion step), builds the same feature row a training example would have
had for one ``(stay_id, hour)``, and returns a probability plus a SHAP-based
reason list: the learned-model analogue of ``risk-engine``'s own
``_explain()`` for NEWS2 ("SHAP attribution surfaced through risk-engine so
every alert carries a reason").

**Known performance limitation, stated rather than hidden:** ``score_one()``
rebuilds the *entire* hourly-grid feature frame (~12,000 rows) on every call
because ``ml/features/engineer.py`` was written for batch training, not
single-row lookup. That costs a few seconds per request -- acceptable for a
demo `/score/ml` endpoint, but not what Phase 7's latency budget targets
(PROJECT_PLAN.md section 15 sizes the *deterministic* NEWS2/SOFA path, which
this is not). An incremental single-stay feature path is the obvious
follow-up if this endpoint needs to be fast, and is explicitly out of scope
here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import shap

from ml.features import ecg, engineer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROMOTED_MODEL_DIR = REPO_ROOT / "ml" / "models" / "promoted"
MODEL_PATH = PROMOTED_MODEL_DIR / "deterioration_model.joblib"
MANIFEST_PATH = PROMOTED_MODEL_DIR / "feature_manifest.json"


def _format_value(value: Any) -> str:
    """Human-readable rendering for a reason string. A bare numpy scalar's
    repr ("np.float64(nan)") is not plain language, a missing vital is itself
    informative (R2/R3) rather than just the string "nan", and a raw Python
    float's repr carries far more decimal places than any vital sign is
    actually measured to (e.g. "6.458333333333333") -- found overflowing a
    mobile screen's width in the dashboard's SHAP-reasons panel.
    """
    if pd.isna(value):
        return "not observed"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.3g}"
    return repr(value)


class PromotedModelUnavailable(Exception):
    """No trained-and-exported model yet -- the caller should serve a real
    503, not fabricate a score."""


class UnknownStayHour(Exception):
    """The requested (stay_id, hour) has no row in the hourly grid."""


def promoted_model_available() -> bool:
    return MODEL_PATH.exists() and MANIFEST_PATH.exists()


def load_promoted_model() -> tuple[Any, dict]:
    if not promoted_model_available():
        raise PromotedModelUnavailable(
            "no promoted model at ml/models/promoted/ -- run "
            "`python ml/evaluation/run_all.py` to train and export one"
        )
    model = joblib.load(MODEL_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text())
    return model, manifest


def _feature_row(
    conn: duckdb.DuckDBPyConnection, stay_id: int, hour: int, feature_columns: list[str]
) -> pd.DataFrame:
    features = engineer.build_feature_frame(conn)
    row = features[(features.stay_id == stay_id) & (features.hour == hour)]
    if row.empty:
        raise UnknownStayHour(f"no hourly_grid row for stay_id={stay_id} hour={hour}")

    ecg_columns = [c for c in feature_columns if c.startswith("ecg_")]
    if ecg_columns:
        ecg_cache = REPO_ROOT / "data" / "processed" / "ecg_features.parquet"
        raw_ecg_dataset_present = ecg.DEFAULT_RECORD_LIST.exists()
        if ecg_cache.exists() or raw_ecg_dataset_present:
            if ecg_cache.exists():
                ecg_features = pd.read_parquet(ecg_cache)
            else:
                subs = {
                    int(s)
                    for s in conn.execute(
                        "select distinct subject_id from mimiciv_derived.icustay_detail"
                    ).fetchdf()["subject_id"]
                }
                ecg_features = ecg.build_ecg_feature_table(
                    ecg.load_record_list(cohort_subject_ids=subs)
                )[0]
            intime = conn.execute(
                "select stay_id, subject_id, icu_intime from mimiciv_derived.icustay_detail"
                " where stay_id = ?",
                [stay_id],
            ).fetchdf()
            keyed = intime.copy()
            keyed["hour"] = hour
            keyed["row_abs_time"] = keyed.icu_intime + pd.to_timedelta(hour, unit="h")
            attached = ecg.attach_nearest_ecg(
                keyed[["stay_id", "hour", "subject_id", "row_abs_time"]], ecg_features
            )
            row = row.merge(attached, on=["stay_id", "hour"], how="left")
        else:
            # Neither the ECG feature cache nor the raw waveform dataset is
            # present in this deployment (e.g. a container image, which
            # deliberately never bundles the multi-hundred-MB raw ECG
            # corpus). Rather than fail the whole request, leave the ECG
            # columns missing -- LightGBM's native NaN handling (R2/R3) is
            # exactly the mechanism for "this signal wasn't available."
            for col in ecg_columns:
                row = row.copy()
                row[col] = float("nan")

    return row[feature_columns]


def score_one(conn: duckdb.DuckDBPyConnection, stay_id: int, hour: int) -> dict:
    model, manifest = load_promoted_model()
    feature_columns = manifest["feature_columns"]
    x = _feature_row(conn, stay_id, hour, feature_columns).copy()
    for col in manifest["categorical_columns"]:
        x[col] = x[col].astype("category")

    proba = float(model.predict_proba(x)[:, 1][0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    if isinstance(shap_values, list):  # some SHAP/LightGBM version combos return [class0, class1]
        shap_values = shap_values[1]
    contributions = pd.Series(shap_values[0], index=feature_columns)
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(5)
    reasons = [
        f"{name}={_format_value(x.iloc[0][str(name)])} contributes {value:+.3f} to predicted risk"
        for name, value in top.items()
    ]

    return {
        "probability": proba,
        "model_name": manifest["model_name"],
        "horizon_h": manifest["horizon_h"],
        "cv_auprc_point_estimate": manifest["cv_auprc_point_estimate"],
        "reasons": reasons,
    }
