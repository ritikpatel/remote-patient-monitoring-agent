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

from ml.features import engineer

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

    return row[feature_columns]


# The model's validated scope, measured -- not asserted. ml/evaluation/report.md's
# fairness audit stratifies held-out performance by hours since ICU admission and finds
# the model discriminates well early and poorly later:
#
#   hour 0-5    AUROC 0.899 (0.81-0.96)   AUPRC 0.715
#   hour 6-23   AUROC 0.668 (0.51-0.88)   AUPRC 0.074
#   hour 24+    withheld -- CI too wide to mean anything
#
# 57% of the training positives fall in the first two hours, so the headline AUPRC is
# carried almost entirely by early-stay rows. A consumer that reads only the headline
# number will over-trust a late-stay score by a wide margin, and nothing in the payload
# used to say so. This is the actionable output of the subgroup audit: not a per-unit
# patch (with 120 positives across nine care units that is fitting to noise) but an
# honest statement of where the number applies.
VALIDATED_SCOPE_MAX_HOUR = 6
OUT_OF_SCOPE_NOTE = (
    "Outside the model's validated scope: held-out AUPRC falls from 0.715 in the first "
    "6 ICU hours to 0.074 from hour 6, and is unmeasurable past hour 24. Treat this "
    "probability as indicative only and use the deterministic NEWS2/SOFA path "
    "(/score/{stay_id}/{hour}), which is what the alerting engine actually escalates on."
)
IN_SCOPE_NOTE = "Within the model's validated scope (first 6 ICU hours, held-out AUPRC 0.715)."


def scope_for_hour(hour: int) -> dict:
    """Whether a prediction at this hour falls inside the measured scope, and why."""
    in_scope = hour < VALIDATED_SCOPE_MAX_HOUR
    return {
        "in_validated_scope": in_scope,
        "validated_scope_max_hour": VALIDATED_SCOPE_MAX_HOUR,
        "scope_note": IN_SCOPE_NOTE if in_scope else OUT_OF_SCOPE_NOTE,
    }


# --- Severity grading ----------------------------------------------------------
# The alerting chain needs the model to say *how bad*, not just *how likely*: a
# raised alert graded "high" is what triggers the agent's care-plan node and the
# escalation email, and grading everything "high" (which is what the pipeline did
# before this existed -- EscalationLoop hardcoded severity="high" on every alert it
# raised) makes the grade carry no information at all.
#
# The cut-points are percentiles of the promoted model's own **out-of-fold** score
# distribution, computed at promotion time and written into the manifest. Three
# deliberate choices in that sentence:
#
# * **Percentiles, not calibrated probabilities.** Deliberately the same device E5
#   used for NEWS2, and it inherits the same caveat: this grades a patient relative
#   to this cohort, and is not a clinically validated severity scale. A raw
#   probability threshold would imply a calibration claim this model has not earned
#   (see ml/evaluation/report.md's Brier/reliability section).
# * **Out-of-fold, not in-sample.** In-sample predictions on a boosted model are
#   sharply optimistic, so in-sample percentiles would put the "high" cut-point far
#   too high and grade real deterioration as medium.
# * **Persisted in the manifest, not recomputed at serving time.** Same reason
#   warehouse/news2.py persists its NEWS2 cut-points: a threshold recomputed
#   independently by each consumer is a threshold that drifts between them.
SEVERITY_PERCENTILE_MEDIUM, SEVERITY_PERCENTILE_HIGH = 0.75, 0.90
SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH = "low", "medium", "high"


def severity_cutpoints_from_scores(oof_scores: np.ndarray) -> dict:
    """The two cut-points, from a vector of out-of-fold predicted probabilities."""
    return {
        "medium": float(np.quantile(oof_scores, SEVERITY_PERCENTILE_MEDIUM)),
        "high": float(np.quantile(oof_scores, SEVERITY_PERCENTILE_HIGH)),
        "percentile_medium": SEVERITY_PERCENTILE_MEDIUM,
        "percentile_high": SEVERITY_PERCENTILE_HIGH,
        "n_oof_scores": int(len(oof_scores)),
    }


def grade_severity(probability: float, cutpoints: dict | None) -> str | None:
    """Probability -> low/medium/high, or None when the manifest carries no
    cut-points.

    Returning None rather than defaulting to "low" (or to "high") is the point: an
    export predating this feature genuinely cannot grade, and a consumer must be able
    to tell "the model says low" apart from "the model was never asked". The alerting
    chain treats None as "not gradeable" and falls back to the deterministic path
    rather than silently suppressing or silently paging.
    """
    if not cutpoints:
        return None
    if probability >= cutpoints["high"]:
        return SEVERITY_HIGH
    if probability >= cutpoints["medium"]:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


def score_one(conn: duckdb.DuckDBPyConnection, stay_id: int, hour: int) -> dict:
    model, manifest = load_promoted_model()
    feature_columns = manifest["feature_columns"]
    x = _feature_row(conn, stay_id, hour, feature_columns).copy()
    # Tolerate a manifest listing a column the model no longer uses: the promoted
    # feature set does change between runs, and a stale entry here should not
    # crash serving. `gender` has now been on both sides of this -- dropped by
    # finding F4, restored when F6's CV-grouping fix moved its ablation back over
    # the bar -- which is exactly why this loop follows the frame rather than a
    # hardcoded column list.
    for col in manifest["categorical_columns"]:
        if col in x.columns:
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

    cutpoints = manifest.get("severity_cutpoints")
    return {
        "probability": proba,
        "severity": grade_severity(proba, cutpoints),
        "severity_cutpoints": cutpoints,
        "model_name": manifest["model_name"],
        "horizon_h": manifest["horizon_h"],
        "cv_auprc_point_estimate": manifest["cv_auprc_point_estimate"],
        "disease_features": manifest.get("disease_features"),
        "reasons": reasons,
        **scope_for_hour(hour),
    }
