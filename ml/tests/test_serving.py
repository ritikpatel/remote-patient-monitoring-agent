"""Tests for ml/models/serving.py."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest

from ml.models import serving

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"


def test_format_value_renders_missing_as_plain_language_not_nan_repr() -> None:
    assert serving._format_value(float("nan")) == "not observed"
    assert serving._format_value(np.float64("nan")) == "not observed"
    assert serving._format_value(pd.NA) == "not observed"


def test_format_value_unwraps_numpy_scalars_to_plain_python_repr() -> None:
    # Found for real: a bare numpy scalar's repr leaked into /score/ml's
    # "reasons" field as e.g. "np.float64(93.0)" instead of "93".
    assert serving._format_value(np.float64(93.0)) == "93"
    assert serving._format_value(np.float32(1)) == "1"
    assert serving._format_value(np.int64(5)) == "5"
    assert serving._format_value("MICU") == "'MICU'"


def test_format_value_rounds_a_float_to_3_significant_figures() -> None:
    # Found for real: a raw float's full repr ("6.458333333333333") overflowed
    # the dashboard's SHAP-reasons panel on a mobile viewport.
    assert serving._format_value(6.458333333333333) == "6.46"
    assert serving._format_value(0.029181829198511387) == "0.0292"


def test_promoted_model_unavailable_raises_a_specific_exception(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(serving, "MODEL_PATH", tmp_path / "does_not_exist.joblib")
    monkeypatch.setattr(serving, "MANIFEST_PATH", tmp_path / "does_not_exist.json")
    assert serving.promoted_model_available() is False
    with pytest.raises(serving.PromotedModelUnavailable):
        serving.load_promoted_model()


def _train_tiny_model_and_export(tmp_path: Path) -> None:
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    n = 200
    x = pd.DataFrame(
        {
            "hr": rng.uniform(50, 150, n),
            "gender": rng.choice(["M", "F"], n),
            "first_careunit": rng.choice(["MICU", "SICU"], n),
        }
    )
    x_cat = x.copy()
    x_cat["gender"] = x_cat["gender"].astype("category")
    x_cat["first_careunit"] = x_cat["first_careunit"].astype("category")
    y = (x.hr > 100).astype(int)
    model = lgb.LGBMClassifier(n_estimators=10, verbosity=-1, n_jobs=1)
    model.fit(x_cat, y, categorical_feature=["gender", "first_careunit"])

    joblib.dump(model, tmp_path / "deterioration_model.joblib")
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps(
            {
                "model_name": "lightgbm",
                "horizon_h": 6,
                "feature_columns": ["hr", "gender", "first_careunit"],
                "categorical_columns": ["gender", "first_careunit"],
                "cv_auprc_point_estimate": 0.5,
                "top_shap_features": {},
            }
        )
    )


def test_score_one_returns_a_real_probability_and_reasons(tmp_path, monkeypatch) -> None:
    if not WAREHOUSE_DB.exists():
        pytest.skip("real warehouse db not built")

    _train_tiny_model_and_export(tmp_path)
    monkeypatch.setattr(serving, "MODEL_PATH", tmp_path / "deterioration_model.joblib")
    monkeypatch.setattr(serving, "MANIFEST_PATH", tmp_path / "feature_manifest.json")

    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    row = conn.execute("select stay_id, hour from capstone.hourly_grid limit 1").fetchone()
    assert row is not None
    stay_id, hour = row
    result = serving.score_one(conn, stay_id, hour)

    assert 0.0 <= result["probability"] <= 1.0
    assert result["model_name"] == "lightgbm"
    assert result["horizon_h"] == 6
    assert len(result["reasons"]) > 0
    assert "hr" in result["reasons"][0] or any("hr" in r for r in result["reasons"])


def test_score_one_raises_for_unknown_stay_hour(tmp_path, monkeypatch) -> None:
    if not WAREHOUSE_DB.exists():
        pytest.skip("real warehouse db not built")

    _train_tiny_model_and_export(tmp_path)
    monkeypatch.setattr(serving, "MODEL_PATH", tmp_path / "deterioration_model.joblib")
    monkeypatch.setattr(serving, "MANIFEST_PATH", tmp_path / "feature_manifest.json")

    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    with pytest.raises(serving.UnknownStayHour):
        serving.score_one(conn, 999999999, 0)


@pytest.mark.skipif(
    not serving.promoted_model_available(), reason="no promoted Phase 5 model exported"
)
def test_score_one_against_the_real_promoted_model() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    row = conn.execute("select stay_id, hour from capstone.hourly_grid limit 1").fetchone()
    assert row is not None
    stay_id, hour = row
    result = serving.score_one(conn, stay_id, hour)
    assert 0.0 <= result["probability"] <= 1.0
    assert len(result["reasons"]) > 0


def test_ml_predictions_declare_their_measured_validated_scope():
    """The actionable output of the subgroup audit. Held-out AUPRC falls from 0.715 in
    the first 6 ICU hours to 0.074 after, and 57% of training positives are in the
    first two hours -- so a consumer reading only the headline number over-trusts a
    late-stay score badly. The payload now says so instead of leaving it to be assumed.
    """
    from ml.models import serving

    early = serving.scope_for_hour(0)
    late = serving.scope_for_hour(50)
    assert early["in_validated_scope"] is True
    assert late["in_validated_scope"] is False
    assert "validated scope" in late["scope_note"]
    # The note must point somewhere useful, not merely warn.
    assert "/score/" in late["scope_note"]
    assert serving.scope_for_hour(serving.VALIDATED_SCOPE_MAX_HOUR)["in_validated_scope"] is False
