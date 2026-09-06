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
