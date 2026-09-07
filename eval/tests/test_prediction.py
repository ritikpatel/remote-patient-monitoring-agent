from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest

from eval.prediction import collect_holdout_predictions, decision_curve, summarize_prediction_axis

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"

pytestmark = pytest.mark.skipif(not WAREHOUSE_DB.exists(), reason="real warehouse db not built")


def test_decision_curve_treat_none_is_always_zero() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.6, 0.8, 0.3, 0.9])
    dc = decision_curve(y_true, y_score, thresholds=np.array([0.1, 0.3, 0.5]))
    assert (dc.net_benefit_treat_none == 0.0).all()


def test_decision_curve_perfect_model_beats_treat_all_near_prevalence() -> None:
    # A perfect classifier's net benefit at low thresholds should equal the
    # prevalence itself (every positive caught, zero false positives).
    y_true = np.array([0] * 8 + [1] * 2)
    y_score = y_true.astype(float)
    dc = decision_curve(y_true, y_score, thresholds=np.array([0.05]))
    assert dc.net_benefit_model.iloc[0] == pytest.approx(0.2)


def test_collect_holdout_predictions_against_real_warehouse() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    predictions = collect_holdout_predictions(conn)
    assert set(predictions) == {
        "news2",
        "sofa",
        "age_vitals_lr",
        "logistic_full",
        "lightgbm",
        "lightgbm_ecg",
    }
    n = len(predictions["news2"].y_true)
    for pred in predictions.values():
        assert len(pred.y_score) == n
        assert len(pred.groups) == n
        # No group (patient) should appear in more than this one holdout set
        # twice with different labels -- a sanity check that groups came from
        # the same split, not independently reshuffled per model.
        assert (pred.y_true == predictions["news2"].y_true).all()


def test_summarize_prediction_axis_ranks_lightgbm_above_news2_on_auprc() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    predictions = collect_holdout_predictions(conn)
    summary = summarize_prediction_axis(predictions)
    assert summary["lightgbm"]["auprc"].point > summary["news2"]["auprc"].point
    assert "decision_curve" in summary["lightgbm"]
    assert "decision_curve" not in summary["news2"]  # DCA needs a real probability, not a raw score
