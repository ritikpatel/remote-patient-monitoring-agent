import sys
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from ml.models import serving  # noqa: E402
from services.common.testing import load_service_app  # noqa: E402

_module = load_service_app("risk-engine", REPO_ROOT)
DEFAULT_DB_PATH, app = _module.DEFAULT_DB_PATH, _module.app

pytestmark = pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="warehouse not built")

client = TestClient(app)


@pytest.fixture(scope="module")
def known_stay_hour() -> tuple[int, int]:
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    row = conn.execute("SELECT stay_id, hour FROM capstone.news2 LIMIT 1").fetchone()
    conn.close()
    return row


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_score_matches_warehouse_exactly(known_stay_hour):
    stay_id, hour = known_stay_hour
    resp = client.get(f"/score/{stay_id}/{hour}")
    assert resp.status_code == 200
    body = resp.json()

    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    expected_news2, expected_tier = conn.execute(
        "SELECT news2, tier_icu FROM capstone.news2 WHERE stay_id = ? AND hour = ?", [stay_id, hour]
    ).fetchone()
    conn.close()

    assert body["news2"] == expected_news2
    assert body["news2_tier_icu"] == expected_tier


def test_score_404_for_unknown_stay():
    resp = client.get("/score/999999999/0")
    assert resp.status_code == 404


def test_score_ml_returns_503_when_no_model_exported():
    if serving.promoted_model_available():
        pytest.skip("a promoted Phase 5 model is exported in this environment")
    resp = client.post("/score/ml/1/0")
    assert resp.status_code == 503


def test_score_ml_returns_a_real_prediction_when_a_model_is_exported(known_stay_hour):
    if not serving.promoted_model_available():
        pytest.skip("no promoted Phase 5 model -- run ml/evaluation/run_all.py")
    stay_id, hour = known_stay_hour
    resp = client.post(f"/score/ml/{stay_id}/{hour}")
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert len(body["reasons"]) > 0


def test_score_ml_404_for_unknown_stay():
    if not serving.promoted_model_available():
        pytest.skip("no promoted Phase 5 model -- run ml/evaluation/run_all.py")
    resp = client.post("/score/ml/999999999/0")
    assert resp.status_code == 404


def test_high_news2_stay_has_nonempty_reason(known_stay_hour):
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    row = conn.execute(
        "SELECT stay_id, hour FROM capstone.news2 WHERE news2 >= 7 LIMIT 1"
    ).fetchone()
    conn.close()
    stay_id, hour = row
    resp = client.get(f"/score/{stay_id}/{hour}")
    assert resp.status_code == 200
    assert len(resp.json()["reason"]) > 0
