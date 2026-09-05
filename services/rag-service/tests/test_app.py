import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from corpus import FACT_LEDGER_PATH  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from services.common.testing import load_service_app  # noqa: E402

app = load_service_app("rag-service", REPO_ROOT).app

pytestmark = pytest.mark.skipif(
    not FACT_LEDGER_PATH.exists(), reason="notes_synth output not generated"
)

client = TestClient(app)


def test_health_reports_corpus_size():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["corpus_size"] > 0


def test_search_endpoint():
    resp = client.get("/search", params={"q": "sepsis antibiotics", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) <= 3
    assert all("score" in r and "fact_ids" in r for r in body)


def test_search_endpoint_filters_by_source():
    resp = client.get("/search", params={"q": "SOFA sepsis", "k": 5, "source": "guideline"})
    assert resp.status_code == 200
    body = resp.json()
    assert body
    assert all(r["source"] == "guideline" for r in body)
