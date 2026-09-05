import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from services.common.testing import load_service_app  # noqa: E402

_module = load_service_app("fhir-mapper", REPO_ROOT)
DEFAULT_DB_PATH, app = _module.DEFAULT_DB_PATH, _module.app

pytestmark = pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="warehouse not built")

client = TestClient(app)


@pytest.fixture(scope="module")
def real_ids():
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    hadm_id, subject_id = conn.execute(
        "SELECT hadm_id, subject_id FROM mimiciv_hosp.admissions LIMIT 1"
    ).fetchone()
    conn.close()
    return hadm_id, subject_id


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_post_observation():
    payload = {
        "patient_ref": "ICUStay/1",
        "device_id": "monitor-1",
        "source": "icu_monitor",
        "code": "8867-4",
        "code_system": "http://loinc.org",
        "display": "Heart rate",
        "value": 88.0,
        "unit": "/min",
        "effective_time": datetime(2110, 1, 1).isoformat(),
        "ingest_time": datetime(2110, 1, 1).isoformat(),
        "quality_flags": [],
    }
    resp = client.post("/fhir/Observation", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["resourceType"] == "Observation"
    assert body["subject"]["reference"] == "ICUStay/1"


def test_get_patient(real_ids):
    _, subject_id = real_ids
    resp = client.get(f"/fhir/Patient/{subject_id}")
    assert resp.status_code == 200
    assert resp.json()["resourceType"] == "Patient"


def test_get_patient_404():
    resp = client.get("/fhir/Patient/999999999")
    assert resp.status_code == 404


def test_get_encounter(real_ids):
    hadm_id, _ = real_ids
    resp = client.get(f"/fhir/Encounter/{hadm_id}")
    assert resp.status_code == 200
    assert resp.json()["resourceType"] == "Encounter"


def test_get_conditions_list(real_ids):
    hadm_id, _ = real_ids
    resp = client.get("/fhir/Condition", params={"hadm_id": hadm_id})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    if body:
        assert body[0]["resourceType"] == "Condition"


def test_get_device():
    resp = client.get("/fhir/Device/empatica-e4", params={"device_type": "wearable"})
    assert resp.status_code == 200
    assert resp.json()["id"] == "empatica-e4"
