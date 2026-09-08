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


def test_publish_returns_503_without_hapi_configured(monkeypatch):
    monkeypatch.setattr(_module, "HAPI_FHIR_BASE_URL", None)
    resp = client.post("/fhir/_publish", json={"resourceType": "Patient"})
    assert resp.status_code == 503


# --------------------------------------------------------------------------
# Real HAPI FHIR: self-skips if no server is actually reachable, the same
# pattern eval/tests/test_latency.py and test_kafka_consumer.py use.
# --------------------------------------------------------------------------

import socket  # noqa: E402

HAPI_BASE_URL = "http://localhost:8090/fhir"


def _hapi_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 8090), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _hapi_reachable(),
    reason="no HAPI FHIR server reachable at localhost:8090 -- see infra/compose/README.md",
)
def test_publish_patient_against_a_real_hapi_fhir_server(real_ids, monkeypatch):
    """The real, Phase-8-infra-dependent path fhir-mapper's own module
    docstring names: map a real warehouse Patient, POST it to a live HAPI FHIR
    server, and prove HAPI itself -- not just fhir.resources locally --
    accepted and stored it (a real assigned id/meta comes back).
    """
    monkeypatch.setattr(_module, "HAPI_FHIR_BASE_URL", HAPI_BASE_URL)
    _, subject_id = real_ids
    resource = client.get(f"/fhir/Patient/{subject_id}").json()

    resp = client.post("/fhir/_publish", json=resource)
    assert resp.status_code == 200, resp.text
    stored = resp.json()
    assert stored["resourceType"] == "Patient"
    assert "id" in stored
    # Not `== "1"`: publishing is a conditional update since finding F2, so a resource
    # already present from an earlier run is matched and versioned up rather than
    # duplicated. That idempotence is the fix; asserting version 1 would assert an
    # empty server, which is no longer required.
    assert stored["meta"]["versionId"].isdigit()


@pytest.mark.skipif(
    not _hapi_reachable(),
    reason="no HAPI FHIR server reachable at localhost:8090 -- see infra/compose/README.md",
)
def test_publish_resolves_references_across_a_transaction(real_ids, monkeypatch):
    """Review finding F2, end to end against a real server.

    Publishing an Encounter used to fail with
    ``HAPI-1094: Resource Patient/<subject_id> not found, specified in path:
    Encounter.subject`` because the Patient had been POSTed and given a server id.
    Published as one transaction with conditional references, the Encounter's subject
    must now resolve to whatever id HAPI actually assigned the Patient.
    """
    monkeypatch.setattr(_module, "HAPI_FHIR_BASE_URL", HAPI_BASE_URL)
    hadm_id, subject_id = real_ids
    patient = client.get(f"/fhir/Patient/{subject_id}").json()
    encounter = client.get(f"/fhir/Encounter/{hadm_id}").json()

    resp = client.post("/fhir/_publish", json=[patient, encounter])
    assert resp.status_code == 200, resp.text
    stored_patient, stored_encounter = resp.json()

    assert stored_encounter["resourceType"] == "Encounter"
    # The reference points at the id HAPI assigned, not the MIMIC subject_id.
    assert stored_encounter["subject"]["reference"] == f"Patient/{stored_patient['id']}"


@pytest.mark.skipif(
    not _hapi_reachable(),
    reason="no HAPI FHIR server reachable at localhost:8090 -- see infra/compose/README.md",
)
def test_publishing_the_same_resource_twice_is_idempotent(real_ids, monkeypatch):
    """Conditional update keys on the business identifier, so a republish must match
    the existing resource rather than creating a second copy of the same patient."""
    monkeypatch.setattr(_module, "HAPI_FHIR_BASE_URL", HAPI_BASE_URL)
    _, subject_id = real_ids
    resource = client.get(f"/fhir/Patient/{subject_id}").json()

    first = client.post("/fhir/_publish", json=resource).json()
    second = client.post("/fhir/_publish", json=resource).json()
    assert first["id"] == second["id"]
