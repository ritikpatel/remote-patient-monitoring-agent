"""fhir-mapper: HTTP surface over mappers.py.

Every endpoint pulls real rows from the warehouse (or takes a real Observation
payload) and returns the FHIR R4 resource `mappers.py` builds from it -- see that
module's docstring for the R4B/HAPI FHIR version note and the two real bugs it
documents (patient_ref double-prefixing, naive-datetime rejection) that were found
by exercising this code, not by inspection.

Phase 8 point of contact: POST the returned JSON to a live HAPI FHIR server's
`/fhir/<ResourceType>` endpoint for the actual create-and-validate round trip this
service is ultimately for. Nothing in this module needs to change to do that --
it already returns exactly the resource HAPI would receive.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
from fastapi import Body, FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hapi_client import (  # noqa: E402
    HapiValidationError,
    post_resource,
    publish_resources,
)
from mappers import (  # noqa: E402
    condition_to_fhir,
    device_to_fhir,
    diagnostic_report_ecg_to_fhir,
    document_reference_to_fhir,
    encounter_to_fhir,
    medication_administration_to_fhir,
    observation_to_fhir,
    patient_to_fhir,
    procedure_to_fhir,
    risk_assessment_to_fhir,
)
from services.common.observability import instrument_metrics, instrument_tracing  # noqa: E402
from services.contracts.observation import Observation  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
# e.g. "http://hapi-fhir:8080/fhir" -- Phase 8 infra (infra/compose/docker-compose.yml).
# Unset in every test and any standalone run without it: /fhir/_publish then returns a
# real 503, not a fabricated "validated" response.
HAPI_FHIR_BASE_URL = os.environ.get("HAPI_FHIR_BASE_URL")

app = FastAPI(title="fhir-mapper", version="0.1.0")
instrument_metrics(app)
instrument_tracing(app, "fhir-mapper")


def get_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "fhir-mapper"}


@app.post("/fhir/_publish")
def publish_to_hapi(payload: dict | list = Body(...)) -> dict | list:
    """Takes any resource this service already mapped (the JSON any /fhir/* route
    above returns, unchanged) and publishes it to a live HAPI FHIR server -- the real
    persist-and-validate round trip, not the local fhir.resources/Pydantic
    construction mappers.py already does. A real 503, not a fabricated pass, when no
    HAPI is configured.

    Accepts either a single resource or a **list** of resources. A list is published
    as one FHIR transaction, so a resource and the resources it references land
    together and the references resolve (review finding F2 -- publishing an Encounter
    on its own still fails, correctly, if its Patient was never published: FHIR will
    not let you reference what does not exist).
    """
    if not HAPI_FHIR_BASE_URL:
        raise HTTPException(503, "HAPI_FHIR_BASE_URL not configured -- Phase 8 infra not running")
    try:
        if isinstance(payload, list):
            return publish_resources(payload, HAPI_FHIR_BASE_URL)
        return post_resource(payload, HAPI_FHIR_BASE_URL)
    except HapiValidationError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/fhir/Observation")
def map_observation(obs: Observation) -> dict:
    return observation_to_fhir(obs).model_dump(mode="json", exclude_none=True)


@app.get("/fhir/Patient/{subject_id}")
def map_patient(subject_id: int) -> dict:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT gender, anchor_age FROM mimiciv_hosp.patients WHERE subject_id = ?",
            [subject_id],
        ).fetchone()
        if row is None:
            raise HTTPException(404, f"subject_id {subject_id} not found")
        return patient_to_fhir(subject_id, row[0], row[1]).model_dump(
            mode="json", exclude_none=True
        )
    finally:
        conn.close()


@app.get("/fhir/Encounter/{hadm_id}")
def map_encounter(hadm_id: int) -> dict:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT subject_id, admission_type, admittime, dischtime FROM mimiciv_hosp.admissions "
            "WHERE hadm_id = ?",
            [hadm_id],
        ).fetchone()
        if row is None:
            raise HTTPException(404, f"hadm_id {hadm_id} not found")
        return encounter_to_fhir(hadm_id, *row).model_dump(mode="json", exclude_none=True)
    finally:
        conn.close()


@app.get("/fhir/Condition")
def map_conditions(hadm_id: int) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT d.hadm_id, a.subject_id, d.icd_code, d.icd_version, dd.long_title, d.seq_num
            FROM mimiciv_hosp.diagnoses_icd d
            JOIN mimiciv_hosp.d_icd_diagnoses dd
              ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
            JOIN mimiciv_hosp.admissions a ON d.hadm_id = a.hadm_id
            WHERE d.hadm_id = ?
            ORDER BY d.seq_num
            """,
            [hadm_id],
        ).fetchall()
        return [condition_to_fhir(*row).model_dump(mode="json", exclude_none=True) for row in rows]
    finally:
        conn.close()


@app.get("/fhir/MedicationAdministration")
def map_medications(hadm_id: int) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT hadm_id, subject_id, drug, route, starttime FROM mimiciv_hosp.prescriptions "
            "WHERE hadm_id = ? AND drug IS NOT NULL AND starttime IS NOT NULL",
            [hadm_id],
        ).fetchall()
        return [
            medication_administration_to_fhir(*row).model_dump(mode="json", exclude_none=True)
            for row in rows
        ]
    finally:
        conn.close()


@app.get("/fhir/Procedure")
def map_procedures(hadm_id: int) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT p.hadm_id, a.subject_id, p.icd_code, p.icd_version, dp.long_title,
                   p.chartdate, p.seq_num
            FROM mimiciv_hosp.procedures_icd p
            JOIN mimiciv_hosp.d_icd_procedures dp
              ON p.icd_code = dp.icd_code AND p.icd_version = dp.icd_version
            JOIN mimiciv_hosp.admissions a ON p.hadm_id = a.hadm_id
            WHERE p.hadm_id = ?
            ORDER BY p.seq_num
            """,
            [hadm_id],
        ).fetchall()
        return [procedure_to_fhir(*row).model_dump(mode="json", exclude_none=True) for row in rows]
    finally:
        conn.close()


@app.get("/fhir/RiskAssessment/{stay_id}/{hour}")
def map_risk_assessment(stay_id: int, hour: int) -> dict:
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT n.stay_id, ie.subject_id, n.news2, s.sofa_24hours, n.tier_icu
            FROM capstone.news2 n
            JOIN mimiciv_icu.icustays ie ON n.stay_id = ie.stay_id
            LEFT JOIN mimiciv_derived.sofa s ON s.stay_id = n.stay_id AND s.hr = n.hour
            WHERE n.stay_id = ? AND n.hour = ?
            """,
            [stay_id, hour],
        ).fetchone()
        if row is None:
            raise HTTPException(404, f"no score for stay_id={stay_id} hour={hour}")
        stay_id_, subject_id, news2, sofa_24h, tier_icu = row
        return risk_assessment_to_fhir(
            stay_id_, subject_id, news2, sofa_24h, tier_icu, []
        ).model_dump(mode="json", exclude_none=True)
    finally:
        conn.close()


@app.get("/fhir/DocumentReference")
def map_document_reference(hadm_id: int, subject_id: int, note_path: str) -> dict:
    path = Path(note_path)
    if not path.is_file():
        raise HTTPException(404, f"no note file at {note_path}")
    note_type = path.stem.split("_", 1)[1] if "_" in path.stem else "note"
    return document_reference_to_fhir(hadm_id, subject_id, note_type, path.read_text()).model_dump(
        mode="json", exclude_none=True
    )


@app.get("/fhir/Device/{device_id}")
def map_device(device_id: str, device_type: str = "unknown") -> dict:
    return device_to_fhir(device_id, device_type).model_dump(mode="json", exclude_none=True)


@app.get("/fhir/DiagnosticReport/ecg")
def map_ecg_report(subject_id: int, study_id: int, ecg_time: str) -> dict:
    from datetime import datetime

    return diagnostic_report_ecg_to_fhir(
        subject_id, study_id, datetime.fromisoformat(ecg_time)
    ).model_dump(mode="json", exclude_none=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
