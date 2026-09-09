import copy
import sys
from datetime import date, datetime
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mappers import (  # noqa: E402
    alert_communication_to_fhir,
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from services.contracts.observation import Observation, ObservationSource  # noqa: E402
from warehouse.build_duckdb import DEFAULT_DB_PATH  # noqa: E402

pytestmark = pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="warehouse not built")


@pytest.fixture(scope="module")
def conn():
    c = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    yield c
    c.close()


def test_observation_uses_patient_ref_verbatim_as_the_reference():
    """Regression test for a real bug: patient_ref is already a full FHIR reference
    ("ICUStay/1"), not a bare id -- an earlier version prefixed it again, producing
    the nonsensical reference "Patient/ICUStay/1"."""
    obs = Observation.for_channel(
        channel="hr",
        patient_ref="ICUStay/1",
        device_id="monitor-1",
        source=ObservationSource.icu_monitor,
        value=88.0,
        effective_time=datetime(2110, 1, 1),
    )
    fhir_obs = observation_to_fhir(obs)
    assert fhir_obs.subject.reference == "ICUStay/1"
    assert fhir_obs.device.reference == "Device/monitor-1"
    assert fhir_obs.code.coding[0].code == "8867-4"
    assert fhir_obs.valueQuantity.value == 88.0


def test_observation_effective_time_gets_a_timezone():
    """FHIR dateTime requires a timezone on any value with a time component --
    caught by actually constructing a resource and reading fhir.resources' own
    validation error, not by inspection."""
    obs = Observation.for_channel(
        channel="hr",
        patient_ref="ICUStay/1",
        device_id="monitor-1",
        source=ObservationSource.icu_monitor,
        value=88.0,
        effective_time=datetime(2110, 1, 1),  # naive
    )
    fhir_obs = observation_to_fhir(obs)
    assert fhir_obs.effectiveDateTime.tzinfo is not None


def test_quality_flags_survive_as_a_note():
    from services.contracts.observation import QualityFlag

    obs = Observation.for_channel(
        channel="hr",
        patient_ref="ICUStay/1",
        device_id="monitor-1",
        source=ObservationSource.icu_monitor,
        value=88.0,
        effective_time=datetime(2110, 1, 1),
        quality_flags=[QualityFlag.imputed],
    )
    fhir_obs = observation_to_fhir(obs)
    assert "imputed" in fhir_obs.note[0].text


def test_patient_and_encounter_from_real_admission(conn):
    row = conn.execute(
        "SELECT hadm_id, subject_id, admission_type, admittime, dischtime "
        "FROM mimiciv_hosp.admissions LIMIT 1"
    ).fetchone()
    hadm_id, subject_id, admission_type, admittime, dischtime = row
    gender = conn.execute(
        "SELECT gender, anchor_age FROM mimiciv_hosp.patients WHERE subject_id = ?", [subject_id]
    ).fetchone()

    patient = patient_to_fhir(subject_id, gender[0], gender[1])
    assert patient.id == str(subject_id)

    encounter = encounter_to_fhir(hadm_id, subject_id, admission_type, admittime, dischtime)
    assert encounter.status == "finished"
    assert encounter.subject.reference == f"Patient/{subject_id}"


def test_condition_from_real_diagnosis(conn):
    row = conn.execute("""
        SELECT d.hadm_id, a.subject_id, d.icd_code, d.icd_version, dd.long_title, d.seq_num
        FROM mimiciv_hosp.diagnoses_icd d
        JOIN mimiciv_hosp.d_icd_diagnoses dd
          ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
        JOIN mimiciv_hosp.admissions a ON d.hadm_id = a.hadm_id
        LIMIT 1
        """).fetchone()
    hadm_id, subject_id, icd_code, icd_version, long_title, seq_num = row
    condition = condition_to_fhir(hadm_id, subject_id, icd_code, icd_version, long_title, seq_num)
    assert condition.code.coding[0].code == icd_code
    assert condition.code.coding[0].display == long_title


def test_medication_administration_from_real_prescription(conn):
    row = conn.execute(
        "SELECT hadm_id, subject_id, drug, route, starttime FROM mimiciv_hosp.prescriptions "
        "WHERE drug IS NOT NULL AND starttime IS NOT NULL LIMIT 1"
    ).fetchone()
    hadm_id, subject_id, drug, route, starttime = row
    med = medication_administration_to_fhir(hadm_id, subject_id, drug, route, starttime)
    assert med.medicationCodeableConcept.coding[0].code == drug


def test_procedure_from_real_procedure(conn):
    row = conn.execute("""
        SELECT p.hadm_id, a.subject_id, p.icd_code, p.icd_version, dp.long_title,
               p.chartdate, p.seq_num
        FROM mimiciv_hosp.procedures_icd p
        JOIN mimiciv_hosp.d_icd_procedures dp
          ON p.icd_code = dp.icd_code AND p.icd_version = dp.icd_version
        JOIN mimiciv_hosp.admissions a ON p.hadm_id = a.hadm_id
        LIMIT 1
        """).fetchone()
    hadm_id, subject_id, icd_code, icd_version, long_title, chartdate, seq_num = row
    proc = procedure_to_fhir(
        hadm_id, subject_id, icd_code, icd_version, long_title, chartdate, seq_num
    )
    assert proc.code.coding[0].code == icd_code


def test_risk_assessment_from_real_news2(conn):
    row = conn.execute("""
        SELECT n.stay_id, ie.subject_id, n.news2, n.tier_icu
        FROM capstone.news2 n JOIN mimiciv_icu.icustays ie ON n.stay_id = ie.stay_id
        WHERE n.news2 >= 7 LIMIT 1
        """).fetchone()
    stay_id, subject_id, news2, tier_icu = row
    ra = risk_assessment_to_fhir(stay_id, subject_id, news2, None, tier_icu, ["HR contributes 2"])
    assert ra.prediction[0].qualitativeRisk.coding[0].code == tier_icu
    assert ra.note[0].text == "HR contributes 2"


def test_document_reference_round_trips_the_watermark_and_citation():
    """fhir.resources stores Attachment.data (FHIR's base64Binary type) as already-
    decoded bytes -- it re-encodes to base64 only at JSON serialization time. This
    mapper base64-encodes before assignment (matching the type's declared shape);
    what should come back out of the *model* is the plain bytes, not double-decoded
    base64 -- found by writing this test naively (decoding again) and getting a
    padding error, not by reading fhir.resources' source."""
    note_text = "SYNTHETIC -- generated from MIMIC-IV demo structured data\n\n[F001] Test."
    doc = document_reference_to_fhir(1, 1, "discharge_summary", note_text)
    decoded = doc.content[0].attachment.data.decode("utf-8")
    assert decoded == note_text
    assert "SYNTHETIC" in decoded
    assert "[F001]" in decoded

    # and the actual wire format (model_dump_json) IS base64, as FHIR requires
    import base64
    import json

    wire = json.loads(doc.model_dump_json())
    assert base64.b64decode(wire["content"][0]["attachment"]["data"]).decode("utf-8") == note_text


def test_device_and_diagnostic_report_and_communication_build():
    d = device_to_fhir("empatica-e4", "wearable")
    assert d.id == "empatica-e4"

    dg = diagnostic_report_ecg_to_fhir(1, 555, datetime(2110, 1, 3))
    assert dg.identifier[0].value == "555"

    com = alert_communication_to_fhir("alert-1", 1, "NEWS2 escalation", datetime(2110, 1, 3))
    assert com.payload[0].contentString == "NEWS2 escalation"


def test_procedure_accepts_date_not_just_datetime():
    proc = procedure_to_fhir(1, 1, "0210", 10, "CABG", date(2110, 1, 2), 1)
    assert proc.performedDateTime is not None


# --- Review finding F2: identity and references on publish -------------------
# A plain POST let HAPI assign its own id, so Patient/10006053 became Patient/2 and
# every resource referencing it was rejected with HAPI-1094. These pin the conditional
# -update / conditional-reference behaviour that replaced it.

from hapi_client import (  # noqa: E402
    IDENTIFIER_SYSTEMS,
    build_transaction_bundle,
    conditional_url,
    rewrite_reference,
    rewrite_references,
)


def test_conditional_url_keys_on_the_business_identifier_not_a_server_id():
    patient = {
        "resourceType": "Patient",
        "id": "10006053",
        "identifier": [{"system": "urn:mimic-iv:subject_id", "value": "10006053"}],
    }
    assert conditional_url(patient) == "Patient?identifier=urn:mimic-iv:subject_id|10006053"


def test_conditional_url_is_none_for_types_without_a_business_identifier():
    """Leaf resources nothing references are created, not conditionally updated."""
    assert conditional_url({"resourceType": "Observation"}) is None


def test_references_are_rewritten_to_conditional_form():
    assert (
        rewrite_reference("Patient/10006053")
        == "Patient?identifier=urn:mimic-iv:subject_id|10006053"
    )
    assert (
        rewrite_reference("Encounter/22942076")
        == "Encounter?identifier=urn:mimic-iv:hadm_id|22942076"
    )


def test_non_fhir_reference_strings_are_left_alone():
    """observation_to_fhir legitimately carries ICUStay/... and Subject/... (see its
    docstring). Rewriting those would invent a mapping that does not exist."""
    for ref in ("ICUStay/34547401", "Subject/S05", "Observation/34617352"):
        assert rewrite_reference(ref) == ref


def test_already_conditional_references_are_not_double_rewritten():
    ref = "Patient?identifier=urn:mimic-iv:subject_id|10006053"
    assert rewrite_reference(ref) == ref


def test_rewrite_references_recurses_and_does_not_mutate_the_caller():
    original = {
        "resourceType": "MedicationAdministration",
        "subject": {"reference": "Patient/10039708"},
        "context": {"reference": "Encounter/28258130"},
        "partOf": [{"reference": "Patient/10039708"}],
    }
    before = copy.deepcopy(original)
    out = rewrite_references(original)
    assert original == before, "the caller's resource was mutated"
    assert out["subject"]["reference"].startswith("Patient?identifier=")
    assert out["context"]["reference"].startswith("Encounter?identifier=")
    assert out["partOf"][0]["reference"].startswith("Patient?identifier=")


def test_transaction_bundle_strips_client_ids_and_picks_the_right_verb():
    """HAPI-0960: a client may not assign a purely numeric id, so a conditional update
    must not carry one. Identity comes from the identifier, not the id."""
    patient = {
        "resourceType": "Patient",
        "id": "10006053",
        "identifier": [{"system": "urn:mimic-iv:subject_id", "value": "10006053"}],
    }
    observation = {"resourceType": "Observation", "status": "final"}
    bundle = build_transaction_bundle([patient, observation])

    assert bundle["type"] == "transaction"
    pat_entry, obs_entry = bundle["entry"]
    assert "id" not in pat_entry["resource"]
    assert pat_entry["request"] == {
        "method": "PUT",
        "url": "Patient?identifier=urn:mimic-iv:subject_id|10006053",
    }
    assert obs_entry["request"] == {"method": "POST", "url": "Observation"}


def test_every_referenced_type_has_an_identifier_system():
    """Guard: if a mapper starts referencing a new resource type, that type needs an
    entry in IDENTIFIER_SYSTEMS or its references silently stop resolving again."""
    assert {"Patient", "Encounter"} <= set(IDENTIFIER_SYSTEMS)
