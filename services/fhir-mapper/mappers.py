"""Projects this project's real, already-shaped data onto FHIR R4 resources.

PROJECT_PLAN.md section 10: "Projects onto HAPI FHIR: Patient, Encounter,
Observation, Condition, MedicationAdministration, Procedure, DiagnosticReport,
DocumentReference, RiskAssessment, Communication, Device."

Every mapper here is a pure function over data this project already produces for
real (services/contracts/observation.py's Observation, the warehouse's admissions/
diagnoses/prescriptions/procedures tables, risk-engine's score response,
notes_synth's generated notes) -- there is no synthetic FHIR-shaped fixture data
anywhere in this module.

Uses the R4B (FHIR R4, backported) resource classes rather than the package's R5
default: HAPI FHIR servers most commonly run R4, and PROJECT_PLAN.md's "HAPI FHIR"
without a version qualifier is read as that convention. Every resource returned by
these functions is a real `fhir.resources.R4B.*` Pydantic model -- constructing one
successfully is a genuine (if server-independent) structural validation; Phase 8's
live HAPI FHIR server is the deployment target this code is already shaped for, not
a prerequisite for checking the shape is right.
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fhir.resources.R4B.codeableconcept import CodeableConcept
from fhir.resources.R4B.coding import Coding
from fhir.resources.R4B.communication import Communication
from fhir.resources.R4B.condition import Condition
from fhir.resources.R4B.device import Device
from fhir.resources.R4B.diagnosticreport import DiagnosticReport
from fhir.resources.R4B.documentreference import DocumentReference, DocumentReferenceContent
from fhir.resources.R4B.encounter import Encounter
from fhir.resources.R4B.identifier import Identifier
from fhir.resources.R4B.medicationadministration import MedicationAdministration
from fhir.resources.R4B.observation import Observation as FHIRObservation
from fhir.resources.R4B.patient import Patient
from fhir.resources.R4B.procedure import Procedure
from fhir.resources.R4B.quantity import Quantity
from fhir.resources.R4B.reference import Reference
from fhir.resources.R4B.riskassessment import RiskAssessment, RiskAssessmentPrediction
from services.contracts.observation import Observation as OurObservation  # noqa: E402


def _cc(system: str, code: str, display: str) -> CodeableConcept:
    return CodeableConcept(coding=[Coding(system=system, code=code, display=display)])


def _ref(reference: str) -> Reference:
    return Reference(reference=reference)


def _dt(value: datetime) -> datetime:
    """FHIR's dateTime type requires a timezone offset on any value with a time
    component -- MIMIC timestamps (and this project's Observation.effective_time
    before its own validator normalizes it) are naive. Found by constructing a real
    Encounter and getting "DateTime value string does not match spec regex" back
    from fhir.resources' own validation -- exactly the kind of structural check this
    module exists to run.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


# --- Observation -------------------------------------------------------------


def observation_to_fhir(obs: OurObservation) -> FHIRObservation:
    """The one message schema (services/contracts/observation.py) projected onto
    FHIR, field for field, exactly as intended in Phase 2's design: patient_ref ->
    subject, code/code_system/display -> code, value/unit -> valueQuantity,
    effective_time -> effectiveDateTime, device_id -> device.

    patient_ref is already a full FHIR reference string ("ICUStay/34547401",
    "Subject/S05", or "Patient/10005866" -- see its docstring in
    services/contracts/observation.py), not a bare id to prefix with "Patient/".
    An earlier version of this mapper did exactly that, producing the nonsensical
    reference "Patient/ICUStay/1" -- caught by actually constructing one and
    reading the output, not by re-reading the code.
    """
    note = None
    if obs.quality_flags:
        from fhir.resources.R4B.annotation import Annotation

        note = [Annotation(text=f"quality_flags: {', '.join(f.value for f in obs.quality_flags)}")]
    return FHIRObservation(
        status="final",
        code=_cc(obs.code_system, obs.code, obs.display or obs.code),
        subject=_ref(obs.patient_ref),
        effectiveDateTime=_dt(obs.effective_time),
        valueQuantity=Quantity(value=obs.value, unit=obs.unit),
        device=_ref(f"Device/{obs.device_id}"),
        note=note,
    )


# --- Patient / Encounter (warehouse admissions + patients) --------------------


def patient_to_fhir(subject_id: int, gender: str | None, anchor_age: int | None) -> Patient:
    return Patient(
        id=str(subject_id),
        identifier=[Identifier(system="urn:mimic-iv:subject_id", value=str(subject_id))],
        gender={"M": "male", "F": "female"}.get(gender or "", "unknown"),
        # MIMIC de-identifies birth date to an anchor age at an anchor year, not a
        # real DOB -- deliberately not populated here rather than fabricating one.
        extension=(
            [
                {
                    "url": "urn:mimic-iv:anchor_age",
                    "valueInteger": anchor_age,
                }
            ]
            if anchor_age is not None
            else None
        ),
    )


def encounter_to_fhir(
    hadm_id: int,
    subject_id: int,
    admission_type: str,
    admittime: datetime,
    dischtime: datetime | None,
) -> Encounter:
    return Encounter(
        id=str(hadm_id),
        identifier=[Identifier(system="urn:mimic-iv:hadm_id", value=str(hadm_id))],
        status="finished" if dischtime else "in-progress",
        class_fhir=Coding(
            system="http://terminology.hl7.org/CodeSystem/v3-ActCode",
            code="EMER" if "EMER" in admission_type.upper() else "IMP",
            display=admission_type,
        ),
        subject=_ref(f"Patient/{subject_id}"),
        period={
            "start": _dt(admittime).isoformat(),
            "end": _dt(dischtime).isoformat() if dischtime else None,
        },
    )


# --- Condition (diagnoses_icd x d_icd_diagnoses) ------------------------------


def condition_to_fhir(
    hadm_id: int, subject_id: int, icd_code: str, icd_version: int, long_title: str, seq_num: int
) -> Condition:
    system = (
        "http://hl7.org/fhir/sid/icd-10-cm"
        if icd_version == 10
        else "http://hl7.org/fhir/sid/icd-9-cm"
    )
    return Condition(
        id=f"{hadm_id}-{seq_num}",
        clinicalStatus=_cc(
            "http://terminology.hl7.org/CodeSystem/condition-clinical", "active", "Active"
        ),
        code=_cc(system, icd_code, long_title),
        subject=_ref(f"Patient/{subject_id}"),
        encounter=_ref(f"Encounter/{hadm_id}"),
    )


# --- MedicationAdministration (prescriptions) ---------------------------------


def medication_administration_to_fhir(
    hadm_id: int, subject_id: int, drug: str, route: str | None, starttime: datetime
) -> MedicationAdministration:
    return MedicationAdministration(
        status="completed",
        medicationCodeableConcept=_cc("urn:mimic-iv:drug-name", drug, drug),
        subject=_ref(f"Patient/{subject_id}"),
        context=_ref(f"Encounter/{hadm_id}"),
        effectiveDateTime=_dt(starttime),
        dosage={"route": _cc("urn:mimic-iv:route", route, route)} if route else None,
    )


# --- Procedure (procedures_icd x d_icd_procedures) ----------------------------


def procedure_to_fhir(
    hadm_id: int,
    subject_id: int,
    icd_code: str,
    icd_version: int,
    long_title: str,
    chartdate: date,
    seq_num: int,
) -> Procedure:
    system = (
        "http://hl7.org/fhir/sid/icd-10-pcs"
        if icd_version == 10
        else "http://hl7.org/fhir/sid/icd-9-cm"
    )
    return Procedure(
        id=f"{hadm_id}-{seq_num}",
        status="completed",
        code=_cc(system, icd_code, long_title),
        subject=_ref(f"Patient/{subject_id}"),
        encounter=_ref(f"Encounter/{hadm_id}"),
        performedDateTime=chartdate,
    )


# --- RiskAssessment (risk-engine's /score response) ---------------------------


def risk_assessment_to_fhir(
    stay_id: int,
    subject_id: int,
    news2: int,
    sofa_24h: int | None,
    tier_icu: str,
    reason: list[str],
) -> RiskAssessment:
    """R4's RiskAssessment.basis is exactly where the deterministic risk-engine
    output and its plain-language `reason` trace belong -- this is the same object
    the agent-orchestrator's EscalationDecider reads (Phase 4's second constraint:
    the policy engine reads risk-engine, not the other way round).
    """
    predictions = [
        RiskAssessmentPrediction(
            outcome=_cc("urn:capstone-rpm:score", "news2", "NEWS2 (ICU-recalibrated)"),
            qualitativeRisk=_cc(
                "http://terminology.hl7.org/CodeSystem/risk-probability", tier_icu, tier_icu
            ),
        )
    ]
    if sofa_24h is not None:
        predictions.append(
            RiskAssessmentPrediction(
                outcome=_cc("urn:capstone-rpm:score", "sofa_24h", "SOFA (24h)"),
            )
        )
    return RiskAssessment(
        status="final",
        subject=_ref(f"Patient/{subject_id}"),
        basis=[_ref(f"Observation/{stay_id}")],
        prediction=predictions,
        note=[{"text": r} for r in reason] if reason else None,
    )


# --- DocumentReference (notes_synth generated notes) --------------------------


def document_reference_to_fhir(
    hadm_id: int, subject_id: int, note_type: str, note_text: str
) -> DocumentReference:
    """notes_synth's watermarked, fact-cited notes -- content is base64'd verbatim
    into the DocumentReference, not paraphrased, so the SYNTHETIC watermark and
    every [F0xx] citation survive the FHIR round trip (R7)."""
    import base64

    return DocumentReference(
        status="current",
        type=_cc("urn:capstone-rpm:note-type", note_type, note_type),
        subject=_ref(f"Patient/{subject_id}"),
        context={"encounter": [_ref(f"Encounter/{hadm_id}")]},
        content=[
            DocumentReferenceContent(
                attachment={
                    "contentType": "text/plain",
                    "data": base64.b64encode(note_text.encode("utf-8")).decode("ascii"),
                }
            )
        ],
    )


# --- Device --------------------------------------------------------------------


def device_to_fhir(device_id: str, device_type: str) -> Device:
    return Device(
        id=device_id,
        identifier=[Identifier(system="urn:capstone-rpm:device_id", value=device_id)],
        type=_cc("urn:capstone-rpm:device-type", device_type, device_type),
    )


# --- DiagnosticReport (ECG studies) --------------------------------------------


def diagnostic_report_ecg_to_fhir(
    subject_id: int, study_id: int, ecg_time: datetime
) -> DiagnosticReport:
    return DiagnosticReport(
        status="final",
        code=_cc("http://loinc.org", "11524-6", "EKG study"),
        subject=_ref(f"Patient/{subject_id}"),
        effectiveDateTime=_dt(ecg_time),
        identifier=[Identifier(system="urn:mimic-iv-ecg:study_id", value=str(study_id))],
        conclusion="Waveform recorded; no automated measurements in this demo dataset (E9).",
    )


# --- Communication (alert-service escalations) ---------------------------------


def alert_communication_to_fhir(
    alert_id: str, subject_id: int, message: str, sent: datetime, status: str = "completed"
) -> Communication:
    return Communication(
        id=alert_id,
        status=status,
        subject=_ref(f"Patient/{subject_id}"),
        sent=_dt(sent),
        payload=[{"contentString": message}],
    )


def to_json(resource: Any) -> str:
    return resource.model_dump_json(indent=2, exclude_none=True)
