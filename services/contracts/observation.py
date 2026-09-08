"""The one message schema every producer satisfies.

PROJECT_PLAN.md section 8, item 1. Fields: patient_ref, device_id, source, code,
value, unit, effective_time, ingest_time, quality_flags. Deliberately shaped like a
FHIR R4 `Observation` (patient_ref -> Observation.subject, code -> Observation.code,
value/unit -> Observation.valueQuantity, effective_time -> Observation.effectiveDateTime,
device_id -> Observation.device) so that services/fhir-mapper in Phase 4 is a
projection of already-FHIR-shaped fields onto a FHIR resource, not a translation
between two different data models.

Three producers satisfy this contract: simulators/real_event_replay.py (the PRIMARY
test input -- every channel the model trains on), simulators/wearable_replay.py
(true-rate and morphed), and the Wear OS edge agent (edge/edge_agent/). All three
import Observation and CHANNELS from here rather than redefining fields.

Wire format: Avro, schemaless (single-object encoding via fastavro's
schemaless_writer/reader). A real deployment would prefix messages with a Confluent
schema-registry id; that prefixing is ingest-gateway's job (Phase 4), not this
contract's.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from enum import StrEnum

from fastavro import schemaless_reader, schemaless_writer
from pydantic import BaseModel, Field, field_validator


class ObservationSource(StrEnum):
    icu_monitor = "icu_monitor"
    wearable = "wearable"
    manual = "manual"
    lab = "lab"


class QualityFlag(StrEnum):
    imputed = "imputed"  # carried forward from a previous hour, not a fresh reading (R2)
    out_of_range = "out_of_range"
    device_fault = "device_fault"  # a data_constraints.txt fault fixture (e.g. wearable f07)
    synthetic = "synthetic"  # morphed/generated, never measured (R7)
    duplicate = "duplicate"  # a data_constraints.txt duplicated block (e.g. wearable S02)


class Channel(BaseModel):
    """One row of the channel registry: what a code means, in what unit, and its
    LOINC code where a standard one exists. Device-native signals (raw PPG, 3-axis
    accelerometer, EDA, inter-beat interval) have no LOINC equivalent -- they are
    waveform/sensor channels, not clinical observations -- and use a local code
    instead. That gap is itself meaningful for the FHIR mapper: only LOINC-coded
    channels project directly onto Observation.code; the rest need a local
    CodeableConcept.
    """

    code: str
    code_system: str
    display: str
    unit: str


# LOINC codes are the standard ones used by the HL7 FHIR "Vital Signs" profile.
# Device-native wearable channels (bvp, acc_*, eda, ibi) have no LOINC equivalent and
# are registered under a local code system instead.
LOCAL_CODE_SYSTEM = "urn:capstone-rpm:device-signal"
LOINC = "http://loinc.org"

CHANNELS: dict[str, Channel] = {
    "hr": Channel(code="8867-4", code_system=LOINC, display="Heart rate", unit="/min"),
    "rr": Channel(code="9279-1", code_system=LOINC, display="Respiratory rate", unit="/min"),
    "spo2": Channel(
        code="59408-5", code_system=LOINC, display="Oxygen saturation, pulse oximetry", unit="%"
    ),
    "sbp": Channel(
        code="8480-6", code_system=LOINC, display="Systolic blood pressure", unit="mmHg"
    ),
    "map": Channel(code="8478-0", code_system=LOINC, display="Mean blood pressure", unit="mmHg"),
    "temp_c": Channel(code="8310-5", code_system=LOINC, display="Body temperature", unit="Cel"),
    "gcs_total": Channel(
        code="9269-2", code_system=LOINC, display="Glasgow coma score total", unit="{score}"
    ),
    "fio2": Channel(
        code="3150-0", code_system=LOINC, display="Inhaled oxygen concentration", unit="%"
    ),
    "glucose": Channel(
        code="2339-0", code_system=LOINC, display="Glucose [Mass/volume] in Blood", unit="mg/dL"
    ),
    # Wearable device-native signals -- no LOINC equivalent (module docstring above).
    "bvp": Channel(
        code="bvp", code_system=LOCAL_CODE_SYSTEM, display="Blood volume pulse (PPG)", unit="nW"
    ),
    "acc_x": Channel(
        code="acc_x", code_system=LOCAL_CODE_SYSTEM, display="Acceleration, X axis", unit="g"
    ),
    "acc_y": Channel(
        code="acc_y", code_system=LOCAL_CODE_SYSTEM, display="Acceleration, Y axis", unit="g"
    ),
    "acc_z": Channel(
        code="acc_z", code_system=LOCAL_CODE_SYSTEM, display="Acceleration, Z axis", unit="g"
    ),
    "eda": Channel(
        code="eda", code_system=LOCAL_CODE_SYSTEM, display="Electrodermal activity", unit="uS"
    ),
    # Wrist skin temperature is NOT body temperature and must not share its code.
    # The Empatica E4's TEMP channel reads 31.5-33.9 degC on a healthy wrist; NEWS2's
    # temperature component expects a core measurement and scores <=35 degC as a red
    # flag, so mapping this onto "temp_c" made every wearable subject raise a
    # hypothermia alert. Found once finding F3's escalation loop let a wearable replay
    # actually reach the scoring engine -- before that, the wearable path never met
    # NEWS2 at all. Registered device-native so no scorer can mistake it for core.
    "temp_skin": Channel(
        code="temp_skin",
        code_system=LOCAL_CODE_SYSTEM,
        display="Skin temperature (wrist)",
        unit="Cel",
    ),
    "ibi": Channel(
        code="ibi", code_system=LOCAL_CODE_SYSTEM, display="Inter-beat interval", unit="s"
    ),
    # Edge-computed feature (edge/edge_agent), not a raw device signal: RMS deviation
    # of 3-axis accelerometer magnitude from 1g over one batch window.
    "activity_index": Channel(
        code="activity_index",
        code_system=LOCAL_CODE_SYSTEM,
        display="Activity index (RMS accelerometer deviation)",
        unit="g",
    ),
}


class Observation(BaseModel):
    schema_version: int = 1
    patient_ref: str  # e.g. "ICUStay/34547401" or "Subject/S05" -- FHIR-reference-shaped
    device_id: str
    source: ObservationSource
    code: str
    code_system: str = LOINC
    display: str = ""
    value: float
    unit: str
    effective_time: datetime  # when the measurement was taken (Observation.effectiveDateTime)
    ingest_time: datetime  # when the pipeline produced this message
    quality_flags: list[QualityFlag] = Field(default_factory=list)

    @field_validator("effective_time", "ingest_time")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @classmethod
    def for_channel(
        cls,
        *,
        channel: str,
        patient_ref: str,
        device_id: str,
        source: ObservationSource,
        value: float,
        effective_time: datetime,
        ingest_time: datetime | None = None,
        quality_flags: list[QualityFlag] | None = None,
    ) -> Observation:
        """Build an Observation from a registered channel name, filling in its
        code/code_system/display/unit so producers never hand-type LOINC codes.
        """
        ch = CHANNELS[channel]
        return cls(
            patient_ref=patient_ref,
            device_id=device_id,
            source=source,
            code=ch.code,
            code_system=ch.code_system,
            display=ch.display,
            value=value,
            unit=ch.unit,
            effective_time=effective_time,
            ingest_time=ingest_time or datetime.now(UTC),
            quality_flags=quality_flags or [],
        )


# --- Avro wire format -------------------------------------------------------

AVRO_SCHEMA = {
    "type": "record",
    "name": "Observation",
    "namespace": "capstone_rpm.contracts",
    "fields": [
        {"name": "schema_version", "type": "int", "default": 1},
        {"name": "patient_ref", "type": "string"},
        {"name": "device_id", "type": "string"},
        {
            "name": "source",
            "type": {
                "type": "enum",
                "name": "Source",
                "symbols": [s.value for s in ObservationSource],
            },
        },
        {"name": "code", "type": "string"},
        {"name": "code_system", "type": "string"},
        {"name": "display", "type": "string"},
        {"name": "value", "type": "double"},
        {"name": "unit", "type": "string"},
        {"name": "effective_time", "type": {"type": "long", "logicalType": "timestamp-micros"}},
        {"name": "ingest_time", "type": {"type": "long", "logicalType": "timestamp-micros"}},
        {
            "name": "quality_flags",
            "type": {
                "type": "array",
                "items": {
                    "type": "enum",
                    "name": "QualityFlag",
                    "symbols": [f.value for f in QualityFlag],
                },
            },
            "default": [],
        },
    ],
}


def to_avro_bytes(obs: Observation) -> bytes:
    buf = io.BytesIO()
    record = obs.model_dump(mode="python")
    record["source"] = obs.source.value
    record["quality_flags"] = [f.value for f in obs.quality_flags]
    schemaless_writer(buf, AVRO_SCHEMA, record)
    return buf.getvalue()


def from_avro_bytes(data: bytes) -> Observation:
    record = schemaless_reader(io.BytesIO(data), AVRO_SCHEMA)
    return Observation.model_validate(record)
