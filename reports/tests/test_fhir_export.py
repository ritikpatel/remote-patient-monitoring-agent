from __future__ import annotations

import base64

from reports.fhir_export import report_to_document_reference


def test_report_to_document_reference_round_trips_the_text_verbatim() -> None:
    doc_ref = report_to_document_reference(
        hadm_id=123, subject_id=456, report_type="daily-summary", report_text="A real summary."
    )
    assert doc_ref.subject.reference == "Patient/456"
    assert doc_ref.context.encounter[0].reference == "Encounter/123"
    encoded = doc_ref.content[0].attachment.data
    # fhir.resources stores Attachment.data as decoded bytes internally (see
    # services/fhir-mapper/tests/test_mappers.py's own note on this) --
    # re-encodes to base64 only at model_dump_json() time.
    assert encoded.decode("utf-8") == "A real summary."

    wire_json = doc_ref.model_dump_json()
    import json

    wire = json.loads(wire_json)
    assert base64.b64decode(wire["content"][0]["attachment"]["data"]).decode("utf-8") == (
        "A real summary."
    )
