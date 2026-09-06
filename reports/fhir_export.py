"""FHIR DocumentReference export for generated reports (PROJECT_PLAN.md
section 12: "...exported to PDF and FHIR DocumentReference").

Calls fhir-mapper's real ``document_reference_to_fhir`` mapper directly (a
plain Python function, loaded via ``services/common/testing.py`` since
``services/fhir-mapper`` is a hyphenated directory) rather than duplicating
it -- a report's FHIR shape must be the exact same shape every other note in
this system produces, not a second, subtly different mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.common.testing import load_module

REPO_ROOT = Path(__file__).resolve().parent.parent
_mappers = load_module(
    REPO_ROOT / "services" / "fhir-mapper" / "mappers.py", "reports_fhir_mapper_mappers"
)


def report_to_document_reference(
    hadm_id: int, subject_id: int, report_type: str, report_text: str
) -> Any:
    """``report_type`` becomes the DocumentReference's type code (e.g.
    "shift-handover", "daily-summary", "post-discharge-digest") -- the same
    field notes_synth's actual clinical notes use for their note_type.
    """
    return _mappers.document_reference_to_fhir(
        hadm_id=hadm_id, subject_id=subject_id, note_type=report_type, note_text=report_text
    )
