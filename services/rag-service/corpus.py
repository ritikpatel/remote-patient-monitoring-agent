"""What rag-service retrieves over: notes_synth's real generated notes, at sentence
granularity, plus a small illustrative guideline corpus.

PROJECT_PLAN.md section 10: "pgvector over synthetic notes + guideline corpus;
returns passages with ledger IDs." The "ledger IDs" are notes_synth's real
`[F0xx]` fact citations (fact_ledger.py) -- indexing at sentence granularity, not
whole-note, means a retrieved passage carries exactly the fact ids it was allowed
to state, so ContextRetriever's citations trace back to the same fact ledger
Phase 7 already checks mechanically.

Guideline corpus: a small set of short, original summaries of well-known public
clinical scoring conventions already used elsewhere in this project (NEWS2, SOFA,
Sepsis-3) -- written here rather than reproduced from any single copyrighted
guideline document. This is illustrative scaffolding for the retrieval mechanism,
not a certified or complete clinical guideline corpus (PROJECT_PLAN.md section 17).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOTES_DIR = REPO_ROOT / "notes_synth" / "output" / "notes"
FACT_LEDGER_PATH = REPO_ROOT / "notes_synth" / "output" / "fact_ledger.parquet"

GUIDELINE_PASSAGES: list[tuple[str, str]] = [
    (
        "G001",
        "NEWS2 (RCP 2017) scores seven physiological parameters -- respiratory rate, "
        "SpO2, systolic blood pressure, heart rate, temperature, consciousness, and "
        "supplemental oxygen use -- into a single 0-20 score, with an aggregate score "
        "of 5-6 conventionally treated as medium risk and 7 or above as high risk on a "
        "general ward population.",
    ),
    (
        "G002",
        "SOFA (Sequential Organ Failure Assessment) scores six organ systems -- "
        "respiratory, coagulation, hepatic, cardiovascular, neurological (via GCS), "
        "and renal -- each 0-4, and is the organ-dysfunction component of the Sepsis-3 "
        "definition (an acute increase of >=2 points in a patient with suspected "
        "infection).",
    ),
    (
        "G003",
        "Sepsis-3 defines sepsis as life-threatening organ dysfunction (an acute SOFA "
        "increase of 2 or more) caused by a dysregulated host response to infection; "
        "suspected infection is typically operationalised as a culture drawn with an "
        "antibiotic ordered within a clinically appropriate window of each other.",
    ),
    (
        "G004",
        "SIRS (Systemic Inflammatory Response Syndrome) criteria -- abnormal "
        "temperature, heart rate, respiratory rate, and white blood cell count -- "
        "predate Sepsis-3 and are more sensitive but less specific for sepsis than the "
        "SOFA-based definition.",
    ),
    (
        "G005",
        "Charlson Comorbidity Index weights a fixed set of chronic conditions "
        "(e.g. diabetes, malignancy, renal disease) to predict 10-year mortality risk "
        "independent of the acute presenting illness, and is commonly used to adjust "
        "outcome comparisons across patients with different baseline health.",
    ),
]


@dataclass
class Passage:
    passage_id: str
    text: str
    source: str  # "note" | "guideline"
    fact_ids: list[str]
    note_id: str | None = None
    hadm_id: int | None = None


def load_guideline_passages() -> list[Passage]:
    return [Passage(pid, text, "guideline", []) for pid, text in GUIDELINE_PASSAGES]


def load_note_passages(
    notes_dir: Path = NOTES_DIR, ledger_path: Path = FACT_LEDGER_PATH
) -> list[Passage]:
    if not ledger_path.exists():
        return []
    ledger = pd.read_parquet(ledger_path)
    passages = []
    for _, row in ledger.iterrows():
        text = row["text"]
        # Drop markdown noise (bare headers, bullet markers) that carries no
        # retrievable clinical content -- a passage worth returning states something.
        stripped = text.lstrip("#*- ").strip()
        if len(stripped) < 15:
            continue
        passages.append(
            Passage(
                passage_id=f"{row['note_id']}#{row['sentence_idx']}",
                text=text,
                source="note",
                fact_ids=list(row["cited_fact_ids"]),
                note_id=row["note_id"],
                hadm_id=int(row["hadm_id"]),
            )
        )
    return passages


def load_corpus(notes_dir: Path = NOTES_DIR, ledger_path: Path = FACT_LEDGER_PATH) -> list[Passage]:
    return load_guideline_passages() + load_note_passages(notes_dir, ledger_path)
