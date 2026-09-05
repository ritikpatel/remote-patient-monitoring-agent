"""The fact ledger: what every generated sentence is allowed to claim, and a
mechanical check of whether it stayed inside that.

PROJECT_PLAN.md section 9: "emit a fact ledger alongside every note. Each generated
sentence carries the (table, row_id, value) tuples it derives from... Real MIMIC
notes could not give you this. It means Phase 7 measures summarisation faithfulness
mechanically -- every claim either traces to a ledger fact or is a hallucination."

The mechanism: generate.py instructs the model to PREFIX any sentence that states
something from the facts list with its citation, e.g. `[F003] Sentence.` or
`[F003,F017] Sentence.` for a sentence drawing on more than one fact. This module
splits generated text into sentences, extracts those citations, and checks each one
against the FactSet that was actually given to the model for that admission -- a
citation to a fact id that doesn't exist (or wasn't in this admission's fact set) is
caught here, not trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from notes_synth.facts import Fact, FactSet

CITATION_RE = re.compile(r"\[(F\d{3}(?:\s*,\s*F\d{3})*)\]")
# generate.py's prompt asks the model to PREFIX each clinical-fact sentence with its
# citation ("[F003] Sentence text."), not suffix it -- so splitting right after
# sentence-ending punctuation groups a citation with the sentence it introduces,
# which is the sentence it's actually citing. Also split on blank lines / newlines:
# real model output is markdown (headers, bullet lists), where a fact-bearing clause
# is very often its own line rather than a ". "-delimited sentence -- found by
# running a real batch and seeing n_sentences implausibly low (a ~20-line bulleted
# note counted as 5 "sentences") because bullet lines don't end in ". " before the
# next capital letter.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])|\n+")

# generate.py's SYSTEM_PROMPT forbids range/dash citation shorthand like "[F003-F010]"
# specifically because CITATION_RE (and this whole mechanism) has no way to safely
# expand it -- a real batch run produced "[F078‑F087]" (a Unicode non-breaking
# hyphen, not even ASCII "-"), which this regex does not match. Preventing the model
# from writing it is simpler and more robust than trying to parse every dash
# variant a model might produce.


@dataclass
class Sentence:
    note_id: str
    hadm_id: int
    note_type: str
    sentence_idx: int
    text: str  # includes the citation markers, verbatim from the model
    cited_fact_ids: list[str]

    @property
    def is_cited(self) -> bool:
        return len(self.cited_fact_ids) > 0


def split_sentences(note_body: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(note_body.strip()) if s.strip()]


def extract_citations(sentence: str) -> list[str]:
    ids: list[str] = []
    for match in CITATION_RE.finditer(sentence):
        ids.extend(f.strip() for f in match.group(1).split(","))
    return ids


def ledger_from_note(note_id: str, hadm_id: int, note_type: str, note_body: str) -> list[Sentence]:
    return [
        Sentence(note_id, hadm_id, note_type, i, s, extract_citations(s))
        for i, s in enumerate(split_sentences(note_body))
    ]


@dataclass
class FaithfulnessReport:
    note_id: str
    n_sentences: int
    n_cited: int
    n_uncited: int
    n_invalid_citations: int  # citations to a fact_id not in this admission's FactSet
    invalid_examples: list[tuple[str, str]]  # (sentence, bad_fact_id)

    @property
    def faithfulness_rate(self) -> float:
        """Fraction of sentences that are either uncited (no clinical claim, e.g. a
        transition phrase) or fully and validly cited. A sentence with even one
        invalid citation counts against this."""
        bad_sentences = len({s for s, _ in self.invalid_examples})
        return (self.n_sentences - bad_sentences) / self.n_sentences if self.n_sentences else 1.0


def check_faithfulness(sentences: list[Sentence], fact_set: FactSet) -> FaithfulnessReport:
    valid_ids = set(fact_set.by_id())
    n_cited = sum(1 for s in sentences if s.is_cited)
    invalid_examples: list[tuple[str, str]] = []
    for s in sentences:
        for fid in s.cited_fact_ids:
            if fid not in valid_ids:
                invalid_examples.append((s.text, fid))
    note_id = sentences[0].note_id if sentences else ""
    return FaithfulnessReport(
        note_id=note_id,
        n_sentences=len(sentences),
        n_cited=n_cited,
        n_uncited=len(sentences) - n_cited,
        n_invalid_citations=len(invalid_examples),
        invalid_examples=invalid_examples,
    )


# --- Persistence -------------------------------------------------------------

FACT_COLUMNS = ["fact_id", "hadm_id", "table", "row_id", "column", "text"]
LEDGER_COLUMNS = ["note_id", "hadm_id", "note_type", "sentence_idx", "text", "cited_fact_ids"]


def facts_to_dataframe(fact_sets: list[FactSet]) -> pd.DataFrame:
    rows: list[Fact] = [f for fs in fact_sets for f in fs.facts]
    return pd.DataFrame(
        [(f.fact_id, f.hadm_id, f.table, f.row_id, f.column, f.text) for f in rows],
        columns=FACT_COLUMNS,
    )


def ledger_to_dataframe(sentences: list[Sentence]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (s.note_id, s.hadm_id, s.note_type, s.sentence_idx, s.text, s.cited_fact_ids)
            for s in sentences
        ],
        columns=LEDGER_COLUMNS,
    )


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
