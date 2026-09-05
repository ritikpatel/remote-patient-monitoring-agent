from notes_synth.fact_ledger import (
    check_faithfulness,
    extract_citations,
    ledger_from_note,
    split_sentences,
)
from notes_synth.facts import Fact, FactSet


def make_fact_set() -> FactSet:
    facts = [
        Fact("F001", 1, "t", "1", "c", "Admitted electively."),
        Fact("F002", 1, "t", "2", "c", "Principal diagnosis: sepsis."),
        Fact("F003", 1, "t", "3", "c", "Peak SOFA score: 6."),
    ]
    return FactSet(1, facts)


def test_split_sentences_keeps_citation_with_the_sentence_it_introduces():
    text = "[F001] Admitted electively. [F002] Principal diagnosis: sepsis."
    sents = split_sentences(text)
    assert sents == ["[F001] Admitted electively.", "[F002] Principal diagnosis: sepsis."]


def test_extract_citations_handles_multiple_ids():
    assert extract_citations("[F001,F002] Two facts in one sentence.") == ["F001", "F002"]


def test_extract_citations_returns_empty_for_uncited_sentence():
    assert extract_citations("This is a transition sentence with no citation.") == []


def test_faithfulness_report_all_valid():
    note = "[F001] Admitted electively. [F002] Principal diagnosis: sepsis."
    sentences = ledger_from_note("note-1", 1, "discharge_summary", note)
    report = check_faithfulness(sentences, make_fact_set())
    assert report.n_sentences == 2
    assert report.n_invalid_citations == 0
    assert report.faithfulness_rate == 1.0


def test_faithfulness_report_catches_invented_citation():
    note = "[F001] Admitted electively. [F999] This fact does not exist."
    sentences = ledger_from_note("note-1", 1, "discharge_summary", note)
    report = check_faithfulness(sentences, make_fact_set())
    assert report.n_invalid_citations == 1
    assert report.invalid_examples[0][1] == "F999"
    assert report.faithfulness_rate == 0.5


def test_split_sentences_handles_markdown_bullets_and_headers():
    """Real model output is markdown: headers and bullet lists put each fact-bearing
    clause on its own line rather than ". "-delimited -- found by running a real
    batch where a ~20-line bulleted note counted as 5 "sentences"."""
    text = "**Diagnoses**\n\n- [F001] First finding.\n- [F002] Second finding.\n"
    sents = split_sentences(text)
    assert sents == ["**Diagnoses**", "- [F001] First finding.", "- [F002] Second finding."]


def test_uncited_transition_sentences_do_not_count_against_faithfulness():
    note = "The patient's course was otherwise unremarkable. [F001] Admitted electively."
    sentences = ledger_from_note("note-1", 1, "discharge_summary", note)
    report = check_faithfulness(sentences, make_fact_set())
    assert report.n_uncited == 1
    assert report.faithfulness_rate == 1.0
