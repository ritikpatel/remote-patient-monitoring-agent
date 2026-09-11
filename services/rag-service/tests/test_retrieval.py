import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from corpus import FACT_LEDGER_PATH, Passage, load_corpus, load_guideline_passages  # noqa: E402
from retrieval import TfidfIndex  # noqa: E402

pytestmark = pytest.mark.skipif(
    not FACT_LEDGER_PATH.exists(), reason="notes_synth output not generated"
)


def test_guideline_passages_have_no_fact_ids():
    for p in load_guideline_passages():
        assert p.source == "guideline"
        assert p.fact_ids == []


def test_load_corpus_includes_both_sources():
    corpus = load_corpus()
    sources = {p.source for p in corpus}
    assert sources == {"guideline", "note"}


def test_note_passages_carry_real_fact_ids_when_cited():
    corpus = load_corpus()
    cited = [p for p in corpus if p.source == "note" and p.fact_ids]
    assert cited, "expected at least one real cited passage from the Phase 3 output"
    assert all(fid.startswith("F") for p in cited for fid in p.fact_ids)


def test_search_returns_relevant_results_for_a_real_topic():
    corpus = load_corpus()
    idx = TfidfIndex(corpus)
    results = idx.search("sepsis and antibiotics", k=5)
    assert results
    assert any(
        "sepsis" in r.passage.text.lower() or "antibiotic" in r.passage.text.lower()
        for r in results
    )


def test_search_can_filter_to_guideline_source():
    corpus = load_corpus()
    idx = TfidfIndex(corpus)
    results = idx.search("SOFA organ dysfunction sepsis", k=3, source="guideline")
    assert results
    assert all(r.passage.source == "guideline" for r in results)


def test_search_scores_are_descending():
    corpus = load_corpus()
    idx = TfidfIndex(corpus)
    results = idx.search("NEWS2 escalation", k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_index_returns_nothing():
    idx = TfidfIndex([])
    assert idx.search("anything") == []


def test_passage_dataclass_defaults():
    p = Passage("id1", "text", "guideline", [])
    assert p.note_id is None
    assert p.hadm_id is None


# ---------------------------------------------------------------------------
# Patient scoping (hadm_id) -- added for the agent's ContextRetriever
# ---------------------------------------------------------------------------


def _two_patient_index() -> TfidfIndex:
    """Two admissions whose notes use the same clinical vocabulary, plus a guideline
    that uses it too. Deliberately confusable: that is the situation where an
    unscoped search returns the wrong patient's chart.

    The two filler passages are load-bearing, not padding. `TfidfIndex` builds its
    vectoriser with `max_df=0.9`, so a term appearing in *every* document is dropped
    from the vocabulary entirely -- in a three-document corpus where all three say
    "sepsis", the query term that matters scores nothing anywhere. The fillers keep
    document frequency below that ceiling so this fixture tests the hadm_id filter
    rather than an artefact of its own size.
    """
    return TfidfIndex(
        [
            Passage(
                "n1#0",
                "Patient has severe sepsis with hypotension.",
                "note",
                ["F1"],
                note_id="n1",
                hadm_id=111,
            ),
            Passage(
                "n2#0",
                "Patient has severe sepsis with hypotension and shock.",
                "note",
                ["F2"],
                note_id="n2",
                hadm_id=222,
            ),
            Passage("G003", "Sepsis is organ dysfunction caused by infection.", "guideline", []),
            Passage(
                "n3#0",
                "Wound dressing changed, dry and intact.",
                "note",
                [],
                note_id="n3",
                hadm_id=333,
            ),
            Passage("G099", "Falls risk assessment is repeated on transfer.", "guideline", []),
        ]
    )


def test_hadm_id_restricts_notes_to_one_admission():
    """The bug this prevents was real: the agent retrieved another patient's
    discharge summary and summarised it under the alerting patient's name."""
    results = _two_patient_index().search("sepsis hypotension", k=5, hadm_id=111)

    note_hadm_ids = {r.passage.hadm_id for r in results if r.passage.source == "note"}
    assert note_hadm_ids == {111}


def test_guidelines_survive_the_patient_filter():
    """A care plan needs the general clinical convention alongside this patient's
    specifics. Filtering guidelines out by hadm_id would leave the LLM with nothing
    to ground a recommendation in."""
    results = _two_patient_index().search("sepsis infection", k=5, hadm_id=111)

    assert any(r.passage.source == "guideline" for r in results)


def test_an_unscoped_search_is_unchanged():
    """hadm_id=None must behave exactly as before the parameter existed -- every
    pre-existing caller passes nothing."""
    index = _two_patient_index()

    unscoped = index.search("sepsis hypotension", k=5)

    assert {r.passage.hadm_id for r in unscoped if r.passage.source == "note"} == {111, 222}
