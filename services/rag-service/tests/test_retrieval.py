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
