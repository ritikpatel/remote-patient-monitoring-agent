from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from services.common.audit import AuditLog

from eval.rag_agent import (
    build_dependencies,
    build_manual_review_sheet,
    check_faithfulness,
    escalation_agreement,
    llm_cost_per_patient_day,
    retrieval_recall_at_k,
    run_corpus,
    sample_stay_hour_pairs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"

pytestmark = pytest.mark.skipif(not WAREHOUSE_DB.exists(), reason="real warehouse db not built")


def test_retrieval_recall_at_k_finds_most_passages_by_their_own_text() -> None:
    result = retrieval_recall_at_k(k=5, n_samples=50, seed=1)
    assert result.n_samples == 50
    # A self-retrieval check should recall the vast majority of a working
    # index's own content -- a low number here would mean the index or
    # corpus loading is broken, not that queries are merely hard.
    assert result.recall > 0.8


def test_check_faithfulness_flags_a_fabricated_number_not_in_source_facts() -> None:
    facts = {"vitals": {"hr": 90.0}, "risk_score": {"news2": 6}}
    result = check_faithfulness("HR is 90, NEWS2 is 6, but SpO2 is a fabricated 999.", facts)
    assert 999.0 in result.numbers_unsupported
    assert 90.0 in result.numbers_grounded
    assert 6.0 in result.numbers_grounded


def test_check_faithfulness_does_not_misparse_score_names_as_numbers() -> None:
    # Found for real: "NEWS2" and "SpO2" contain a literal digit and were
    # being extracted as the fabricated number "2".
    facts = {"vitals": {"hr": 90.0}}
    result = check_faithfulness("HR 90. NEWS2 and SpO2 were both normal.", facts)
    assert 2.0 not in result.numbers_in_summary


def test_check_faithfulness_perfectly_grounded_summary_has_rate_one() -> None:
    facts = {"vitals": {"hr": 90.0, "rr": 18.0}}
    result = check_faithfulness("HR 90, RR 18.", facts)
    assert result.faithfulness_rate == 1.0


def test_sample_stay_hour_pairs_spans_multiple_tiers() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    pairs = sample_stay_hour_pairs(conn, n=9)
    assert len(pairs) > 0
    for stay_id, hour in pairs:
        assert isinstance(stay_id, int) or hasattr(stay_id, "item")
        assert hour >= 0


def test_run_corpus_and_escalation_agreement_without_an_llm(tmp_path) -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    deps = build_dependencies(WAREHOUSE_DB, tmp_path / "audit.db", use_llm=False)
    pairs = sample_stay_hour_pairs(conn, n=4)
    results = run_corpus(conn, pairs, deps)
    assert len(results) == len(pairs)

    agreement = escalation_agreement(results)
    # escalation_decider computes escalate from the tier alone, before any
    # LLM is consulted -- agreement must be 100% by construction, with or
    # without an LLM configured.
    assert agreement.agreement_rate == 1.0

    sheet = build_manual_review_sheet(results)
    assert len(sheet) == len(results)
    assert "REVIEW_grounding_ok" in sheet.columns


def test_llm_cost_per_patient_day_is_zero_with_no_llm_configured(tmp_path) -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    deps = build_dependencies(WAREHOUSE_DB, tmp_path / "audit.db", use_llm=False)
    pairs = sample_stay_hour_pairs(conn, n=3)
    results = run_corpus(conn, pairs, deps)
    cost = llm_cost_per_patient_day(deps.audit_log, n_runs=len(results))
    assert cost.total_tokens == 0
    assert cost.total_cost_usd_upper_bound == 0.0


def test_llm_cost_per_patient_day_reads_from_the_audit_log_directly(tmp_path) -> None:
    audit_log = AuditLog(tmp_path / "audit.db")
    audit_log.record_agent_step(
        "Summarizer", input_data={}, output="x", tokens=1000, model_id="test"
    )
    result = llm_cost_per_patient_day(audit_log, n_runs=1, hours_per_run=1.0)
    assert result.total_tokens == 1000
    # 1000 tokens at the $0.60/1M upper-bound rate.
    assert result.total_cost_usd_upper_bound == pytest.approx(0.0006)
    assert result.projected_cost_per_patient_day_usd == pytest.approx(0.0006 * 24)
