"""Axis 4 -- RAG and agent (PROJECT_PLAN.md section 13): "retrieval recall@k,
faithfulness against fact_ledger.parquet, escalation agreement with the
rule-based policy, LLM cost per patient-day, and manual review of 50
summaries."

Loads rag-service's and agent-orchestrator's real modules directly (via
``services/common/testing.py``, since both live in hyphenated directories)
rather than re-implementing retrieval or the graph -- this axis measures the
real production code, not a parallel approximation of it.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.common.audit import AuditLog  # noqa: E402
from services.common.testing import load_service_app  # noqa: E402
from warehouse.news2 import should_escalate  # noqa: E402

_rag_module = load_service_app("rag-service", REPO_ROOT)
_agent_module = load_service_app("agent-orchestrator", REPO_ROOT)

# Groq pricing for openai/gpt-oss-120b, fetched 2026-09-06 from
# https://console.groq.com/docs/model/openai/gpt-oss-120b -- $0.15/1M input
# tokens, $0.60/1M output tokens. This is Groq's rate for the model this
# capstone actually ran against (no ANTHROPIC_API_KEY was available -- see
# notes_synth/README.md); PROJECT_PLAN.md's own target backend is
# claude-sonnet-5, whose real per-token cost differs and is not what this
# number reports.
GROQ_INPUT_COST_PER_1M = 0.15
GROQ_OUTPUT_COST_PER_1M = 0.60
assert GROQ_OUTPUT_COST_PER_1M >= GROQ_INPUT_COST_PER_1M, (
    "GROQ_UPPER_BOUND_COST_PER_1M below assumes the output rate is the more "
    "expensive of the two -- if Groq's real pricing ever flips this, the "
    "'upper bound' this axis reports would silently become a lower bound instead."
)
# agent-orchestrator's audit rows record combined tokens (input + output), not
# split -- record_agent_step's "tokens" field is deps.llm's
# input_tokens + output_tokens (nodes.py's summarizer). Costing the combined
# total at the (more expensive) output rate is therefore a deliberate upper
# bound, not an average -- stated so this number is never read as more
# precise than it is.
GROQ_UPPER_BOUND_COST_PER_1M = GROQ_OUTPUT_COST_PER_1M


# --------------------------------------------------------------------------
# Retrieval recall@k
# --------------------------------------------------------------------------


@dataclass
class RecallResult:
    k: int
    n_samples: int
    hits: int

    @property
    def recall(self) -> float:
        return self.hits / self.n_samples if self.n_samples else 0.0


def retrieval_recall_at_k(k: int = 5, n_samples: int = 100, seed: int = 0) -> RecallResult:
    """Self-retrieval sanity check: for a random sample of real fact-ledger
    passages actually in the index, query using the passage's own text and
    confirm the index finds that exact passage within the top k. This is
    deliberately not a claim about *relevance* to novel clinical queries
    (there is no independently labelled relevance-judgement set for that) --
    it proves the index can re-find content it was actually built from,
    which is the failure mode a broken vectoriser or a corpus-loading bug
    would produce (queries returning irrelevant results even for a document's
    own text is a real, checkable failure mode, not a strawman).
    """
    index = _rag_module.get_index()
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(
        len(index.passages), size=min(n_samples, len(index.passages)), replace=False
    )

    hits = 0
    for i in sample_idx:
        passage = index.passages[i]
        results = index.search(passage.text, k=k)
        if any(r.passage.passage_id == passage.passage_id for r in results):
            hits += 1
    return RecallResult(k=k, n_samples=len(sample_idx), hits=hits)


# --------------------------------------------------------------------------
# Faithfulness
# --------------------------------------------------------------------------

#  Excludes digits embedded in an alphanumeric token via the lookaround
# guards -- found for real: "NEWS2" and "SpO2" (score/measurement *names*
# that happen to contain a digit) were being extracted as the numbers "2",
# poisoning the faithfulness check with false "unsupported number" flags
# that have nothing to do with the LLM inventing a value.
_NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?(?![A-Za-z])")


@dataclass
class FaithfulnessResult:
    numbers_in_summary: list[float]
    numbers_grounded: list[float]
    numbers_unsupported: list[float]

    @property
    def faithfulness_rate(self) -> float:
        total = len(self.numbers_in_summary)
        return len(self.numbers_grounded) / total if total else 1.0


def check_faithfulness(
    summary_text: str, source_facts: dict, tolerance: float = 0.5
) -> FaithfulnessResult:
    """Extracts every numeric literal the LLM's summary states and checks it
    against every numeric value present anywhere in the structured facts it
    was given (`source_facts`, e.g. vitals/labs/risk_score) -- a summary
    inventing a vital sign or lab value not in its own input is exactly the
    failure mode PROJECT_PLAN.md section 10's Summarizer prompt
    ("state only what is given") exists to prevent, and this makes that
    checkable at corpus scale rather than by spot-reading transcripts.

    A crude but real check: it catches numeric fabrication (the most
    dangerous kind for a clinical summary) but not qualitative
    misstatements -- documented as a known limitation, not silently absorbed
    into the "faithfulness" name.
    """
    source_numbers = _extract_numbers_recursively(source_facts)
    summary_numbers = [float(m) for m in _NUMBER_RE.findall(summary_text)]

    grounded, unsupported = [], []
    for n in summary_numbers:
        if any(abs(n - s) <= tolerance for s in source_numbers):
            grounded.append(n)
        else:
            unsupported.append(n)
    return FaithfulnessResult(summary_numbers, grounded, unsupported)


def _extract_numbers_recursively(obj: object) -> list[float]:
    numbers: list[float] = []
    if isinstance(obj, bool):
        return numbers
    if isinstance(obj, int | float):
        numbers.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            numbers.extend(_extract_numbers_recursively(v))
    elif isinstance(obj, list | tuple):
        for v in obj:
            numbers.extend(_extract_numbers_recursively(v))
    elif isinstance(obj, str):
        numbers.extend(float(m) for m in _NUMBER_RE.findall(obj))
    return numbers


# --------------------------------------------------------------------------
# Escalation agreement + corpus-scale run harness
# --------------------------------------------------------------------------


def build_dependencies(conn_db_path: Path, audit_db_path: Path, use_llm: bool = True) -> Any:
    """Returns agent-orchestrator's real ``Dependencies`` -- typed ``Any``
    because that class is dynamically loaded (hyphenated package directory,
    see ``services.common.testing.load_service_app``) and so has no static
    type mypy can see through.
    """

    risk_engine_module = load_service_app("risk-engine", REPO_ROOT)
    rag_service_module = load_service_app("rag-service", REPO_ROOT)
    from starlette.testclient import TestClient as SyncASGIClient

    llm = None
    if use_llm:
        import os

        if os.environ.get("GROQ_API_KEY"):
            from notes_synth.backends import GroqBackend

            llm = GroqBackend()

    return _agent_module.Dependencies(
        db_path=conn_db_path,
        risk_engine_client=SyncASGIClient(risk_engine_module.app, base_url="http://risk-engine"),
        rag_client=SyncASGIClient(rag_service_module.app, base_url="http://rag-service"),
        audit_log=AuditLog(audit_db_path),
        llm=llm,
    )


@dataclass
class AgreementResult:
    n: int
    n_agree: int

    @property
    def agreement_rate(self) -> float:
        return self.n_agree / self.n if self.n else 1.0


def run_corpus(
    conn: duckdb.DuckDBPyConnection,
    stay_hour_pairs: list[tuple[int, int]],
    deps: object,
) -> list[dict]:
    """Runs the real graph once per (stay_id, hour) pair. Returns one dict
    per run with everything the other functions in this module need:
    escalate/tier agreement, the summary text (for faithfulness), the state
    dict (for faithfulness's source facts), and token usage (for cost).
    """
    graph = _agent_module.build_graph(deps)
    results = []
    for stay_id, hour in stay_hour_pairs:
        patient_ref = f"ICUStay/{stay_id}"
        state = graph.invoke(
            {"stay_id": stay_id, "hour": hour, "patient_ref": patient_ref, "audit_rows": []}
        )
        results.append(dict(state))
    return results


def escalation_agreement(results: list[dict]) -> AgreementResult:
    """Does the agent's escalate flag match the deterministic policy it claims to obey?

    The reference is ``warehouse.news2.should_escalate`` -- imported, not restated.
    This function previously inlined ``tier == "high"``, which was a *second* copy of
    the rule; when finding F1 added NEWS2's single-parameter limb to the policy node,
    this metric silently began reporting 67% agreement for an agent that was in fact
    obeying its policy perfectly. A hand-copied reference implementation measures
    drift between two copies of a rule, not agent fidelity.
    """
    n_agree = 0
    for state in results:
        risk = state.get("risk_score", {})
        expected = should_escalate(
            risk.get("news2_tier_icu"),
            risk.get("max_component_nongcs"),
            risk.get("gcs_drop", False),
        )
        n_agree += int(state.get("escalate") == expected)
    return AgreementResult(n=len(results), n_agree=n_agree)


@dataclass
class CostResult:
    n_runs: int
    total_tokens: int
    total_cost_usd_upper_bound: float
    mean_tokens_per_run: float
    mean_cost_per_run_usd: float
    projected_cost_per_patient_day_usd: float


def llm_cost_per_patient_day(
    audit_log: AuditLog, n_runs: int, hours_per_run: float = 1.0
) -> CostResult:
    """Token usage comes from the audit log, not the graph's returned state --
    ``nodes.py``'s ``audited()`` wrapper pops ``_tokens`` into the audit row
    before it ever reaches ``state`` (PROJECT_PLAN.md section 10's required
    audit shape), so the audit trail is the actual source of truth for cost,
    not a side channel.

    ``hours_per_run`` is how many hourly_grid hours of monitoring one graph
    run's audit trail corresponds to -- if the agent is invoked once per
    patient-hour in production (the natural cadence, since risk-engine's own
    score is hourly-charted, E1), that is 1.0; a less frequent invocation
    policy would change this, so it is a parameter, not a hidden constant.
    """
    rows = audit_log.all_rows()
    total_tokens = sum(r.payload.get("tokens") or 0 for r in rows if r.action == "agent_step")
    total_cost = total_tokens / 1_000_000 * GROQ_UPPER_BOUND_COST_PER_1M

    mean_tokens = total_tokens / n_runs if n_runs else 0.0
    mean_cost = total_cost / n_runs if n_runs else 0.0
    runs_per_patient_day = 24 / hours_per_run
    return CostResult(
        n_runs=n_runs,
        total_tokens=total_tokens,
        total_cost_usd_upper_bound=total_cost,
        mean_tokens_per_run=mean_tokens,
        mean_cost_per_run_usd=mean_cost,
        projected_cost_per_patient_day_usd=mean_cost * runs_per_patient_day,
    )


# --------------------------------------------------------------------------
# Manual review scaffold
# --------------------------------------------------------------------------


def sample_stay_hour_pairs(
    conn: duckdb.DuckDBPyConnection, n: int, seed: int = 0
) -> list[tuple[int, int]]:
    """A mix across severity tiers, not just the easy (already-obviously-
    high) cases -- a reviewer judging summary quality only on the clearest
    escalations would never see whether the agent is faithful when the
    picture is more ambiguous."""
    rows = conn.execute(
        "select stay_id, hour, tier_icu from capstone.news2 order by random()"
    ).fetchdf()
    per_tier = max(1, n // 3)
    picked = pd.concat(
        [rows[rows.tier_icu == tier].head(per_tier) for tier in ("low", "medium", "high")]
    )
    picked = picked.head(n)
    return list(zip(picked.stay_id, picked.hour, strict=True))


def build_manual_review_sheet(results: list[dict]) -> pd.DataFrame:
    """One row per real agent run, with everything a human reviewer needs to
    judge grounding and escalation appropriateness, and blank columns for
    their verdict -- PROJECT_PLAN.md section 13's "manual review of 50
    summaries" needs a human's clinical judgement, which this script cannot
    supply; it only prepares real content for that review to be done against.
    """
    rows = []
    for state in results:
        risk = state.get("risk_score", {})
        faithfulness = check_faithfulness(
            state.get("summary") or "",
            {
                "vitals": state.get("vitals"),
                "abnormal_labs": state.get("abnormal_labs"),
                "risk_score": risk,
            },
        )
        rows.append(
            {
                "patient_ref": state.get("patient_ref"),
                "hour": state.get("hour"),
                "news2": risk.get("news2"),
                "news2_tier_icu": risk.get("news2_tier_icu"),
                "escalate": state.get("escalate"),
                "escalation_reason": state.get("escalation_reason"),
                "summary": state.get("summary"),
                "n_context_passages": len(state.get("context_passages", [])),
                "auto_check_unsupported_numbers": faithfulness.numbers_unsupported,
                "auto_check_faithfulness_rate": round(faithfulness.faithfulness_rate, 3),
                "REVIEW_grounding_ok": "",
                "REVIEW_escalation_appropriate": "",
                "REVIEW_notes": "",
            }
        )
    return pd.DataFrame(rows)
