"""CLI entry point: runs all four evaluation axes for real and writes the
single cross-cutting HTML report PROJECT_PLAN.md section 13 requires.

Usage:
    python eval/run_eval.py                  # everything, including k6 (needs
                                              # the 5 pipeline services running
                                              # locally -- see eval/README.md)
    python eval/run_eval.py --skip-latency    # skip axis 3 if services aren't up
    python eval/run_eval.py --n-review 50     # manual-review sample size
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import ml  # noqa: E402,F401 -- sets KMP_DUPLICATE_LIB_OK/OMP_NUM_THREADS

from eval import alerting, latency, prediction, rag_agent, report  # noqa: E402
from eval.prediction import PRIMARY_HORIZON  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
OUTPUT_DIR = REPO_ROOT / "eval" / "output"
REPORT_PATH = OUTPUT_DIR / "report.html"
REVIEW_SHEET_PATH = OUTPUT_DIR / "manual_review_sample.csv"

BUDGETS = (0.05, 0.10, 0.20)
BUDGET_MODELS = ("news2", "lightgbm", "lightgbm_ecg")


def run_prediction_axis(conn: duckdb.DuckDBPyConnection) -> tuple[str, dict]:
    print("=== Axis 1: Prediction ===")
    predictions = prediction.collect_holdout_predictions(conn, horizon=PRIMARY_HORIZON)
    summary = prediction.summarize_prediction_axis(predictions)
    for name, entry in summary.items():
        print(f"  {name}: AUROC={entry['auroc'].point:.3f} AUPRC={entry['auprc'].point:.3f}")
    section = report.render_prediction_section(summary, PRIMARY_HORIZON)
    return section, predictions


def run_alerting_axis(conn: duckdb.DuckDBPyConnection, predictions: dict) -> str:
    print("\n=== Axis 2: Alerting ===")
    history = alerting.simulate_alert_history(conn)
    per_day = alerting.alerts_per_patient_day(history, conn)
    lead_time = alerting.median_lead_time_to_event(history, conn)
    false_alarm = alerting.false_alarm_rate_by_hour_of_day(history, conn)
    print(f"  {len(history)} alerts raised, {per_day:.2f}/patient-day")
    print(f"  lead time: {lead_time}")

    sensitivity_by_budget = {}
    for name in BUDGET_MODELS:
        pred = predictions[name]
        sensitivity_by_budget[name] = {
            b: alerting.sensitivity_at_alert_budget(pred.y_true, pred.y_score, b) for b in BUDGETS
        }
        print(f"  sensitivity[{name}]: {sensitivity_by_budget[name]}")

    return report.render_alerting_section(
        history, per_day, lead_time, false_alarm, sensitivity_by_budget
    )


def run_latency_axis(skip: bool) -> str:
    print("\n=== Axis 3: Latency ===")
    if skip:
        print("  skipped (--skip-latency)")
        return '<h2 id="latency">3. Latency</h2><p>Skipped for this run (--skip-latency).</p>'
    if not latency.k6_available():
        print("  k6 not installed -- skipping")
        return '<h2 id="latency">3. Latency</h2><p>k6 is not installed in this environment.</p>'
    results = latency.run_all_tiers(duration="20s")
    for r in results:
        print(f"  {r.device_count}x: p50={r.p50_ms} p95={r.p95_ms} p99={r.p99_ms} error={r.error}")
    return report.render_latency_section(results)


def run_rag_agent_axis(conn: duckdb.DuckDBPyConnection, n_review: int) -> str:
    print("\n=== Axis 4: RAG and agent ===")
    recall_result = rag_agent.retrieval_recall_at_k(k=5, n_samples=100)
    print(f"  recall@{recall_result.k}: {recall_result.recall:.3f}")

    audit_db_path = OUTPUT_DIR / "eval_agent_audit.db"
    audit_db_path.unlink(missing_ok=True)
    deps = rag_agent.build_dependencies(DEFAULT_DB_PATH, audit_db_path, use_llm=True)

    pairs = rag_agent.sample_stay_hour_pairs(conn, n=n_review)
    print(f"  running the real agent graph on {len(pairs)} (stay, hour) pairs...")
    results = rag_agent.run_corpus(conn, pairs, deps)

    agreement_result = rag_agent.escalation_agreement(results)
    faithfulness_rates = []
    for state in results:
        risk = state.get("risk_score", {})
        fr = rag_agent.check_faithfulness(
            state.get("summary") or "",
            {
                "vitals": state.get("vitals"),
                "abnormal_labs": state.get("abnormal_labs"),
                "risk_score": risk,
            },
        )
        faithfulness_rates.append(fr.faithfulness_rate)

    cost_result = rag_agent.llm_cost_per_patient_day(deps.audit_log, n_runs=len(results))
    print(
        f"  agreement: {agreement_result.agreement_rate:.3f}, "
        f"cost/run: ${cost_result.mean_cost_per_run_usd:.5f}"
    )

    review_sheet = rag_agent.build_manual_review_sheet(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    review_sheet.to_csv(REVIEW_SHEET_PATH, index=False)
    print(f"  wrote {REVIEW_SHEET_PATH}")

    return report.render_rag_agent_section(
        recall_result, faithfulness_rates, agreement_result, cost_result, review_sheet
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--n-review", type=int, default=50)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)

    prediction_section, predictions = run_prediction_axis(conn)
    alerting_section = run_alerting_axis(conn, predictions)
    latency_section = run_latency_axis(args.skip_latency)
    rag_agent_section = run_rag_agent_axis(conn, args.n_review)

    html_doc = report.render_report(
        {
            "prediction": prediction_section,
            "alerting": alerting_section,
            "latency": latency_section,
            "rag_agent": rag_agent_section,
        }
    )
    REPORT_PATH.write_text(html_doc)
    print(f"\nWrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
