"""CLI entry point: generate one of the three report types end to end --
data assembly -> LLM narrative -> PDF -> FHIR DocumentReference.

Usage:
    python reports/run_reports.py shift-handover --ward "Medical Intensive Care Unit (MICU)"
    python reports/run_reports.py daily-summary --stay-id 30057454 --day 0
    python reports/run_reports.py post-discharge-digest --activity STRESS --participant S01
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from notes_synth.backends import GroqBackend  # noqa: E402

from reports.daily_summary import build_daily_summary  # noqa: E402
from reports.fhir_export import report_to_document_reference  # noqa: E402
from reports.pdf import ReportSection, render_pdf  # noqa: E402
from reports.post_discharge_digest import build_post_discharge_digest  # noqa: E402
from reports.shift_handover import build_ward_shift_handover  # noqa: E402

DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
OUTPUT_DIR = REPO_ROOT / "reports" / "output"


def _default_llm():
    return GroqBackend() if os.environ.get("GROQ_API_KEY") else None


def _fact_ids_from_passages(passages: list[dict]) -> list[str]:
    ids: list[str] = []
    for p in passages:
        ids.extend(p.get("fact_ids", []))
    return sorted(set(ids))


def run_shift_handover(ward: str, conn: duckdb.DuckDBPyConnection) -> Path:
    result = build_ward_shift_handover(conn, ward, llm=_default_llm())
    lines = "\n".join(
        f"{p.patient_ref}: NEWS2 {p.news2_start}->{p.news2_end} ({p.tier_end}), "
        f"{p.active_alerts} active alert(s)"
        for p in result.patients
    )
    sections = [
        ReportSection("Shift narrative", result.narrative),
        ReportSection("Patients this shift", lines or "(none currently monitored)"),
    ]
    out_path = OUTPUT_DIR / f"shift_handover_{ward.split()[0].lower()}.pdf"
    generated_note = (
        f"Covers each patient's most recent 4-hourly wall-clock block "
        f"({len(result.patients)} patients)."
    )
    render_pdf(
        out_path,
        title=f"Shift handover — {ward}",
        generated_note=generated_note,
        sections=sections,
        citations=[],
    )
    print(f"Wrote {out_path}")
    return out_path


def run_daily_summary(stay_id: int, day: int, conn: duckdb.DuckDBPyConnection) -> Path:
    result = build_daily_summary(conn, stay_id, day, llm=_default_llm())
    sections = [
        ReportSection("Summary", result.narrative),
        ReportSection(
            "NEWS2 trajectory",
            " -> ".join(
                f"{n} ({t})"
                for n, t in zip(result.news2_trajectory, result.tier_trajectory, strict=True)
            ),
        ),
        ReportSection(
            "Interventions today",
            f"Vasopressor started: {result.vasopressor_started_today}. "
            f"Ventilation started: {result.ventilation_started_today}.",
        ),
        ReportSection("Outstanding risk", f"Tier as of end of day: {result.outstanding_tier}"),
    ]
    out_path = OUTPUT_DIR / f"daily_summary_{stay_id}_day{day}.pdf"
    render_pdf(
        out_path,
        title=f"Daily summary — ICUStay/{stay_id}, day {day}",
        generated_note=f"Covers admission hours {result.hour_range[0]}-{result.hour_range[1]}.",
        sections=sections,
        citations=[],
    )
    print(f"Wrote {out_path}")

    # The one report type with an unambiguous single FHIR subject -- see
    # reports/README.md for why shift-handover (ward-wide, no single patient)
    # and the post-discharge digest (wearable participants have no MIMIC
    # patient link, E10) do not export a DocumentReference.
    ids = conn.execute(
        "select subject_id, hadm_id from mimiciv_icu.icustays where stay_id = ?", [stay_id]
    ).fetchone()
    if ids is not None:
        subject_id, hadm_id = ids
        doc_ref = report_to_document_reference(
            hadm_id, subject_id, "daily-summary", result.narrative
        )
        print(
            f"FHIR DocumentReference: subject=Patient/{subject_id}, "
            f"encounter=Encounter/{hadm_id}"
        )
        _ = doc_ref  # constructed for real (fhir.resources validates it); not persisted here
    return out_path


def run_post_discharge_digest(activity: str, participant: str) -> Path:
    result = build_post_discharge_digest(activity, participant, llm=_default_llm())
    lines = "\n".join(
        f"Day {b.day_index}: HR {b.mean_hr:.0f} bpm, SpO2 {b.mean_spo2:.0f}%, "
        f"HRV {b.hrv_rmssd_ms:.0f} ms"
        for b in result.daily_buckets
    )
    sections = [
        ReportSection("Digest narrative", result.narrative),
        ReportSection("Daily wearable summary (morphed, demo-compressed)", lines),
    ]
    out_path = OUTPUT_DIR / f"post_discharge_digest_{participant}.pdf"
    render_pdf(
        out_path,
        title=f"Post-discharge weekly digest — {participant}",
        generated_note=(
            "Morphed wearable telemetry, compressed into a demo session standing in "
            "for a week (R7) -- not a claim that a week was actually recorded."
        ),
        sections=sections,
        citations=[],
    )
    print(f"Wrote {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="report_type", required=True)

    p1 = sub.add_parser("shift-handover")
    p1.add_argument("--ward", required=True)

    p2 = sub.add_parser("daily-summary")
    p2.add_argument("--stay-id", type=int, required=True)
    p2.add_argument("--day", type=int, default=0)

    p3 = sub.add_parser("post-discharge-digest")
    p3.add_argument("--activity", default="STRESS")
    p3.add_argument("--participant", default="S01")

    args = parser.parse_args()

    if args.report_type == "post-discharge-digest":
        run_post_discharge_digest(args.activity, args.participant)
        return 0

    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    if args.report_type == "shift-handover":
        run_shift_handover(args.ward, conn)
    elif args.report_type == "daily-summary":
        run_daily_summary(args.stay_id, args.day, conn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
