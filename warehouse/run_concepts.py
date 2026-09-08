"""Derive every mimic-code concept table into mimiciv_derived, one file at a time.

PROJECT_PLAN.md section 7, items 2-4: run the vendored concept SQL in dependency
order, expect partial failure on this 100-patient demo, and never abort the whole
run over one broken concept. Each file is executed independently (try/except) and
the outcome -- succeeded / failed / row count / error -- is written to
concept_status.md. The dependency order itself is not re-derived here: it is read
straight out of mimic-code's own duckdb.sql, which is the authoritative build order
upstream ships and tests against (mimic-iv/VENDORED.md).

Usage:
    python warehouse/run_concepts.py [--db PATH]
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
CONCEPTS_DIR = REPO_ROOT / "mimic-iv" / "concepts_duckdb"
ORDER_FILE = CONCEPTS_DIR / "duckdb.sql"
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
STATUS_FILE = REPO_ROOT / "warehouse" / "concept_status.md"

# PROJECT_PLAN.md section 7, item 4 -- must-work concepts, plan name -> table name
# where they differ (e.g. the plan's shorthand "cbc" for complete_blood_count).
MUST_WORK = {
    "icustay_detail": "icustay_detail",
    "vitalsign": "vitalsign",
    "bg": "bg",
    "chemistry": "chemistry",
    "cbc": "complete_blood_count",
    "coagulation": "coagulation",
    "ventilation": "ventilation",
    "norepinephrine_equivalent_dose": "norepinephrine_equivalent_dose",
    "urine_output": "urine_output",
    "kdigo_stages": "kdigo_stages",
    "sofa": "sofa",
    "sapsii": "sapsii",
    "oasis": "oasis",
    "sirs": "sirs",
    "charlson": "charlson",
    "suspicion_of_infection": "suspicion_of_infection",
    "sepsis3": "sepsis3",
}


def status_path_for(db: Path) -> Path:
    """Where this build's status report goes.

    Derived from the database rather than fixed, because more than one warehouse
    can now exist (``build_duckdb.py --cohort-subjects`` builds alongside the
    demo). A hardcoded path meant running concepts against a cohort database
    silently overwrote the committed demo report with a different cohort's
    numbers -- observed, not hypothesised. The default database keeps the
    historical filename so nothing already referencing it breaks.
    """
    if db.resolve() == DEFAULT_DB_PATH.resolve():
        return STATUS_FILE
    return STATUS_FILE.with_name(f"concept_status_{db.stem}.md")


def build_order() -> list[str]:
    """Extract the .read sequence from duckdb.sql, e.g. 'demographics/icustay_times.sql'."""
    text = ORDER_FILE.read_text()
    return re.findall(r"^\.read (\S+)$", text, flags=re.MULTILINE)


def run_one(conn: duckdb.DuckDBPyConnection, rel_path: str) -> dict:
    """Execute one concept file. Each file is 'DROP TABLE IF EXISTS x; CREATE TABLE x AS ...'
    (verified across all 65 vendored files). Split on the FIRST semicolon only -- some
    files (e.g. measurement/rhythm.sql) embed a semicolon inside a string literal in the
    CREATE statement, so a naive full split would corrupt the query.
    """
    text = (CONCEPTS_DIR / rel_path).read_text()
    idx = text.index(";")
    drop_stmt, create_stmt = text[: idx + 1], text[idx + 1 :].strip()

    table = rel_path.rsplit("/", 1)[-1].removesuffix(".sql")
    phase = rel_path.split("/", 1)[0]
    result: dict = {"concept": table, "phase": phase, "path": rel_path}
    try:
        conn.execute(drop_stmt)
        conn.execute(create_stmt)
        row = conn.execute(f"SELECT count(*) FROM mimiciv_derived.{table}").fetchone()
        assert row is not None
        result.update(status="ok", rows=row[0], error=None)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: log and move on
        result.update(status="failed", rows=None, error=str(exc).strip())
    return result


def describe_cohort(conn: duckdb.DuckDBPyConnection) -> str:
    """What the concepts were actually built against.

    Read from the warehouse rather than hardcoded: with
    ``build_duckdb.py --cohort-subjects`` this file can now describe a sample of
    any size, and a status report that always claims "100 patients" would be
    wrong for every build except one.
    """
    try:
        row = conn.execute(
            "select count(*), count(distinct subject_id) from mimiciv_icu.icustays"
        ).fetchone()
    except Exception:  # noqa: BLE001 -- status text must never break the build
        return "an unidentified MIMIC-IV warehouse"
    if row is None:
        return "an unidentified MIMIC-IV warehouse"
    stays, subjects = int(row[0]), int(row[1])
    demo = " (the MIMIC-IV Clinical Database Demo)" if subjects == 100 and stays == 140 else ""
    return f"a MIMIC-IV warehouse of {subjects:,} patients / {stays:,} ICU stays{demo}"


def write_status(results: list[dict], cohort: str, status_file: Path) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]

    lines = [
        "# Concept build status",
        "",
        f"{len(ok)}/{len(results)} concepts built successfully ({len(failed)} failed) on "
        f"{cohort}.",
        "",
        "| Phase | Concept | Status | Rows | Note |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        note = "" if r["status"] == "ok" else (r["error"] or "").splitlines()[0][:120]
        rows = f"{r['rows']:,}" if r["rows"] is not None else "-"
        lines.append(f"| {r['phase']} | `{r['concept']}` | {r['status']} | {rows} | {note} |")

    lines += ["", "## Must-work concepts (PROJECT_PLAN.md section 7, item 4)", ""]
    by_table = {r["concept"]: r for r in results}
    missing_must_work = []
    for plan_name, table in MUST_WORK.items():
        rec = by_table.get(table)
        if rec is None:
            lines.append(f"- `{plan_name}` ({table}): **never ran** -- not in the build order")
            missing_must_work.append(plan_name)
        elif rec["status"] != "ok":
            lines.append(f"- `{plan_name}` ({table}): **FAILED** -- {rec['error']}")
            missing_must_work.append(plan_name)
        else:
            lines.append(f"- `{plan_name}` ({table}): ok, {rec['rows']:,} rows")

    if failed:
        lines += ["", "## Failed concepts -- full error", ""]
        for r in failed:
            lines.append(f"### `{r['concept']}` ({r['phase']}/{r['path']})")
            lines.append("```")
            lines.append(r["error"])
            lines.append("```")

    status_file.write_text("\n".join(lines) + "\n")
    rel = status_file.relative_to(REPO_ROOT)
    print(f"\n{len(ok)}/{len(results)} concepts ok. Status written to {rel}")
    if missing_must_work:
        print(f"WARNING: must-work concepts not ok: {', '.join(missing_must_work)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    order = build_order()
    print(f"{len(order)} concepts in dependency order (from {ORDER_FILE.relative_to(REPO_ROOT)})")

    conn = duckdb.connect(str(args.db))
    conn.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_derived")
    cohort = describe_cohort(conn)
    print(f"Building against {cohort}")

    results = []
    t0 = time.time()
    for rel_path in order:
        r = run_one(conn, rel_path)
        mark = "ok" if r["status"] == "ok" else "FAILED"
        detail = f"{r['rows']:,} rows" if r["status"] == "ok" else r["error"].splitlines()[0][:100]
        print(f"  [{mark:>6}] {rel_path:<45} {detail}")
        results.append(r)
    conn.close()

    write_status(results, cohort, status_path_for(args.db))
    n_failed = sum(1 for r in results if r["status"] == "failed")
    print(f"Done in {time.time() - t0:.1f}s. {n_failed} of {len(results)} concepts failed.")
    return 0  # partial failure is expected and reported, not a build error


if __name__ == "__main__":
    raise SystemExit(main())
