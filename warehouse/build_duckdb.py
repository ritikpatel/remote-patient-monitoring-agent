"""Load a MIMIC-IV release into a DuckDB warehouse -- demo or full.

Four steps, each idempotent (safe to re-run):
  1. detect_variant  -- demo (100 patients) or full, from the source directory.
  2. create_schema   -- (re)create mimiciv_hosp / mimiciv_icu / mimiciv_derived from the
                        vendored postgres schema, patched for DuckDB (see patch_create_sql).
  3. load_data       -- every hosp/*.csv.gz and icu/*.csv.gz into its table. Loose
                        .csv duplicates (icu/chartevents.csv, icu/outputevents.csv,
                        icu/datetimeevents 2.csv) are ignored by construction: we only
                        glob *.csv.gz.
  4. validate        -- exact published row counts for an unfiltered demo build;
                        structural checks (referential integrity, cohort containment)
                        for anything else, where no such published counts exist.

PROJECT_PLAN.md section 7, item 1. Schema naming rationale: mimic-iv/VENDORED.md.

**Scale.** Full MIMIC-IV's ``chartevents`` is ~432M rows against the demo's 668,862,
and a whole-release warehouse does not fit on every machine this project is developed
on. ``--cohort-subjects N`` loads a seeded uniform sample of N ICU patients instead,
carrying each one's complete hospital history; see ``warehouse/mimic_source.py`` for
why the sample is drawn over subjects rather than stays, and why prevalence is left
alone. Sizing that N is what ``ml/evaluation/reliability.py`` computes.

Usage:
    # demo, exactly as before
    python warehouse/build_duckdb.py

    # full release, 8,000-patient cohort into its own database
    python warehouse/build_duckdb.py \\
        --data-dir data/raw/mimic-iv-3.1 --db warehouse/mimic4_full.db \\
        --cohort-subjects 8000 --force
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from warehouse import mimic_source  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "mimic-iv-clinical-database-demo-2.2"
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
CREATE_SQL = REPO_ROOT / "mimic-iv" / "buildmimic" / "postgres" / "create.sql"
VALIDATE_SQL = REPO_ROOT / "mimic-iv" / "buildmimic" / "postgres" / "validate_demo.sql"

# Tables that are legitimately empty on some releases/cohorts, so an empty one is
# reported rather than failed. The demo's own thin areas are documented in
# mimic-iv/VENDORED.md; a cohort sample can additionally miss any low-frequency
# table by chance.
MAY_BE_EMPTY = {"mimiciv_hosp.hcpcsevents", "mimiciv_hosp.drgcodes", "mimiciv_icu.caregiver"}


def scalar(conn: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> object:
    row = conn.execute(sql, params or [])
    result = row.fetchone()
    assert result is not None
    return result[0]


def patch_create_sql(text: str) -> str:
    """Apply the same three DuckDB-compatibility patches as upstream's build_mimic.sh.

    1. TIMESTAMP(N) -- DuckDB does not accept a precision argument.
    2. microbiologyevents.spec_type_desc NOT NULL -- the demo has one zero-length
       string here, which the CSV loader treats as NULL.
    3. prescriptions.drug NOT NULL -- likewise, a zero-length string loads as NULL.
    """
    text = re.sub(r"TIMESTAMP\([0-9]+\)", "TIMESTAMP", text)
    text = re.sub(r"(spec_type_desc\s+VARCHAR\(\d+\))\s+NOT NULL", r"\1", text)
    text = re.sub(r"(drug\s+VARCHAR\(\d+\))\s+NOT NULL", r"\1", text)
    return text


def run_script(conn: duckdb.DuckDBPyConnection, sql_text: str) -> None:
    """Execute a semicolon-separated DDL script one statement at a time.

    Safe for create.sql / validate_demo.sql: neither contains a semicolon inside a
    string literal (verified by inspection). Concept SQL files are NOT run through
    this — see run_concepts.py, which needs to preserve embedded semicolons.
    """
    for stmt in sql_text.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    patched = patch_create_sql(CREATE_SQL.read_text())
    run_script(conn, patched)


def table_name_for(path: Path) -> str:
    """hosp/admissions.csv.gz -> mimiciv_hosp.admissions"""
    return mimic_source.table_name_for(path)


def load_data(
    conn: duckdb.DuckDBPyConnection, data_dir: Path, cohort: bool = False
) -> list[tuple[str, int]]:
    """Load every source file into its table.

    ``cohort=False`` keeps the original wholesale ``COPY``; ``cohort=True``
    filters each subject-keyed table to ``capstone.cohort_subjects`` while
    loading reference tables whole (mimic_source.load_table_filtered).
    """
    loaded = []
    for f in mimic_source.source_files(data_dir):
        table = table_name_for(f)
        if not mimic_source.table_exists(conn, table):
            print(f"   {table}: not in the schema, skipping {f.name}")
            continue
        t0 = time.time()
        if cohort:
            n = mimic_source.load_table_filtered(conn, f, table)
        else:
            n = mimic_source.load_table_whole(conn, f, table)
        print(f"   {table}: loaded {n:,} rows ({time.time() - t0:.1f}s)")
        loaded.append((table, n))
    return loaded


def validate_demo_row_counts(conn: duckdb.DuckDBPyConnection) -> bool:
    rows = conn.execute(VALIDATE_SQL.read_text()).fetchall()
    cols = [d[0] for d in conn.description]
    ok = True
    for row in rows:
        rec = dict(zip(cols, row, strict=True))
        if any(str(v).upper() == "FAILED" for v in rec.values()):
            ok = False
        print("   " + "  ".join(f"{k}={v}" for k, v in rec.items()))
    return ok


def validate_structure(conn: duckdb.DuckDBPyConnection, loaded: list[tuple[str, int]]) -> bool:
    """Checks that hold for any release or cohort, since no published row
    counts exist outside the demo.

    Referential integrity is the substantive one: a cohort filter that dropped
    a subject's rows from one table but not another would produce a warehouse
    that loads cleanly and then yields silently wrong labels downstream. These
    queries would catch that; a row count never would.
    """
    ok = True

    empty = [t for t, n in loaded if n == 0 and t not in MAY_BE_EMPTY]
    if empty:
        ok = False
        print(f"   FAILED: unexpectedly empty tables: {', '.join(empty)}")
    else:
        print(f"   ok: all {len(loaded)} loaded tables non-empty (or known-thin)")

    checks = [
        (
            "every icustays.subject_id exists in patients",
            "SELECT count(*) FROM mimiciv_icu.icustays s "
            "LEFT JOIN mimiciv_hosp.patients p USING (subject_id) WHERE p.subject_id IS NULL",
        ),
        (
            "every icustays.hadm_id exists in admissions",
            "SELECT count(*) FROM mimiciv_icu.icustays s "
            "LEFT JOIN mimiciv_hosp.admissions a USING (hadm_id) WHERE a.hadm_id IS NULL",
        ),
        (
            "every chartevents.stay_id exists in icustays",
            "SELECT count(*) FROM mimiciv_icu.chartevents c "
            "LEFT JOIN mimiciv_icu.icustays s USING (stay_id) WHERE s.stay_id IS NULL",
        ),
    ]
    n_cohort = mimic_source.cohort_size(conn)
    if n_cohort is not None:
        checks.append(
            (
                "every icustays.subject_id is in the cohort",
                "SELECT count(*) FROM mimiciv_icu.icustays "
                f"WHERE subject_id NOT IN (SELECT subject_id FROM {mimic_source.COHORT_TABLE})",
            )
        )
        checks.append(
            (
                "every chartevents.subject_id is in the cohort",
                "SELECT count(*) FROM mimiciv_icu.chartevents "
                f"WHERE subject_id NOT IN (SELECT subject_id FROM {mimic_source.COHORT_TABLE})",
            )
        )

    for label, sql in checks:
        orphans = int(scalar(conn, sql))  # type: ignore[call-overload]
        if orphans:
            ok = False
            print(f"   FAILED: {label} -- {orphans:,} violations")
        else:
            print(f"   ok: {label}")

    n_subjects = int(scalar(conn, "SELECT count(DISTINCT subject_id) FROM mimiciv_icu.icustays"))  # type: ignore[call-overload]
    n_stays = int(scalar(conn, "SELECT count(*) FROM mimiciv_icu.icustays"))  # type: ignore[call-overload]
    print(f"   cohort: {n_subjects:,} ICU subjects, {n_stays:,} ICU stays")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--force", action="store_true", help="rebuild schema even if the db exists")
    ap.add_argument(
        "--cohort-subjects",
        type=int,
        default=0,
        help="load a seeded random sample of N ICU patients instead of the whole release "
        "(0 = load everything, the default and the only sensible setting for the demo)",
    )
    ap.add_argument("--cohort-seed", type=int, default=0, help="seed for --cohort-subjects")
    args = ap.parse_args()

    if not (args.data_dir / "hosp").is_dir() or not (args.data_dir / "icu").is_dir():
        print(f"ERROR: {args.data_dir} must contain hosp/ and icu/ subfolders.", file=sys.stderr)
        return 1

    variant = mimic_source.detect_variant(args.data_dir)
    use_cohort = args.cohort_subjects > 0
    if use_cohort and variant.is_demo:
        print(
            "NOTE: --cohort-subjects on the demo release is allowed but pointless "
            "(it holds 100 patients); proceeding.",
            file=sys.stderr,
        )

    if args.force and args.db.exists():
        args.db.unlink()

    t0 = time.time()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(args.db))

    print(f"== variant: {variant.name} ({args.data_dir})")

    print(f"== create_schema: {CREATE_SQL.relative_to(REPO_ROOT)}")
    create_schema(conn)

    if use_cohort:
        print(
            f"== cohort: sampling {args.cohort_subjects:,} ICU subjects (seed={args.cohort_seed})"
        )
        n = mimic_source.create_cohort(
            conn, args.data_dir, args.cohort_subjects, seed=args.cohort_seed
        )
        print(f"   {n:,} subjects selected")

    print(f"== load_data: {args.data_dir}")
    loaded = load_data(conn, args.data_dir, cohort=use_cohort)
    print(f"   {len(loaded)} tables loaded")

    print("== validate")
    if variant.supports_exact_row_validation and not use_cohort:
        ok = validate_demo_row_counts(conn)
    else:
        reason = "cohort-filtered build" if use_cohort else f"{variant.name} release"
        print(f"   no published row counts for a {reason} -- structural checks instead")
        ok = validate_structure(conn, loaded)

    conn.close()
    dt = time.time() - t0
    print(
        f"\nBuild {'OK' if ok else 'completed with validation failures'} in {dt:.1f}s -> {args.db}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
