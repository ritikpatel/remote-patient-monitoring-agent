"""Load the MIMIC-IV Clinical Database Demo into a DuckDB warehouse.

Three steps, each idempotent (safe to re-run):
  1. create_schema  -- (re)create mimiciv_hosp / mimiciv_icu / mimiciv_derived from the
                        vendored postgres schema, patched for DuckDB (see patch_create_sql).
  2. load_data      -- COPY every hosp/*.csv.gz and icu/*.csv.gz into its table. Loose
                        .csv duplicates (icu/chartevents.csv, icu/outputevents.csv,
                        icu/datetimeevents 2.csv) are ignored by construction: we only
                        glob *.csv.gz.
  3. validate       -- check every loaded table's row count against the known MIMIC-IV
                        demo counts (mimic-iv/buildmimic/postgres/validate_demo.sql).

PROJECT_PLAN.md section 7, item 1. Schema naming rationale: mimic-iv/VENDORED.md.

Usage:
    python warehouse/build_duckdb.py [--data-dir PATH] [--db PATH] [--force]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "mimic-iv-clinical-database-demo-2.2"
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
CREATE_SQL = REPO_ROOT / "mimic-iv" / "buildmimic" / "postgres" / "create.sql"
VALIDATE_SQL = REPO_ROOT / "mimic-iv" / "buildmimic" / "postgres" / "validate_demo.sql"

MODULES = ("hosp", "icu")


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
    module = path.parent.name
    table = path.name.split(".")[0]
    return f"mimiciv_{module}.{table}"


def load_data(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> list[tuple[str, int]]:
    loaded = []
    files = sorted(f for module in MODULES for f in (data_dir / module).glob("*.csv.gz"))
    for f in files:
        table = table_name_for(f)
        exists = scalar(
            conn,
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema || '.' || table_name = ?",
            [table],
        )
        if not exists:
            print(f"   {table}: not in the schema, skipping {f.name}")
            continue
        conn.execute(f"DELETE FROM {table}")
        conn.execute(f"COPY {table} FROM '{f}' (HEADER, DELIM ',', QUOTE '\"', ESCAPE '\"')")
        n = int(scalar(conn, f"SELECT count(*) FROM {table}"))  # type: ignore[call-overload]
        print(f"   {table}: loaded {n:,} rows")
        loaded.append((table, n))
    return loaded


def validate(conn: duckdb.DuckDBPyConnection) -> bool:
    rows = conn.execute(VALIDATE_SQL.read_text()).fetchall()
    cols = [d[0] for d in conn.description]
    ok = True
    for row in rows:
        rec = dict(zip(cols, row, strict=True))
        if any(str(v).upper() == "FAILED" for v in rec.values()):
            ok = False
        print("   " + "  ".join(f"{k}={v}" for k, v in rec.items()))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--force", action="store_true", help="rebuild schema even if the db exists")
    args = ap.parse_args()

    if not (args.data_dir / "hosp").is_dir() or not (args.data_dir / "icu").is_dir():
        print(f"ERROR: {args.data_dir} must contain hosp/ and icu/ subfolders.", file=sys.stderr)
        return 1

    if args.force and args.db.exists():
        args.db.unlink()

    t0 = time.time()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(args.db))

    print(f"== create_schema: {CREATE_SQL.relative_to(REPO_ROOT)}")
    create_schema(conn)

    print(f"== load_data: {args.data_dir}")
    loaded = load_data(conn, args.data_dir)
    print(f"   {len(loaded)} tables loaded")

    print("== validate")
    ok = validate(conn)

    conn.close()
    dt = time.time() - t0
    print(
        f"\nBuild {'OK' if ok else 'completed with row-count mismatches'} in {dt:.1f}s -> {args.db}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
