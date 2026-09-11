"""Describe a MIMIC-IV source directory and load it at any scale.

``warehouse/build_duckdb.py`` was written against the 100-patient demo, where
"load it" means ``COPY`` every file wholesale. Full MIMIC-IV is roughly three
orders of magnitude larger (``chartevents`` alone is ~432M rows against the
demo's 668,862), which breaks that assumption in two places: the machine may
not have room for the whole warehouse, and the demo's exact-row-count
validation is meaningless against a different release. This module supplies
what the bigger source needs, without disturbing the demo path that every
existing number in this project was produced on.

**Cohort sampling is by subject, never by stay.** Sampling ICU *stays* would
be the obvious reading of "take 10,000 stays", and it is wrong here for three
concrete reasons, each of which silently corrupts a label rather than raising
an error:

* ``ml/features/labels.py`` attributes death to ``max(icustay_seq)`` per
  ``hadm_id``. Drop one of a patient's stays and the surviving stay can become
  the maximum by accident, moving a death onto a stay that ended in a live
  transfer.
* Unplanned ICU readmission is defined as a *consecutive pair* of stays inside
  one admission. Split the pair and the event disappears from the data
  entirely.
* ``ml/evaluation/reliability.py`` establishes that the subject is the unit of
  statistical independence here -- 21 of 93 subjects in the demo's at-risk set
  have more than one stay, carrying 44% of its positives. A stay-level sample
  would put the same patient's other stay in a different fold of whatever is
  trained on the result.

So the cohort is a seeded random sample of subjects **that have at least one
ICU stay**, and every table is filtered to those subjects -- which keeps each
sampled patient's complete hospital history, including admissions that never
reached the ICU. The sample is drawn uniformly and prevalence is left alone:
case-control sampling on the outcome would wreck calibration, and calibration
is what an alerting threshold is set from.

Two loading strategies, chosen by whether a cohort filter applies:

* **Unfiltered** -- the existing ``COPY`` path, untouched. This is what every
  committed demo number was produced with, so it stays the proven default.
* **Filtered** -- ``INSERT ... SELECT ... WHERE subject_id IN (cohort)``,
  reading each CSV with column types taken from the *declared schema* rather
  than from DuckDB's sniffer. Type inference on a sample of a 2.6GB gzip is a
  real hazard on MIMIC (``chartevents.value`` and ``labevents.value`` are text
  columns whose leading rows look numeric), and the vendored ``create.sql`` is
  the authoritative answer. Reading this way also tolerates a release whose
  column order differs, or which adds columns the vendored schema predates:
  columns are matched by name, extras are read but not inserted, and absent
  ones land as NULL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

MODULES = ("hosp", "icu")
COHORT_SCHEMA = "capstone"
COHORT_TABLE = f"{COHORT_SCHEMA}.cohort_subjects"

# The demo ships this file at its root; no full release does. Checked before
# any size heuristic because it is an exact marker rather than a guess.
DEMO_MARKER = "demo_subject_id.csv"


@dataclass(frozen=True)
class MimicVariant:
    """Which MIMIC-IV release a directory holds, and how to check it loaded."""

    name: str  # "demo" | "full"
    data_dir: Path

    @property
    def is_demo(self) -> bool:
        return self.name == "demo"

    @property
    def supports_exact_row_validation(self) -> bool:
        """Only the demo has published per-table row counts to check against."""
        return self.is_demo


def detect_variant(data_dir: Path) -> MimicVariant:
    """Identify the release in ``data_dir``.

    The demo marker file is definitive. Everything else is treated as a full
    release, which is the safe direction to be wrong in: the full path applies
    structural validation rather than asserting demo row counts, so a
    misidentified demo directory produces weaker checks, never false failures.
    """
    name = "demo" if (data_dir / DEMO_MARKER).exists() else "full"
    return MimicVariant(name=name, data_dir=data_dir)


def source_files(data_dir: Path) -> list[Path]:
    """Every module CSV, gzipped only.

    Loose ``.csv`` duplicates that sit beside the gzips in some distributions
    (the demo carries ``icu/chartevents.csv`` and ``icu/datetimeevents 2.csv``)
    are excluded by construction rather than by name, exactly as the original
    loader did.
    """
    return sorted(f for module in MODULES for f in (data_dir / module).glob("*.csv.gz"))


def table_name_for(path: Path) -> str:
    """``hosp/admissions.csv.gz`` -> ``mimiciv_hosp.admissions``"""
    return f"mimiciv_{path.parent.name}.{path.name.split('.')[0]}"


def _quote_path(path: Path) -> str:
    text = str(path)
    if "'" in text:
        raise ValueError(f"refusing to interpolate a path containing a quote: {text}")
    return text


def table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = conn.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema || '.' || table_name = ?",
        [table],
    ).fetchone()
    return bool(row and row[0])


def declared_columns(conn: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    """Column -> declared SQL type, from the schema ``create.sql`` built."""
    schema, name = table.split(".", 1)
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
        [schema, name],
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def csv_columns(conn: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    """Column names in *file* order, so a release that reorders or adds
    columns still loads by name rather than by position.
    """
    conn.execute(
        f"SELECT * FROM read_csv('{_quote_path(path)}', {CSV_DIALECT}, all_varchar=true) LIMIT 0"
    )
    assert conn.description is not None
    return [d[0] for d in conn.description]


# --------------------------------------------------------------------------
# Cohort selection
# --------------------------------------------------------------------------


def icu_subject_ids(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> list[int]:
    """Distinct subjects with at least one ICU stay, read straight from the
    source CSV -- the cohort has to be chosen *before* anything is loaded.

    Sampling from ``hosp/patients`` instead would be a mistake worth naming:
    MIMIC-IV holds several hundred thousand patients, only a minority of whom
    ever reach an ICU, so a uniform sample of patients would spend most of the
    cohort budget on subjects that contribute no ICU stay at all.
    """
    icustays = data_dir / "icu" / "icustays.csv.gz"
    if not icustays.exists():
        raise FileNotFoundError(f"cannot select a cohort without {icustays}")
    rows = conn.execute(
        "SELECT DISTINCT CAST(subject_id AS BIGINT) AS subject_id "
        f"FROM read_csv('{_quote_path(icustays)}', {CSV_DIALECT}, all_varchar=true) "
        "ORDER BY subject_id"
    ).fetchall()
    return [int(r[0]) for r in rows]


def create_cohort(
    conn: duckdb.DuckDBPyConnection,
    data_dir: Path,
    n_subjects: int,
    seed: int = 0,
) -> int:
    """Materialise ``capstone.cohort_subjects`` as a seeded uniform sample.

    Returns the cohort size actually taken, which is capped at the number of
    ICU subjects available -- asking for more subjects than the release holds
    is a no-op, not an error.
    """
    available = icu_subject_ids(conn, data_dir)
    take = min(n_subjects, len(available))

    # Drawn in numpy rather than through DuckDB's USING SAMPLE, for two
    # reasons. DuckDB rejects a bound parameter in a sample clause outright
    # ("Only constants are supported in sample clause currently"), and its
    # sampler is seeded by a global setseed() whose reach across a connection
    # is easy to get subtly wrong. A local Generator makes the draw depend on
    # nothing but (seed, available), so the same seed and release reproduce
    # the same cohort no matter what else ran on this connection first.
    rng = np.random.default_rng(seed)
    chosen = sorted(int(s) for s in rng.choice(available, size=take, replace=False))

    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {COHORT_SCHEMA}")
    conn.execute(f"DROP TABLE IF EXISTS {COHORT_TABLE}")
    conn.execute(f"CREATE TABLE {COHORT_TABLE} (subject_id BIGINT PRIMARY KEY)")
    conn.executemany(f"INSERT INTO {COHORT_TABLE} VALUES (?)", [[s] for s in chosen])

    row = conn.execute(f"SELECT count(*) FROM {COHORT_TABLE}").fetchone()
    assert row is not None
    return int(row[0])


def cohort_size(conn: duckdb.DuckDBPyConnection) -> int | None:
    if not table_exists(conn, COHORT_TABLE):
        return None
    row = conn.execute(f"SELECT count(*) FROM {COHORT_TABLE}").fetchone()
    return int(row[0]) if row else None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


# The dialect the demo build has always used. Repeated here rather than left to
# auto-detection: the sniffer reads only the first `sample_size` rows of a gzip,
# and on labevents it settles on an empty escape character and then fails partway
# through the real file. These four settings are the ones COPY was already
# passing, so the filtered path parses byte-for-byte like the proven one.
CSV_DIALECT = "header=true, delim=',', quote='\"', escape='\"'"


def load_table_filtered(
    conn: duckdb.DuckDBPyConnection,
    path: Path,
    table: str,
) -> int:
    """Load one CSV, keeping only rows whose ``subject_id`` is in the cohort.

    Tables with no ``subject_id`` (``d_items``, ``d_labitems``, ``provider``,
    ``caregiver``, the ICD dictionaries) are reference data and load whole --
    they are small, and filtering them would break every join that resolves a
    code to its label.

    Every column is read as VARCHAR and cast to its **declared** type on
    insert, which takes DuckDB's type sniffer out of the loop entirely. That
    matters more than it sounds: the sniffer samples the head of the file, and
    several MIMIC columns are text whose leading rows look numeric
    (``chartevents.value``, ``labevents.value`` with entries like ``<0.1``).
    A sniffed type that is right for the first 20k rows and wrong for row 30M
    fails the load hours in. The vendored ``create.sql`` already states the
    right answer, so it is the one used. Empty fields arrive from ``read_csv``
    as NULL, not as an empty string, so the casts pass them through untouched.
    """
    declared = declared_columns(conn, table)
    file_cols = csv_columns(conn, path)
    shared = [c for c in file_cols if c in declared]
    if not shared:
        raise ValueError(f"{path.name} shares no columns with {table}")

    col_list = ", ".join(shared)
    cast_list = ", ".join(f'CAST("{c}" AS {declared[c]}) AS "{c}"' for c in shared)
    where = ""
    if "subject_id" in declared:
        where = f"WHERE CAST(subject_id AS BIGINT) IN (SELECT subject_id FROM {COHORT_TABLE})"

    conn.execute(f"DELETE FROM {table}")
    conn.execute(
        f"INSERT INTO {table} ({col_list}) SELECT {cast_list} "
        f"FROM read_csv('{_quote_path(path)}', {CSV_DIALECT}, all_varchar=true) {where}"
    )
    row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def load_table_whole(conn: duckdb.DuckDBPyConnection, path: Path, table: str) -> int:
    """The original ``COPY`` load, byte-for-byte the behaviour every committed
    demo number was produced with. Kept as the unfiltered path deliberately
    rather than folded into the filtered one: it is the proven route, and the
    demo build should not change because the full build was added.
    """
    conn.execute(f"DELETE FROM {table}")
    conn.execute(
        f"COPY {table} FROM '{_quote_path(path)}' (HEADER, DELIM ',', QUOTE '\"', ESCAPE '\"')"
    )
    row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])
