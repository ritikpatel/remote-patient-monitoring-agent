"""Tests for the scale-out ingestion path (warehouse/mimic_source.py).

Two things need holding still here. The demo build is what every committed
number in this project was produced on, so it must keep loading byte-for-byte
as it did before cohort support existed. And the cohort path -- which is the
only part that will ever run against a release nobody here can test against --
must be shown to preserve exactly the referential structure that the labels
depend on, rather than merely to run without raising.

These run against the real demo dataset when it is present, and skip cleanly
when it is not (a fresh checkout has the symlinks but not necessarily the
data), matching how the rest of this project's warehouse-backed tests behave.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from warehouse import build_duckdb, mimic_source

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_DIR = REPO_ROOT / "data" / "raw" / "mimic-iv-clinical-database-demo-2.2"

pytestmark = pytest.mark.skipif(
    not (DEMO_DIR / "icu" / "icustays.csv.gz").exists(),
    reason="MIMIC-IV demo dataset not present",
)


@pytest.fixture(scope="module")
def cohort_db(tmp_path_factory) -> Path:
    """A real cohort-filtered build over the demo: 40 of its 100 subjects."""
    db = tmp_path_factory.mktemp("warehouse") / "cohort.db"
    conn = duckdb.connect(str(db))
    build_duckdb.create_schema(conn)
    mimic_source.create_cohort(conn, DEMO_DIR, n_subjects=40, seed=7)
    build_duckdb.load_data(conn, DEMO_DIR, cohort=True)
    conn.close()
    return db


def test_per_database_output_paths_keep_the_demo_artefacts_intact():
    """Every derived report/parquet is keyed to its warehouse.

    Regression test for a real incident: running `run_concepts.py --db <cohort>`
    overwrote the committed `warehouse/concept_status.md` with a 40-patient
    cohort's numbers, because the output path was fixed while the input was not.
    `hourly_grid.parquet` had the same flaw and is worse -- `ml/features` and the
    GRU read it, so the next training run would have silently used another
    cohort's data.
    """
    from warehouse import hourly_grid, news2, run_concepts

    other = Path("/tmp/mimic4_full.db")
    cases = [
        (run_concepts.status_path_for, run_concepts.DEFAULT_DB_PATH, run_concepts.STATUS_FILE),
        (hourly_grid.parquet_path_for, hourly_grid.DEFAULT_DB_PATH, hourly_grid.PARQUET_OUT),
        (news2.report_path_for, news2.DEFAULT_DB_PATH, news2.REPORT_FILE),
    ]
    for fn, default_db, historical in cases:
        assert fn(default_db) == historical, "the default database must keep its historical name"
        derived = fn(other)
        assert derived != historical, f"{fn.__name__} still collides with the demo artefact"
        assert "mimic4_full" in derived.name
        assert derived.parent == historical.parent


def test_detect_variant_identifies_the_demo_by_its_marker_file():
    assert mimic_source.detect_variant(DEMO_DIR).name == "demo"


def test_detect_variant_treats_an_unmarked_directory_as_full(tmp_path):
    (tmp_path / "hosp").mkdir()
    (tmp_path / "icu").mkdir()
    variant = mimic_source.detect_variant(tmp_path)
    assert variant.name == "full"
    # The consequence that matters: no exact row-count assertion is attempted
    # against a release that has no published counts.
    assert not variant.supports_exact_row_validation


def test_source_files_ignores_the_loose_csv_duplicates():
    """The demo ships icu/chartevents.csv beside chartevents.csv.gz; loading
    both would double every chartevents row."""
    names = [f.name for f in mimic_source.source_files(DEMO_DIR)]
    assert "chartevents.csv.gz" in names
    assert not any(n.endswith(".csv") and not n.endswith(".csv.gz") for n in names)


def test_cohort_selection_is_reproducible_for_a_seed():
    conn = duckdb.connect()
    first = mimic_source.icu_subject_ids(conn, DEMO_DIR)
    conn.close()

    def draw(seed: int) -> list[int]:
        c = duckdb.connect()
        mimic_source.create_cohort(c, DEMO_DIR, n_subjects=25, seed=seed)
        got = [
            r[0]
            for r in c.execute(f"SELECT subject_id FROM {mimic_source.COHORT_TABLE}").fetchall()
        ]
        c.close()
        return sorted(got)

    assert draw(3) == draw(3), "same seed must reproduce the same cohort"
    assert draw(3) != draw(4), "different seeds must draw different cohorts"
    assert set(draw(3)) <= set(first)


def test_cohort_is_capped_at_the_subjects_available():
    """Asking for more patients than the release holds is a no-op, not an error."""
    conn = duckdb.connect()
    n = mimic_source.create_cohort(conn, DEMO_DIR, n_subjects=10_000, seed=0)
    available = len(mimic_source.icu_subject_ids(conn, DEMO_DIR))
    conn.close()
    assert n == available


def test_cohort_load_keeps_only_cohort_subjects(cohort_db):
    conn = duckdb.connect(str(cohort_db), read_only=True)
    for table in ("mimiciv_icu.icustays", "mimiciv_icu.chartevents", "mimiciv_hosp.admissions"):
        orphans = conn.execute(
            f"SELECT count(*) FROM {table} "
            f"WHERE subject_id NOT IN (SELECT subject_id FROM {mimic_source.COHORT_TABLE})"
        ).fetchone()[0]
        assert orphans == 0, f"{table} leaked rows outside the cohort"
    conn.close()


def test_cohort_load_keeps_reference_tables_whole(cohort_db):
    """d_items has no subject_id and must not be filtered -- every itemid lookup
    downstream (including the hourly grid's 14 vitals) resolves through it."""
    conn = duckdb.connect(str(cohort_db), read_only=True)
    n_items = conn.execute("SELECT count(*) FROM mimiciv_icu.d_items").fetchone()[0]
    conn.close()
    assert n_items == 4014, "reference table was filtered when it should load whole"


def test_cohort_keeps_every_stay_of_a_sampled_subject(cohort_db):
    """Sampling subjects rather than stays is what keeps readmission events and
    icustay_seq intact; a subject must never arrive with only some of their stays.
    """
    conn = duckdb.connect(str(cohort_db), read_only=True)
    cohort_stays = conn.execute(
        "SELECT subject_id, count(*) AS n FROM mimiciv_icu.icustays GROUP BY subject_id"
    ).fetchdf()
    conn.close()

    full = duckdb.connect()
    true_stays = full.execute(
        "SELECT subject_id, count(*) AS n FROM "
        f"read_csv('{DEMO_DIR / 'icu' / 'icustays.csv.gz'}', {mimic_source.CSV_DIALECT}, "
        "all_varchar=true) GROUP BY subject_id"
    ).fetchdf()
    full.close()

    merged = cohort_stays.merge(
        true_stays.astype({"subject_id": int}), on="subject_id", suffixes=("_cohort", "_true")
    )
    assert len(merged) == len(cohort_stays)
    assert (merged.n_cohort == merged.n_true).all(), "a sampled subject lost some of their stays"


def test_filtered_load_preserves_declared_column_types(cohort_db):
    """The filtered path reads every column as VARCHAR and casts to the declared
    type. If that cast were skipped, chartevents.charttime would land as text and
    every downstream time computation would silently fail."""
    conn = duckdb.connect(str(cohort_db), read_only=True)
    types = mimic_source.declared_columns(conn, "mimiciv_icu.chartevents")
    row = conn.execute(
        "SELECT charttime, valuenum, value FROM mimiciv_icu.chartevents "
        "WHERE charttime IS NOT NULL LIMIT 1"
    ).fetchone()
    conn.close()
    assert types["charttime"].upper().startswith("TIMESTAMP")
    assert not isinstance(row[0], str), "charttime came back as text, not a timestamp"
    assert row[2] is None or isinstance(row[2], str), "value should stay textual"
