import duckdb
import pytest
from warehouse.build_duckdb import DEFAULT_DB_PATH

from notes_synth.backends import OfflineTemplateBackend
from notes_synth.facts import extract_facts
from notes_synth.generate import (
    WATERMARK,
    build_note_body,
    eligible_note_types,
    generate_one,
    sample_hadm_ids,
)

pytestmark = pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="warehouse not built")


@pytest.fixture(scope="module")
def conn():
    c = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    yield c
    c.close()


def test_discharge_summary_is_always_eligible(conn):
    hadm_id = conn.execute(
        "SELECT hadm_id FROM mimiciv_hosp.admissions ORDER BY hadm_id LIMIT 1"
    ).fetchone()[0]
    fs = extract_facts(conn, hadm_id)
    assert "discharge_summary" in eligible_note_types(fs)


def test_ecg_stub_only_eligible_with_an_ecg_fact(conn):
    hadm_id = conn.execute(
        "SELECT hadm_id FROM mimiciv_hosp.admissions ORDER BY hadm_id LIMIT 1"
    ).fetchone()[0]
    fs_without_ecg = extract_facts(conn, hadm_id)  # no ecg_index passed
    assert "ecg_report_stub" not in eligible_note_types(fs_without_ecg)


def test_build_note_body_always_starts_with_the_watermark():
    backend = OfflineTemplateBackend()
    hadm_id = 1
    from notes_synth.facts import Fact, FactSet

    fs = FactSet(hadm_id, [Fact("F001", hadm_id, "t", "1", "c", "Something happened.")])
    result = backend.generate_from_facts(fs, "discharge_summary")
    body = build_note_body(result)
    assert body.startswith(WATERMARK)


def test_generate_one_offline_produces_only_facts_that_exist(conn):
    hadm_id = conn.execute(
        "SELECT hadm_id FROM mimiciv_hosp.admissions ORDER BY hadm_id LIMIT 1"
    ).fetchone()[0]
    fs = extract_facts(conn, hadm_id)
    backend = OfflineTemplateBackend()
    result = generate_one(backend, fs, "discharge_summary", max_tokens=1024)
    for f in fs.facts:
        assert f"[{f.fact_id}]" in result.text


def test_sample_hadm_ids_is_reproducible_with_the_same_seed(conn):
    a = sample_hadm_ids(conn, 5, seed=42)
    b = sample_hadm_ids(conn, 5, seed=42)
    assert a == b
    assert len(a) == 5


def test_sample_hadm_ids_differs_across_seeds(conn):
    a = sample_hadm_ids(conn, 5, seed=1)
    b = sample_hadm_ids(conn, 5, seed=2)
    assert a != b
