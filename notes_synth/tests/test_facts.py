import duckdb
import pytest
from warehouse.build_duckdb import DEFAULT_DB_PATH

from notes_synth.facts import ECG_RECORD_LIST, extract_facts, load_ecg_index

pytestmark = pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="warehouse not built")


@pytest.fixture(scope="module")
def conn():
    c = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    yield c
    c.close()


@pytest.fixture(scope="module")
def sample_hadm_id(conn) -> int:
    return conn.execute(
        "SELECT hadm_id FROM mimiciv_hosp.admissions ORDER BY hadm_id LIMIT 1"
    ).fetchone()[0]


def test_extract_facts_returns_nonempty_set(conn, sample_hadm_id):
    fs = extract_facts(conn, sample_hadm_id)
    assert fs.hadm_id == sample_hadm_id
    assert len(fs.facts) > 0


def test_fact_ids_are_unique_and_sequential(conn, sample_hadm_id):
    fs = extract_facts(conn, sample_hadm_id)
    ids = [f.fact_id for f in fs.facts]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_no_fact_contains_a_redacted_placeholder(conn, sample_hadm_id):
    fs = extract_facts(conn, sample_hadm_id)
    for f in fs.facts:
        assert "___" not in f.text, f"redacted value leaked into a fact: {f.text}"


def test_every_fact_row_id_is_a_real_row_for_diagnoses(conn, sample_hadm_id):
    fs = extract_facts(conn, sample_hadm_id)
    diag_facts = [f for f in fs.facts if f.table == "mimiciv_hosp.diagnoses_icd"]
    assert diag_facts
    for f in diag_facts:
        hadm_id, seq_num = f.row_id.split(":")
        row = conn.execute(
            "SELECT 1 FROM mimiciv_hosp.diagnoses_icd WHERE hadm_id = ? AND seq_num = ?",
            [int(hadm_id), int(seq_num)],
        ).fetchone()
        assert row is not None, f"fact {f.fact_id} does not trace to a real row"


def test_to_prompt_block_contains_every_fact_id(conn, sample_hadm_id):
    fs = extract_facts(conn, sample_hadm_id)
    block = fs.to_prompt_block()
    for f in fs.facts:
        assert f"[{f.fact_id}]" in block


@pytest.mark.skipif(not ECG_RECORD_LIST.exists(), reason="ECG dataset not found under data/raw/")
def test_some_admissions_get_an_ecg_fact(conn):
    """E9: 92/100 clinical-demo patients have a linked ECG -- among the first 20
    admissions there should be several with a linked study."""
    ecg_index = load_ecg_index()
    hadm_ids = conn.execute(
        "SELECT hadm_id FROM mimiciv_hosp.admissions ORDER BY hadm_id LIMIT 20"
    ).fetchdf()["hadm_id"]
    n_with_ecg = sum(
        any("ECG" in f.text for f in extract_facts(conn, h, ecg_index).facts) for h in hadm_ids
    )
    assert n_with_ecg > 0
