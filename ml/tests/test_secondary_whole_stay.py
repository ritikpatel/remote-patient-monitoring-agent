"""Tests for ml/evaluation/secondary_whole_stay.py."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from ml.evaluation import secondary_whole_stay as sws

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"


def _synthetic_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute("""
        create table admissions as select * from (values
            (1, 100, timestamp '2100-01-01', timestamp '2100-01-05', 0),
            (1, 101, timestamp '2100-01-20', timestamp '2100-01-25', 0),
            (2, 200, timestamp '2100-01-01', timestamp '2100-01-10', 1)
        ) as t(subject_id, hadm_id, admittime, dischtime, hospital_expire_flag)
        """)
    conn.execute("create schema mimiciv_hosp")
    conn.execute("create table mimiciv_hosp.admissions as select * from admissions")
    return conn


def test_readmission_labels_flag_only_alive_discharges_within_30_days() -> None:
    conn = _synthetic_conn()
    labels = sws.build_readmission_labels(conn)
    # hadm 100: readmitted 19 days later, alive discharge -> 1.
    assert labels.set_index("hadm_id").loc[100, "readmitted_30d"] == 1
    # hadm 101: no next admission -> 0.
    assert labels.set_index("hadm_id").loc[101, "readmitted_30d"] == 0
    # hadm 200: died -> excluded entirely (a death cannot be "readmitted").
    assert 200 not in labels.hadm_id.values


def test_representative_stay_per_admission_takes_the_last_stay() -> None:
    stays = pd.DataFrame({"hadm_id": [1, 1, 2], "stay_id": [10, 20, 30], "x": ["a", "b", "c"]})
    rep = sws.representative_stay_per_admission(stays)
    assert set(rep.stay_id) == {20, 30}


@pytest.mark.skipif(not WAREHOUSE_DB.exists(), reason="real warehouse db not built")
def test_build_stay_features_against_real_warehouse_has_one_row_per_stay() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    features = sws.build_stay_features(conn)
    assert len(features) == 140
    assert features.stay_id.is_unique
    assert int(features.hospital_expire_flag.sum()) == 20  # E4/E6's stay-level count


@pytest.mark.skipif(not WAREHOUSE_DB.exists(), reason="real warehouse db not built")
def test_evaluate_whole_stay_task_runs_end_to_end_on_real_mortality_labels() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    features = sws.build_stay_features(conn)
    result = sws.evaluate_whole_stay_task(
        features, features["hospital_expire_flag"], features["subject_id"], n_repeats=2
    )
    assert result["n"] == 140
    assert result["n_positive"] == 20
    assert 0.0 <= result["auroc_point"] <= 1.0
    assert result["auroc_lo"] <= result["auroc_point"] <= result["auroc_hi"]
