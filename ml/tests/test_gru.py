"""Tests for ml/models/gru.py -- sequence construction and a smoke test of
the training loop (kept tiny: this is not where the real 5x5 evaluation
protocol runs, that happens in ml/evaluation/run_all.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.models import gru


def _synthetic_hourly_grid() -> pd.DataFrame:
    rows = []
    for stay_id, n_hours in [(1, 30), (2, 5)]:
        for h in range(n_hours):
            row: dict[str, float | bool] = {"stay_id": stay_id, "hour": h}
            for v in gru.CORE_VITALS:
                row[v] = 70.0 + h
            for flag in gru.FLAG_COLUMNS:
                row[flag] = False
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_sequences_left_pads_short_stays_and_truncates_long_ones() -> None:
    grid = _synthetic_hourly_grid()
    labels_df = pd.DataFrame({"stay_id": [1, 1, 2], "hour": [29, 5, 4], "label_6h": [1, 0, 0]})
    batch = gru.build_sequences(grid, labels_df, "label_6h", seq_len=24)

    assert batch.x.shape == (3, 24, len(gru.SEQUENCE_FEATURE_COLUMNS))
    assert list(batch.y) == [1, 0, 0]
    assert list(batch.stay_ids) == [1, 1, 2]

    # Stay 1, hour 29: 24-hour window is full -> length 24, no padding.
    assert batch.lengths[0] == 24
    # Stay 2, hour 4: only 5 hours of history exist -> length 5, left-padded.
    assert batch.lengths[2] == 5
    hr_col = gru.SEQUENCE_FEATURE_COLUMNS.index("hr")
    assert (batch.x[2, :19, hr_col] == 0).all()  # padded region is zero
    assert batch.x[2, 19, hr_col] != 0  # first real timestep


def test_build_sequences_skips_rows_for_stays_absent_from_the_grid() -> None:
    grid = _synthetic_hourly_grid()
    labels_df = pd.DataFrame({"stay_id": [999], "hour": [0], "label_6h": [1]})
    batch = gru.build_sequences(grid, labels_df, "label_6h")
    assert batch.x.shape[0] == 0


def test_fit_predict_proba_runs_a_full_training_loop_without_error() -> None:
    rng = np.random.default_rng(0)
    n_train, seq_len, n_features = 40, 24, len(gru.SEQUENCE_FEATURE_COLUMNS)
    x_train = rng.normal(size=(n_train, seq_len, n_features)).astype(np.float32)
    len_train = np.full(n_train, seq_len)
    y_train = (rng.uniform(size=n_train) < 0.3).astype(np.int64)
    x_test = rng.normal(size=(5, seq_len, n_features)).astype(np.float32)
    len_test = np.full(5, seq_len)

    model, proba = gru.fit_predict_proba(
        x_train, len_train, y_train, x_test, len_test, epochs=2, batch_size=8
    )
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.skipif(
    not (
        __import__("pathlib").Path(__file__).resolve().parent.parent.parent
        / "warehouse"
        / "mimic4_demo.db"
    ).exists(),
    reason="real warehouse db not built",
)
def test_build_sequences_against_real_warehouse() -> None:
    import duckdb

    from ml.features import engineer, labels

    db_path = (
        __import__("pathlib").Path(__file__).resolve().parent.parent.parent
        / "warehouse"
        / "mimic4_demo.db"
    )
    conn = duckdb.connect(str(db_path), read_only=True)
    grid = conn.execute("select stay_id, hour from capstone.hourly_grid").fetchdf()
    lab = labels.build_labels(conn, grid)
    raw = engineer.load_hourly_grid_raw(conn)
    batch = gru.build_sequences(raw, lab, "label_6h")
    assert batch.x.shape[0] == len(lab)
    assert batch.x.shape[1:] == (gru.SEQUENCE_LEN_H, len(gru.SEQUENCE_FEATURE_COLUMNS))
