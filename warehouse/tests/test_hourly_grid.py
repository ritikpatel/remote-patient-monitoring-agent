"""The hourly grid's SQL pivot must equal the pandas one it replaced.

``warehouse/hourly_grid.py`` originally pulled every matching ``chartevents``
row into pandas and called ``pivot_table``. That does not survive a cohort
larger than the demo, so the aggregation moved into DuckDB. The port is only
safe if it is *identical*, and "identical" is checked here against a literal
re-implementation of the original pandas algorithm rather than against a
committed parquet, so the test keeps working after the artefact is regenerated.

One real difference is expected and asserted for rather than against:
``mimiciv_icu.chartevents.valuenum`` is declared ``FLOAT`` (32-bit), so the
pandas path averaged in float32 while DuckDB's ``AVG`` accumulates in double.
The SQL result is therefore *more* precise, and the comparison uses a float32
round-trip tolerance instead of pretending the two agree bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from warehouse.hourly_grid import (
    CORE,
    EXPECTED_DEMO_STAYS,
    EXPECTED_ROWS,
    GRID_ITEMS,
    build_raw_grid,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"

# chartevents.valuenum is FLOAT: ~7 decimal digits. Anything tighter would be
# asserting that double-precision accumulation agrees with single-precision.
FLOAT32_TOL = 1e-6

pytestmark = pytest.mark.skipif(
    not DEMO_DB.exists(), reason="demo warehouse not built (run warehouse/build_duckdb.py)"
)


def pandas_reference_grid(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The original implementation, verbatim in behaviour: fetch every event
    row, pivot in pandas, then unify temperature / GCS / blood pressure.
    """
    itemids = ", ".join(str(i) for i in GRID_ITEMS)
    events = conn.execute(f"""
        SELECT ce.stay_id, ce.itemid, ce.valuenum,
               CAST(FLOOR(DATE_DIFF('second', ie.intime, ce.charttime) / 3600.0) AS BIGINT) AS hour
        FROM mimiciv_icu.chartevents ce
        JOIN mimiciv_icu.icustays ie ON ce.stay_id = ie.stay_id
        WHERE ce.itemid IN ({itemids})
        """).fetchdf()

    events = events[events.hour >= 0].copy()
    events["var"] = events.itemid.map(GRID_ITEMS)
    grid = events.pivot_table(
        index=["stay_id", "hour"], columns="var", values="valuenum", aggfunc="mean"
    ).reset_index()

    grid["temp_c"] = grid.get("temp_c", pd.Series(dtype=float)).fillna(
        (grid.get("temp_f", pd.Series(dtype=float)) - 32) * 5 / 9
    )
    gcs_cols = [c for c in ("gcs_eye", "gcs_verbal", "gcs_motor") if c in grid]
    grid["gcs_total"] = grid[gcs_cols].sum(axis=1, min_count=3) if gcs_cols else pd.NA
    grid["sbp"] = grid.get("sbp_art", pd.Series(dtype=float)).fillna(
        grid.get("sbp_ni", pd.Series(dtype=float))
    )
    grid["map"] = grid.get("map_art", pd.Series(dtype=float)).fillna(
        grid.get("map_ni", pd.Series(dtype=float))
    )
    for c in CORE:
        if c not in grid.columns:
            grid[c] = pd.NA
    return grid


@pytest.fixture(scope="module")
def grids() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = duckdb.connect(str(DEMO_DB), read_only=True)
    sql_grid = build_raw_grid(conn)
    ref_grid = pandas_reference_grid(conn)
    conn.close()
    key = ["stay_id", "hour"]
    return (
        sql_grid.sort_values(key).reset_index(drop=True),
        ref_grid.sort_values(key).reset_index(drop=True),
    )


def test_reproduces_the_eda_row_count(grids):
    sql_grid, _ = grids
    assert len(sql_grid) == EXPECTED_ROWS
    assert sql_grid.stay_id.nunique() == EXPECTED_DEMO_STAYS


def test_sql_and_pandas_pivots_have_the_same_shape(grids):
    sql_grid, ref_grid = grids
    assert len(sql_grid) == len(ref_grid)
    assert set(sql_grid.columns) == set(ref_grid.columns)


def test_sql_pivot_matches_the_pandas_pivot_value_for_value(grids):
    sql_grid, ref_grid = grids
    mismatched = {}
    for col in ref_grid.columns:
        got = sql_grid[col].astype(float)
        want = ref_grid[col].astype(float)
        close = np.isclose(got, want, rtol=FLOAT32_TOL, atol=FLOAT32_TOL, equal_nan=True)
        if not close.all():
            mismatched[col] = int((~close).sum())
    assert not mismatched, f"columns differ from the pandas reference: {mismatched}"


def test_all_null_hours_are_dropped_exactly_as_pivot_table_dropped_them(grids):
    """``pivot_table`` emitted no row for a (stay, hour) whose every valuenum was
    NULL; a bare GROUP BY would emit an all-NULL one. The HAVING clause is what
    keeps the two in step, and this is the assertion that would catch its loss.
    """
    sql_grid, _ = grids
    value_cols = [c for c in sql_grid.columns if c not in ("stay_id", "hour")]
    all_null = sql_grid[value_cols].isna().all(axis=1)
    assert not all_null.any(), f"{int(all_null.sum())} all-NULL rows leaked into the grid"


def test_gcs_total_requires_all_three_components(grids):
    """min_count=3 in pandas: a partial GCS must not silently sum to a low score,
    which would read as neurological deterioration that never happened."""
    sql_grid, _ = grids
    components = ["gcs_eye", "gcs_verbal", "gcs_motor"]
    partial = sql_grid[components].notna().sum(axis=1).between(1, 2)
    assert sql_grid.loc[partial, "gcs_total"].isna().all()
