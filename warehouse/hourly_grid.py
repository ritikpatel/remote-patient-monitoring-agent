"""Build the hourly patient-state grid: one row per (stay_id, hour_since_intime).

Ported from notebooks/01_capstone_eda.ipynb section 5, reading from the DuckDB
warehouse instead of raw CSVs, and extended per PROJECT_PLAN.md R2 ("every imputed
value carries a flag") with a `<col>_was_imputed` and `<col>_hours_since_last_obs`
pair for every core vital. The raw grid must reproduce the EDA's 12,004 rows before
any forward-fill is applied -- that reproduction is the regression test for the port.

Output: table `capstone.hourly_grid` in the warehouse db, plus a parquet copy at
data/processed/hourly_grid.parquet for downstream ML code that would rather not open
a DuckDB connection.

Usage:
    python warehouse/hourly_grid.py [--db PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
PARQUET_OUT = REPO_ROOT / "data" / "processed" / "hourly_grid.parquet"

EXPECTED_ROWS = 12_004  # notebooks/01_capstone_eda.ipynb section 5
EXPECTED_DEMO_STAYS = 140  # the cohort that expectation was measured on


def parquet_path_for(db: Path) -> Path:
    """Where this warehouse's grid copy goes.

    Derived from the database rather than fixed. More than one warehouse can now
    exist (``build_duckdb.py --cohort-subjects``), and a single hardcoded path
    meant building a cohort grid silently replaced the demo's parquet -- which
    `ml/features/engineer.py` and the GRU read, so the next training run would
    have used a different cohort's data without anything looking wrong. The
    default database keeps the historical filename.
    """
    if db.resolve() == DEFAULT_DB_PATH.resolve():
        return PARQUET_OUT
    return PARQUET_OUT.with_name(f"hourly_grid_{db.stem}.parquet")


# Canonical vital-sign itemids (MIMIC-IV metavision) -- identical to the EDA.
GRID_ITEMS = {
    220045: "hr",
    220210: "rr",
    220277: "spo2",
    220179: "sbp_ni",
    220181: "map_ni",
    220050: "sbp_art",
    220052: "map_art",
    223761: "temp_f",
    223762: "temp_c",
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
    223835: "fio2",
    225664: "glucose",
}

# The features the deterioration models and NEWS2 both consume (E3, R2).
CORE = ["hr", "rr", "spo2", "sbp", "map", "temp_c", "gcs_total", "fio2", "glucose"]


def build_raw_grid_sql() -> str:
    """The pivot, as SQL.

    This used to be ``fetchdf()`` of every matching ``chartevents`` row followed
    by a pandas ``pivot_table``. That is fine for the demo's 668,862-row
    ``chartevents`` and untenable for anything larger: the intermediate frame is
    one row per charted value, so a cohort a hundred times the demo's size pulls
    tens of millions of rows through pandas purely to collapse them again.
    Aggregating in DuckDB means pandas only ever sees the *output* grid, one row
    per (stay_id, hour).

    Three details preserve the pandas behaviour exactly, because
    ``EXPECTED_ROWS`` below is a regression test on this function and it has to
    keep passing:

    * ``aggfunc="mean"`` over ``valuenum`` becomes ``AVG``, which likewise
      ignores NULLs rather than propagating them.
    * ``pivot_table`` drops an (index, column) cell whose values are all NaN, so
      a (stay_id, hour) with nothing but NULL ``valuenum`` produced no row at
      all. ``GROUP BY`` would emit an all-NULL row instead, so
      ``HAVING count(valuenum) > 0`` restores the original behaviour.
    * ``events[events.hour >= 0]`` filtered before the pivot, so the hour
      expression is computed in a CTE and filtered there, not after grouping.
    """
    itemids = ", ".join(str(i) for i in GRID_ITEMS)
    pivots = ",\n               ".join(
        f"AVG(CASE WHEN itemid = {itemid} THEN valuenum END) AS {name}"
        for itemid, name in GRID_ITEMS.items()
    )
    return f"""
        WITH ev AS (
            SELECT ce.stay_id, ce.itemid, ce.valuenum,
                   CAST(FLOOR(DATE_DIFF('second', ie.intime, ce.charttime) / 3600.0)
                        AS BIGINT) AS hour
            FROM mimiciv_icu.chartevents ce
            JOIN mimiciv_icu.icustays ie ON ce.stay_id = ie.stay_id
            WHERE ce.itemid IN ({itemids})
        ),
        pivoted AS (
            SELECT stay_id, hour,
               {pivots}
            FROM ev
            WHERE hour >= 0
            GROUP BY stay_id, hour
            HAVING count(valuenum) > 0
        )
        -- Column set and order match what the pandas pivot produced, exactly:
        -- the raw per-itemid columns in alphabetical order, then the three
        -- derived ones appended. `engineer.load_hourly_grid_raw` reads this
        -- table with SELECT *, so the schema is part of the contract.
        SELECT stay_id, hour,
               fio2, gcs_eye, gcs_motor, gcs_verbal, glucose, hr,
               map_art, map_ni, rr, sbp_art, sbp_ni, spo2,
               -- Unify the two temperature scales, in place (temp_c keeps its
               -- alphabetical slot, as `grid["temp_c"] = ...fillna(...)` did).
               COALESCE(temp_c, (temp_f - 32) * 5.0 / 9.0) AS temp_c,
               temp_f,
               CASE WHEN gcs_eye IS NOT NULL
                     AND gcs_verbal IS NOT NULL
                     AND gcs_motor IS NOT NULL
                    THEN gcs_eye + gcs_verbal + gcs_motor END AS gcs_total,
               -- Prefer arterial pressure where a line exists, else non-invasive
               -- (E3: arterial-line presence is itself a signal).
               COALESCE(sbp_art, sbp_ni) AS sbp,
               COALESCE(map_art, map_ni) AS map
        FROM pivoted
        ORDER BY stay_id, hour
    """


def build_raw_grid(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    grid = conn.execute(build_raw_grid_sql()).fetchdf()
    for c in CORE:
        if c not in grid.columns:
            grid[c] = pd.NA
    return grid


def add_imputation_features(grid: pd.DataFrame) -> pd.DataFrame:
    """R2: every carried-forward value is flagged, and every channel carries a
    recency counter. Modifies CORE columns in place to their forward-filled value;
    the two engineered columns preserve what was original vs. imputed.
    """
    grid = grid.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    for c in CORE:
        observed_hour = grid["hour"].where(grid[c].notna())
        last_obs_hour = observed_hour.groupby(grid["stay_id"]).ffill()
        grid[f"{c}_hours_since_last_obs"] = grid["hour"] - last_obs_hour

        ffilled = grid.groupby("stay_id")[c].ffill()
        grid[f"{c}_was_imputed"] = grid[c].isna() & ffilled.notna()
        grid[c] = ffilled
    return grid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db))
    raw_grid = build_raw_grid(conn)

    n_stays = raw_grid.stay_id.nunique()
    print(f"Raw hourly grid: {len(raw_grid):,} patient-hours across {n_stays} ICU stays")
    # The 12,004-row reproduction is the regression test for this port, and it
    # is only meaningful against the demo cohort it was measured on. A larger or
    # cohort-sampled warehouse has no published expectation to check, so assert
    # nothing there rather than emit a warning that is guaranteed to fire.
    if n_stays == EXPECTED_DEMO_STAYS:
        if len(raw_grid) != EXPECTED_ROWS:
            print(
                f"WARNING: expected {EXPECTED_ROWS:,} rows (EDA section 5) "
                f"-- got {len(raw_grid):,}"
            )
        else:
            print(f"   reproduces the EDA's {EXPECTED_ROWS:,} patient-hours exactly")
    else:
        print(f"   (not the {EXPECTED_DEMO_STAYS}-stay demo cohort -- no row-count expectation)")

    filled = add_imputation_features(raw_grid)

    conn.execute("CREATE SCHEMA IF NOT EXISTS capstone")
    conn.execute("DROP TABLE IF EXISTS capstone.hourly_grid")
    conn.register("filled_df", filled)
    conn.execute("CREATE TABLE capstone.hourly_grid AS SELECT * FROM filled_df")
    conn.unregister("filled_df")
    conn.close()

    parquet_out = parquet_path_for(args.db)
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    filled.to_parquet(parquet_out, index=False)

    raw_completeness = raw_grid[CORE].notna().mean() * 100
    filled_completeness = filled[CORE].notna().mean() * 100
    print("\nCompleteness before / after carry-forward (%):")
    for c in CORE:
        print(f"  {c:<10} {raw_completeness[c]:>5.1f} -> {filled_completeness[c]:>5.1f}")

    print(f"\nWrote capstone.hourly_grid ({len(filled):,} rows) to {args.db}")
    print(f"Wrote {parquet_out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
