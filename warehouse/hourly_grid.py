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


def load_raw_events(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    itemids = ", ".join(str(i) for i in GRID_ITEMS)
    return conn.execute(
        f"""
        SELECT ce.stay_id, ce.itemid, ce.valuenum,
               CAST(FLOOR(DATE_DIFF('second', ie.intime, ce.charttime) / 3600.0) AS BIGINT) AS hour
        FROM mimiciv_icu.chartevents ce
        JOIN mimiciv_icu.icustays ie ON ce.stay_id = ie.stay_id
        WHERE ce.itemid IN ({itemids})
        """
    ).fetchdf()


def build_raw_grid(events: pd.DataFrame) -> pd.DataFrame:
    events = events[events.hour >= 0].copy()
    events["var"] = events.itemid.map(GRID_ITEMS)

    grid = events.pivot_table(
        index=["stay_id", "hour"], columns="var", values="valuenum", aggfunc="mean"
    ).reset_index()

    # Unify the two temperature scales and the three GCS components.
    grid["temp_c"] = grid.get("temp_c", pd.Series(dtype=float)).fillna(
        (grid.get("temp_f", pd.Series(dtype=float)) - 32) * 5 / 9
    )
    gcs_cols = [c for c in ("gcs_eye", "gcs_verbal", "gcs_motor") if c in grid]
    grid["gcs_total"] = grid[gcs_cols].sum(axis=1, min_count=3) if gcs_cols else pd.NA
    # Prefer arterial pressure where a line exists, else non-invasive
    # (E3: arterial-line presence is itself a signal).
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
    events = load_raw_events(conn)
    raw_grid = build_raw_grid(events)

    n_stays = raw_grid.stay_id.nunique()
    print(f"Raw hourly grid: {len(raw_grid):,} patient-hours across {n_stays} ICU stays")
    if len(raw_grid) != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS:,} rows (EDA section 5) -- got {len(raw_grid):,}")

    filled = add_imputation_features(raw_grid)

    conn.execute("CREATE SCHEMA IF NOT EXISTS capstone")
    conn.execute("DROP TABLE IF EXISTS capstone.hourly_grid")
    conn.register("filled_df", filled)
    conn.execute("CREATE TABLE capstone.hourly_grid AS SELECT * FROM filled_df")
    conn.unregister("filled_df")
    conn.close()

    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    filled.to_parquet(PARQUET_OUT, index=False)

    raw_completeness = raw_grid[CORE].notna().mean() * 100
    filled_completeness = filled[CORE].notna().mean() * 100
    print("\nCompleteness before / after carry-forward (%):")
    for c in CORE:
        print(f"  {c:<10} {raw_completeness[c]:>5.1f} -> {filled_completeness[c]:>5.1f}")

    print(f"\nWrote capstone.hourly_grid ({len(filled):,} rows) to {args.db}")
    print(f"Wrote {PARQUET_OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
