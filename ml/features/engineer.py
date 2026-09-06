"""Feature matrix for the composite deterioration model (PROJECT_PLAN.md section 11).

Every feature group here traces to a specific design rule from section 4, not
just "things that seemed predictive":

* **Carried-forward vitals + imputation flags + recency** (R2, E3) come
  straight from ``capstone.hourly_grid`` -- already built in Phase 1.
* **Rolling trend over the trailing 4h and 24h** for each core vital (mean,
  std, OLS slope) reuses ``services/stream-processor/windowing.py``'s
  ``rolling_stats``/``trend_slope`` -- the same functions the live streaming
  path uses, loaded through ``services/common/testing.py`` because that
  module's directory is hyphenated and not import-able by dotted path.
* **Severity scores as features, not just baselines to beat** -- NEWS2
  (``capstone.news2``) and the rolling 24h SOFA total
  (``mimiciv_derived.sofa.sofa_24hours``) are joined in as inputs. This is
  standard "augmented early-warning score" practice, and legitimate because
  the baselines each model must still separately beat are evaluated as
  stand-alone scores, not with these engineered features attached.
* **Arterial-line presence** (R3, E3 -- "Arterial-line presence (46% of
  stays)... Ordering behaviour is a feature") -- an hour-resolved boolean from
  ``mimiciv_derived.invasive_line``, not inferred indirectly from which blood
  pressure channel happened to be charted.
* **Lab-ordering intensity, normalised to the admission's own baseline rate**
  (R3, R4, E15 -- "measurement intensity tracks outcome but is confounded...
  event-rate features are legitimate but must be normalised by care
  setting") -- trailing lab-order counts divided by that admission's own
  mean rate, so the feature reads as "busier than usual for this patient
  right now," not "in the ICU" (which every row already is).
* **Static demographics** -- age, gender, first ICU care unit -- from
  ``mimiciv_derived.icustay_detail`` / ``mimiciv_icu.icustays``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from services.common.testing import load_module

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_windowing = load_module(
    REPO_ROOT / "services" / "stream-processor" / "windowing.py",
    "ml_features_engineer_windowing",
)
trend_slope = _windowing.trend_slope

CORE_VITALS = ["hr", "rr", "spo2", "sbp", "map", "temp_c", "gcs_total", "fio2", "glucose"]
ROLLING_WINDOWS_H = (4, 24)


def load_hourly_grid_raw(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute("select * from capstone.hourly_grid order by stay_id, hour").fetchdf()


def add_rolling_features(
    grid: pd.DataFrame, windows: tuple[int, ...] = ROLLING_WINDOWS_H
) -> pd.DataFrame:
    """Trailing (causal, current hour included) mean/std/slope per vital per
    window, computed per stay so no window ever crosses a stay boundary.
    """
    grid = grid.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    out = grid.copy()
    for w in windows:
        grouped = grid.groupby("stay_id")
        for var in CORE_VITALS:
            roll = grouped[var].rolling(window=w, min_periods=1)
            out[f"{var}_{w}h_mean"] = roll.mean().reset_index(level=0, drop=True)
            out[f"{var}_{w}h_std"] = roll.std(ddof=0).fillna(0.0).reset_index(level=0, drop=True)

        # Slope needs (time, value) pairs, not a single series -- compute per
        # stay with the shared trend_slope() from the streaming path.
        for var in CORE_VITALS:
            out[f"{var}_{w}h_slope"] = 0.0
        for _stay_id, g in grid.groupby("stay_id"):
            hours = g["hour"].to_numpy(dtype=float)
            for var in CORE_VITALS:
                values = g[var].to_numpy(dtype=float)
                slopes = np.zeros(len(g))
                for i in range(len(g)):
                    lo = max(0, i - w + 1)
                    window_hours = hours[lo : i + 1].tolist()
                    window_values = values[lo : i + 1].tolist()
                    slopes[i] = trend_slope(window_hours, window_values)
                out.loc[g.index, f"{var}_{w}h_slope"] = slopes
    return out


def add_severity_scores(grid: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    news2 = conn.execute(
        "select stay_id, hour, news2, tier_ward, tier_icu from capstone.news2"
    ).fetchdf()
    sofa = conn.execute(
        "select stay_id, hr as hour, sofa_24hours from mimiciv_derived.sofa"
    ).fetchdf()
    out = grid.merge(news2, on=["stay_id", "hour"], how="left")
    out = out.merge(sofa, on=["stay_id", "hour"], how="left")
    out["news2"] = out["news2"].fillna(0)
    out["sofa_24hours"] = out["sofa_24hours"].fillna(0)
    for tier_col in ("tier_ward", "tier_icu"):
        out[tier_col] = out[tier_col].astype("object").fillna("low")
    return out


def add_arterial_line_feature(grid: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    lines = conn.execute(
        """
        select il.stay_id, il.starttime, il.endtime, d.icu_intime
        from mimiciv_derived.invasive_line il
        join mimiciv_derived.icustay_detail d using (stay_id)
        where il.line_type = 'Arterial'
        """
    ).fetchdf()
    out = grid.copy()
    out["has_arterial_line"] = 0
    if lines.empty:
        return out
    lines["start_h"] = (lines.starttime - lines.icu_intime).dt.total_seconds() / 3600.0
    lines["end_h"] = (lines.endtime - lines.icu_intime).dt.total_seconds() / 3600.0
    for stay_id, stay_lines in lines.groupby("stay_id"):
        mask = out.stay_id == stay_id
        row_hours = out.loc[mask, "hour"]
        active = pd.Series(False, index=row_hours.index)
        for _, line in stay_lines.iterrows():
            active |= (row_hours >= line.start_h) & (row_hours <= line.end_h)
        out.loc[mask, "has_arterial_line"] = active.astype(int)
    return out


def add_lab_order_intensity(grid: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """R3/R4/R5: lab-ordering rate in the trailing 4h/24h, normalised by this
    admission's own mean rate across its ICU stay so far -- "busier than
    usual for this patient," not "in the ICU."
    """
    labs = conn.execute(
        """
        select l.hadm_id, l.charttime, d.stay_id, d.icu_intime
        from mimiciv_hosp.labevents l
        join mimiciv_derived.icustay_detail d using (hadm_id)
        where l.hadm_id is not null
        """
    ).fetchdf()
    out = grid.copy()
    for w in ROLLING_WINDOWS_H:
        out[f"lab_orders_{w}h"] = 0
    out["lab_order_rate_ratio"] = 1.0
    if labs.empty:
        return out

    labs["lab_hour"] = (labs.charttime - labs.icu_intime).dt.total_seconds() / 3600.0
    for stay_id, stay_labs in labs.groupby("stay_id"):
        mask = out.stay_id == stay_id
        if not mask.any():
            continue
        row_hours = out.loc[mask, "hour"].to_numpy(dtype=float)
        lab_hours = stay_labs["lab_hour"].to_numpy(dtype=float)
        counts_by_window = {}
        for w in ROLLING_WINDOWS_H:
            counts = np.array([np.sum((lab_hours <= h) & (lab_hours > h - w)) for h in row_hours])
            counts_by_window[w] = counts
            out.loc[mask, f"lab_orders_{w}h"] = counts
        stay_duration_h = max(row_hours.max() - row_hours.min() + 1, 1.0)
        admission_mean_rate = len(stay_labs) / stay_duration_h
        current_rate_24h = counts_by_window[24] / 24.0
        ratio = np.where(admission_mean_rate > 0, current_rate_24h / admission_mean_rate, 1.0)
        out.loc[mask, "lab_order_rate_ratio"] = ratio
    return out


def add_static_features(grid: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    static = conn.execute(
        """
        select d.stay_id, d.admission_age, d.gender, i.first_careunit
        from mimiciv_derived.icustay_detail d
        join mimiciv_icu.icustays i using (stay_id)
        """
    ).fetchdf()
    return grid.merge(static, on="stay_id", how="left")


FEATURE_COLUMNS_BASE = [
    *CORE_VITALS,
    *[f"{v}_was_imputed" for v in CORE_VITALS],
    *[f"{v}_hours_since_last_obs" for v in CORE_VITALS],
    "news2",
    "sofa_24hours",
    "has_arterial_line",
    "lab_orders_4h",
    "lab_orders_24h",
    "lab_order_rate_ratio",
    "admission_age",
]


def rolling_feature_columns(windows: tuple[int, ...] = ROLLING_WINDOWS_H) -> list[str]:
    cols = []
    for w in windows:
        for var in CORE_VITALS:
            cols += [f"{var}_{w}h_mean", f"{var}_{w}h_std", f"{var}_{w}h_slope"]
    return cols


def build_feature_frame(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per (stay_id, hour) with the full feature set attached. Does
    NOT apply label censoring -- join against ``labels.build_labels()``'s
    output on (stay_id, hour) to get the at-risk, labelled subset.
    """
    grid = load_hourly_grid_raw(conn)
    grid = add_rolling_features(grid)
    grid = add_severity_scores(grid, conn)
    grid = add_arterial_line_feature(grid, conn)
    grid = add_lab_order_intensity(grid, conn)
    grid = add_static_features(grid, conn)
    return grid


def feature_matrix_for_training(
    features: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_col: str,
    include_ecg: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Inner-join the full feature frame to the at-risk labelled rows, and
    return (X, y, groups) ready for ``models.splits``. ``groups`` is
    ``stay_id`` -- grouped CV must never let one patient span train and test.
    """
    merged = labels_df[["stay_id", "hour", label_col]].merge(
        features, on=["stay_id", "hour"], how="inner"
    )
    if include_ecg is not None:
        merged = merged.merge(include_ecg, on=["stay_id", "hour"], how="left")

    cols = list(FEATURE_COLUMNS_BASE) + rolling_feature_columns()
    if include_ecg is not None:
        cols += [c for c in include_ecg.columns if c not in ("stay_id", "hour")]

    x = merged[cols].copy()
    x["gender"] = merged["gender"]
    x["first_careunit"] = merged["first_careunit"]
    y = merged[label_col]
    groups = merged["stay_id"]
    return x, y, groups
