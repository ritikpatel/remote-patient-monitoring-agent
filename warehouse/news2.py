"""NEWS2 (RCP 2017, Scale 1), computed on the hourly grid, then recalibrated.

Ported from notebooks/01_capstone_eda.ipynb section 7 (news2_row / cells 38-40).
PROJECT_PLAN.md section 7, item 6: after reproducing the ward-standard score, E5
requires recalibrating the *escalation thresholds* to this ICU population before it
can be used as an alert trigger -- 79% of ICU stays trip the ward "medium" threshold
at some point, which is not clinically discriminating within a population that is, by
definition, already critically ill.

What is NOT recalibrated: the component scoring itself (respiratory rate, SpO2, SBP,
heart rate, temperature, consciousness, supplemental O2) is the validated NEWS2
formula and is left untouched -- ward-derived component thresholds are a clinical
scoring instrument, not the thing E5 flags as wrong. What IS recalibrated is the
*aggregate cut-point* that turns a score into an escalation tier: "medium" and "high"
are redefined as the 75th and 90th percentile of this cohort's own patient-hour score
distribution, rather than the ward-standard 5 and 7. This is a cohort-relative
recalibration, not a clinically validated one -- flagged as such in every output,
per PROJECT_PLAN.md section 17.

Output: table `capstone.news2` in the warehouse db (one row per patient-hour, with
both the ward-standard and ICU-recalibrated tier), plus warehouse/news2_report.md
comparing alert rates under both threshold sets.

Usage:
    python warehouse/news2.py [--db PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
REPORT_FILE = REPO_ROOT / "warehouse" / "news2_report.md"

# EDA section 7 regression targets, read off the executed figure
# (eda_figures/08_news2.png, right panel) rather than PROJECT_PLAN.md's E4 prose
# ("110/140 reach >=5; 68 reach >=7"): the executed notebook is the ground truth
# per its own stated methodology ("every number quoted in this plan traces to a
# cell in that notebook"), and the actual rendered figure reads 128 and 105. E4's
# figures appear to predate the final executed run and were not updated in the
# planning doc -- flagged here rather than silently reconciled.
EXPECTED_STAYS_GE5 = 128
EXPECTED_STAYS_GE7 = 105
N_STAYS = 140

WARD_MEDIUM, WARD_HIGH = 5, 7
ICU_PERCENTILE_MEDIUM, ICU_PERCENTILE_HIGH = 0.75, 0.90


def news2_row(r: pd.Series) -> pd.Series:
    """NEWS2 (RCP 2017), Scale 1. Returns (score, n_components_available).
    Verbatim port of the EDA's news2_row -- see notebooks/01_capstone_eda.ipynb cell 38.
    """
    s, n = 0, 0
    if pd.notna(r.rr):
        n += 1
        s += 3 if r.rr <= 8 else 1 if r.rr <= 11 else 0 if r.rr <= 20 else 2 if r.rr <= 24 else 3
    if pd.notna(r.spo2):
        n += 1
        s += 3 if r.spo2 <= 91 else 2 if r.spo2 <= 93 else 1 if r.spo2 <= 95 else 0
    if pd.notna(r.sbp):
        n += 1
        s += (
            3
            if r.sbp <= 90
            else 2 if r.sbp <= 100 else 1 if r.sbp <= 110 else 0 if r.sbp <= 219 else 3
        )
    if pd.notna(r.hr):
        n += 1
        s += (
            3
            if r.hr <= 40
            else (
                1
                if r.hr <= 50
                else 0 if r.hr <= 90 else 1 if r.hr <= 110 else 2 if r.hr <= 130 else 3
            )
        )
    if pd.notna(r.temp_c):
        n += 1
        s += (
            3
            if r.temp_c <= 35
            else 1 if r.temp_c <= 36 else 0 if r.temp_c <= 38 else 1 if r.temp_c <= 39 else 2
        )
    if pd.notna(r.gcs_total):
        n += 1
        s += 0 if r.gcs_total >= 15 else 3  # NEWS2 scores any non-alert state as 3
    if pd.notna(r.fio2):
        n += 1
        s += 2 if r.fio2 > 21 else 0  # supplemental oxygen
    return pd.Series({"news2": s, "components": n})


def tier(score: pd.Series, medium: float, high: float) -> pd.Series:
    return pd.cut(score, bins=[-1, medium - 1, high - 1, 999], labels=["low", "medium", "high"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db))
    grid = conn.execute(
        "SELECT stay_id, hour, hr, rr, spo2, sbp, temp_c, gcs_total, fio2 FROM capstone.hourly_grid"
    ).fetchdf()

    grid[["news2", "components"]] = grid.apply(news2_row, axis=1)
    print(f"NEWS2 computed for {len(grid):,} patient-hours")
    print(f"Median components available per hour: {grid.components.median():.0f} / 7")

    # --- Ward-standard thresholds ---
    grid["tier_ward"] = tier(grid.news2, WARD_MEDIUM, WARD_HIGH)
    stay_max = grid.groupby("stay_id").news2.max()
    stays_ge5 = int((stay_max >= 5).sum())
    stays_ge7 = int((stay_max >= 7).sum())
    print(f"Stays ever reaching ward NEWS2 >= 5: {stays_ge5}/{N_STAYS}")
    print(f"Stays ever reaching ward NEWS2 >= 7: {stays_ge7}/{N_STAYS}")
    if (stays_ge5, stays_ge7) != (EXPECTED_STAYS_GE5, EXPECTED_STAYS_GE7):
        print(
            f"WARNING: expected {EXPECTED_STAYS_GE5}/{EXPECTED_STAYS_GE7} "
            f"(EDA section 7) -- got {stays_ge5}/{stays_ge7}"
        )

    # --- ICU-recalibrated thresholds (E5) ---
    icu_medium = int(grid.news2.quantile(ICU_PERCENTILE_MEDIUM))
    icu_high = int(grid.news2.quantile(ICU_PERCENTILE_HIGH))
    icu_high = max(
        icu_high, icu_medium + 1
    )  # keep tiers non-degenerate if the distribution is flat
    grid["tier_icu"] = tier(grid.news2, icu_medium, icu_high)

    TIER_RANK = {"low": 0, "medium": 1, "high": 2}

    def alert_rates(tier_col: str) -> pd.DataFrame:
        """Per tier: % of patient-hours at that tier, and how many of the 140 stays
        ever reach *at least* that tier (monotonically non-increasing down the rows --
        every stay has a low hour, so 'low' is always 140/140).
        """
        hourly = (grid[tier_col].value_counts(normalize=True) * 100).reindex(
            ["low", "medium", "high"]
        )
        stay_max_rank = grid.groupby("stay_id")[tier_col].agg(lambda s: s.map(TIER_RANK).max())
        ever_at_least = {t: int((stay_max_rank >= r).sum()) for t, r in TIER_RANK.items()}
        return pd.DataFrame(
            {
                "pct_patient_hours": hourly.round(1),
                "stays_ever_at_least_this_tier": pd.Series(ever_at_least),
            },
            index=["low", "medium", "high"],
        )

    ward_rates = alert_rates("tier_ward")
    icu_rates = alert_rates("tier_icu")

    conn.execute("CREATE SCHEMA IF NOT EXISTS capstone")
    conn.execute("DROP TABLE IF EXISTS capstone.news2")
    conn.register("news2_df", grid)
    conn.execute("CREATE TABLE capstone.news2 AS SELECT * FROM news2_df")
    conn.unregister("news2_df")
    conn.close()

    report = f"""# NEWS2 -- ward-standard vs. ICU-recalibrated

Computed on {len(grid):,} patient-hours across {N_STAYS} ICU stays (`capstone.hourly_grid`).
Component scoring is the unmodified NEWS2 (RCP 2017, Scale 1) formula; only the
aggregate escalation cut-points differ between the two columns below (see module
docstring in `warehouse/news2.py` for why).

**This recalibration is cohort-relative (75th/90th percentile of this 140-stay
sample), not a clinically validated threshold.** Per PROJECT_PLAN.md section 17:
this platform is validated on a 100-patient demo subset of MIMIC-IV; the clinical
performance figures demonstrate pipeline validity and do not transfer to clinical
practice.

## Ward-standard (medium >= {WARD_MEDIUM}, high >= {WARD_HIGH})

{ward_rates.to_markdown()}

## ICU-recalibrated (medium >= {icu_medium}, high >= {icu_high})

{icu_rates.to_markdown()}

## Why recalibrate (E5)

{stays_ge5}/{N_STAYS} ICU stays ({100 * stays_ge5 / N_STAYS:.0f}%) cross the ward
"medium" threshold at some point in the stay. A trigger this population trips this
often carries little discriminating information for an ICU-only alert engine --
everyone here is already sick enough to be in the ICU. The ICU-recalibrated cut-points
instead flag the {100 * (1 - ICU_PERCENTILE_MEDIUM):.0f}% of this cohort's own
patient-hours with the highest NEWS2, i.e. relative deterioration within an ICU
population rather than absolute deterioration relative to a ward population.
"""
    REPORT_FILE.write_text(report)
    print(f"\nWrote capstone.news2 ({len(grid):,} rows) to {args.db}")
    print(f"Wrote {REPORT_FILE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
