"""NEWS2 (RCP 2017, Scale 1), computed on the hourly grid, then recalibrated.

Ported from notebooks/01_capstone_eda.ipynb section 7 (news2_row / cells 38-40).
PROJECT_PLAN.md section 7, item 6: after reproducing the ward-standard score, E5
requires recalibrating the *escalation thresholds* to this ICU population before it
can be used as an alert trigger -- 79% of ICU stays trip the ward "medium" threshold
at some point, which is not clinically discriminating within a population that is, by
definition, already critically ill.

NEWS2's SECOND escalation trigger (added after review finding F1). RCP 2017 defines
two independent triggers, not one: the aggregate score *and* "a score of 3 in any
single parameter", which mandates urgent review regardless of the total. An earlier
version of this module computed the component subscores and then discarded them,
tiering on the aggregate alone -- a silent deviation from the standard this module
claims to implement. A real case: stay 34617352 hour 35 (GCS 3, the deepest possible
coma, with SOFA-24h 12) scored NEWS2 8, which is 'medium' on the ICU-recalibrated
cut-point, and did not escalate. That patient died.

The strict rule cannot be applied as written in an ICU, though: GCS 3 is routine in
sedated patients, so scoring it red fires on 71.5% of all patient-hours. Variants were
measured against all 78 composite deterioration events (coverage / share of
patient-hours alerted / efficiency = coverage per unit alert burden):

    aggregate only (previous behaviour)        15.4%  / 12.8%  -> 1.20
    + red non-GCS parameter                    41.0%  / 31.8%  -> 1.29
    + red non-GCS + GCS DROP off sedation      41.0%  / 32.9%  -> 1.25   ADOPTED
    + red non-GCS + red GCS level not sedated  47.4%  / 51.1%  -> 0.93
    strict RCP, all parameters                 55.1%  / 71.5%  -> 0.77
    (ward-standard aggregate, reference)       39.7%  / 48.8%  -> 0.81

Every variant here beats the ward-standard reference. The adopted rule is not the
most efficient by a hair -- dropping the GCS limb scores 1.29 against its 1.25 -- and
that trade was made deliberately, because *the efficient variant does not catch the
patient who exposed the bug*. Stay 34617352's only red parameter was GCS, so a rule
that simply ignores GCS leaves that death exactly as unflagged as the original bug
did. What distinguishes that patient from a sedated one is the trajectory: GCS 7 for
six hours, then 3, with no sedative running. Escalating on a GCS *drop* rather than a
GCS *level* costs about one extra percentage point of alert burden, catches that case
and every case like it, and is the clinically defensible answer -- "we ignore GCS" is
not something to tell a clinician when a falling GCS is the textbook deterioration
sign. Coverage of the 78-event metric is unchanged because those stays were already
covered by another limb; the metric counts first composite events, and neurological
deterioration is not one of its components.

`max_component` (all seven parameters, GCS included) is stored regardless, so the
level-vs-trajectory decision stays reviewable in the data rather than only in prose.

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
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
REPORT_FILE = REPO_ROOT / "warehouse" / "news2_report.md"


def report_path_for(db: Path) -> Path:
    """Where this warehouse's NEWS2 report goes.

    Derived from the database for the same reason as
    ``run_concepts.status_path_for`` and ``hourly_grid.parquet_path_for``: with
    ``build_duckdb.py --cohort-subjects`` there can be several warehouses, and a
    fixed path means the last build silently overwrites the committed demo
    report with another cohort's numbers.
    """
    if db.resolve() == DEFAULT_DB_PATH.resolve():
        return REPORT_FILE
    return REPORT_FILE.with_name(f"news2_report_{db.stem}.md")


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

# RCP 2017's single-parameter trigger: any one component scoring this much mandates
# urgent review on its own. GCS is excluded from the *escalation* limb (see the module
# docstring for the measurement that decided this) but still counts toward
# `max_component`, which is reported for review.
RED_COMPONENT_SCORE = 3
ESCALATION_COMPONENTS = ("rr", "spo2", "sbp", "hr", "temp_c", "fio2")

# GCS enters escalation as a *change*, not a level. A patient sedated at GCS 3 for
# days is not deteriorating; a patient whose GCS falls 7 -> 3 in an hour is, and that
# is precisely the case that exposed F1 (stay 34617352, hour 35: not sedated, GCS 7
# for six hours then 3, dead within two days). A drop of this many points below the
# preceding window's best, while no sedative is running, escalates.
GCS_DROP_POINTS = 2
GCS_DROP_LOOKBACK_H = 4

# inputevents itemids for continuous/bolus sedation and analgesia. A GCS drop while
# any of these is running is attributed to the drug, not to the brain.
SEDATION_ITEMIDS = (
    222168,  # Propofol
    227210,  # Propofol (Intubation)
    226224,  # Propofol Ingredient
    221668,  # Midazolam (Versed)
    229420,  # Dexmedetomidine (Precedex)
    225150,  # Dexmedetomidine (Precedex)
    221744,  # Fentanyl
    225942,  # Fentanyl (Concentrate)
    225972,  # Fentanyl (Push)
)
ICU_PERCENTILE_MEDIUM, ICU_PERCENTILE_HIGH = 0.75, 0.90


# Per-component NEWS2 (RCP 2017, Scale 1) subscores, extracted as standalone
# functions so other code (simulators/morphing.py, which needs to target a specific
# HR/SpO2 subscore when it morphs a wearable segment) can reuse the exact same
# thresholds instead of re-transcribing them. news2_row below is a thin sum over
# these -- verbatim port of the EDA's news2_row, see notebooks/01_capstone_eda.ipynb
# cell 38.
def rr_score(rr: float) -> int:
    return 3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20 else 2 if rr <= 24 else 3


def spo2_score(spo2: float) -> int:
    return 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0


def sbp_score(sbp: float) -> int:
    return 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0 if sbp <= 219 else 3


def hr_score(hr: float) -> int:
    return (
        3
        if hr <= 40
        else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2 if hr <= 130 else 3
    )


def temp_score(temp_c: float) -> int:
    return (
        3
        if temp_c <= 35
        else 1 if temp_c <= 36 else 0 if temp_c <= 38 else 1 if temp_c <= 39 else 2
    )


def gcs_score(gcs_total: float) -> int:
    return 0 if gcs_total >= 15 else 3  # NEWS2 scores any non-alert state as 3


def fio2_score(fio2: float) -> int:
    return 2 if fio2 > 21 else 0  # supplemental oxygen


def news2_row(r: pd.Series) -> pd.Series:
    """NEWS2 (RCP 2017), Scale 1. Returns (score, n_components_available)."""
    s, n = 0, 0
    if pd.notna(r.rr):
        n += 1
        s += rr_score(r.rr)
    if pd.notna(r.spo2):
        n += 1
        s += spo2_score(r.spo2)
    if pd.notna(r.sbp):
        n += 1
        s += sbp_score(r.sbp)
    if pd.notna(r.hr):
        n += 1
        s += hr_score(r.hr)
    if pd.notna(r.temp_c):
        n += 1
        s += temp_score(r.temp_c)
    if pd.notna(r.gcs_total):
        n += 1
        s += gcs_score(r.gcs_total)
    if pd.notna(r.fio2):
        n += 1
        s += fio2_score(r.fio2)
    return pd.Series({"news2": s, "components": n})


def component_scores(r: pd.Series) -> dict[str, int]:
    """Every available NEWS2 component subscore for one patient-hour, keyed by the
    grid's own column name. news2_row sums these; the single-parameter rule needs
    them individually, which is exactly what the pre-F1 code threw away.
    """
    scorers = {
        "rr": rr_score,
        "spo2": spo2_score,
        "sbp": sbp_score,
        "hr": hr_score,
        "temp_c": temp_score,
        "gcs_total": gcs_score,
        "fio2": fio2_score,
    }
    return {name: f(r[name]) for name, f in scorers.items() if pd.notna(r[name])}


def red_flags(r: pd.Series) -> pd.Series:
    """The single-parameter limb of NEWS2, as three reviewable columns.

    ``max_component``          -- highest subscore across all seven parameters.
    ``max_component_nongcs``   -- highest across ESCALATION_COMPONENTS only; this is
                                  the one the escalation predicate reads.
    ``red_params``             -- comma-joined names of every parameter at or above
                                  RED_COMPONENT_SCORE, GCS included, so an alert can
                                  say *which* parameter tripped it and a reviewer can
                                  see the GCS-only cases that deliberately do not fire.
    """
    scores = component_scores(r)
    red = [k for k, v in scores.items() if v >= RED_COMPONENT_SCORE]
    non_gcs = [v for k, v in scores.items() if k in ESCALATION_COMPONENTS]
    return pd.Series(
        {
            "max_component": max(scores.values()) if scores else 0,
            "max_component_nongcs": max(non_gcs) if non_gcs else 0,
            "red_params": ",".join(sorted(red)),
        }
    )


def should_escalate(
    tier_icu: str | None,
    max_component_nongcs: int | None,
    gcs_drop: bool | None = False,
) -> bool:
    """The escalation predicate, defined once and imported by every consumer
    (agent-orchestrator's EscalationDecider, eval/alerting.py's replay,
    eval/rag_agent.py's agreement metric, and risk-engine's response). Three limbs,
    OR'd:

      1. the ICU-recalibrated aggregate tier reaches 'high'      (E5)
      2. any single non-GCS parameter scores 3                   (RCP 2017, finding F1)
      3. GCS falls >= GCS_DROP_POINTS with no sedative running    (finding F1)

    Limb 3 exists because limb 2 alone did not catch the case that exposed the bug:
    that patient's only red parameter was GCS, and a level-based GCS trigger is
    unusable in an ICU (71.5% of patient-hours). A *falling* GCS off sedation is both
    specific and clinically the textbook deterioration signal.
    """
    return (
        tier_icu == "high" or (max_component_nongcs or 0) >= RED_COMPONENT_SCORE or bool(gcs_drop)
    )


def escalation_reason(
    tier_icu: str | None,
    max_component_nongcs: int | None,
    red_params: str | None,
    gcs_drop: bool | None = False,
) -> str:
    """Which limb fired, in words, for the alert payload and the audit log."""
    limbs = []
    if tier_icu == "high":
        limbs.append("ICU-recalibrated NEWS2 tier is 'high'")
    if (max_component_nongcs or 0) >= RED_COMPONENT_SCORE:
        named = [p for p in (red_params or "").split(",") if p and p != "gcs_total"]
        limbs.append(
            f"single-parameter red flag (RCP 2017): {', '.join(named)} "
            f"scoring {max_component_nongcs}"
        )
    if gcs_drop:
        limbs.append(
            f"GCS fell >= {GCS_DROP_POINTS} points within {GCS_DROP_LOOKBACK_H}h "
            f"with no sedative running"
        )
    if not limbs:
        gcs_red = "gcs_total" in (red_params or "")
        base = f"ICU-recalibrated NEWS2 tier is '{tier_icu}', below the high threshold"
        return (
            base + "; GCS is red but stable and/or sedated, so it does not escalate on level alone"
            if gcs_red
            else base
        )
    return " AND ".join(limbs)


@dataclass(frozen=True)
class Thresholds:
    """The aggregate cut-points that turn a NEWS2 score into a tier.

    These are derived from this cohort's own score distribution at build time (E5),
    which means anything scoring a *live* observation has to read them rather than
    recompute them -- otherwise the streaming path and the warehouse path silently
    disagree about what "high" means. Before F3 they existed only as two local
    variables in main() and two numbers in news2_report.md, so live scoring was not
    expressible at all.
    """

    ward_medium: int
    ward_high: int
    icu_medium: int
    icu_high: int


def load_thresholds(conn: duckdb.DuckDBPyConnection) -> Thresholds:
    row = conn.execute(
        "SELECT ward_medium, ward_high, icu_medium, icu_high FROM capstone.news2_thresholds"
    ).fetchone()
    if row is None:
        raise RuntimeError("capstone.news2_thresholds is empty -- run warehouse/news2.py")
    return Thresholds(*(int(v) for v in row))


def tier_for_score(score: int, medium: int, high: int) -> str:
    """Scalar counterpart of ``tier()`` for scoring one live observation."""
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


def sedation_intervals(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """stay_id + start/end of every sedative or analgesic administration."""
    placeholders = ",".join(str(i) for i in SEDATION_ITEMIDS)
    df = conn.execute(
        f"SELECT stay_id, starttime, endtime FROM mimiciv_icu.inputevents "
        f"WHERE itemid IN ({placeholders})"
    ).fetchdf()
    df["starttime"] = pd.to_datetime(df.starttime)
    df["endtime"] = pd.to_datetime(df.endtime)
    return df


def add_gcs_drop(grid: pd.DataFrame, sedation: pd.DataFrame) -> pd.DataFrame:
    """Adds ``sedated`` and ``gcs_drop`` to a grid that already carries stay_id, hour,
    gcs_total and abs_time. Sorted by (stay_id, hour) on the way in, because the
    rolling lookback is only meaningful in time order.
    """
    grid = grid.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    by_stay = {s: v[["starttime", "endtime"]].to_numpy() for s, v in sedation.groupby("stay_id")}

    def _sedated(stay_id: int, t: pd.Timestamp) -> bool:
        intervals = by_stay.get(stay_id)
        if intervals is None:
            return False
        start = np.datetime64(t)
        end = start + np.timedelta64(1, "h")
        return bool(((intervals[:, 0] < end) & (intervals[:, 1] > start)).any())

    grid["sedated"] = [_sedated(s, t) for s, t in zip(grid.stay_id, grid.abs_time, strict=True)]
    prev_best = grid.groupby("stay_id").gcs_total.transform(
        lambda s: s.shift(1).rolling(GCS_DROP_LOOKBACK_H, min_periods=1).max()
    )
    grid["gcs_drop"] = (
        (prev_best - grid.gcs_total >= GCS_DROP_POINTS) & (~grid.sedated) & grid.gcs_total.notna()
    )
    return grid


def tier(score: pd.Series, medium: float, high: float) -> pd.Series:
    return pd.cut(score, bins=[-1, medium - 1, high - 1, 999], labels=["low", "medium", "high"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db))
    grid = conn.execute(
        "SELECT g.stay_id, g.hour, g.hr, g.rr, g.spo2, g.sbp, g.temp_c, g.gcs_total, g.fio2, "
        "d.icu_intime "
        "FROM capstone.hourly_grid g "
        "JOIN mimiciv_derived.icustay_detail d USING (stay_id)"
    ).fetchdf()
    grid["abs_time"] = pd.to_datetime(grid.icu_intime) + pd.to_timedelta(grid.hour, unit="h")

    grid[["news2", "components"]] = grid.apply(news2_row, axis=1)
    grid[["max_component", "max_component_nongcs", "red_params"]] = grid.apply(red_flags, axis=1)
    grid["max_component"] = grid["max_component"].astype(int)
    grid["max_component_nongcs"] = grid["max_component_nongcs"].astype(int)
    grid = add_gcs_drop(grid, sedation_intervals(conn))
    grid = grid.drop(columns=["icu_intime", "abs_time"])
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
        hourly: pd.Series = (grid[tier_col].value_counts(normalize=True) * 100).reindex(
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

    # Finding F1: the two-limb escalation predicate, and what each limb contributes.
    grid["escalates"] = [
        should_escalate(t, m, d)
        for t, m, d in zip(grid.tier_icu, grid.max_component_nongcs, grid.gcs_drop, strict=True)
    ]
    aggregate_limb = int((grid.tier_icu == "high").sum())
    red_limb = int((grid.max_component_nongcs >= RED_COMPONENT_SCORE).sum())
    drop_limb = int(grid.gcs_drop.sum())
    both = int(grid.escalates.sum())
    gcs_level_only = int(((grid.max_component >= RED_COMPONENT_SCORE) & (~grid.escalates)).sum())
    print(
        f"Escalating hours: {both:,} ({both / len(grid) * 100:.1f}%) "
        f"[aggregate {aggregate_limb:,}, red-parameter {red_limb:,}, GCS-drop {drop_limb:,}]"
    )
    print(f"Red GCS by level only (sedated and/or stable): {gcs_level_only:,} hours")

    conn.execute("CREATE SCHEMA IF NOT EXISTS capstone")
    conn.execute("DROP TABLE IF EXISTS capstone.news2_thresholds")
    conn.execute(
        "CREATE TABLE capstone.news2_thresholds AS SELECT "
        f"{WARD_MEDIUM} AS ward_medium, {WARD_HIGH} AS ward_high, "
        f"{icu_medium} AS icu_medium, {icu_high} AS icu_high"
    )
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

## Single-parameter escalation (finding F1)

NEWS2 (RCP 2017) has two independent escalation triggers, not one. Alongside the
aggregate tier above, **a score of {RED_COMPONENT_SCORE} in any single parameter**
mandates urgent review on its own. `capstone.news2` now stores `max_component`,
`max_component_nongcs` and `red_params` so both limbs are computable, and
`should_escalate()` is the single definition every consumer imports.

| Limb | Patient-hours |
|---|---|
| ICU-recalibrated aggregate tier == high | {aggregate_limb:,} |
| Any non-GCS parameter scoring {RED_COMPONENT_SCORE} | {red_limb:,} |
| GCS falling >= {GCS_DROP_POINTS} points off sedation | {drop_limb:,} |
| **Any of the three (the escalation predicate)** | **{both:,} ({100 * both / len(grid):.1f}%)** |

GCS enters as a *change*, not a level. A red GCS level alone accounts for
{gcs_level_only:,} further patient-hours -- overwhelmingly sedated patients -- and
escalating on it fires on 71.5% of the cohort. A GCS *drop* off sedation is specific
enough to cost roughly one extra percentage point of alert burden. See the module
docstring for the measurement of every variant against the 78 composite events.
"""
    report_file = report_path_for(args.db)
    report_file.write_text(report)
    print(f"\nWrote capstone.news2 ({len(grid):,} rows) to {args.db}")
    print(f"Wrote {report_file.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
