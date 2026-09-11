# Warehouse

One queryable DuckDB store (`mimic4_demo.db`, gitignored — rebuild it, don't commit it) plus every
standard ICU severity score. PROJECT_PLAN.md section 7.

## Build order

```bash
python warehouse/build_duckdb.py    # schema + load 31 CSVs -> mimiciv_hosp / mimiciv_icu
python warehouse/run_concepts.py    # 65 mimic-code concepts -> mimiciv_derived
python warehouse/hourly_grid.py     # capstone.hourly_grid    (12,004 patient-hours)
python warehouse/disease.py         # capstone.disease_context (140 stays, 14 dx chapters)
python warehouse/news2.py           # capstone.news2          (ward, ICU, and per-disease)
```

Each script is idempotent — re-running it drops and rebuilds only its own tables.
`disease.py` must run **before** `news2.py`: the per-disease recalibration groups on
`capstone.disease_context.dx_chapter`.

## Disease context and per-disease thresholds

`disease.py` builds one row per ICU stay carrying the primary diagnosis's ICD chapter
and the 17 Charlson chronic-comorbidity flags. The two are deliberately kept apart,
because they sit on opposite sides of a leakage line: ICD codes are assigned by billing
coders **after discharge**, while Charlson scores *pre-existing chronic* burden.
`ml/evaluation/disease_leakage.py` measures both rather than assuming either — see that
report for the result (no detectable leak, and no AUPRC gain at this cohort size).

`news2.py` then recalibrates the ICU escalation cut-points **within each diagnosis
chapter**, guarded by a minimum of 20 stays — counted as *stays*, not patient-hours,
since hours within one stay are strongly correlated. In this 140-stay demo exactly one
chapter clears that bar:

| Chapter | Stays | medium | high | vs pooled |
|---|---|---|---|---|
| `__pooled__` (fallback) | 140 | 9 | 10 | — |
| Circulatory | 41 | 7 | 9 | escalates earlier |

Every other chapter falls back to the pooled cut-point. `capstone.news2` carries both
`tier_icu` (disease-specific where earned — this is what `should_escalate` reads) and
`tier_icu_pooled` (the previous behaviour), plus `threshold_is_disease_specific` so a
consumer can say *which* threshold escalated a patient. See `news2_report.md` for every
chapter's cut-points and their stay-level bootstrap intervals.

## Building from the full MIMIC-IV release

The same four scripts build a full release. What changes is scale: full
MIMIC-IV's `chartevents` is ~432M rows against the demo's 668,862, and a
whole-release warehouse is a tens-of-gigabytes commitment.

```bash
python warehouse/fetch_mimic4.py                      # check the landing zone + disk
python warehouse/build_duckdb.py \
    --data-dir data/raw/mimic-iv-3.1 \
    --db warehouse/mimic4_full.db \
    --cohort-subjects 5000 --force                    # seeded sample of 5,000 ICU patients
python warehouse/run_concepts.py  --db warehouse/mimic4_full.db
python warehouse/hourly_grid.py   --db warehouse/mimic4_full.db
python warehouse/news2.py         --db warehouse/mimic4_full.db
```

**Getting the data is your step, not this repo's.** MIMIC-IV is credentialed
access — a PhysioNet account, CITI training, a signed DUA. `fetch_mimic4.py`
downloads nothing and never touches a credential; it checks whether the
directory is complete, checks whether this machine has the disk, and prints the
`wget` command for you to run (with an interactive password prompt).

**`--cohort-subjects N` samples patients, not stays**, and that distinction is
load-bearing rather than stylistic. Dropping one of a patient's ICU stays would
move a death onto a stay that ended in a live transfer (labels are attributed to
`max(icustay_seq)` per admission), delete readmission events outright (they are
defined on *consecutive pairs* of stays), and split one patient across CV folds.
Sampling subjects keeps each patient's complete hospital history. The sample is
uniform and leaves prevalence alone — case-control sampling would wreck the
calibration that alert thresholds are set from. See `mimic_source.py`.

**How large a cohort?** Not a guess — [`ml/evaluation/reliability_report.md`](../ml/evaluation/reliability_report.md)
measures the learning curve and extrapolates it. Its answer at the time of
writing: a 0.10-wide AUPRC confidence interval needs on the order of 860
positive subjects against the demo's 49, which is a **4,000–5,000 patient
cohort** at a plausible event rate.

Validation adapts to what was built. An unfiltered demo build is checked against
`validate_demo.sql`'s exact published row counts, as before. Anything else has
no published counts to check, so it gets structural checks instead —
referential integrity across `patients`/`admissions`/`icustays`/`chartevents`,
and containment of every loaded row within the cohort.

## Schemas

| Schema | Contents | Source |
|---|---|---|
| `mimiciv_hosp`, `mimiciv_icu` | Raw MIMIC-IV demo tables, unmodified | `build_duckdb.py` |
| `mimiciv_derived` | 65 vendored mimic-code concepts (SOFA, SAPS-II, OASIS, SIRS, Charlson, Sepsis-3, kdigo, vasoactive dosing, …) | `run_concepts.py`, SQL vendored in `mimic-iv/concepts_duckdb/` — see `mimic-iv/VENDORED.md` |
| `capstone` | This project's own derived tables: `hourly_grid`, `disease_context`, `news2`, `news2_thresholds`, `news2_group_thresholds` | `hourly_grid.py`, `disease.py`, `news2.py` |

## Current status (last full rebuild)

- 31/31 tables load with row counts matching the MIMIC-IV demo exactly (`validate_demo.sql`).
- **65/65 concepts build successfully** — see `concept_status.md` (regenerated on every run of
  `run_concepts.py`). PROJECT_PLAN.md anticipated partial failure on this demo subset (rrt/crrt,
  blood_differential, note-derived concepts); in practice, every vendored concept builds on the
  100-patient / 140-stay demo. `sofa` covers all 140 stays.
- `capstone.hourly_grid`: 12,004 patient-hours across 140 stays, reproducing
  `notebooks/01_capstone_eda.ipynb` section 5 exactly, extended with `<col>_was_imputed` and
  `<col>_hours_since_last_obs` per core vital (R2). The pivot now runs **in DuckDB** rather than
  pulling every charted value into a pandas `pivot_table` — the old shape was one intermediate row
  per charted value, which does not survive a cohort larger than the demo.
  `tests/test_hourly_grid.py` asserts the port against a literal re-implementation of the pandas
  algorithm and finds them identical on all 37 columns, at float32 tolerance: `chartevents.valuenum`
  is declared `FLOAT`, so pandas averaged in single precision while DuckDB's `AVG` accumulates in
  double. The SQL result is the more precise of the two.

## What is verified, and what is not

The cohort path is exercised for real, not by inspection: `tests/test_mimic_source.py` builds a
40-of-100-subject cohort from the actual demo dataset and asserts cohort containment, that reference
tables load whole, that every stay of a sampled subject survives, and that declared column types
land correctly.

**It has never been run against a real full MIMIC-IV release, because that data is not on this
machine.** Two things are therefore structurally complete but unproven: the vendored `create.sql`
(from a 2026-09-01 `mimic-code` commit) is assumed to match the release you download, and the
wall-clock cost of a multi-hundred-million-row load is unmeasured. The load is written to survive
schema drift — columns are matched by name, extras are read but not inserted, absent ones land as
NULL, and every type comes from the declared schema rather than DuckDB's sniffer — but "written to
survive it" is not "observed surviving it". The first real full-release build should be treated as
the test it is.
- `capstone.news2`: NEWS2 computed per patient-hour, reproducing section 7 exactly (component
  distribution, 128/140 stays reaching ward "medium", 105/140 reaching ward "high" — see
  `news2_report.md` for why these differ from the *prose* figures in PROJECT_PLAN.md's E4, and for
  the ICU-recalibrated cut-points required by E5).

## A note on PROJECT_PLAN.md's quoted figures

The plan states in its header that "every number quoted in this plan traces to a cell in that
notebook." Two of the E4 figures in section 2 (110/140 and 68/140 stays reaching NEWS2 thresholds)
do not match the executed notebook's own rendered figure (`eda_figures/08_news2.png`: 128 and 105).
The scripts here reproduce the *executed notebook* exactly and use its numbers as the regression
target — the plan's prose appears to predate the final run and was not updated.
