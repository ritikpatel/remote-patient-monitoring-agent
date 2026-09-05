# Warehouse

One queryable DuckDB store (`mimic4_demo.db`, gitignored — rebuild it, don't commit it) plus every
standard ICU severity score. PROJECT_PLAN.md section 7.

## Build order

```bash
python warehouse/build_duckdb.py    # schema + load 31 CSVs -> mimiciv_hosp / mimiciv_icu
python warehouse/run_concepts.py    # 65 mimic-code concepts -> mimiciv_derived
python warehouse/hourly_grid.py     # capstone.hourly_grid   (12,004 patient-hours)
python warehouse/news2.py           # capstone.news2         (ward-standard + ICU-recalibrated)
```

Each script is idempotent — re-running it drops and rebuilds only its own tables.

## Schemas

| Schema | Contents | Source |
|---|---|---|
| `mimiciv_hosp`, `mimiciv_icu` | Raw MIMIC-IV demo tables, unmodified | `build_duckdb.py` |
| `mimiciv_derived` | 65 vendored mimic-code concepts (SOFA, SAPS-II, OASIS, SIRS, Charlson, Sepsis-3, kdigo, vasoactive dosing, …) | `run_concepts.py`, SQL vendored in `mimic-iv/concepts_duckdb/` — see `mimic-iv/VENDORED.md` |
| `capstone` | This project's own derived tables: `hourly_grid`, `news2` | `hourly_grid.py`, `news2.py` |

## Current status (last full rebuild)

- 31/31 tables load with row counts matching the MIMIC-IV demo exactly (`validate_demo.sql`).
- **65/65 concepts build successfully** — see `concept_status.md` (regenerated on every run of
  `run_concepts.py`). PROJECT_PLAN.md anticipated partial failure on this demo subset (rrt/crrt,
  blood_differential, note-derived concepts); in practice, every vendored concept builds on the
  100-patient / 140-stay demo. `sofa` covers all 140 stays.
- `capstone.hourly_grid`: 12,004 patient-hours across 140 stays, reproducing
  `notebooks/01_capstone_eda.ipynb` section 5 exactly, extended with `<col>_was_imputed` and
  `<col>_hours_since_last_obs` per core vital (R2).
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
