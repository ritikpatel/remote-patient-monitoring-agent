# Vendored from MIT-LCP/mimic-code

Source: https://github.com/MIT-LCP/mimic-code
Commit: `303d26c623dcc9c49cc0f204468d4acc2f063797` (2026-09-01)
License: MIT — full text in [`LICENSE-mimic-code`](LICENSE-mimic-code); copyright MIT Laboratory for
Computational Physiology.

## What was pulled and why

| Path here | Source path | Purpose |
|---|---|---|
| `concepts_duckdb/` | `mimic-iv/concepts_duckdb/` | 65 concept SQL files (DuckDB dialect) computing every standard severity score, plus `duckdb.sql`, which encodes the dependency-ordered build sequence we parse in [`warehouse/run_concepts.py`](../warehouse/run_concepts.py) |
| `buildmimic/postgres/create.sql` | `mimic-iv/buildmimic/postgres/create.sql` | Canonical schema definition (`mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived`) — the DuckDB build patches three lines at load time rather than editing this file, per upstream's own `build_mimic.sh` (see [`warehouse/build_duckdb.py`](../warehouse/build_duckdb.py)) |
| `buildmimic/postgres/validate_demo.sql` | `mimic-iv/buildmimic/postgres/validate_demo.sql` | Expected row counts for the MIMIC-IV **demo** subset (100 patients / 140 ICU stays), used to confirm the load is complete |

**Not vendored:** the postgres/mysql/sqlite/bigquery build variants, the `concepts` (BigQuery dialect) and
`concepts_postgres` directories, `mapping/`, `notebooks/`, and `tests/` — none are used by this project.

## Why the upstream schema names are kept as-is

`PROJECT_PLAN.md` §7 describes the target schemas as "`mimic_hosp` / `mimic_icu`". This build instead keeps
the upstream names `mimiciv_hosp` / `mimiciv_icu` / `mimiciv_derived` verbatim, because all 65 concept SQL
files are schema-qualified against those exact names. Renaming would mean either hand-editing every
vendored file (defeating "reuse, don't rebuild") or maintaining a fork that drifts from upstream. The plan's
wording is read as descriptive shorthand for "the hosp-module schema and the icu-module schema," not a
literal naming requirement.

## Known thin areas (per PROJECT_PLAN.md E7 / §7.3)

`rrt`/`crrt` (9 stays in the demo), `blood_differential`, and any concept indirectly dependent on modules
absent from the demo (e.g. free-text notes) are expected to build with very little data, or occasionally
fail outright on this 100-patient subset. `warehouse/run_concepts.py` treats each concept file
independently (try/except) and records the outcome in `concept_status.md` rather than aborting the whole
run — see PROJECT_PLAN.md §7 item 3, "expect partial failure."
