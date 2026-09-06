# Phase 5 -- Predictive models

PROJECT_PLAN.md section 11: composite hourly deterioration, served through
`risk-engine` (Phase 4), attributed with SHAP, and logged to MLflow.

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice (PROJECT_PLAN.md section 17).

## Running it

```bash
pytest ml/ -v                          # unit + real-warehouse regression tests
python ml/evaluation/run_all.py        # full protocol, ~5 min on a laptop CPU
python ml/evaluation/run_all.py --quick  # 2-repeat smoke run while developing
python ml/evaluation/secondary_whole_stay.py   # the underpowered secondary task
```

`run_all.py` writes [`ml/evaluation/report.md`](evaluation/report.md) (the
committed results snapshot), a SHAP summary CSV, an MLflow run under
`mlruns.db` (gitignored, local SQLite-backed tracking + model registry), and
the promoted model export at `ml/models/promoted/` (also gitignored -- see
below).

## The task (`ml/features/labels.py`)

Composite event = death, vasopressor initiation, invasive ventilation
initiation, or unplanned ICU readmission, predicted within the next 6h
(primary) or 12h (secondary) from every `capstone.hourly_grid` row. This is
the only adequately-powered task in this cohort (E2), but three design
decisions correct a naive reading of the plan's own wording -- all three are
mechanically tested, not just asserted:

- **Death is attributed to the last ICU stay of the hospitalisation**, not
  every stay under it (the warehouse's `hospital_expire_flag` propagates to
  every stay of a hadm; a patient's earlier stay in the same admission ended
  in a live transfer, not death). Collapses the naive 20 flag-carrying rows to
  15 actual death *events*.
- **Ventilation means invasive ventilation** (`ventilation_status =
  'InvasiveVent'`), not any oxygen-delivery device.
- **Unplanned ICU readmission** is a same-admission ICU bounce-back within
  72h -- a different, ICU-specific concept from the EDA's 53 "30-day
  readmissions" (which is a whole-*hospitalisation* metric across all 275
  admissions, most of which never touched the ICU).

**R1 in practice, and the finding it surfaced:** a row is only a valid
training instance strictly *before* its stay's first composite event --
"windows ending strictly before the outcome." Applying that discovered that
78 of 140 stays already needed vasopressor/ventilator support within their
first 1-3 hours (this cohort skews post-operative/cardiac-surgery, admitted
*for* the intervention rather than deteriorating into it). Correctly censored,
that shrinks the raw 12,004-row hourly grid to **2,979 at-risk patient-hours**
-- still ~21x the whole-stay outcome count (E2's underlying argument), just a
different number than the raw grid size. This is reported here rather than
worked around, because the alternative (a looser event definition that keeps
more rows) would mean modelling "already needed intervention," not
"about to deteriorate."

| Horizon | At-risk hours | Positives | Positive stays |
|---|---|---|---|
| 6h | 2,979 | 120 (4.0%) | 60 |
| 12h | 2,979 | 161 (5.4%) | 60 |

## Features (`ml/features/engineer.py`, `ml/features/ecg.py`)

Every group traces to a design rule (see the module docstring for the full
mapping): carried-forward vitals + imputation flags + recency (R2), rolling
4h/24h mean/std/OLS-slope per vital (reusing
`services/stream-processor/windowing.py`'s `trend_slope` -- the same function
the live streaming path uses), NEWS2 + rolling-24h SOFA as *input features*
(not just baselines), hour-resolved arterial-line presence (R3), lab-ordering
intensity normalised to the admission's own baseline rate (R3/R4/R5), static
demographics, and ECG fusion (E9: rate/QRS/QTc/a rhythm-regularity heuristic
via neurokit2, joined on subject_id + nearest **preceding** `ecg_time` within
72h -- future ECGs cannot inform a past hour).

The rhythm classification is a rate/RR-variability heuristic, not a clinical
arrhythmia diagnosis -- there is no ground truth to validate one against
(MIMIC-IV-ECG's demo subset ships no `machine_measurements`).

## Baselines and models (`ml/models/`)

Baselines the plan requires beating before any claim: recalibrated NEWS2 and
SOFA (read straight from the warehouse, used as monotonic risk scores) and
logistic regression on literally just age + last HR/RR/SpO2. Models: L2
logistic regression on the full feature set, LightGBM, and a small GRU over
the raw 24h vitals sequence (not the pre-aggregated features -- it learns its
own temporal representation).

**Evaluation** (`ml/evaluation/metrics.py`): grouped (by `subject_id`)
repeated stratified 5-fold CV, AUPRC as the headline metric (base rates are
~4-5%), AUROC/calibration/Brier alongside it, and a group-level bootstrap 95%
CI (resampling whole patients, not individual patient-hours -- verified by a
dedicated test that a naive row-level bootstrap would fail).

**GRU compute-budget note, stated rather than hidden:** LR and LightGBM run
the full 20-repeat protocol (each fit is well under a second). A GRU fit is a
training loop; the plan explicitly permits "GRU -> stop at LightGBM" as the
*first* item in the descope order. Rather than drop it, `ml/models/gru.py`
runs it for real over a reduced 5-repeat x 5-fold budget (25 fits, ~5 CPU
minutes) and says so plainly in the code and the report, instead of silently
presenting it as the same-size protocol as the other two.

## Results (6h, primary horizon -- see [`evaluation/report.md`](evaluation/report.md) for both horizons in full)

| model | AUROC | AUPRC | Brier |
|---|---|---|---|
| NEWS2 (recalibrated) | 0.673 (0.58-0.76) | 0.095 (0.05-0.19) | n/a (ordinal score) |
| SOFA | 0.711 (0.61-0.79) | 0.106 (0.06-0.25) | n/a |
| age + HR/RR/SpO2 LR | 0.674 (0.57-0.77) | 0.094 (0.05-0.21) | 0.218 |
| Logistic regression (full) | 0.844 (0.75-0.93) | 0.358 (0.23-0.55) | 0.068 |
| **LightGBM** | **0.855 (0.76-0.94)** | **0.448 (0.32-0.64)** | **0.035** |
| LightGBM + ECG fusion | 0.825 (0.73-0.92) | 0.462 (0.34-0.64) | 0.032 |
| GRU (24h raw sequence) | 0.800 (0.70-0.91) | 0.264 (0.18-0.38) | 0.131 |

**Verification (PROJECT_PLAN.md section 15): LightGBM beats recalibrated
NEWS2 on AUPRC in 20 of 20 CV repeats at the 6h horizon -- meets the >=15/20
bar.** ECG fusion adds a small AUPRC gain at 6h (+0.014) but a loss at
12h (-0.050) -- reported both ways rather than only the favourable one. The
promoted model (LightGBM + ECG, chosen because it won on the primary 6h
horizon) is exported for `risk-engine` to serve; see below.

Top SHAP features (see [`evaluation/shap_summary.csv`](evaluation/shap_summary.csv)):
`sofa_24hours`, `hr_24h_mean`, `gender`, `rr_24h_std`, `hr_4h_mean`,
`first_careunit`, `temp_c_24h_mean`, `rr_24h_slope`, `ecg_hours_since`
(i.e. *whether an ECG exists nearby at all* carries signal, independent of
its content).

## Secondary, underpowered outcomes (`ml/evaluation/secondary_whole_stay.py`)

ICU mortality and 30-day readmission, exactly as the plan requires: reported
with wide CIs and an explicit underpowered caveat, **not** as a second
headline result and **not** compared against NEWS2/SOFA the way the primary
task is (E6: only 20 ICU deaths and 53 whole-cohort 30-day readmissions).

| Task | n | Positives | AUROC (95% CI) | AUPRC (95% CI) |
|---|---|---|---|---|
| ICU mortality (whole stay) | 140 | 20 (14.3%) | 0.734 (0.62-0.87) | 0.378 (0.23-0.59) |
| 30-day readmission (whole admission) | 113 | 22 (19.5%) | 0.452 (0.33-0.59) | 0.181 (0.12-0.29) |

The readmission AUROC sitting at/below chance, with a CI straddling 0.5, is
not a bug -- it is what "no usable signal at this n" looks like, reported as
such. (The 113/22 here is a different, ICU-restricted slice of the EDA's
53-readmission figure, not a discrepancy -- see the script's own docstring.)

## Serving the model (`ml/models/serving.py` -> `services/risk-engine`)

"SHAP attribution surfaced through risk-engine so every alert carries a
reason" (PROJECT_PLAN.md section 11). `risk-engine`'s `/score/ml/{stay_id}/
{hour}` loads the promoted model + a per-request SHAP explanation and returns
a probability plus the top-5 contributing features in plain language --
tested end-to-end against the *real* trained model (`ml/tests/test_serving.py`,
`services/risk-engine/tests/test_app.py`), not a mock.

Two things are deliberately imperfect and said so in the code, not hidden:

1. **The promoted model is not committed to git** -- "the promoted model is a
   registry artefact, not a pickle in a folder" (section 11). It's an MLflow
   registered model (`mlruns.db`, gitignored) plus a local export at
   `ml/models/promoted/` (also gitignored) that `risk-engine` reads directly,
   analogous to pulling an artefact from a real registry at deploy time. A
   fresh checkout that hasn't run `run_all.py` gets a genuine 503 from
   `/score/ml`, never a fabricated number.
2. **`score_one()` rebuilds the entire ~12,000-row feature frame per request**
   (a few seconds of latency) because `ml/features/engineer.py` was written
   for batch training, not single-row lookup. Phase 7's latency budget sizes
   the *deterministic* NEWS2/SOFA path (`/score`), not this extension point --
   stated as a known limitation, not silently absorbed into that budget.

**Docker verification: done for real, on a second pass.** The first attempt
(same session, before the host disk was cleared) surfaced a real missing
`.dockerignore` (a plain `docker build` was sending the multi-gigabyte
raw-dataset and `.venv` directories as build context) and then hit a host
disk that was genuinely full, which corrupted Docker Desktop's own image
store mid-build. Once the disk had room, a clean rebuild surfaced two more
real, previously-invisible bugs -- both fixed and re-verified in a running
container, not just patched and assumed:

- **`libgomp.so.1: cannot open shared object file`.** LightGBM's compiled
  core dynamically links GNU OpenMP; `python:3.11-slim` doesn't ship it.
  `/score/ml` 500'd the instant `joblib.load()` tried to import `lightgbm`
  inside the container. Every local test had passed because macOS already
  has an OpenMP runtime on the library search path. Fixed with
  `apt-get install libgomp1` in `services/risk-engine/Dockerfile`.
- **`torch` pulling the entire CUDA toolkit into a CPU-only container.**
  PyPI's default Linux `torch` wheel is CUDA-enabled, so `uv sync` inside the
  Linux container resolved ~2.5GB of `nvidia-*` packages (cublas, cudnn,
  cusolver, nccl, triton, ...) that nothing in this project uses -- the image
  never runs on a GPU, and GRU training only ever happens on the host during
  `run_all.py`. Fixed with a `[tool.uv.sources]` entry in `pyproject.toml`
  routing `torch` to PyTorch's own CPU-only index for `platform_system ==
  'Linux'` only (macOS keeps resolving from PyPI, whose wheel has no CUDA
  dependency to begin with). Dropped the image's content size from 3.55GB to
  552MB and regenerated `uv.lock` accordingly -- `uv sync` on macOS was
  re-verified to still resolve the identical local dependency set.

With both fixed, a real container (`docker build` + `docker run`, the real
warehouse and the real exported model volume-mounted in, no live sockets or
mocks) answered `/health`, `/score`, and `/score/ml` correctly for two
different patients, and `/score/ml` 404s correctly for an unknown stay. A
third, cosmetic bug turned up in that same pass -- `/score/ml`'s "reasons"
field was rendering missing vitals and numpy scalars as `np.float64(nan)`
instead of plain language -- fixed in `ml/models/serving.py::_format_value`
with a regression test, verified by rebuilding and re-curling the container
a second time.

## Real bugs found while building this (not by inspection)

- **LightGBM + PyTorch SIGSEGV on macOS.** A clean run of ~10-20 repeated
  LightGBM CV fits, immediately followed by GRU training in the *same
  process*, crashed with exit code 139 and no Python traceback -- both
  libraries bundle their own OpenMP runtime, and macOS does not tolerate two
  copies cleanly. Reproduced directly (isolated to the LightGBM-then-torch
  ordering, not LightGBM alone -- a 120-fit LightGBM-only stress test ran
  clean). Fixed by pinning `n_jobs=1` on every LightGBM fit
  (`ml/models/gbm.py`) and setting `KMP_DUPLICATE_LIB_OK=TRUE` /
  `OMP_NUM_THREADS=1` at `ml/__init__.py` import time, before any submodule
  can import numpy/lightgbm/torch.
- **`ecg_time` leaking into the feature matrix.** `attach_nearest_ecg()`
  selected every column starting with `"ecg_"` as a feature -- which also
  matched its own join key, `ecg_time`, a raw timestamp. LightGBM's pandas
  ingestion rejected the resulting mixed-dtype frame outright
  (`DTypePromotionError`). Fixed by excluding the join key explicitly, with a
  regression test (`test_attach_nearest_ecg_excludes_the_join_key_from_feature_columns`).
- **DuckDB int32 vs. pandas int64 breaking `merge_asof`'s `by=` key.**
  `subject_id` from a DuckDB query and from `record_list.csv` (pandas' own
  CSV inference) carried different integer widths; `merge_asof` requires
  matching dtypes on `by`. Fixed by normalising both sides to `int64` before
  the join, with a regression test for the dtype mismatch specifically.
