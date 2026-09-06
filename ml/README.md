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

**Docker verification status, honestly:** the updated `services/risk-engine/
Dockerfile` (now `COPY`ing `ml/` and `stream-processor/windowing.py`) was
*not* re-verified with a live container build this session. Building it
surfaced two real, separate problems worth recording even though the second
one blocked the attempt: (1) this repo had no `.dockerignore`, so a plain
`docker build` was sending the multi-gigabyte raw-dataset and `.venv`
directories as build context -- fixed, real fix, unrelated to whether the
build itself completes; (2) the host disk was found to be essentially full
during this attempt (56Mi free of 228Gi), which corrupted Docker Desktop's
local image/layer store mid-build (`input/output error` extracting a layer,
then the same error from `docker system prune`). That is a host-machine
condition, not a defect in this Dockerfile or in `ml/`'s code, and resolving
it (freeing disk space, likely including Docker Desktop's own 18GB VM disk
image) needs the user's decision, not an automated fix from inside this
session. The application logic this Dockerfile would serve is still verified
for real, just via `pytest` (`ml/tests/test_serving.py`,
`services/risk-engine/tests/test_app.py`) against the actual trained model,
rather than via a running container.

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
