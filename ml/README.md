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

## Features (`ml/features/engineer.py`)

**67 features**, each group tracing to a design rule (see the module docstring
for the full mapping): carried-forward vitals + imputation flags + recency (R2),
rolling 4h/24h **std and OLS-slope** per vital (reusing
`services/stream-processor/windowing.py`'s `trend_slope` -- the same function
the live streaming path uses), hour-resolved arterial-line presence (R3), static
and static demographics.

**Four groups were removed and are documented in
[`evaluation/feature_pruning_report.md`](evaluation/feature_pruning_report.md)**,
which measured each rather than arguing it:

- **`news2` and `sofa_24hours`** are no longer input features. They are
  deterministic functions of vitals the model already has (measured
  contribution: 0.000), and SOFA's cardiovascular component is scored on
  vasopressor dose -- one of the labels. Eleven at-risk rows carry a
  vasopressor-implied SOFA score and **all eleven are positives**. They remain
  the two baselines every model must beat, fed from their own matrix via
  `feature_matrix_for_training(..., include_severity_scores=True)`.
- **The 18 rolling means** were largely a restatement of the carried-forward raw
  value; removing them was the single largest gain in the study. Std and slope
  stay, because variability and direction are not recoverable from a point value.
- **The 3 lab-ordering-intensity columns** were removed on measurement (10/10
  paired repeats) *and* on design: ordering rate is a proxy for clinician
  concern, so it partly reads out the outcome rather than predicting it.
- **`rr` was proposed for removal and deliberately kept** -- see the report for
  why acting on that one would have been the study overfitting its own noise.

**ECG fusion was built, measured, and removed.** It won only **5 of 20** paired
CV repeats (mean per-repeat delta −0.015), and a post-discharge patient has no
12-lead ECG, so it was cut from the model, the pipeline and the repository
rather than kept as an unused option. PROJECT_PLAN.md's finding E9 is marked
retired with that measurement attached.

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

> **That "by `subject_id`" was not true until recently, and the numbers below
> changed when it became true.** This README claimed subject grouping,
> `ml/models/splits.py` documented it, and `splits.group_key()` existed to
> supply it -- but nothing ever called that helper with a subject id, so every
> CV number was grouped by `stay_id`. In this cohort 21 of 93 at-risk subjects
> have more than one ICU stay, carrying 45% of the rows and **44% of the
> positives**, so nearly half the signal came from patients who could sit in
> the training and test folds simultaneously. Measured cost: **+0.0499 AUPRC**
> of optimism at the 6h horizon on the current feature set
> ([`evaluation/reliability_report.md`](evaluation/reliability_report.md)).
> `feature_matrix_for_training` now returns `subject_id`, `group_key()` is
> deleted, and `ml/tests/test_engineer.py` pins the grouping unit so it cannot
> regress. Three sources agreeing on paper while the code did something else is
> the failure mode worth remembering here.

**GRU compute-budget note, stated rather than hidden:** LR and LightGBM run
the full 20-repeat protocol (each fit is well under a second). A GRU fit is a
training loop; the plan explicitly permits "GRU -> stop at LightGBM" as the
*first* item in the descope order. Rather than drop it, `ml/models/gru.py`
runs it for real over a reduced 5-repeat x 5-fold budget (25 fits, ~5 CPU
minutes) and says so plainly in the code and the report, instead of silently
presenting it as the same-size protocol as the other two.

## Results (6h, primary horizon -- see [`evaluation/report.md`](evaluation/report.md) for both horizons in full)

**These figures are patient-grouped.** They are lower than the ones this file
carried before finding F6 (LightGBM AUPRC 0.448 → 0.398), because those were
grouped by stay and read patients they had trained on. See the note above.

| model | AUROC | AUPRC | Brier |
|---|---|---|---|
| NEWS2 (recalibrated) | 0.673 (0.58-0.76) | 0.095 (0.05-0.19) | n/a (ordinal score) |
| SOFA | 0.711 (0.61-0.80) | 0.106 (0.06-0.24) | n/a |
| age + HR/RR/SpO2 LR | 0.645 (0.53-0.76) | 0.075 (0.04-0.17) | 0.242 |
| Logistic regression (full) | 0.817 (0.74-0.90) | 0.332 (0.23-0.47) | 0.094 |
| **LightGBM** | **0.823 (0.73-0.92)** | **0.493 (0.38-0.63)** | **0.033** |
| GRU (24h raw sequence) | 0.800 (0.70-0.91) | 0.265 (0.18-0.38) | 0.131 |

**Verification (PROJECT_PLAN.md section 15): LightGBM beats recalibrated
NEWS2 on AUPRC in 20 of 20 CV repeats at the 6h horizon -- meets the >=15/20
bar**, and still does under both the corrected grouping and the pruned feature
set. There is now **one model** -- plain LightGBM on 67 features, exported for
`risk-engine` to serve. The previous ECG-fusion variant and the sign test that
chose between them are both gone: a promoted artefact should not change identity
between runs on a noisy point estimate.

**AUPRC rose 0.398 -> 0.493 because the feature set shrank**, not because the
model or the data improved: 90 features against 120 positives was
over-parameterised, and
[`evaluation/feature_pruning_report.md`](evaluation/feature_pruning_report.md)
measures what each removed family was worth. `news2` and `sofa_24hours` are gone
as *features* (they remain the baselines every model must beat, fed from their
own matrix), along with the 18 rolling means and the 3 lab-ordering-intensity
columns.

Top SHAP features (see [`evaluation/shap_summary.csv`](evaluation/shap_summary.csv)):
`hr`, `first_careunit`, `gender`, `rr_24h_std`, `admission_age`, `map_24h_std`, `sbp`, `gcs_total`, `sbp_24h_std`.

The list is worth comparing against the pre-pruning one, which was led by
`sofa_24hours` -- a feature later measured at **0.000** contribution. Removing it
cost nothing and the ranking is now led by `hr`, which is both more plausible and
a standing warning that **SHAP attribution is not necessity** when features are
correlated.

`gender` sitting third is exactly the pattern review finding F4 flagged, and it
is back because F4 was **reversed**: correcting the CV grouping moved its
ablation from 13/20 to 15/20 repeats, over the 75% bar. That was a deliberate
call with its weaknesses recorded (`ml/features/engineer.py`'s docstring and
`VALIDATION_REPORT.md` Appendix 4). The practical consequence: the subgroup
audit below is no longer a check that a protected attribute stayed *out* of the
model -- the model uses it, so that audit is the thing standing between it and
an unequal error distribution.

## What limits reliability (`ml/evaluation/reliability.py`)

```bash
python ml/evaluation/reliability.py               # full protocol, ~15 min
python ml/evaluation/reliability.py --report-only  # rebuild the prose from the last run's CSVs
```

The headline AUPRC carries a confidence interval roughly 0.26 wide -- consistent
with "clearly beats NEWS2" and "barely beats NEWS2" at once. This module asks
which available lever actually narrows it, and answers by measurement rather
than assertion. Full write-up:
[`evaluation/reliability_report.md`](evaluation/reliability_report.md).

| Lever | Verdict |
|---|---|
| **Grouping unit** | Real, fixed. Stay-grouping was inflating AUPRC by **+0.0499**. See the note above. |
| **Feature budget** | **Not a lever.** Tested and refuted. |
| **More patients** | The only real one. **~725 positive subjects** for a 0.10-wide CI, against today's 49. |

Two of these are worth stating plainly because they went against expectation.

**The events-per-variable objection does not apply to this model.** 89 features
against 120 positives is EPV 1.35, against a conventional floor of 10, and the
obvious inference is that the model must be fitting noise. It is not: with
features selected *inside* each training fold and the model refitted on the
top-k, AUPRC rises monotonically with the budget (0.235 at 5 features → 0.399 at
the full set) and the across-repeat spread shows no trend. EPV is a rule about degrees
of freedom in a *linear* model, where every predictor spends a parameter; a
depth-4, 15-leaf LightGBM selects features at every split and spends nothing on
one it does not use. Recorded so the objection is not re-litigated from theory.

**The confidence interval does not shrink like a standard error.** Fitting
`width = a * n^b` over subsampled patients gives **b = -0.37** (R² 0.868), not
the textbook -0.50. Assuming -0.50 would have said 317 positive subjects buy a
0.10-wide interval; the measured exponent says **725**. The report is explicit
that this is an extrapolation roughly 15x beyond the largest measured point, and
that the exponent is itself likely flattered by AUPRC's bounded range -- it
sizes a data pull to an order of magnitude, and does not promise a number.

That sizing is what `warehouse/build_duckdb.py --cohort-subjects N` exists to
load; see [`warehouse/README.md`](../warehouse/README.md).

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
- **DuckDB int32 vs. pandas int64 breaking `merge_asof`'s `by=` key.**
  `subject_id` from a DuckDB query and from `record_list.csv` (pandas' own
  CSV inference) carried different integer widths; `merge_asof` requires
  matching dtypes on `by`. Fixed by normalising both sides to `int64` before
  the join, with a regression test for the dtype mismatch specifically.
