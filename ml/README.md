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

**85 features**, each group tracing to a design rule (see the module docstring
for the full mapping): carried-forward vitals + imputation flags + recency (R2),
rolling 4h/24h **std and OLS-slope** per vital (reusing
`services/stream-processor/windowing.py`'s `trend_slope` -- the same function
the live streaming path uses), hour-resolved arterial-line presence (R3), static
demographics, and **disease context** (18 columns) from
`capstone.disease_context`.

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

### Disease awareness (`evaluation/disease_leakage.py`)

The model was disease-blind until `warehouse/disease.py` existed, and its only
case-mix signal was `first_careunit` -- which ranked **second by mean |SHAP|**,
above every vital except HR. That column was acting as an unexamined diagnosis
proxy ("admitted to CVICU" ≈ "cardiac problem"), learned without anyone deciding
it should be and without the leakage scrutiny a diagnosis feature deserves.

Making the axis explicit forced two questions, both measured rather than assumed
(20 repeats × 5 folds, identical folds across arms):

| arm | disease cols | mean AUPRC | Δ | paired wins vs `none` | mean AUROC |
|---|---|---|---|---|---|
| `none` | 0 | 0.5018 | — | — | 0.8198 |
| `chronic_index_only` | 1 | 0.4936 | −0.0083 | 3 / 20 | 0.8429 |
| **`chronic`** (shipped) | 18 | 0.4879 | −0.0139 | 4 / 20 | 0.8448 |
| `coded_only` | 1 | 0.4907 | −0.0111 | 2 / 20 | 0.8042 |
| `all` | 19 | 0.4757 | −0.0262 | 2 / 20 | 0.8273 |

**There is no detectable leak.** `dx_chapter` alone scores AUPRC **0.0392 against
a base rate of 0.0403** -- at, fractionally below, chance. The discharge-coded
diagnosis carries essentially no information about *which hour* a patient
deteriorates. Nor does any chapter spike on the label component matching its own
organ system. The leakage worry that motivated the two-set split was real enough
to have to check, and it did not materialise.

**Disease features do not improve the point estimate.** No arm beats disease-blind;
the best wins 4 of 20 repeats. Same lesson `feature_pruning.py` already drew: 49
positive subjects will not support more columns. Note the metric split -- AUROC
rises consistently (+0.025) while AUPRC falls. AUPRC is what this project judges
on at a 4% base rate, so AUPRC wins the argument.

**`chronic` ships anyway, and that is a judgement, not a measurement.** The
−0.0139 delta is ~5% of the width of AUPRC's own bootstrap CI (0.3763–0.6319), and
`chronic` is the arm the leakage probes clear unambiguously. Against that: the
platform is disease-aware end to end -- per-chapter NEWS2 cut-points, disease-scoped
retrieval, the care-plan node, the diagnosis on the escalation email -- and a model
that alone stayed blind to it would explain its scores in terms no other component
shares. Reversible in one line: `engineer.DEFAULT_DISEASE_FEATURES = "none"`.

One thing the retrain did change: **`charlson_comorbidity_index` is now the top
SHAP feature (0.832, above `hr` at 0.686), and `first_careunit` fell from 0.542 to
0.300** -- the explicit disease feature absorbed roughly half of what the care-unit
proxy had been carrying. The information was always being used; it is now named.

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

## 30-day readmission, rebuilt (`ml/evaluation/readmission.py`)

`secondary_whole_stay.py` reported **AUROC 0.452** for readmission and blamed sample
size. Two things were wrong before sample size was:

* **The cohort.** Readmission is an outcome of a *hospitalisation*, not of an ICU stay.
  The old cohort was the 140 ICU stays' own admissions (113 rows, 22 positives); the
  matching cohort is every live discharge — **252 admissions, 53 positives (21.0%)**
  after CMS-style competing-risk exclusions (hospice, and death within 30 days without
  readmission). That is 2.4x the positives from the same database.
* **The features.** All eight old columns were ICU physiology — max NEWS2/SOFA, mean
  HR/RR/SpO2, ever-vasopressor, ever-ventilation, age. None described who the patient
  is, what they were treated for, how often they had been admitted, or where they were
  discharged to. Prediction time here is **discharge**, so per finding E22 the whole
  coded record is legitimate — the same ICD code that is leak-suspect for the hourly
  model is fine for this one.

Identical rows, identical folds, 20 repeats x 5 folds, subject-grouped:

| model | cv AUROC | cv AUPRC | AUROC 95% CI | lift over base rate |
|---|---|---|---|---|
| `icu_physiology` (previous) | 0.4757 | 0.2387 | 0.335–0.539 | 1.13x |
| `discharge_context` (24 cols) | 0.5215 | 0.2687 | 0.383–0.609 | 1.28x |
| **`lace_plus`** (7 cols, shipped) | **0.5455** | **0.2815** | 0.412–0.590 | **1.34x** |

**Two findings that point different ways, and both ship.** The feature set genuinely
matters — LACE+ beats the previous set in **19 of 20 paired repeats**, which is not
noise. And the parsimonious seven-column set (the validated LACE index plus discharge
disposition and recent utilisation) beats the twenty-four-column one, which is the same
over-parameterisation lesson `feature_pruning.py` and `disease_leakage.py` already drew,
now on a third task.

**But the previous conclusion still stands.** The AUROC interval **0.412–0.590 includes
0.5**, so better-than-chance discrimination is *not* established at this cohort size.
The defensible pair of statements is: the feature set matters (paired evidence), and the
cohort is too small to certify the model (interval evidence). Reporting the first
without the second would be exactly the error this project exists to avoid.

**What it does not claim:** a readmission-rate *reduction*. Predicting is not reducing;
that needs an intervention and a control arm. 30.6% of the cohort is also right-censored
— the patient's last recorded admission, with no way to know what followed.

## Home-kit transfer (`ml/evaluation/home_kit_transfer.py`)

**MIMIC contains no physiology after discharge** (E20). It records discharge,
readmission and death *times* — outcomes, but no signal. So a post-discharge
deterioration score cannot be trained on post-discharge data here, and the only
defensible construction is a transfer: train on ICU physiology, deploy against what a
home kit can observe, and state the bound.

`channel_dropout.py` measured that bound the cheap way — train on everything, mask at
prediction time — and then argued against its own method: every channel is 74–100%
present in training, so the trees never learned a split direction for absence. This
study does it properly, with **kit-shaped channel dropout applied during training**,
and compares the two on identical folds (10 repeats × 5 folds):

| assumed home kit | dropout-trained | masked-at-inference | Δ | paired wins |
|---|---|---|---|---|
| `icu_full` (all channels) | 0.3857 | **0.4925** | −0.1068 | 0/10 |
| `full_home` (watch + cuff + CGM) | **0.3452** | 0.3078 | +0.0374 | 9/10 |
| `watch_plus_cuff` | **0.3430** | 0.3074 | +0.0356 | 9/10 |
| `watch_only` | **0.3169** | 0.2454 | +0.0716 | 10/10 |

**This reverses a conclusion this repo previously drew.** `channel_dropout_report.md`
concluded that "one model that degrades gracefully covers the same ground" as a
dedicated wrist model — a correct reading of the evidence *then*, because the only
alternative it compared against was `wrist_only.py`'s model trained on a **restricted
feature set**. Training on the **full** feature set with kit-shaped dropout is a
different thing, and it wins every home kit. It also costs the ICU arm 0.107 AUPRC,
losing all 10 repeats.

So the deployment shape is **two fits, one estimator**: the dropout-free model for the
ICU arm, the dropout-trained model for the post-discharge arm, each used only in the
regime it was fitted for. That is the accuracy/robustness trade-off in its plainest
form — a model taught that channels vanish stops leaning as hard on the ones present,
which is what you want at home and not what you want in an ICU.

**The floor no training scheme escapes.** Core temperature, GCS, FiO2 and arterial-line
presence have **no home instrument at any price**, and notebook 03 §5 shows no
home-measurable channel correlates with them above |r| = 0.22. The gap between
`full_home` and `icu_full` is the information those four carry. Closing it means
changing what the patient is wearing, not how the model is fitted.

The kits (`watch_only`, `watch_plus_cuff`, `full_home`) are the **same objects**
`simulators/home_kit_stream.py` streams — imported, not redeclared, so training masks
and streamed channels cannot drift apart.

## Channel dropout (`ml/evaluation/channel_dropout.py`)

What the *deployed* model does when a sensor is missing, masked at prediction
time rather than retrained. The realistic post-discharge channel set retains
**68%** of AUPRC; masking to HR+SpO2 retains 55% (0.274) — statistically the
same as a model trained only on those channels (0.269), which is the evidence
that one generic model beats maintaining a second wrist-specific one. See
[`evaluation/channel_dropout_report.md`](evaluation/channel_dropout_report.md),
including why these are upper bounds.

## Synthetic patients, and what they are worth (`ml/synthetic/`)

```bash
python ml/synthetic/run_tutorial.py      # quality battery + row-level augmentation, ~20 min
python ml/synthetic/run_patients.py      # whole class-balanced patients, ~45 min
python ml/synthetic/report.py            # rebuild the prose from the CSVs
```

[`reliability.py`](evaluation/reliability.py) concludes that the only lever left
on this model's interval is more patients, and full MIMIC-IV is credentialed
access. So the obvious question is whether a generative model can manufacture
the difference. [`evaluation/synthetic_ceiling.py`](evaluation/synthetic_ceiling.py)
answered it with a resampling generator and found nothing — but admitted a hole
in its own text: perturbing channels independently could only reach 100% or 0%
fidelity, never the middle, so "what if a *real* generator hit 90%?" stayed
hypothetical.

`ml/synthetic/` closes that hole by implementing the EMR-WGAN tutorial (Yan et
al., JMIR AI 2024;3:e52615) properly, and the implementation is faithful enough
for the answer to count: dimension-wise distance **1.44** against the 0.52–1.56
the paper reports for its own five runs on 181,294 patients, column-wise
correlation ~5.0 against their 5.0–6.5, and a model trained purely on synthetic
rows scoring AUROC **0.800** against the real model's 0.827 on the same held-out
patients.

It still buys nothing. Across 20 paired comparisons no augmented arm beats the
real-only baseline — best of six is **−0.0006 AUPRC at 10/20 wins** — and the
largest multiplier is the worst arm. Generating whole *patients*, class-balanced,
removes the fair objection that patient-hours cannot raise the positive-**subject**
count: it takes that count from 39 per fold to **539**, fourteen times over, and
AUPRC moves **−0.0337**. Fidelity was never the binding constraint.

Two findings the paper does not lead you to expect:

- **The paradigm it demonstrates breaks on a 4% event rate.** Nonconditional
  training collapses the deterioration label to 0.0041 early on — too few
  positives to fit a classifier. Conditional training holds the rate exactly.
- **Privacy runs the other way.** Membership-inference F1 reaches **0.84**
  against a 0.51 chance floor and 1.0 for publishing the real records, rising
  with training length. [`privacy_control.py`](synthetic/privacy_control.py)
  rules out the confound: refitting on the held-out group swings the verdict on
  identical targets by +0.66. **The synthetic cohort is not de-identified and
  must not leave the project** — the one use the method exists for.

Full write-up, including the paper's use-case weight profiles:
[`synthetic/report.md`](synthetic/report.md).

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
