# 30-day readmission, rebuilt on the right cohort and the right question

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice (PROJECT_PLAN.md section 17).

`ml/evaluation/secondary_whole_stay.py` reports **AUROC 0.452** for this task and
attributes it to sample size. This study holds the protocol fixed and changes two
things: the cohort, and the feature set.

## 1. Cohort

Readmission is an outcome of a *hospitalisation*, not of an ICU stay. The previous
cohort was the 140 ICU stays' own admissions; the matching cohort is every live
discharge, less the CMS-style competing-risk exclusions.

| cohort                                       |   admissions |   readmitted_30d |   rate |
|:---------------------------------------------|-------------:|-----------------:|-------:|
| all live discharges                          |          260 |               53 |   20.4 |
| less hospice                                 |          255 |               53 |   20.8 |
| less died <=30d before readmission (SHIPPED) |          252 |               53 |   21   |
| previous cohort: ICU admissions only         |          113 |               22 |   19.5 |

Shipped: **252 admissions, 53 positives
(21.0%)** across 92 patients — **2.4x the positives** of the
previous cohort, from the same database. Two exclusions, both competing risks rather
than negatives: patients discharged to **hospice** were never readmission candidates,
and patients who **died within 30 days without being readmitted** could not have been.

**The censoring this cannot fix.** 77
(30.6%) of these are the last admission recorded for
that patient. MIMIC shifts dates per patient, so there is no global observation window
to censor against, and "no later admission in the dataset" is the only signal available.
Those rows are treated as negatives — standard practice, and a known source of
one-directional label noise that no protocol removes.

**Grouping.** 92 patients contribute 252 admissions
(mean 2.7, max
20). Every fold is grouped on `subject_id`: a
row-level split would put the same patient on both sides, and `n_prior_admissions` would
become close to a patient identifier.

## 2. Feature set

The previous eight columns are all ICU physiology. The new set is care-transition
facts, every one of them known at the prediction time — which is **discharge**. Finding
E22 matters here: the ICD code that is leak-suspect for the hourly model is entirely
legitimate for this one, because coding *precedes* this prediction rather than following
it.

| family | columns |
|---|---|
| `demographics` | `age`, `gender`, `insurance`, `marital_status` |
| `index_stay` | `los_days`, `admission_type`, `admission_location`, `came_via_ed`, `had_icu_stay`, `icu_los_days` |
| `disposition` | `discharge_location` |
| `history` | `n_prior_admissions`, `days_since_last_discharge`, `n_prior_admissions_365d` |
| `comorbidity` | `charlson_comorbidity_index`, `congestive_heart_failure`, `chronic_pulmonary_disease`, `renal_disease`, `diabetes_with_cc`, `malignant_cancer` |
| `diagnosis` | `dx_chapter`, `n_diagnoses` |
| `treatment_intensity` | `n_procedures`, `n_distinct_drugs` |

## 3. Result

Identical rows, identical folds, 20 repeats x 5 folds, subject-grouped.

| model                     |   cv_mean_auroc |   cv_mean_auprc |   auroc_point |   auroc_lo |   auroc_hi |   auprc_point |   auprc_lo |   auprc_hi |   lift_over_base_rate |
|:--------------------------|----------------:|----------------:|--------------:|-----------:|-----------:|--------------:|-----------:|-----------:|----------------------:|
| icu_physiology (previous) |          0.4757 |          0.2387 |        0.4525 |     0.3345 |     0.5394 |        0.2014 |     0.1168 |     0.2890 |                1.1349 |
| discharge_context (new)   |          0.5215 |          0.2687 |        0.4865 |     0.3832 |     0.6087 |        0.2135 |     0.1373 |     0.3269 |                1.2775 |
| lace_plus (parsimonious)  |          0.5455 |          0.2815 |        0.5132 |     0.4115 |     0.5896 |        0.2254 |     0.1354 |     0.3145 |                1.3385 |

**Two findings, and they point in different directions. Both belong in the report.**

**The previous diagnosis of the cause was wrong.** Sample size was blamed; the feature set was the larger problem. With the same protocol, the same rows and the same folds, changing only the features moves AUROC from 0.476 to **0.546** and AUPRC from 0.239 to **0.282** (1.34x the 21.0% base rate), winning **19 of 20** paired repeats. A 19/20 paired result is not noise: asking the question from care-transition facts rather than ICU physiology reliably helps.

**The previous conclusion nevertheless stands.** The improvement is real and it is not large enough to produce a usable model. Note also that the parsimonious 7-column LACE+ set beats the full 24-column one (0.2815 against 0.2687) -- the same over-parameterisation lesson `feature_pruning.py` and `disease_leakage.py` already drew, now on a third task with 53 positives.

The 95% bootstrap interval on AUROC is **0.412-0.590**, which still includes 0.5. **Better-than-chance discrimination is not established at this sample size**, even with the corrected cohort and framing. The point estimate moved and the paired comparison is decisive, but the interval did not shrink enough to support a standalone claim about this model's accuracy. The defensible statements are: the feature set matters (paired evidence), and the cohort is too small to certify the result (interval evidence). Reporting the first without the second would be the error this project exists to avoid.

## 4. Which information carries the task

Leave-one-family-out, most costly to remove first:

| removed_family      |   n_columns |   mean_auprc |   mean_auroc |   delta_vs_full |
|:--------------------|------------:|-------------:|-------------:|----------------:|
| demographics        |           4 |       0.2571 |       0.4973 |         -0.0116 |
| treatment_intensity |           2 |       0.2634 |       0.5364 |         -0.0052 |
| disposition         |           1 |       0.2657 |       0.5165 |         -0.0030 |
| comorbidity         |           6 |       0.2666 |       0.5151 |         -0.0021 |
| diagnosis           |           2 |       0.2774 |       0.5364 |          0.0087 |
| history             |           3 |       0.2892 |       0.5363 |          0.0205 |
| index_stay          |           6 |       0.2927 |       0.5371 |          0.0240 |

Removing **`demographics`** costs the most
(-0.0116 AUPRC), which is the concrete, actionable output of
this study: it names the information a discharge-planning workflow would need to
collect.

## 5. What this does not claim

**Not a readmission-rate reduction.** Predicting readmission is not reducing it — that
requires an intervention and a control arm, and this project has neither. The honest
claim is a discrimination figure at a stated base rate on a 100-patient cohort, with
30.6% of labels right-censored.

**Not a replacement for `secondary_whole_stay.py`'s in-hospital mortality arm**, which
is a different outcome on a different cohort and is untouched by this.
