# Phase 5 -- predictive models: results

> This platform is validated on a 100-patient demo subset of MIMIC-IV. Clinical narrative is LLM-generated from structured data. Wearable deterioration signals are synthetically morphed from healthy-volunteer recordings. The engineering is real and the methodology is rigorous, and the clinical performance figures below demonstrate pipeline validity -- they do not transfer to clinical practice (PROJECT_PLAN.md section 17).


## Horizon: 6h

- At-risk patient-hours (after R1 censoring at first event): 2979
- Positives: 120 (4.0%), across 60 distinct stays

| model         |   cv_mean_auroc |   cv_mean_auprc |   auroc_point |   auroc_lo |   auroc_hi |   auprc_point |   auprc_lo |   auprc_hi |    brier |
|:--------------|----------------:|----------------:|--------------:|-----------:|-----------:|--------------:|-----------:|-----------:|---------:|
| news2         |          0.6760 |          0.1290 |        0.6729 |     0.5816 |     0.7561 |        0.0953 |     0.0487 |     0.1873 | nan      |
| sofa          |          0.7121 |          0.1745 |        0.7107 |     0.6065 |     0.7962 |        0.1061 |     0.0622 |     0.2445 | nan      |
| age_vitals_lr |          0.6641 |          0.1317 |        0.6446 |     0.5264 |     0.7574 |        0.0752 |     0.0370 |     0.1731 |   0.2418 |
| logistic_full |          0.8167 |          0.3469 |        0.8168 |     0.7359 |     0.9027 |        0.3316 |     0.2305 |     0.4711 |   0.0937 |
| lightgbm      |          0.8198 |          0.5018 |        0.8234 |     0.7269 |     0.9165 |        0.4930 |     0.3763 |     0.6319 |   0.0331 |
| gru           |          0.8008 |          0.2713 |        0.7996 |     0.7028 |     0.9055 |        0.2645 |     0.1831 |     0.3848 |   0.1305 |


## Horizon: 12h

- At-risk patient-hours (after R1 censoring at first event): 2979
- Positives: 161 (5.4%), across 60 distinct stays

| model         |   cv_mean_auroc |   cv_mean_auprc |   auroc_point |   auroc_lo |   auroc_hi |   auprc_point |   auprc_lo |   auprc_hi |    brier |
|:--------------|----------------:|----------------:|--------------:|-----------:|-----------:|--------------:|-----------:|-----------:|---------:|
| news2         |          0.6590 |          0.1493 |        0.6566 |     0.5536 |     0.7555 |        0.1059 |     0.0538 |     0.2054 | nan      |
| sofa          |          0.7548 |          0.2502 |        0.7567 |     0.6501 |     0.8400 |        0.1562 |     0.0839 |     0.3433 | nan      |
| age_vitals_lr |          0.6736 |          0.1852 |        0.6803 |     0.5645 |     0.7906 |        0.0962 |     0.0491 |     0.2212 |   0.2238 |
| logistic_full |          0.7877 |          0.3134 |        0.8118 |     0.7393 |     0.9006 |        0.3130 |     0.2306 |     0.4448 |   0.1054 |
| lightgbm      |          0.7883 |          0.3980 |        0.7956 |     0.7144 |     0.8944 |        0.3736 |     0.2872 |     0.5108 |   0.0524 |


## Fairness audit (review finding F4)

`gender` used to rank third by mean |SHAP|, above most vitals, with no subgroup analysis anywhere in the project. In this cohort the association is real -- 66.2% of male ICU stays reach a composite event against 42.9% of female (Fisher OR 2.62, p=0.007) -- but that is a 100-patient sample, and an effect that size in 140 stays is what sampling noise looks like.

**Ablation** (same model, same folds, 20 repeats): AUPRC **0.5018** with `gender` against **0.4891** without (+0.0127), winning in **16/20** repeats (the bar is 75%).

Decision: **kept**, reversing F4. Two things must be said plainly alongside that.

First, **what moved was the evaluation, not the evidence.** F4 measured 13/20 under CV grouped by `stay_id`. Correcting the grouping to `subject_id` (see `ml/evaluation/reliability_report.md`) moved that comparison to **15/20** -- exactly the bar -- and that is the number the reversal was decided on. The 16/20 above is a *further* comparison, run on the variant that carrying `gender` made the winner; it is not the 15/20 measurement improving. No new information about `gender` arrived at any point in that sequence. A criterion that crosses its own threshold because a grouping bug was fixed is a weak instrument, and the delta (+0.0127) is far inside a bootstrap CI many times its size.

Second, **the subgroup audit below is now load-bearing rather than diagnostic.** The model uses the attribute it is audited on, so the table is no longer a check that a protected characteristic stayed out of the model -- it is the only thing standing between the model and an unequal distribution of errors it is now free to learn. `age` and `first_careunit` remain on their own footing: a validated severity covariate and clinical context respectively, not proxies.


**Subgroup performance** at a single shared alert threshold (top decile of scores, p=0.086). One threshold applied to every subgroup on purpose: a model can be equally accurate overall and still distribute its errors unequally. Subgroups whose 95% CI is wider than 0.40 have their point estimates **withheld**: an interval that wide is consistent with a useless model and an excellent one at once, so publishing the midpoint manufactures a finding the data cannot support. The bootstrap resamples patients, not rows -- rows from one patient are not independent, and a row-level bootstrap reports a CI far narrower than the data earns.

| dimension    | subgroup                                         |   n_rows |   n_positives |   event_rate |   alert_rate |    auroc |    auprc |   auroc_lo |   auroc_hi |   ci_width | uninformative   |
|:-------------|:-------------------------------------------------|---------:|--------------:|-------------:|-------------:|---------:|---------:|-----------:|-----------:|-----------:|:----------------|
| age_band     | 50-64                                            |      952 |            37 |       0.0389 |       0.1261 |   0.9013 |   0.5120 |     0.8060 |     0.9835 |     0.1775 | False           |
| age_band     | 65-79                                            |      638 |            37 |       0.0580 |       0.0815 |   0.8353 |   0.6105 |     0.6858 |     0.9851 |     0.2993 | False           |
| age_band     | 80+                                              |      832 |            24 |       0.0288 |       0.0769 | nan      | nan      |     0.5569 |     0.9648 |     0.4079 | True            |
| age_band     | <50                                              |      557 |            22 |       0.0395 |       0.1113 | nan      | nan      |     0.4541 |     1.0000 |     0.5459 | True            |
| care_unit    | Cardiac Vascular Intensive Care Unit (CVICU)     |      305 |            23 |       0.0754 |       0.1311 |   0.9870 |   0.8950 |     0.9655 |     1.0000 |     0.0345 | False           |
| care_unit    | Coronary Care Unit (CCU)                         |      275 |            19 |       0.0691 |       0.1055 |   0.8988 |   0.5270 |     0.7500 |     0.9699 |     0.2200 | False           |
| care_unit    | Medical Intensive Care Unit (MICU)               |      571 |            24 |       0.0420 |       0.0998 |   0.8397 |   0.5267 |     0.6381 |     0.9991 |     0.3609 | False           |
| care_unit    | Medical/Surgical Intensive Care Unit (MICU/SICU) |      461 |            22 |       0.0477 |       0.1540 | nan      | nan      |     0.3270 |     0.8939 |     0.5669 | True            |
| care_unit    | Neuro Intermediate                               |       19 |             0 |       0.0000 |       0.0526 | nan      | nan      |   nan      |   nan      |   nan      | True            |
| care_unit    | Neuro Stepdown                                   |      169 |             0 |       0.0000 |       0.0059 | nan      | nan      |   nan      |   nan      |   nan      | True            |
| care_unit    | Neuro Surgical Intensive Care Unit (Neuro SICU)  |      121 |             3 |       0.0248 |       0.0496 | nan      | nan      |   nan      |   nan      |   nan      | True            |
| care_unit    | Surgical Intensive Care Unit (SICU)              |      605 |            12 |       0.0198 |       0.0463 |   0.9903 |   0.8818 |     0.9662 |     1.0000 |     0.0338 | False           |
| care_unit    | Trauma SICU (TSICU)                              |      453 |            17 |       0.0375 |       0.1435 | nan      | nan      |     0.1763 |     0.8979 |     0.7216 | True            |
| sex          | F                                                |     1960 |            32 |       0.0163 |       0.0694 |   0.8076 |   0.3941 |     0.6449 |     0.9726 |     0.3277 | False           |
| sex          | M                                                |     1019 |            88 |       0.0864 |       0.1590 |   0.8140 |   0.5587 |     0.7236 |     0.9197 |     0.1961 | False           |
| time_in_stay | hour 0-5                                         |      497 |            78 |       0.1569 |       0.3360 |   0.8876 |   0.7174 |     0.8354 |     0.9408 |     0.1055 | False           |
| time_in_stay | hour 24+                                         |     1377 |            15 |       0.0109 |       0.0508 | nan      | nan      |     0.5806 |     0.9889 |     0.4083 | True            |
| time_in_stay | hour 6-23                                        |     1105 |            27 |       0.0244 |       0.0552 | nan      | nan      |     0.3583 |     0.7649 |     0.4066 | True            |


## Verification (PROJECT_PLAN.md section 15)

LightGBM beats recalibrated NEWS2 on AUPRC in **20/20** repeats at the primary 6h horizon (meets the >=15/20 bar).

## Top SHAP features (promoted model)

|                           |   mean_abs_shap |
|:--------------------------|----------------:|
| hr                        |          0.8461 |
| first_careunit            |          0.5424 |
| gender                    |          0.4232 |
| rr_24h_std                |          0.3531 |
| admission_age             |          0.3474 |
| map_24h_std               |          0.3421 |
| sbp                       |          0.3353 |
| gcs_total                 |          0.2924 |
| sbp_24h_std               |          0.2622 |
| hr_24h_slope              |          0.2329 |
| rr_24h_slope              |          0.2241 |
| spo2_24h_slope            |          0.2226 |
| temp_c_24h_std            |          0.2176 |
| temp_c_4h_slope           |          0.2149 |
| fio2_hours_since_last_obs |          0.1956 |
