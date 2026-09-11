# Phase 5 -- predictive models: results

> This platform is validated on a 100-patient demo subset of MIMIC-IV. Clinical narrative is LLM-generated from structured data. Post-discharge signals are real MIMIC physiology passed through a simulated home sensor layer, and the post-discharge model is a transfer from ICU data with no post-discharge labels. The engineering is real and the methodology is rigorous, and the clinical performance figures below demonstrate pipeline validity -- they do not transfer to clinical practice (PROJECT_PLAN.md section 17).


## Horizon: 6h

- At-risk patient-hours (after R1 censoring at first event): 2979
- Positives: 120 (4.0%), across 60 distinct stays

| model         |   cv_mean_auroc |   cv_mean_auprc |   auroc_point |   auroc_lo |   auroc_hi |   auprc_point |   auprc_lo |   auprc_hi |    brier |
|:--------------|----------------:|----------------:|--------------:|-----------:|-----------:|--------------:|-----------:|-----------:|---------:|
| news2         |          0.6760 |          0.1290 |        0.6729 |     0.5816 |     0.7561 |        0.0953 |     0.0487 |     0.1873 | nan      |
| sofa          |          0.7121 |          0.1745 |        0.7107 |     0.6065 |     0.7962 |        0.1061 |     0.0622 |     0.2445 | nan      |
| age_vitals_lr |          0.6641 |          0.1317 |        0.6446 |     0.5264 |     0.7574 |        0.0752 |     0.0370 |     0.1731 |   0.2418 |
| logistic_full |          0.7862 |          0.3123 |        0.8038 |     0.7301 |     0.8814 |        0.2737 |     0.1798 |     0.4343 |   0.1112 |
| lightgbm      |          0.8448 |          0.4879 |        0.8743 |     0.8112 |     0.9332 |        0.4759 |     0.3704 |     0.6036 |   0.0324 |
| gru           |          0.8008 |          0.2713 |        0.7996 |     0.7028 |     0.9055 |        0.2645 |     0.1831 |     0.3848 |   0.1305 |


## Horizon: 12h

- At-risk patient-hours (after R1 censoring at first event): 2979
- Positives: 161 (5.4%), across 60 distinct stays

| model         |   cv_mean_auroc |   cv_mean_auprc |   auroc_point |   auroc_lo |   auroc_hi |   auprc_point |   auprc_lo |   auprc_hi |    brier |
|:--------------|----------------:|----------------:|--------------:|-----------:|-----------:|--------------:|-----------:|-----------:|---------:|
| news2         |          0.6590 |          0.1493 |        0.6566 |     0.5536 |     0.7555 |        0.1059 |     0.0538 |     0.2054 | nan      |
| sofa          |          0.7548 |          0.2502 |        0.7567 |     0.6501 |     0.8400 |        0.1562 |     0.0839 |     0.3433 | nan      |
| age_vitals_lr |          0.6736 |          0.1852 |        0.6803 |     0.5645 |     0.7906 |        0.0962 |     0.0491 |     0.2212 |   0.2238 |
| logistic_full |          0.7769 |          0.2781 |        0.7794 |     0.6830 |     0.8895 |        0.2272 |     0.1455 |     0.3693 |   0.1302 |
| lightgbm      |          0.8196 |          0.4164 |        0.8443 |     0.7789 |     0.9171 |        0.3789 |     0.2932 |     0.5246 |   0.0515 |


## Fairness audit (review finding F4)

`gender` used to rank third by mean |SHAP|, above most vitals, with no subgroup analysis anywhere in the project. In this cohort the association is real -- 66.2% of male ICU stays reach a composite event against 42.9% of female (Fisher OR 2.62, p=0.007) -- but that is a 100-patient sample, and an effect that size in 140 stays is what sampling noise looks like.

**Ablation** (same model, same folds, 20 repeats): AUPRC **0.4879** with `gender` against **0.4819** without (+0.0060), winning in **13/20** repeats (the bar is 75%).

Decision: **dropped**. A delta this far inside the bootstrap CI, winning barely more often than a coin flip, does not justify carrying a protected attribute into a clinical model. `gender` is excluded from the feature set (`engineer.feature_matrix_for_training(include_demographics=False)`); age and first care unit are kept, being a validated severity covariate and clinical context respectively, not proxies.


**Subgroup performance** at a single shared alert threshold (top decile of scores, p=0.067). One threshold applied to every subgroup on purpose: a model can be equally accurate overall and still distribute its errors unequally. Subgroups whose 95% CI is wider than 0.40 have their point estimates **withheld**: an interval that wide is consistent with a useless model and an excellent one at once, so publishing the midpoint manufactures a finding the data cannot support. The bootstrap resamples patients, not rows -- rows from one patient are not independent, and a row-level bootstrap reports a CI far narrower than the data earns.

| dimension    | subgroup                                         |   n_rows |   n_positives |   event_rate |   alert_rate |    auroc |    auprc |   auroc_lo |   auroc_hi |   ci_width | uninformative   |
|:-------------|:-------------------------------------------------|---------:|--------------:|-------------:|-------------:|---------:|---------:|-----------:|-----------:|-----------:|:----------------|
| age_band     | 50-64                                            |      952 |            37 |       0.0389 |       0.1271 |   0.9111 |   0.5088 |     0.8456 |     0.9719 |     0.1263 | False           |
| age_band     | 65-79                                            |      638 |            37 |       0.0580 |       0.1113 |   0.7939 |   0.5931 |     0.6441 |     0.9584 |     0.3143 | False           |
| age_band     | 80+                                              |      832 |            24 |       0.0288 |       0.0841 |   0.9170 |   0.2435 |     0.8468 |     0.9791 |     0.1323 | False           |
| age_band     | <50                                              |      557 |            22 |       0.0395 |       0.0646 |   0.8709 |   0.5121 |     0.7615 |     0.9997 |     0.2382 | False           |
| care_unit    | Cardiac Vascular Intensive Care Unit (CVICU)     |      305 |            23 |       0.0754 |       0.1279 |   0.9477 |   0.8124 |     0.8747 |     0.9990 |     0.1243 | False           |
| care_unit    | Coronary Care Unit (CCU)                         |      275 |            19 |       0.0691 |       0.1055 |   0.8834 |   0.4199 |     0.8003 |     0.9770 |     0.1767 | False           |
| care_unit    | Medical Intensive Care Unit (MICU)               |      571 |            24 |       0.0420 |       0.1489 |   0.8672 |   0.5115 |     0.7231 |     0.9982 |     0.2752 | False           |
| care_unit    | Medical/Surgical Intensive Care Unit (MICU/SICU) |      461 |            22 |       0.0477 |       0.0998 | nan      | nan      |     0.5665 |     0.9683 |     0.4018 | True            |
| care_unit    | Neuro Intermediate                               |       19 |             0 |       0.0000 |       0.0000 | nan      | nan      |   nan      |   nan      |   nan      | True            |
| care_unit    | Neuro Stepdown                                   |      169 |             0 |       0.0000 |       0.0059 | nan      | nan      |   nan      |   nan      |   nan      | True            |
| care_unit    | Neuro Surgical Intensive Care Unit (Neuro SICU)  |      121 |             3 |       0.0248 |       0.0248 | nan      | nan      |   nan      |   nan      |   nan      | True            |
| care_unit    | Surgical Intensive Care Unit (SICU)              |      605 |            12 |       0.0198 |       0.0727 |   0.9517 |   0.7573 |     0.8505 |     0.9998 |     0.1494 | False           |
| care_unit    | Trauma SICU (TSICU)                              |      453 |            17 |       0.0375 |       0.1126 |   0.7103 |   0.0880 |     0.4970 |     0.8927 |     0.3957 | False           |
| sex          | F                                                |     1960 |            32 |       0.0163 |       0.0541 |   0.8771 |   0.3259 |     0.7997 |     0.9669 |     0.1671 | False           |
| sex          | M                                                |     1019 |            88 |       0.0864 |       0.1884 |   0.8472 |   0.5502 |     0.7637 |     0.9269 |     0.1632 | False           |
| time_in_stay | hour 0-5                                         |      497 |            78 |       0.1569 |       0.3461 |   0.8740 |   0.6569 |     0.8194 |     0.9258 |     0.1065 | False           |
| time_in_stay | hour 24+                                         |     1377 |            15 |       0.0109 |       0.0414 |   0.8918 |   0.1355 |     0.7681 |     0.9982 |     0.2301 | False           |
| time_in_stay | hour 6-23                                        |     1105 |            27 |       0.0244 |       0.0624 |   0.7062 |   0.1292 |     0.5341 |     0.8882 |     0.3541 | False           |


## Verification (PROJECT_PLAN.md section 15)

LightGBM beats recalibrated NEWS2 on AUPRC in **20/20** repeats at the primary 6h horizon (meets the >=15/20 bar).

## Top SHAP features (promoted model)

|                            |   mean_abs_shap |
|:---------------------------|----------------:|
| charlson_comorbidity_index |          0.8315 |
| hr                         |          0.6858 |
| rr_24h_std                 |          0.4352 |
| sbp                        |          0.3929 |
| first_careunit             |          0.3000 |
| spo2_24h_slope             |          0.2961 |
| map_24h_std                |          0.2909 |
| sbp_24h_std                |          0.2505 |
| gender                     |          0.2376 |
| temp_c_24h_std             |          0.2289 |
| gcs_total                  |          0.2288 |
| fio2_hours_since_last_obs  |          0.2146 |
| temp_c_4h_slope            |          0.2067 |
| rr_24h_slope               |          0.1919 |
| rr_4h_slope                |          0.1735 |
