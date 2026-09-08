# Phase 5 -- predictive models: results

> This platform is validated on a 100-patient demo subset of MIMIC-IV. Clinical narrative is LLM-generated from structured data. Wearable deterioration signals are synthetically morphed from healthy-volunteer recordings. The engineering is real and the methodology is rigorous, and the clinical performance figures below demonstrate pipeline validity -- they do not transfer to clinical practice (PROJECT_PLAN.md section 17).


## Horizon: 6h

- At-risk patient-hours (after R1 censoring at first event): 2979
- Positives: 120 (4.0%), across 60 distinct stays
- ECG-fusion delta on AUPRC (with ECG - without): +0.0189

| model         |   cv_mean_auroc |   cv_mean_auprc |   auroc_point |   auroc_lo |   auroc_hi |   auprc_point |   auprc_lo |   auprc_hi |    brier |
|:--------------|----------------:|----------------:|--------------:|-----------:|-----------:|--------------:|-----------:|-----------:|---------:|
| news2         |          0.6769 |          0.1333 |        0.6729 |     0.5847 |     0.7583 |        0.0953 |     0.0514 |     0.1904 | nan      |
| sofa          |          0.7097 |          0.1764 |        0.7107 |     0.6115 |     0.7937 |        0.1061 |     0.0581 |     0.2537 | nan      |
| age_vitals_lr |          0.6545 |          0.1150 |        0.6736 |     0.5678 |     0.7714 |        0.0936 |     0.0460 |     0.2122 |   0.2184 |
| logistic_full |          0.8353 |          0.3642 |        0.8313 |     0.7433 |     0.9193 |        0.3408 |     0.2285 |     0.5060 |   0.0725 |
| lightgbm      |          0.8418 |          0.4842 |        0.8273 |     0.7397 |     0.9219 |        0.4529 |     0.3254 |     0.6308 |   0.0330 |
| lightgbm_ecg  |          0.8225 |          0.4953 |        0.8186 |     0.7289 |     0.9188 |        0.4718 |     0.3440 |     0.6446 |   0.0309 |
| gru           |          0.8008 |          0.2713 |        0.7996 |     0.7028 |     0.9055 |        0.2645 |     0.1831 |     0.3848 |   0.1305 |


## Horizon: 12h

- At-risk patient-hours (after R1 censoring at first event): 2979
- Positives: 161 (5.4%), across 60 distinct stays
- ECG-fusion delta on AUPRC (with ECG - without): -0.0716

| model         |   cv_mean_auroc |   cv_mean_auprc |   auroc_point |   auroc_lo |   auroc_hi |   auprc_point |   auprc_lo |   auprc_hi |    brier |
|:--------------|----------------:|----------------:|--------------:|-----------:|-----------:|--------------:|-----------:|-----------:|---------:|
| news2         |          0.6612 |          0.1508 |        0.6566 |     0.5560 |     0.7524 |        0.1059 |     0.0578 |     0.2005 | nan      |
| sofa          |          0.7529 |          0.2821 |        0.7567 |     0.6554 |     0.8349 |        0.1562 |     0.0812 |     0.3493 | nan      |
| age_vitals_lr |          0.6784 |          0.1589 |        0.6291 |     0.5135 |     0.7447 |        0.0809 |     0.0433 |     0.1631 |   0.2288 |
| logistic_full |          0.8301 |          0.3401 |        0.8055 |     0.7128 |     0.8986 |        0.2484 |     0.1551 |     0.4282 |   0.1061 |
| lightgbm      |          0.8586 |          0.4334 |        0.8648 |     0.7942 |     0.9286 |        0.4171 |     0.3123 |     0.5976 |   0.0473 |
| lightgbm_ecg  |          0.8351 |          0.4239 |        0.8039 |     0.7135 |     0.8975 |        0.3456 |     0.2399 |     0.5380 |   0.0521 |


## Fairness audit (review finding F4)

`gender` used to rank third by mean |SHAP|, above most vitals, with no subgroup analysis anywhere in the project. In this cohort the association is real -- 66.2% of male ICU stays reach a composite event against 42.9% of female (Fisher OR 2.62, p=0.007) -- but that is a 100-patient sample, and an effect that size in 140 stays is what sampling noise looks like.

**Ablation** (same model, same folds, 20 repeats): AUPRC **0.5065** with `gender` against **0.4953** without (+0.0112), winning in only **13/20** repeats.

Decision: **dropped**. A delta this far inside the bootstrap CI, winning barely more often than a coin flip, does not justify carrying a protected attribute into a clinical model. `gender` is excluded from the feature set (`engineer.feature_matrix_for_training(include_demographics=False)`, the default); age and first care unit are kept, being a validated severity covariate and clinical context respectively, not proxies.


**Subgroup performance** at a single shared alert threshold (top decile of scores, p=0.066). One threshold applied to every subgroup on purpose: a model can be equally accurate overall and still distribute its errors unequally. Subgroups whose 95% CI is wider than 0.40 have their point estimates **withheld**: an interval that wide is consistent with a useless model and an excellent one at once, so publishing the midpoint manufactures a finding the data cannot support. The bootstrap resamples patients, not rows -- rows from one patient are not independent, and a row-level bootstrap reports a CI far narrower than the data earns.

| dimension    | subgroup                                         |   n_rows |   n_positives |   event_rate |   alert_rate |    auroc |    auprc |   auroc_lo |   auroc_hi |   ci_width | uninformative   |
|:-------------|:-------------------------------------------------|---------:|--------------:|-------------:|-------------:|---------:|---------:|-----------:|-----------:|-----------:|:----------------|
| age_band     | 50-64                                            |      952 |            37 |       0.0389 |       0.1628 |   0.8494 |   0.5070 |     0.7161 |     0.9959 |     0.2798 | False           |
| age_band     | 65-79                                            |      638 |            37 |       0.0580 |       0.0768 |   0.8411 |   0.5906 |     0.6698 |     0.9836 |     0.3138 | False           |
| age_band     | 80+                                              |      832 |            24 |       0.0288 |       0.0781 | nan      | nan      |     0.5052 |     0.9870 |     0.4818 | True            |
| age_band     | <50                                              |      557 |            22 |       0.0395 |       0.0521 |   0.8433 |   0.6256 |     0.6040 |     0.9997 |     0.3957 | False           |
| care_unit    | Cardiac Vascular Intensive Care Unit (CVICU)     |      305 |            23 |       0.0754 |       0.1180 |   0.9915 |   0.8874 |     0.9706 |     1.0000 |     0.0294 | False           |
| care_unit    | Coronary Care Unit (CCU)                         |      275 |            19 |       0.0691 |       0.1055 | nan      | nan      |     0.4829 |     0.9811 |     0.4982 | True            |
| care_unit    | Medical Intensive Care Unit (MICU)               |      571 |            24 |       0.0420 |       0.1086 | nan      | nan      |     0.4913 |     0.9989 |     0.5076 | True            |
| care_unit    | Medical/Surgical Intensive Care Unit (MICU/SICU) |      461 |            22 |       0.0477 |       0.0803 | nan      | nan      |     0.5785 |     0.9944 |     0.4159 | True            |
| care_unit    | Neuro Intermediate                               |       19 |             0 |       0.0000 |       0.0000 | nan      | nan      |   nan      |   nan      |   nan      | True            |
| care_unit    | Neuro Stepdown                                   |      169 |             0 |       0.0000 |       0.0118 | nan      | nan      |   nan      |   nan      |   nan      | True            |
| care_unit    | Neuro Surgical Intensive Care Unit (Neuro SICU)  |      121 |             3 |       0.0248 |       0.0248 | nan      | nan      |   nan      |   nan      |   nan      | True            |
| care_unit    | Surgical Intensive Care Unit (SICU)              |      605 |            12 |       0.0198 |       0.0959 |   0.9793 |   0.6597 |     0.9559 |     0.9997 |     0.0438 | False           |
| care_unit    | Trauma SICU (TSICU)                              |      453 |            17 |       0.0375 |       0.1567 | nan      | nan      |     0.2024 |     0.9307 |     0.7283 | True            |
| sex          | F                                                |     1960 |            32 |       0.0163 |       0.0658 |   0.8633 |   0.5008 |     0.6821 |     0.9939 |     0.3119 | False           |
| sex          | M                                                |     1019 |            88 |       0.0864 |       0.1658 |   0.8033 |   0.4931 |     0.6897 |     0.9124 |     0.2227 | False           |
| time_in_stay | hour 0-5                                         |      497 |            78 |       0.1569 |       0.3541 |   0.8991 |   0.7151 |     0.8140 |     0.9604 |     0.1464 | False           |
| time_in_stay | hour 24+                                         |     1377 |            15 |       0.0109 |       0.0370 | nan      | nan      |     0.2952 |     0.7664 |     0.4712 | True            |
| time_in_stay | hour 6-23                                        |     1105 |            27 |       0.0244 |       0.0643 |   0.6681 |   0.0737 |     0.5124 |     0.8775 |     0.3651 | False           |


## Verification (PROJECT_PLAN.md section 15)

LightGBM beats recalibrated NEWS2 on AUPRC in **20/20** repeats at the primary 6h horizon (meets the >=15/20 bar).

## Top SHAP features (promoted model)

|                           |   mean_abs_shap |
|:--------------------------|----------------:|
| sofa_24hours              |          1.0014 |
| hr_24h_mean               |          0.6321 |
| rr_24h_std                |          0.5595 |
| first_careunit            |          0.3745 |
| hr_4h_mean                |          0.3034 |
| rr_24h_slope              |          0.2852 |
| map_24h_std               |          0.2510 |
| sbp_24h_std               |          0.2205 |
| ecg_qrs_ms                |          0.2118 |
| ecg_hours_since           |          0.2043 |
| temp_c_24h_mean           |          0.1899 |
| spo2_24h_slope            |          0.1845 |
| gcs_total_24h_mean        |          0.1677 |
| fio2_hours_since_last_obs |          0.1582 |
| map_4h_mean               |          0.1435 |
