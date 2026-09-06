# Phase 5 -- predictive models: results

> This platform is validated on a 100-patient demo subset of MIMIC-IV. Clinical narrative is LLM-generated from structured data. Wearable deterioration signals are synthetically morphed from healthy-volunteer recordings. The engineering is real and the methodology is rigorous, and the clinical performance figures below demonstrate pipeline validity -- they do not transfer to clinical practice (PROJECT_PLAN.md section 17).


## Horizon: 6h

- At-risk patient-hours (after R1 censoring at first event): 2979
- Positives: 120 (4.0%), across 60 distinct stays
- ECG-fusion delta on AUPRC (with ECG - without): +0.0144

| model         |   cv_mean_auroc |   cv_mean_auprc |   auroc_point |   auroc_lo |   auroc_hi |   auprc_point |   auprc_lo |   auprc_hi |    brier |
|:--------------|----------------:|----------------:|--------------:|-----------:|-----------:|--------------:|-----------:|-----------:|---------:|
| news2         |          0.6769 |          0.1333 |        0.6729 |     0.5847 |     0.7583 |        0.0953 |     0.0514 |     0.1904 | nan      |
| sofa          |          0.7097 |          0.1764 |        0.7107 |     0.6115 |     0.7937 |        0.1061 |     0.0581 |     0.2537 | nan      |
| age_vitals_lr |          0.6545 |          0.1150 |        0.6736 |     0.5678 |     0.7714 |        0.0936 |     0.0460 |     0.2122 |   0.2184 |
| logistic_full |          0.8524 |          0.3898 |        0.8438 |     0.7486 |     0.9309 |        0.3581 |     0.2342 |     0.5546 |   0.0677 |
| lightgbm      |          0.8644 |          0.4895 |        0.8550 |     0.7643 |     0.9410 |        0.4479 |     0.3232 |     0.6394 |   0.0352 |
| lightgbm_ecg  |          0.8515 |          0.5065 |        0.8247 |     0.7304 |     0.9235 |        0.4623 |     0.3354 |     0.6435 |   0.0317 |
| gru           |          0.8008 |          0.2713 |        0.7996 |     0.7028 |     0.9055 |        0.2645 |     0.1831 |     0.3848 |   0.1305 |


## Horizon: 12h

- At-risk patient-hours (after R1 censoring at first event): 2979
- Positives: 161 (5.4%), across 60 distinct stays
- ECG-fusion delta on AUPRC (with ECG - without): -0.0496

| model         |   cv_mean_auroc |   cv_mean_auprc |   auroc_point |   auroc_lo |   auroc_hi |   auprc_point |   auprc_lo |   auprc_hi |    brier |
|:--------------|----------------:|----------------:|--------------:|-----------:|-----------:|--------------:|-----------:|-----------:|---------:|
| news2         |          0.6612 |          0.1508 |        0.6566 |     0.5560 |     0.7524 |        0.1059 |     0.0578 |     0.2005 | nan      |
| sofa          |          0.7529 |          0.2821 |        0.7567 |     0.6554 |     0.8349 |        0.1562 |     0.0812 |     0.3493 | nan      |
| age_vitals_lr |          0.6784 |          0.1589 |        0.6291 |     0.5135 |     0.7447 |        0.0809 |     0.0433 |     0.1631 |   0.2288 |
| logistic_full |          0.8607 |          0.3807 |        0.8507 |     0.7796 |     0.9246 |        0.3043 |     0.1981 |     0.4895 |   0.0953 |
| lightgbm      |          0.8828 |          0.4675 |        0.8827 |     0.8110 |     0.9418 |        0.4225 |     0.3211 |     0.5976 |   0.0453 |
| lightgbm_ecg  |          0.8629 |          0.4534 |        0.8386 |     0.7620 |     0.9160 |        0.3729 |     0.2666 |     0.5700 |   0.0473 |


## Verification (PROJECT_PLAN.md section 15)

LightGBM beats recalibrated NEWS2 on AUPRC in **20/20** repeats at the primary 6h horizon (meets the >=15/20 bar).

## Top SHAP features (promoted model)

|                    |   mean_abs_shap |
|:-------------------|----------------:|
| sofa_24hours       |          0.8937 |
| hr_24h_mean        |          0.5764 |
| gender             |          0.4849 |
| rr_24h_std         |          0.4586 |
| hr_4h_mean         |          0.3142 |
| first_careunit     |          0.3011 |
| temp_c_24h_mean    |          0.2256 |
| rr_24h_slope       |          0.2206 |
| ecg_hours_since    |          0.2149 |
| map_24h_std        |          0.2138 |
| spo2_24h_slope     |          0.2041 |
| map_4h_mean        |          0.1925 |
| spo2_24h_mean      |          0.1825 |
| ecg_quality_mean   |          0.1696 |
| gcs_total_24h_mean |          0.1685 |
