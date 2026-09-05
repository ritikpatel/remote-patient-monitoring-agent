# Concept build status

65/65 concepts built successfully (0 failed) on the MIMIC-IV Clinical Database Demo (100 patients, 140 ICU stays).

| Phase | Concept | Status | Rows | Note |
|---|---|---|---|---|
| demographics | `icustay_times` | ok | 140 |  |
| demographics | `icustay_hourly` | ok | 15,615 |  |
| demographics | `weight_durations` | ok | 578 |  |
| measurement | `urine_output` | ok | 7,317 |  |
| organfailure | `kdigo_uo` | ok | 7,317 |  |
| demographics | `age` | ok | 275 |  |
| demographics | `icustay_detail` | ok | 140 |  |
| measurement | `bg` | ok | 889 |  |
| measurement | `blood_differential` | ok | 2,763 |  |
| measurement | `cardiac_marker` | ok | 283 |  |
| measurement | `chemistry` | ok | 3,289 |  |
| measurement | `coagulation` | ok | 1,630 |  |
| measurement | `complete_blood_count` | ok | 2,959 |  |
| measurement | `creatinine_baseline` | ok | 275 |  |
| measurement | `enzyme` | ok | 1,411 |  |
| measurement | `gcs` | ok | 3,279 |  |
| measurement | `height` | ok | 69 |  |
| measurement | `icp` | ok | 303 |  |
| measurement | `inflammation` | ok | 42 |  |
| measurement | `oxygen_delivery` | ok | 1,154 |  |
| measurement | `rhythm` | ok | 12,439 |  |
| measurement | `urine_output_rate` | ok | 7,317 |  |
| measurement | `ventilator_setting` | ok | 2,064 |  |
| measurement | `vitalsign` | ok | 21,086 |  |
| comorbidity | `charlson` | ok | 275 |  |
| medication | `acei` | ok | 107 |  |
| medication | `antibiotic` | ok | 903 |  |
| medication | `arb` | ok | 35 |  |
| medication | `dobutamine` | ok | 44 |  |
| medication | `dopamine` | ok | 28 |  |
| medication | `epinephrine` | ok | 36 |  |
| medication | `milrinone` | ok | 15 |  |
| medication | `neuroblock` | ok | 0 |  |
| medication | `norepinephrine` | ok | 947 |  |
| medication | `nsaid` | ok | 202 |  |
| medication | `phenylephrine` | ok | 625 |  |
| medication | `vasopressin` | ok | 55 |  |
| treatment | `code_status` | ok | 389 |  |
| treatment | `crrt` | ok | 580 |  |
| treatment | `invasive_line` | ok | 216 |  |
| treatment | `rrt` | ok | 5,134 |  |
| treatment | `ventilation` | ok | 209 |  |
| firstday | `first_day_bg` | ok | 140 |  |
| firstday | `first_day_bg_art` | ok | 140 |  |
| firstday | `first_day_gcs` | ok | 140 |  |
| firstday | `first_day_height` | ok | 140 |  |
| firstday | `first_day_lab` | ok | 140 |  |
| firstday | `first_day_rrt` | ok | 140 |  |
| firstday | `first_day_urine_output` | ok | 140 |  |
| firstday | `first_day_vitalsign` | ok | 140 |  |
| firstday | `first_day_weight` | ok | 140 |  |
| organfailure | `kdigo_creatinine` | ok | 1,272 |  |
| organfailure | `meld` | ok | 140 |  |
| score | `apsiii` | ok | 140 |  |
| score | `lods` | ok | 140 |  |
| score | `oasis` | ok | 140 |  |
| score | `sapsii` | ok | 140 |  |
| score | `sirs` | ok | 140 |  |
| score | `sofa` | ok | 12,255 |  |
| sepsis | `suspicion_of_infection` | ok | 903 |  |
| organfailure | `kdigo_stages` | ok | 8,729 |  |
| firstday | `first_day_sofa` | ok | 140 |  |
| sepsis | `sepsis3` | ok | 62 |  |
| medication | `vasoactive_agent` | ok | 1,874 |  |
| medication | `norepinephrine_equivalent_dose` | ok | 1,748 |  |

## Must-work concepts (PROJECT_PLAN.md section 7, item 4)

- `icustay_detail` (icustay_detail): ok, 140 rows
- `vitalsign` (vitalsign): ok, 21,086 rows
- `bg` (bg): ok, 889 rows
- `chemistry` (chemistry): ok, 3,289 rows
- `cbc` (complete_blood_count): ok, 2,959 rows
- `coagulation` (coagulation): ok, 1,630 rows
- `ventilation` (ventilation): ok, 209 rows
- `norepinephrine_equivalent_dose` (norepinephrine_equivalent_dose): ok, 1,748 rows
- `urine_output` (urine_output): ok, 7,317 rows
- `kdigo_stages` (kdigo_stages): ok, 8,729 rows
- `sofa` (sofa): ok, 12,255 rows
- `sapsii` (sapsii): ok, 140 rows
- `oasis` (oasis): ok, 140 rows
- `sirs` (sirs): ok, 140 rows
- `charlson` (charlson): ok, 275 rows
- `suspicion_of_infection` (suspicion_of_infection): ok, 903 rows
- `sepsis3` (sepsis3): ok, 62 rows
