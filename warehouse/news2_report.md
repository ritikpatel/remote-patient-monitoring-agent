# NEWS2 -- ward-standard vs. ICU-recalibrated

Computed on 12,004 patient-hours across 140 ICU stays (`capstone.hourly_grid`).
Component scoring is the unmodified NEWS2 (RCP 2017, Scale 1) formula; only the
aggregate escalation cut-points differ between the two columns below (see module
docstring in `warehouse/news2.py` for why).

**This recalibration is cohort-relative (75th/90th percentile of this 140-stay
sample), not a clinically validated threshold.** Per PROJECT_PLAN.md section 17:
this platform is validated on a 100-patient demo subset of MIMIC-IV; the clinical
performance figures demonstrate pipeline validity and do not transfer to clinical
practice.

## Ward-standard (medium >= 5, high >= 7)

|        |   pct_patient_hours |   stays_ever_at_least_this_tier |
|:-------|--------------------:|--------------------------------:|
| low    |                26   |                             140 |
| medium |                25.2 |                             128 |
| high   |                48.8 |                             105 |

## ICU-recalibrated (medium >= 8, high >= 10)

|        |   pct_patient_hours |   stays_ever_at_least_this_tier |
|:-------|--------------------:|--------------------------------:|
| low    |                61.3 |                             140 |
| medium |                24.3 |                              97 |
| high   |                14.4 |                              73 |

## Why recalibrate (E5)

128/140 ICU stays (91%) cross the ward
"medium" threshold at some point in the stay. A trigger this population trips this
often carries little discriminating information for an ICU-only alert engine --
everyone here is already sick enough to be in the ICU. The ICU-recalibrated cut-points
instead flag the 25% of this cohort's own
patient-hours with the highest NEWS2, i.e. relative deterioration within an ICU
population rather than absolute deterioration relative to a ward population.

## Per-disease recalibration

E5 fixed the base rate but left one cut-point for every patient: a post-cardiac-surgery
stay and a septic stay escalate on the same number. The same 75th/90th-percentile rule
is therefore evaluated *within* each primary-diagnosis chapter
(`capstone.disease_context`, built by `warehouse/disease.py`).

A chapter gets its own cut-point only if it has at least
**20 stays**; otherwise it falls back to the pooled ICU
numbers above. The guard counts *stays*, not patient-hours, because hours within a
stay are strongly correlated -- a 40-hour stay is much closer to one observation than
to forty, so a row-count bar would wave through chapters with three patients in them.
The `high_ci_lo`/`high_ci_hi` columns are a 500-sample bootstrap of the high cut-point
**resampled over stays**, which is what makes that claim checkable rather than
asserted: read down the rejected rows and the intervals visibly blow out.

| dx_group            |   stays |   patient_hours |   icu_medium |   icu_high |   high_ci_lo |   high_ci_hi | own_threshold   |
|:--------------------|--------:|----------------:|-------------:|-----------:|-------------:|-------------:|:----------------|
| Circulatory         |      41 |            2883 |            7 |          9 |        8     |            9 | True            |
| Infectious          |      18 |            2052 |            9 |         10 |        9     |           11 | False           |
| Digestive           |      15 |            1126 |            9 |         10 |        9     |           11 | False           |
| Injury              |      15 |            1364 |            9 |         11 |        8     |           11 | False           |
| Respiratory         |      13 |            1520 |            9 |         10 |        9     |           11 | False           |
| Neoplasm            |      12 |             943 |            8 |         10 |        8     |           11 | False           |
| Nervous/Sense       |       6 |             401 |            7 |          9 |        3     |           10 | False           |
| Endocrine/Metabolic |       5 |             171 |            6 |          8 |        4     |            8 | False           |
| Injury/Poisoning    |       4 |             659 |            9 |         11 |        8     |           12 | False           |
| Mental              |       3 |              71 |            6 |          7 |        5     |            8 | False           |
| Neoplasm/Blood      |       3 |             482 |            9 |         10 |        3.475 |           11 | False           |
| Symptoms/Signs      |       3 |             266 |            8 |          9 |        4     |           10 | False           |
| Factors/Encounter   |       1 |              49 |            6 |          7 |        7     |            7 | False           |
| Musculoskeletal     |       1 |              17 |            3 |          4 |        4     |            4 | False           |

At this cohort size exactly **1 of 14 chapters** clears the bar. That
is the honest result at n=140, not a defect in the mechanism -- the code path is
general, and on the 4,000-5,000 patient build `ml/evaluation/reliability.py` sizes,
most chapters clear it comfortably. Applying the disease-specific cut-points moved
**541 of 12,004 patient-hours (4.5%)** into a
different tier than the pooled cut-point would have given them.

`capstone.news2` carries both `tier_icu` (disease-specific where one was earned, which
is what `should_escalate` reads) and `tier_icu_pooled` (the previous behaviour), plus
`threshold_is_disease_specific` so a consumer can say *which* threshold escalated a
patient rather than having to guess.

## Single-parameter escalation (finding F1)

NEWS2 (RCP 2017) has two independent escalation triggers, not one. Alongside the
aggregate tier above, **a score of 3 in any single parameter**
mandates urgent review on its own. `capstone.news2` now stores `max_component`,
`max_component_nongcs` and `red_params` so both limbs are computable, and
`should_escalate()` is the single definition every consumer imports.

| Limb | Patient-hours |
|---|---|
| ICU-recalibrated aggregate tier == high | 1,726 |
| Any non-GCS parameter scoring 3 | 3,435 |
| GCS falling >= 2 points off sedation | 255 |
| **Any of the three (the escalation predicate)** | **4,020 (33.5%)** |

GCS enters as a *change*, not a level. A red GCS level alone accounts for
4,561 further patient-hours -- overwhelmingly sedated patients -- and
escalating on it fires on 71.5% of the cohort. A GCS *drop* off sedation is specific
enough to cost roughly one extra percentage point of alert burden. See the module
docstring for the measurement of every variant against the 78 composite events.
