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
| low    |                64.2 |                             140 |
| medium |                23   |                              92 |
| high   |                12.8 |                              66 |

## Why recalibrate (E5)

128/140 ICU stays (91%) cross the ward
"medium" threshold at some point in the stay. A trigger this population trips this
often carries little discriminating information for an ICU-only alert engine --
everyone here is already sick enough to be in the ICU. The ICU-recalibrated cut-points
instead flag the 25% of this cohort's own
patient-hours with the highest NEWS2, i.e. relative deterioration within an ICU
population rather than absolute deterioration relative to a ward population.

## Single-parameter escalation (finding F1)

NEWS2 (RCP 2017) has two independent escalation triggers, not one. Alongside the
aggregate tier above, **a score of 3 in any single parameter**
mandates urgent review on its own. `capstone.news2` now stores `max_component`,
`max_component_nongcs` and `red_params` so both limbs are computable, and
`should_escalate()` is the single definition every consumer imports.

| Limb | Patient-hours |
|---|---|
| ICU-recalibrated aggregate tier == high | 1,535 |
| Any non-GCS parameter scoring 3 | 3,435 |
| GCS falling >= 2 points off sedation | 255 |
| **Any of the three (the escalation predicate)** | **3,948 (32.9%)** |

GCS enters as a *change*, not a level. A red GCS level alone accounts for
4,633 further patient-hours -- overwhelmingly sedated patients -- and
escalating on it fires on 71.5% of the cohort. A GCS *drop* off sedation is specific
enough to cost roughly one extra percentage point of alert burden. See the module
docstring for the measurement of every variant against the 78 composite events.
