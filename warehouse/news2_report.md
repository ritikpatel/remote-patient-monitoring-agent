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
