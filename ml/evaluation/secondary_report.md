# Phase 5 -- secondary whole-stay outcomes (underpowered by design)

> E6: only 20 ICU deaths and 53 30-day readmissions across 140 stays. These numbers are reported for completeness and because the plan requires it (section 11) -- **not** as a second headline result, and not compared against a NEWS2/SOFA baseline the way the primary hourly task is. Confidence intervals this wide are not a modelling failure; they are the correct, honest consequence of n this small (section 17).


## ICU mortality

- n=140, positives=20 (14.3%)
- AUROC: 0.734 (95% CI 0.619-0.867)
- AUPRC: 0.378 (95% CI 0.226-0.591)


## 30-day readmission

- n=113, positives=22 (19.5%)
- AUROC: 0.452 (95% CI 0.326-0.587)
- AUPRC: 0.181 (95% CI 0.118-0.290)


**Why 22 readmissions here, not the EDA's 53:** notebooks/01_capstone_eda.ipynb section 3 counted 30-day readmission across all 275 admissions in the demo (most of which never had an ICU stay at all). This script's cohort is restricted to the 140 ICU stays' admissions, one row per admission -- a smaller, ICU-specific slice of the same phenomenon, not a different calculation of the same number.


The readmission AUROC point estimate (0.45) sitting at or below chance, with a CI straddling 0.5, is not a bug -- it is what 'no usable signal at this n' looks like, and is reported as such rather than reframed.
