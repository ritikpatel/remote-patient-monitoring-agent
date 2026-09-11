# Is the disease-aware model learning physiology or reading the coder's notes?

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice (PROJECT_PLAN.md section 17).

`diagnoses_icd` is assigned by billing coders **after discharge**, so a primary
diagnosis summarises how an admission turned out rather than what was knowable at
the bedside. Charlson's comorbidity flags are pre-existing chronic conditions by
construction. This study measures the two separately -- see the module docstring in
`ml/evaluation/disease_leakage.py` for the full argument.

Protocol: 20 repeats x 5 folds, subject-grouped stratified CV, identical
folds across arms, LightGBM throughout. Judged on the **paired per-repeat win count**
against the disease-blind arm, not on mean AUPRC alone -- with 49 positive subjects
the difference between two arms is routinely smaller than the CI on either.

## Arms

| arm                |   n_features |   disease_columns |   mean_auprc |   mean_auroc |   wins_vs_none |   repeats |   delta_auprc |
|:-------------------|-------------:|------------------:|-------------:|-------------:|---------------:|----------:|--------------:|
| none               |           67 |                 0 |     0.501832 |     0.819814 |            nan |        20 |    0          |
| chronic_index_only |           68 |                 1 |     0.493574 |     0.842947 |              3 |        20 |   -0.00825841 |
| chronic            |           85 |                18 |     0.487918 |     0.844758 |              4 |        20 |   -0.013914   |
| coded_only         |           68 |                 1 |     0.490703 |     0.804185 |              2 |        20 |   -0.0111285  |
| all                |           86 |                19 |     0.475677 |     0.827325 |              2 |        20 |   -0.026155   |

Best mean AUPRC: **`none`** at 0.5018
(+0.0000 against disease-blind).

## Probe 1 -- `dx_chapter` alone

A model given nothing but the discharge-coded chapter -- no vitals, no trend, no age:

| | AUPRC |
|---|---|
| `dx_chapter` alone | 0.0392 |
| Label base rate | 0.0403 |
| Ratio | **0.97x** |

A patient-constant billing code contains no physiology. Whatever it scores above the
base rate is either genuine case mix (sicker diagnoses really do deteriorate more) or
outcome information written into the chart after the fact. Probe 2 separates them.

## Probe 2 -- lift by label component

Share of each chapter's stays that ever have each composite component, divided by the
cohort-wide share. Chapters with fewer than 5 stays are omitted (a rate over four
patients is not a rate).

| dx_chapter          |   death |   icu_readmission |   vasopressor |   ventilation |
|:--------------------|--------:|------------------:|--------------:|--------------:|
| Circulatory         |    0.68 |              0.38 |          1.38 |          1    |
| Digestive           |    2.49 |              1.04 |          1.08 |          0.8  |
| Endocrine/Metabolic |    0    |              0    |          0.54 |          0    |
| Infectious          |    0.52 |              3.46 |          1.35 |          1.34 |
| Injury              |    1.24 |              1.04 |          0.36 |          0.97 |
| Neoplasm            |    0.78 |              2.59 |          0.45 |          0.8  |
| Nervous/Sense       |    0    |              0    |          0.9  |          0.8  |
| Respiratory         |    2.15 |              0    |          0.83 |          1.3  |

The shape is what matters. Broad, modest lift across all four components is case mix.
A spike in the single component matching the chapter's own organ system is the coder
describing the event. The largest single cell here is
**Infectious x icu_readmission at 3.46x**
(18 stays, 22% against a cohort 6%).

## What ships

`ml/features/engineer.DEFAULT_DISEASE_FEATURES` is set **by hand** from this report,
not automatically from the winning arm. An arm can win on AUPRC precisely *because*
it leaks, so promoting the top row mechanically would be the exact error this study
exists to prevent.
