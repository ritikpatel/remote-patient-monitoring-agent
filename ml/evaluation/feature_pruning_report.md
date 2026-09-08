# Feature pruning: what the model actually needs

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice (PROJECT_PLAN.md section 17).

Full rationale for every removal and every deliberate keep is in this
module's docstring (`ml/evaluation/feature_pruning.py`).

| feature set | features | AUPRC (95% CI) | AUROC |
|---|---|---|---|
| pre-pruning | 90 | 0.398 (0.296-0.540) | 0.826 |
| **pruned (current)** | **67** | **0.493** (0.376-0.632) | 0.823 |

The pruned set beats the pre-pruning set in **17 of 20** paired CV
repeats (identical folds in both arms).

Removed: `news2`, `sofa_24hours`, the 18 rolling means, and the 3
lab-ordering-intensity columns. Kept after explicit testing:
`admission_age`, `gender`, `first_careunit`, and every vital channel --
including `rr`, which the screen wanted to drop and which the data-quality
check protected (99.9% present, 1.1% carried forward).

The honest headline is not that the model got better. It is that a
90-feature model was **over-parameterised for 120 positives**, and most of
the measured gain is variance removed rather than signal added.
