# What limits this model's reliability

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice (PROJECT_PLAN.md section 17).


Primary task (6h horizon): **2,979 at-risk rows, 120 positives across 49 subjects, 67 features** -- 1.79 events per variable.


## Lever 1 -- the grouping unit (correctness fix, applied)

`ml/models/splits.py` had always documented grouping by `subject_id`, and shipped a `group_key()` helper to do it. Nothing ever passed that helper a `subject_id`: `run_all.py` handed it `feature_matrix_for_training`'s `groups`, which was `stay_id`, and `eval/prediction.py` called `group_key(groups)` with the optional argument omitted. Every cross-validated number this project produced was therefore stay-grouped, while the helper made it look handled.

That matters here because repeat patients are not rare: **21 of 93 subjects have more than one ICU stay in the at-risk set**, carrying **1,353 rows (45.4%)** and **53 of the 120 positives (44.2%)**. Under stay-grouping those patients appeared in train and test at once.

**This is now fixed**: `feature_matrix_for_training` returns `subject_id`, `group_key()` is deleted, and `ml/tests/test_engineer.py` pins the grouping unit so it cannot regress. The table below is the measurement that justified it.

| grouping                 |   auprc |   auprc_lo |   auprc_hi |   ci_width |   across_repeat_sd |
|:-------------------------|--------:|-----------:|-----------:|-----------:|-------------------:|
| stay_id (before the fix) |  0.5034 |     0.3836 |     0.6381 |     0.2545 |             0.0187 |
| subject_id (current)     |  0.4930 |     0.3762 |     0.6257 |     0.2495 |             0.0195 |

**Optimism from stay-grouping: -0.0104 AUPRC.** The correct grouping scores lower, which is the expected direction: the reported figure was partly reading patients it had already trained on. Fixing this does not improve the model -- it corrects what the model's score means.


## Lever 2 -- the feature budget (gain-ranked pruning is not a lever; targeted pruning is)

67 features against 120 positives is **1.79 events per variable**, against the conventional floor of 10. That was the hypothesis going in: at an EPV this low the model should be fitting noise, and pruning should buy back stability. Features are ranked by split gain *inside each training fold* and the model refitted on the top-k, so the held-out rows never influence the ranking. Subject-grouped throughout.

|   n_features |   events_per_variable |   auprc |   auprc_lo |   auprc_hi |   ci_width |   across_repeat_sd |
|-------------:|----------------------:|--------:|-----------:|-----------:|-----------:|-------------------:|
|       5.0000 |               24.0000 |  0.2904 |     0.1935 |     0.4250 |     0.2314 |             0.0305 |
|      10.0000 |               12.0000 |  0.3250 |     0.1919 |     0.5194 |     0.3274 |             0.0318 |
|      20.0000 |                6.0000 |  0.4681 |     0.3539 |     0.6013 |     0.2474 |             0.0203 |
|      40.0000 |                3.0000 |  0.4959 |     0.3769 |     0.6293 |     0.2524 |             0.0178 |
|      67.0000 |                1.7910 |  0.4930 |     0.3762 |     0.6257 |     0.2495 |             0.0195 |

**The hypothesis is wrong, and the measurement says so.** AUPRC rises monotonically with the feature budget -- 0.2904 at 5 features up to **0.4930 at all 67** -- and the across-repeat spread does not improve as features are removed (0.0178-0.0318 with no trend). Pruning *by this method* costs accuracy and buys nothing.

Part of the reason is that events-per-variable is a rule about *degrees of freedom in a linear model*, where every predictor spends a parameter. This model is a depth-4, 15-leaf, 200-tree LightGBM (`ml/models/gbm.py`): it already performs its own feature selection at every split and never spends a parameter on a feature it does not use. Quoting EPV 1.79 as though it condemned this model would have been borrowing a logistic-regression diagnostic for an estimator it does not describe.

**But read the scope of that result carefully, because a later study ([`feature_pruning_report.md`](feature_pruning_report.md)) pruned the feature set from 90 to 67 and *gained* AUPRC in 17 of 20 paired repeats.** There is no contradiction: the two studies remove different things. The curve above ranks features by **split gain** and keeps the top-k, so it can only ever discard what the model already uses least. It is structurally incapable of removing a feature that is heavily used *and* redundant -- which is exactly what `sofa_24hours` was, the single highest-attribution feature in the SHAP summary and worth 0.000 AUPRC when removed. The lesson is narrower and more useful than 'pruning does not work': **a model's own importance ranking is a poor guide to what it can afford to lose**, because importance and necessity come apart wherever features are correlated.


## Lever 3 -- how many events a target precision needs

Subjects (not rows) subsampled at each fraction, several independent draws each, full protocol re-run per draw. The CI width is then fitted as `width = a * n_positive_subjects ** b`, with **both** the constant and the exponent estimated from these measurements rather than the exponent being pinned at the textbook -0.5.

|   fraction |   n_subjects |   n_positive_subjects |   n_positive_rows |   auprc |   ci_width |
|-----------:|-------------:|----------------------:|------------------:|--------:|-----------:|
|      0.250 |       23.000 |                10.833 |            27.500 |   0.295 |      0.483 |
|      0.400 |       37.000 |                19.667 |            43.667 |   0.424 |      0.402 |
|      0.550 |       51.000 |                29.333 |            71.167 |   0.475 |      0.317 |
|      0.700 |       65.000 |                35.333 |            84.667 |   0.505 |      0.294 |
|      0.850 |       79.000 |                41.667 |           100.000 |   0.480 |      0.290 |
|      1.000 |       93.000 |                49.000 |           120.000 |   0.493 |      0.249 |

Fitted scaling: **width = 1.38 x n^-0.43** (R² = 0.979). The textbook standard-error exponent is -0.50; the measured one is **-0.43**, i.e. this interval narrows *more slowly* than a standard error would. Assuming -0.50 instead (fit: 1.74 x n^-0.50, R² = 0.931) would have been the convenient choice, so both are shown below and the larger requirement is the one to plan against.


|   target 95% CI width |   positive subjects (fitted exponent) |   positive subjects (if -0.50) | multiple of current   |
|----------------------:|--------------------------------------:|-------------------------------:|:----------------------|
|                  0.2  |                                    90 |                             76 | 1.8x                  |
|                  0.1  |                                   448 |                            304 | 9.1x                  |
|                  0.05 |                                  2246 |                           1213 | 45.8x                 |

Current cohort: **49 positive subjects**.

The fit itself is good -- R² 0.979 across 6 measured points -- but a good fit *within* the measured range is not what is being asked of it, and the extrapolation is this analysis's weak point in three named ways.

* **Reach.** The points span a 4.5x range of patient counts, and the 0.10 target sits about 9x beyond the largest of them. A power law fitted over one order of magnitude and read off two is an estimate of size, not a prediction.
* **The exponent may itself be flattered.** AUPRC is bounded in [0, 1], so at the smallest subsamples the interval cannot widen indefinitely and the measured decay is pushed shallower than the underlying one. That biases `b` toward zero and the requirement upward -- conservative, but not free of assumption.
* **Case mix.** It assumes a larger MIMIC-IV cohort resembles this one. This cohort is 53% positive at the subject level, which is extraordinarily high and reflects a deliberately post-surgical 100-patient subset admitted *for* intervention. A uniform MIMIC-IV sample will almost certainly have a lower event rate, meaning **more** patients per positive than the conversion below assumes.


## What this means, in order

1. **Group by `subject_id`** -- done. Worth 0.0104 AUPRC of removed optimism, with 44% of the positives exposed to the leak beforehand. This does not make the model better; it makes the number true, which is the only kind of improvement available for free. Every figure in `ml/evaluation/report.md` was regenerated under the corrected grouping.
2. **Leave the feature set alone.** The EPV objection was tested and does not hold for this estimator. Recorded here so it does not get re-raised and re-litigated later from theory.
3. **Get more patients is the only real lever -- and it is closed to this project.** Nothing in the modelling narrows an interval this wide; only events do. Reaching a 0.10-wide AUPRC CI needs on the order of **448 positive subjects** against today's 49, and there is no route to them here: the full MIMIC-IV database requires PhysioNet credentialing (CITI human-subjects training plus a signed data use agreement), while this project uses only the open-access demo subsets and will not be seeking credentialed access (`docs/DATA_USE.md`). The requirement below is therefore a measurement of the gap, not a backlog item.


Turning that into a cohort size depends entirely on the event rate, which is the assumption least likely to carry over:

| assumed positive-subject rate | ICU patients to load |
|---|---|
| 53% (this cohort, post-surgical) | ~900 |
| 30% | ~1,500 |
| 20% | ~2,300 |

A uniform MIMIC-IV sample will not reproduce a 53% event rate, so the realistic figure is the lower rows: **a 4,000-5,000 patient cohort**. That is what closing this interval would take, and it is recorded so the size of the gap is on the record rather than implied. For anyone working from credentialed MIMIC-IV, `warehouse/build_duckdb.py --cohort-subjects N` loads it, and this analysis is worth re-running there because the fitted exponent is the thing most likely to change.


### What follows for this capstone

The interval does not close, so the honest move is to say so and stop treating it as pending. Three consequences, in order of how easily each is got wrong:

1. **The ~0.25-wide AUPRC CI is a structural property of having 49 positive subjects, not an outstanding task.** Every headline number in `ml/evaluation/report.md` should be read with it attached. The model genuinely beats recalibrated NEWS2 on AUPRC in 20 of 20 CV repeats -- that result is a *paired* comparison on the same folds, which is exactly why it survives a sample this small when the absolute AUPRC does not.
2. **Synthetic patients cannot substitute, and this was measured rather than assumed** -- see [`synthetic_ceiling_report.md`](synthetic_ceiling_report.md). A perfect generator adds nothing on real held-out patients, because the information in it is bounded by the same 49 positive subjects it was fitted to. Evaluating on synthetic patients would collapse the reported interval to nothing while changing no actual knowledge.
3. **The project's existing framing was right, and now has a number behind it.** PROJECT_PLAN.md section 17's "pipeline validity, not clinical performance" is not a hedge covering an unfinished result -- it is the correct reading of a cohort this size, and this analysis is what makes it quantitative.
