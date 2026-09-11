# What synthetic patients would buy

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice (PROJECT_PLAN.md section 17).


The question this answers: **if a generator reached ~90% similarity to the real cohort, could it substitute for the patients [`reliability_report.md`](reliability_report.md) says are needed?** Similarity is measured by a classifier two-sample test (C2ST) and reported as `100 * (2 - 2 * AUC)`, so **90% similar means C2ST AUC 0.55** and 100% means a discriminator cannot separate real from synthetic at all.


Setup: 5 subject-grouped folds over the real cohort (93 subjects, 120 positives). Synthetic patients are built from the **training** subjects only and evaluated on **real, held-out** subjects. The generator resamples whole patients with replacement, preserving every temporal correlation exactly, then optionally perturbs them -- at zero noise it reproduces the training distribution perfectly, which no learned generator can beat.


## 1. A perfect generator adds no information

Noise 0.0: synthetic patients are exact resamples, so C2ST cannot separate them. `k` is the multiplier on training patients -- k=10 means nine synthetic patients per real one.

|   multiplier |   measured similarity % |   AUPRC on REAL held-out |   CI width (real, honest) |   CI width (synthetic, believed) |
|-------------:|------------------------:|-------------------------:|--------------------------:|---------------------------------:|
|        1.000 |                 100.000 |                    0.422 |                     0.606 |                            0.606 |
|        3.000 |                  94.919 |                    0.458 |                     0.623 |                            0.000 |
|       10.000 |                 100.000 |                    0.442 |                     0.616 |                            0.000 |

**Real-world performance does not move: 0.422 at k=1 against 0.458 at the best multiplier (+0.036).** Ten times the training rows, drawn from a flawless generator, buys nothing on real patients -- because a model fitted to 74 training subjects and then resampled cannot learn about patients it never saw.

The last two columns are the point. The honest interval, measured on real held-out patients, stays around **0.606**. The interval you would *report* if you evaluated on synthetic patients is **0.000** -- it collapses to an interval with no measurable width at all, and it shrinks further with every synthetic patient added. That is manufactured precision: the number moves, the knowledge does not.


## 2. Lower fidelity does not rescue it either

Fidelity walked down by perturbing the resampled patients. Similarity is measured, not assumed.

Read this table for the *direction*, not for a 90% row: the C2ST goes from unable to separate (100%) to perfectly separating (0%) as soon as any noise is added, with nothing in between. Perturbing every channel independently breaks the inter-vital correlations a discriminator finds trivially, so this axis cannot land on 90% by construction. The measured answer to the question as posed is the k=3 row of section 1 (92.0% similar), and it is the same answer: no gain on real patients.

|   noise (x SD) |   C2ST AUC |   measured similarity % |   AUPRC on REAL held-out |
|---------------:|-----------:|------------------------:|-------------------------:|
|          0.000 |      0.369 |                 100.000 |                    0.442 |
|          0.005 |      1.000 |                   0.000 |                    0.440 |
|          0.010 |      1.000 |                   0.000 |                    0.438 |
|          0.025 |      1.000 |                   0.000 |                    0.421 |
|          0.050 |      1.000 |                   0.000 |                    0.456 |

Across this whole fidelity range the AUPRC on real held-out patients moves by only 0.035 (0.421-0.456), and it does **not** fall monotonically: the most heavily perturbed generator scores 0.456 against the perfect generator's 0.442. That is the honest reading -- differences this small sit inside cross-validation noise, so fidelity is not the binding constraint and degrading it is not what costs you. What does not happen at any level is augmentation overtaking the un-augmented baseline by a margin this protocol could detect.


## What this means for the capstone

**Synthetic data cannot close the sample-size gap, and 90% fidelity would not change that.** The binding constraint is not how realistic the rows look -- it is that the information in them is bounded by the 49 real positive subjects any generator would be fitted to. Reporting a narrow CI computed over synthetic patients would be the single most misleading thing this project could do, and section 1 shows exactly how large that misstatement would be.

**What synthetic data is legitimately for here, and already used for:**

- **Simulating the instrument, not the patient.** `simulators/home_kit_stream.py` replays a MIMIC stay that genuinely deteriorated and simulates only what a home kit would see of it -- channel availability, cadence, noise, non-wear gaps. It replaced `morphing.py`, which synthesised a deterioration onto a healthy volunteer because that cohort contained no deterioration at all (E10); the physiology is now real and only the device is not.
- **Load and robustness testing.** `eval/load/ramp.js` needs volume, not truth.
- **Narrative generation.** `notes_synth/` writes clinical prose from real structured rows, with a fact ledger tying every sentence back to a source row.

**What it must never do:** appear in a training or evaluation set whose metrics are reported as performance. Train-on-synthetic/test-on-real is the only defensible protocol, and section 1 measures what it yields here: nothing.

The honest conclusion stands, now with a number behind it rather than a hedge: **this cohort's confidence interval is a structural property of having 49 positive subjects, and it is not fixable within this project.** Reporting it plainly is the correct result, not a shortfall.

## Follow-up: the middle of the fidelity range, measured

Section 2 above carries an admitted hole. Perturbing channels independently could only produce cohorts a discriminator finds identical or trivially separable, never the intermediate fidelity the question was posed at, so the answer rested on the perfect-generator argument of section 1 rather than on a generator that actually lives in that band.

[`ml/synthetic/`](../synthetic/report.md) closes it. EMR-WGAN (Yan et al., JMIR AI 2024) trained on this cohort reaches a dimension-wise distance inside the range that paper reports for its own runs on 181,294 patients, and it changes nothing here: across 20 paired comparisons no augmented arm beats the real-only baseline, the best of six sitting at -0.0006 AUPRC with 10 of 20 wins. Fidelity was never the binding constraint, and now that has been measured with a real generator rather than argued from a resampler.

That work also surfaced something this experiment could not have: the EMR-WGAN cohort carries **membership-inference risk two-thirds of the way from chance to publishing the real records**, verified against a swap control. Synthetic data from a cohort this small is not de-identified data.
