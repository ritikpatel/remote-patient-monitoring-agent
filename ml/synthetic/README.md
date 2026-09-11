# `ml/synthetic` — EMR-WGAN synthetic EHR generation

An implementation of Yan C, Zhang Z, Nyemba S, Li Z. *Generating Synthetic
Electronic Health Record Data Using Generative Adversarial Networks: Tutorial.*
JMIR AI 2024;3:e52615 ([doi:10.2196/52615](https://doi.org/10.2196/52615)) —
the paper saved at `mimic-synthetic-generator-tutorial.pdf` in the repo root —
applied to this project's ICU deterioration cohort.

Results live in [`report.md`](report.md), regenerated from the CSVs by
`python ml/synthetic/report.py`.

## What it found

**The implementation is faithful and the generator is good.** Dimension-wise
distance reaches 1.44, inside the 0.52–1.56 band the paper reports for its own
five runs on 181,294 patients; column-wise correlation ~5.0 against their
5.0–6.5; clinical knowledge violation 0.006 against their 0.04–0.07. A model
trained purely on synthetic rows scores AUROC 0.800 against the real model's
0.827 on the same held-out patients.

**Augmentation still does not improve the model.** Across 20 paired
comparisons, no augmented arm beats the real-only baseline — the best of six is
−0.0006 AUPRC at 10/20 wins, against this project's 15/20 bar, and the largest
synthetic multiplier is the worst arm. This is the answer measured with a real
generator in the fidelity band the earlier ceiling experiment could not reach.

**The paradigm the tutorial demonstrates does not survive a 4% event rate.**
The nonconditional generator collapses the deterioration label to 0.0041 early
in training — ~10× under-represented, too few positives to fit a classifier on.
Conditional training holds the rate exactly and is what stage 2 uses.

**Whole synthetic patients, class-balanced, do not help either.** The fair
objection to the above is that patient-*hours* are not patients and cannot raise
the positive-**subject** count, which is what `reliability.py` says is binding.
`patients.py` generates real patients — trajectories, subject ids, 50/50 class
balance — and takes that count from 39 per fold to **539**, fourteen times over.
AUPRC moves −0.0337, winning 9 of 20. A synthetic positive subject is not a
positive subject.

**Privacy runs the opposite way from the paper, and this is the finding that
constrains use.** Membership-inference F1 reaches 0.84 against a 0.51 chance
floor and 1.0 for publishing the real records — two-thirds of the way to full
disclosure — and it *rises* with training length, the signature of
memorisation. `privacy_control.py` rules out the confound by refitting on the
held-out group: the attack's verdict on fixed targets swings +0.66 to follow
whichever subjects the generator saw. **The synthetic cohort is not
de-identified and must not leave the project.**

## Read this before using the output

**The tutorial's premise is 181,294 real patients. This cohort has 100.**

That is not a footnote, it is the fact that decides what the generated data may
legitimately be used for. The paper generates a synthetic set *the same size* as
its real one, for privacy-preserving **data sharing** — releasing something that
behaves like MIMIC-IV without releasing MIMIC-IV. It never claims that a GAN
manufactures new information about patients it was not fitted on, and it could
not: a generator fitted on 74 training subjects is an estimate of a distribution,
and sampling an estimate a million times does not tell you anything further about
the population it estimates. That is the data-processing inequality, and
[`../evaluation/synthetic_ceiling_report.md`](../evaluation/synthetic_ceiling_report.md)
already measured its consequence here with a resampling generator.

What this package adds is the part that report could not reach. Its own text
admits the gap:

> Perturbing every channel independently breaks the inter-vital correlations a
> discriminator finds trivially, so this axis cannot land on 90% by construction.

That experiment could only produce synthetic cohorts at 100% fidelity or 0%,
never the middle, so "what if a *real* generator hit 90%?" stayed hypothetical.
EMR-WGAN lands in the middle by construction. `report.md` is the answer measured
rather than assumed.

## Layout

| file | what it is |
| --- | --- |
| [`preprocess.py`](preprocess.py) | The paper's four preprocessing steps: outlier removal, missing-value handling, Equation 1 min-max normalization, low-prevalence pruning. Emits a `Matrix` in [0,1] plus the `MatrixSpec` that inverts it. |
| [`emr_wgan.py`](emr_wgan.py) | The model of the paper's Figure 1. WGAN-GP; batch norm in the generator, layer norm in the critic; per-block SoftMax for one-hot preservation; both training paradigms. Checkpoints along the trajectory. |
| [`evaluate.py`](evaluate.py) | The paper's data-quality battery — six matrix-level utility metrics, TSTR/TRTR, feature-importance overlap, two privacy attacks, plus `temporal_coherence` (this project's own, see below). |
| [`run_tutorial.py`](run_tutorial.py) | Orchestrator. Stage 1 replicates the paper's quality experiment; stage 2 asks whether any of it helps this project's model. |
| [`patients.py`](patients.py) | Whole synthetic **patients**: a GAN over ~70 low-rank per-stay parameters, expanded to hourly trajectories by a calibrated AR(1). Derived features are recomputed, never generated. |
| [`run_patients.py`](run_patients.py) | Does raising the class-balanced patient count move the real numbers? |
| [`privacy_control.py`](privacy_control.py) | Swap test that rules out the confound behind the membership-inference finding. |
| [`report.py`](report.py) | Renders `report.md`, including the paper's Table 3 use-case weight profiles. |

## Running it

```bash
python ml/synthetic/run_tutorial.py && python ml/synthetic/privacy_control.py && python ml/synthetic/report.py
```

Roughly 20 minutes on CPU. Stage 1 trains ten independent generators (five runs
× two paradigms); stage 2 trains twenty more, one per fold across four repeats,
because five paired comparisons cannot settle a small delta and this project
calls a change real at 15 wins of 20.

`--quick` cuts it to about three minutes for smoke-testing, and `--stage
quality` / `--stage augmentation` runs one half. `--repeats` trades runtime
against statistical power.

```bash
python ml/synthetic/run_tutorial.py --quick
```

The patient-level arm is a separate entry point, about 45 minutes:

```bash
python ml/synthetic/run_patients.py && python ml/synthetic/report.py
```

Tests:

```bash
python -m pytest ml/synthetic/tests/ -q
```

## Two things worth knowing about the implementation

**Conditional training is not optional here.** The paper demonstrates the
*nonconditional* paradigm, where the outcome label is generated as an ordinary
column. At this project's 4% event rate that fails in exactly the way the paper
predicts for low-prevalence concepts, and `report.md` quantifies it. The
conditional paradigm — label as an input to both networks — fixes the rate by
construction and is what stage 2 uses throughout.

**Why the patient generator is not one big GAN.** A GAN over the raw 48h x
9-vital sequence is ~460 dimensions fitted on ~100 stays, and `report.md` already
shows this architecture memorising at 89 dimensions with 1,995 rows. So the
patient is parameterised low-rank — the GAN learns ~70 per-stay parameters, and a
calibrated AR(1) expands them into a trajectory. The expansion stage is not
learned adversarially and therefore cannot memorise a real patient. Lag-1
autocorrelation in this cohort runs 0.41–0.69, so AR(1) is a fair description of
how these signals move.

The trade is visible and worth naming: the patient generator scores **worse** on
`temporal_coherence` than the row generator (0.190 against 0.092). Recomputing
derived features guarantees a patient is internally consistent, but AR(1) is a
simplification of real vital dynamics, so the *distributional* relationship
between `hr_4h_std` and `hr_24h_std` is looser than the one the row-level GAN
learned directly. Structurally it is faithful — `4h_std` exceeds `24h_std` in
18.3% of synthetic rows against 19.4% of real ones.

**One metric is added, not from the paper.** 36 of this cohort's 85 features are
rolling-window statistics computed from a patient's trajectory, and a snapshot
generator emits them as free-standing columns with nothing forcing `hr_4h_std`
and `hr_24h_std` to describe the same patient. `evaluate.temporal_coherence`
measures that directly — it is the tutorial's own stated limitation ("neglects
the timestamping of medical events") landing on a time-series project. The
generator does better here than expected: mean correlation error 0.092 against
real correlations averaging 0.552.

**One metric is adapted rather than copied.** The paper measures clinical
knowledge violation as male-specific diagnoses appearing on female records. This
cohort has no sex-specific diagnosis at usable prevalence, and `gender` was
dropped from the feature set by the fairness audit (finding F4), so the test has
nothing to bite on. `evaluate.clinical_knowledge_violation` substitutes
physiological impossibilities the data does contain — MAP above SBP, SpO2 over
100%, GCS outside 3–15. The reasoning is argued in that module's docstring.

## What the output may and may not be used for

The rules the project already operates under
([`synthetic_ceiling_report.md`](../evaluation/synthetic_ceiling_report.md)) are
unchanged by this work, and nothing here supersedes them:

- A synthetic row **must never** appear in a test set whose metrics are reported
  as performance. Every number in `report.md` is measured on real held-out
  subjects; stage 2 refits the generator inside each fold precisely so that a
  test subject cannot leak into the synthetic training rows.
- A confidence interval computed over synthetic rows is **manufactured
  precision**, not evidence. It narrows with every generated row while the
  knowledge behind it stays fixed.
- Train-on-synthetic/test-on-real is the only defensible protocol, and it is what
  the TSTR arm reports.

And one rule this work adds, which did not exist before because nothing had
measured it:

- **The synthetic cohort is not de-identified data and must not be published or
  shared outside the project.** Membership-inference risk sits two-thirds of the
  way from chance to releasing the real records, verified against a swap control.
  Data sharing is the use the tutorial's method exists for, and it is precisely
  the use a 100-patient cohort cannot safely make of it.
