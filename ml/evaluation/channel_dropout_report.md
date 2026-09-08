# Channel dropout: how this model degrades when sensors go missing

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice (PROJECT_PLAN.md section 17).

The promoted model, trained once per fold on the full feature set, then scored with channels masked **at prediction time only**. Full rationale, and why this differs from `wrist_only_report.md`, is in `ml/evaluation/channel_dropout.py`'s docstring.

| masked | columns | AUPRC | delta | retained | baseline wins |
|---|---|---|---|---|---|
| drop glucose | 7 | 0.506 | +0.005 | 101% | 2/20 |
| drop map | 7 | 0.502 | +0.000 | 100% | 8/20 |
| none (baseline) | 0 | 0.502 | +0.000 | 100% | -- |
| drop hr | 7 | 0.501 | -0.001 | 100% | 10/20 |
| drop rr | 7 | 0.499 | -0.003 | 99% | 12/20 |
| drop temp_c | 7 | 0.488 | -0.014 | 97% | 17/20 |
| drop sbp | 7 | 0.475 | -0.027 | 95% | 20/20 |
| drop spo2 | 7 | 0.471 | -0.031 | 94% | 20/20 |
| drop gcs_total | 7 | 0.467 | -0.035 | 93% | 19/20 |
| drop fio2 | 7 | 0.417 | -0.085 | 83% | 20/20 |
| post_discharge_realistic | 22 | 0.340 | -0.162 | 68% | 20/20 |
| wrist_only_channels | 50 | 0.274 | -0.228 | 55% | 20/20 |

Baseline (nothing masked): **0.502** AUPRC over 20 repeats.

## What this says

**The realistic post-discharge case retains 68%** of the model's AUPRC (0.340 against 0.502). That is the honest number to quote for a discharged patient wearing a full home sensor suite, because core temperature, GCS and FiO2 have no home sensor and arterial-line presence is always false once the patient is home.

**Masking every channel but HR and SpO2 retains 55%** (0.274). A model *trained* on only those channels (`wrist_only_report.md`) scores 0.269 against this 0.274 -- so masking the generic model is **as good as training a dedicated one**, and a second wrist-specific model is not worth deploying: one model that degrades gracefully covers the same ground. The wrist study's role is to characterise the loss, not to ship an artefact.

**The single most costly channel to lose is `fio2`** (0.417, -0.085).

## The caveat that limits all of it

Every channel here is 74-100% present in training, so the trees had very little opportunity to learn a sensible default direction for its absence. These numbers therefore measure *an artefact of training-time availability* as much as the clinical value of each signal. A model intended to run with channels routinely missing should be trained that way -- with dropout applied during training, not only measured at inference. That is the natural next step this report argues for, and it is not done here.
