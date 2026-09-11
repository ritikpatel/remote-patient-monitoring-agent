# Home-kit transfer: train for the kit, or mask afterwards?

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice (PROJECT_PLAN.md section 17).

**MIMIC contains no physiology after discharge.** It records discharge, readmission
and death *times* -- outcomes, but no signal. A post-discharge deterioration score
therefore cannot be trained on post-discharge data here, and the only defensible
construction is a transfer: train on ICU physiology, deploy against what a home kit
can observe, and state the bound. Everything below is that bound. It is **not** a
validated post-discharge model, and the label is still the in-ICU composite event.

The kits are the same objects `simulators/home_kit_stream.py` streams, so the regime
measured here is exactly the regime that simulator emits.

## What each kit costs in features

| kit             | channels                        |   columns_masked |   columns_remaining | channels_absent                                |
|:----------------|:--------------------------------|-----------------:|--------------------:|:-----------------------------------------------|
| icu_full        | all                             |                0 |                  85 | none                                           |
| watch_only      | hr, spo2                        |               50 |                  35 | rr, sbp, map, temp_c, gcs_total, fio2, glucose |
| watch_plus_cuff | hr, rr, spo2, sbp, map          |               29 |                  56 | temp_c, gcs_total, fio2, glucose               |
| full_home       | hr, rr, spo2, sbp, map, glucose |               22 |                  63 | temp_c, gcs_total, fio2                        |

## Two training arms, identical folds

* **`masked_at_inference`** -- fit on the full ICU feature set, blank the kit's
  unavailable columns when scoring. The model has never seen a patient without a
  ventilator.
* **`dropout_trained`** -- fit on a training set replicated once per kit, each
  replica masked to that kit (`ml/models/channel_masking.py`). Absence becomes a
  learned pattern rather than a surprise. Only the training side is augmented; both
  arms are scored on identical held-out rows.

Protocol: 10 repeats x 5 folds, subject-grouped stratified CV.
`dropout_wins` is the paired per-repeat count of `dropout_trained` beating
`masked_at_inference` within that kit.

| kit             |   auprc_dropout_trained |   auprc_masked_at_inference |   auroc_dropout_trained |   auroc_masked_at_inference |   dropout_wins |   repeats |   auprc_delta |   retained_vs_icu |
|:----------------|------------------------:|----------------------------:|------------------------:|----------------------------:|---------------:|----------:|--------------:|------------------:|
| full_home       |                  0.3452 |                      0.3078 |                  0.7948 |                      0.8037 |              9 |        10 |        0.0374 |            0.7010 |
| icu_full        |                  0.3857 |                      0.4925 |                  0.8033 |                      0.8504 |              0 |        10 |       -0.1068 |            0.7831 |
| watch_only      |                  0.3169 |                      0.2454 |                  0.7781 |                      0.7121 |             10 |        10 |        0.0716 |            0.6435 |
| watch_plus_cuff |                  0.3430 |                      0.3074 |                  0.7942 |                      0.8092 |              9 |        10 |        0.0356 |            0.6964 |

## What this says

**Dropout training helps every home kit and costs the ICU model, so the answer
is two models -- and that reverses a conclusion this project previously drew.**

`channel_dropout_report.md` concluded that "masking the generic model is as good as
training a dedicated one, and a second wrist-specific model is not worth deploying:
one model that degrades gracefully covers the same ground." That was a correct
reading of the evidence *available at the time*, because the only alternative it
compared against was `wrist_only.py`'s model trained on a **restricted feature set**.
Training on the **full** feature set with kit-shaped dropout is a different thing,
and it wins: every home kit improves by 0.0356-0.0716
AUPRC on 9-10 of 10 paired repeats.

The same change costs the ICU model **0.1068 AUPRC**
(0.4925 -> 0.3857, losing all
10 repeats). That is the accuracy/robustness trade-off in its plainest form:
a model taught that channels vanish stops leaning as hard on the ones that are
present, which is exactly what you want at home and exactly what you do not want in
an ICU where every channel is charted.

So the deployment shape is: **the dropout-free model for the ICU arm, the
dropout-trained model for the post-discharge arm.** One estimator, one feature set,
two fits, each used only in the regime it was fitted for. `risk-engine` already
serves a promoted artefact per arm, so this costs an export, not an architecture.

Best home-kit result: **`full_home`** at AUPRC 0.3452,
retaining 70% of the full-ICU model.

The ordering across kits is the part worth carrying into a design decision: it prices
each additional home instrument in the only currency that matters here. Note that
`watch_only` loses considerably more than the step from `watch_plus_cuff` to
`full_home` gains -- blood pressure is doing most of the work that separates a
usable home signal from a wrist pulse.

## The limit that no arm escapes

Core temperature, GCS, FiO2 and arterial-line presence have **no home instrument**,
at any price. That is not a modelling restriction to be engineered away -- a patient
at home is not ventilated and nobody is performing hourly neurological exams on them.
The gap between `full_home` and `icu_full` is the information those four carry, and
the only way to close it is to change what the patient is wearing, not how the model
is fitted.
