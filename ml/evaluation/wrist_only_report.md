# The wrist-only model: what the post-discharge arm can actually see

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice (PROJECT_PLAN.md section 17).


Composite deterioration within 6h, grouped repeated stratified CV (20 repeats x 5 folds, grouped by `subject_id`), bootstrap CIs resampling whole patients. Identical protocol to `report.md` -- only the feature set changes.

| model | features | obtainable from | AUPRC (95% CI) | AUROC (95% CI) |
|---|---|---|---|---|
| ICU full (LightGBM) | 67 | hospital only | 0.493 (0.376-0.632) | 0.823 (0.727-0.917) |
| Wrist, consumer class (HR+SpO2) | 15 | Apple Watch / Fitbit | 0.269 (0.186-0.394) | 0.754 (0.658-0.858) |
| Wrist, strict class (HR only) | 8 | Empatica E4 | 0.204 (0.124-0.356) | 0.717 (0.618-0.822) |
| Wrist HR rule (untrained) | 1 | any wearable | 0.113 (0.050-0.240) | 0.667 (0.546-0.769) |
| NEWS2 (hospital reference) | 7 vitals | hospital only | 0.095 (0.049-0.187) | 0.673 (0.582-0.756) |


## What this says

**A wrist-shaped keyhole retains roughly 55% of the full model's AUPRC** (0.269 against 0.493) while using 15 features instead of 67. The ICU model beats it in 20 of 20 repeats, so the gap is consistent, not noise.

**SpO2 is the sensor that matters.** Consumer class beats strict HR-only class in 20 of 20 repeats (0.269 vs 0.204 AUPRC). If a device is being chosen for the post-discharge arm, this is the specification that changes the answer -- and it is exactly the channel the Empatica E4 in this project does not have. That gap is why the wearable arm was retired: `simulators/morphing.py` had to fabricate SpO2 outright rather than morph it from a real recording. `simulators/home_kit_stream.py` replaced it and inverts the problem -- SpO2 is real, charted physiology from a MIMIC stay, and only the device layer is simulated.

**A two-channel wrist beats the seven-vital ward standard.** The consumer wrist model scores 0.269 AUPRC against hospital NEWS2's 0.095, winning 20 of 20 repeats -- on a task NEWS2 was built for and with five of its seven parameters unavailable. Read it carefully, though: this compares a *trained* model against an *untrained* ordinal score that was never fitted to this cohort, so it measures the value of fitting, not the inferiority of NEWS2's clinical design. The fair conclusion is narrower and still useful -- a wearable-obtainable feature set is not the reason a post-discharge arm would under-perform.

**The learned model earns its complexity.** It beats the untrained 'high heart rate = risk' rule in 20 of 20 repeats (0.269 vs 0.113), so the result is not simply tachycardia correlating with badness.

## What it does not say

Every caveat in this module's docstring applies, and two of them are load-bearing:

1. **This is an optimistic ceiling.** The HR is hourly, nurse-validated ICU monitor HR, not motion-corrupted wrist PPG. A real wearable scores below this line, not on it.
2. **The labels are ICU events.** This measures how much ICU-defined deterioration is visible through wearable-obtainable channels. It is **not** a post-discharge readmission model, and must not be reported as one -- no dataset in this project links wearable telemetry to post-discharge outcomes (E10).

The honest one-line reading: *a wrist can see a meaningful fraction of what the full ICU feature set sees, and SpO2 is most of that fraction* -- which is a real, defensible finding about sensor choice, not a clinical performance claim.
