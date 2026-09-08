# Aligning the two arms

This project is called *"Remote Patient Monitoring in ICU **and Post-Discharge**
Care"*. The two arms are not equal, and until now nothing in the repo said so.
This document measures the gap, names the one root cause in code, records a
**provable defect** it produces, and sequences the work to close what can be
closed.

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The
> engineering is real and the methodology is rigorous; the clinical performance
> figures demonstrate pipeline validity and do not transfer to clinical
> practice (PROJECT_PLAN.md section 17).

## 1. The asymmetry, measured

| | ICU arm | Post-discharge arm |
|---|---|---|
| Data | MIMIC-IV demo, real | Empatica cohort — healthy volunteers, **no ICU link** (E10) |
| Outcome labels | 120 composite events, 49 positive subjects | **none — no dataset here links wearable telemetry to an outcome** |
| Model | `lightgbm`, 67 features, AUPRC 0.493 | none until now; see §4 |
| Channels available | 9 vitals + labs | HR, IBI/HRV, EDA, accelerometer, skin temperature (+ SpO2 on consumer devices) |
| Escalation limbs that can fire | **3 of 3** | **1 of 3** — see §3 |
| Evaluation | 20-repeat grouped CV, bootstrap CIs, fairness audit | none until now |
| Reporting | `ml/evaluation/report.md` | `reports/post_discharge_digest.py`, a demo compression of minutes into "7 days" |

## 2. The root cause: care setting is not a routing dimension

`ObservationSource` already distinguishes `icu_monitor` from `wearable`, but
**nothing downstream branches on it.** `services/stream-processor/escalation.py`
holds one flat `SCORING_CHANNELS` set, and `risk-engine`'s `/score/live` applies
the ICU-recalibrated NEWS2 policy to whatever arrives. A patient at home wearing
a watch is therefore scored by a policy calibrated on the distribution of
seven-vital ICU patient-hours.

Everything in §3 follows from that single missing distinction.

## 3. The provable defect: two of three escalation limbs are dead post-discharge

The escalation rule has three limbs (finding F1): aggregate tier, any red
non-GCS parameter, and a falling GCS off sedation.

The ICU-recalibrated tiers are **medium ≥ 8, high ≥ 10**
(`warehouse/news2_report.md`). The maximum NEWS2 subscore reachable from heart
rate is 3, and from SpO2 is 3 — swept over the full physiological range against
`warehouse/news2.py`'s own scorers, not read off the breakpoint table:

```
max NEWS2 from HR alone            : 3
max NEWS2 from SpO2 alone          : 3
=> max aggregate for an HR+SpO2 wearable: 6
   medium (>= 8) reachable?  False
   high   (>= 10) reachable? False
```

So for a wearable patient:

- **the aggregate-tier limb can never fire** — 6 < 8, before clinical
  considerations even enter;
- **the GCS-drop limb can never fire** — a wrist has no consciousness sensor;
- only the **single red parameter** limb is live.

This is not a tuning question. It is arithmetic, and it silently removes
two-thirds of the alerting policy for the entire post-discharge arm. It also
explains an observation that looked incidental at the time: when a deteriorating
wearable stream was driven end-to-end through the real chain, the alert it
raised was `single-parameter red flag (RCP 2017)` — not because the aggregate
was close, but because **the aggregate limb is unreachable by construction.**

`news2_row()` already returns `components` (how many of the seven were
available) and `warehouse/news2.py` prints it as a diagnostic. Nothing consumes
it. That value is the hook the fix should hang on.

## 4. What is already closed

**The post-discharge arm now has a model.** `ml/evaluation/wrist_only.py`
retrains the primary task restricted to wearable-obtainable channels, under the
identical protocol (same labels, same grouped repeated CV, same bootstrap CIs):

| model | features | AUPRC (95% CI) |
|---|---|---|
| ICU full (LightGBM) | 67 | 0.493 (0.376–0.632) |
| Wrist, consumer class (HR+SpO2) | 15 | 0.269 (0.186–0.394) |
| Wrist, strict class (HR only) | 8 | 0.204 (0.124–0.356) |
| Wrist HR rule (untrained) | 1 | 0.113 (0.050–0.240) |

A wrist retains **~55%** of the full model's AUPRC on 15 features instead of 67,
and **SpO2 is the sensor that buys it** (consumer beats strict in 20 of 20
repeats). That is a procurement finding, not just a modelling one — and the
device this project actually has data from, the Empatica E4, is the one without
SpO2.

**This is an ICU-label proxy, not a readmission model**, and it is an optimistic
ceiling: the HR is nurse-validated hourly monitor HR, not motion-corrupted wrist
PPG. See the report's own "What it does not say".

## 5. The plan

Ordered by value per unit of effort. Items A and B are the ones that matter.

### A. Make tier thresholds provably reachable — *small, highest value*

Add the invariant the codebase currently lacks, as a **test**, not a comment:

> No configured tier threshold may exceed the maximum score achievable from the
> channel set available in that care setting.

That single test fails today for the post-discharge setting, which is the point.
Fix it by deriving post-discharge tiers from the achievable distribution — the
same 75th/90th-percentile recalibration `warehouse/news2.py` already performs for
the ICU, run over HR+SpO2-only scores — instead of inheriting seven-vital
cut-points. Keeps one method, two calibrations.

### B. Route on care setting — *medium, fixes the root cause*

1. Add `CareSetting` (`icu` | `post_discharge`) to the Observation contract,
   defaulted from `source` but explicit on the wire, because the mapping is not
   always one-to-one (a wearable can be worn on the ward).
2. `escalation.py` selects the threshold set and the scoring channel set by
   setting.
3. `risk-engine` serves the wrist model for `post_discharge` and the promoted
   ICU model for `icu`, behind the existing `/score/live` shape.

The escalation loop already refuses to score channels that cannot move a NEWS2
score; this generalises that instinct from *channels* to *settings*.

### C. Export and serve the wrist model — *small*

`run_all.py` promotes one model. Promote two, keyed by setting, so the
post-discharge path returns an ML probability with SHAP reasons rather than a
bare rule. The serving code needs no change beyond model selection.

### D. Carry the numbers into the digest — *small*

`reports/post_discharge_digest.py` scores a NEWS2 proxy over a synthetic
trajectory. Once C lands it should carry the wrist model's probability **and its
confidence interval**, with the §17 notice and the "ICU-label proxy" caveat
inline — so nobody reads the digest as a validated home-monitoring risk score.

### E. State the permanent limits — *documentation only*

Some of this gap is not an engineering backlog and should stop being written as
though it were:

- **No post-discharge outcome labels exist in any dataset here.** A genuine
  post-discharge risk model cannot be trained or validated in this project. The
  wrist model is a proxy and must always be labelled as one.
- **The digest's "week" is minutes of real recording** divided into seven
  buckets — already disclosed in its docstring, and belongs in the rendered
  report too.
- **The wearable cohort has no deterioration in it at all**, which is why
  `simulators/morphing.py` exists and why every morphed Observation is
  watermarked `synthetic`.

## 6. What "aligned" should mean here

Not parity — the data will not support parity. Aligned means the system is
**explicit about which arm it is operating in**, applies a policy calibrated for
that arm's sensors, serves a model that can actually run on them, and reports
each arm's limits in the arm's own artefacts. A and B get most of the way there;
E is what keeps the claim honest afterwards.
