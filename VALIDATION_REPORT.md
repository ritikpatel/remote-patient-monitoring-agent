# Independent validation report

**Reviewer:** Opus 5 · **Date:** 2026-09-08 · **Subject:** Phases 0–8 as built by Sonnet-5
**Method:** every claim re-run locally from a clean tree. Nothing below is taken from a README
without being independently reproduced, except the three items listed under "Not re-verified".

---

## Verdict

**The build is real and the engineering is honest.** 284 tests pass, the warehouse reconciles
exactly to the EDA, and the architectural guarantees in PROJECT_PLAN.md section 4 are implemented
in code rather than asserted in documentation. The READMEs consistently under-claim rather than
over-claim — including reporting a headline result (15.4% alert coverage) that reflects badly on
the system.

**Four defects found.** One is clinically material and has a quantified fix. None invalidate the
architecture.

> **Update 2026-09-08 — F1 and F2 are fixed and verified.** F1 raised event coverage from 15.4% to
> **41.0%**; F2 makes all seven mapped FHIR resource types publish with references resolving. Fixing
> F1 surfaced two further defects (a hand-copied second definition of the escalation rule in
> `eval/rag_agent.py`, and a FHIR `code` whitespace violation that 500'd every
> MedicationAdministration), both fixed. Suite is now **309 passed, 3 skipped**. F3 and F4 remain
> open. Details in each finding below.

> **Update 2026-09-08 (later) — F3 and F4 are also fixed and verified.** All four findings are
> now closed. Fixing F3 exposed a further defect (wearable skin temperature scored as core body
> temperature, which would have raised a false hypothermia alert for every wearable subject).
> **Appendix 3 corrects an overstatement this report itself made about the Trauma SICU subgroup.**
> Suite is **321 passed, 6 skipped**. See appendices 2 and 3.

**9 of 11 deliverables verified end-to-end by me. 2 are overstated** — deliverable 1 and
deliverable 5 pass in substance but fail their acceptance tests exactly as written.

---

## What I verified

| Area | Evidence reproduced |
|---|---|
| Test suite | 274 passed / 12 skipped with no infra; **284 passed / 2 skipped** with Kafka+Postgres+EMQX+HAPI+Keycloak up. 10 of 12 skips were genuinely infra-gated |
| Warehouse | `chartevents` 668,862 · `labevents` 107,727 · `icustays` 140 · **`hourly_grid` 12,004** — all match the EDA exactly. 65/65 concepts built. SOFA covers all 140 stays |
| E5 recalibration | `capstone.news2` carries both `tier_ward` and `tier_icu`. Confirmed live: NEWS2=8 is `high` on ward, `medium` on ICU |
| R6 (q4h dedup) | `dedup_key` buckets to `…T00:00:00` / `…T08:00:00`. Calendar buckets, with reasoning for why a rolling window would be wrong |
| R1 (leakage guard) | Rows at `hour >= h_event` are **dropped**, not labelled 0. Windows anchored to hours-since-ICU-admission |
| Grouped CV | `StratifiedGroupKFold` grouped on `subject_id` |
| ECG join | `merge_asof(direction="backward")`, 72h tolerance — past studies only, no future leakage |
| Death attribution | Independently confirmed their bug fix: 20 flag-carrying stays → **15 real death events**. Their cited case (hadm 22942076) verified: first ICU stay ends 2111-11-14 00:14, death 2111-11-15 17:20 during the *second* stay |
| Audit chain | Clean chain verifies; tampering row 3 detected at exactly `seq=3` |
| Agent graph | 6 nodes in plan order, every node wrapped in `audited()`. Escalation decided **before** the LLM is called; LLM output stored as advisory only |
| Fact ledger | 1,314 sentences, 1,674 citations, **0 dangling**, 0 inline/list mismatches. Catalog carries full `(table, row_id, column)` provenance |
| FHIR | `Patient` and `Device` mapped and accepted by live HAPI (server-assigned ids, versionId) |
| UI | `tsc -b && vite build` clean, 43 modules |
| K8s / Helm | 9 charts lint clean; 38 resources render — 9 Deployment, 9 Service, 9 NetworkPolicy, 9 PDB, **2 HPA** (exactly the two services the plan named). `default-deny` NetworkPolicy present |
| Latency | k6 passes against 5 live services |

---

## Findings

### F1 — Escalation drops NEWS2's single-parameter rule *(high, clinically material)* — **FIXED**

`services/agent-orchestrator/nodes.py:180` escalates on `tier == "high"` alone, and
`warehouse/news2.py:124` derives that tier purely from the **aggregate** score. NEWS2 (RCP 2017)
specifies a second, independent trigger: **a score of 3 in any single parameter mandates urgent
review regardless of total.** The component scores are computed and then discarded.

Observed live on stay 34617352 hour 35 — GCS 3 (deepest possible coma), SOFA-24h 12, FiO2 60%:

```
escalate            : False
escalation_reason   : ICU-recalibrated NEWS2 tier is 'medium', below the high threshold
llm_advisory        : I disagree, because the extremely low GCS and high SOFA score
                      indicate critical deterioration that warrants ICU escalation...
```

The LLM advisory was right and the policy was wrong. This patient died. The architecture
captured the disagreement in the audit log, which is the design working — but nothing acts on it.

**This is a second cause of the project's headline weakness** (15.4% alert coverage), distinct
from the "events happen 1–3h after admission" explanation in `eval/README.md`.

The naive fix is wrong: applying the strict rule fires on 59% of all patient-hours, because
GCS 3 is routine in sedated ICU patients. Excluding GCS and keeping the rule for the other
components was measured against all 78 composite events:

| Rule | Event coverage | Median lead | Fires on |
|---|---|---|---|
| Current (aggregate ICU tier == high) | 12/78 (15.4%) | 0.75h | 12.8% of hours |
| **+ single red parameter, excl. GCS** | **32/78 (41.0%)** | 0.49h | 31.8% of hours |
| Ward-standard aggregate (reference) | 31/78 (39.7%) | 0.48h | 48.8% of hours |

The proposed rule **dominates the ward-standard baseline** — slightly better coverage at a third
less alert burden. Cost is 2.5× more alerts than today.

Better still: MIMIC records sedation (2,020 administrations across 64 stays), so GCS could be
*qualified by concurrent sedation* rather than dropped outright.

**Fix:** add a `max_component` column in `warehouse/news2.py`, escalate on
`tier == "high" or max_component_excluding_gcs >= 3`, and document the deviation either way. The
current silent departure from the standard the code claims to implement is the actual defect.

### F2 — FHIR publish breaks referential integrity *(medium-high)* — **FIXED**

`fhir-mapper` POSTs resources, so HAPI assigns its own ids. `Patient/10006053` becomes
`Patient/2`, and every reference to it then dangles:

```
Encounter    -> HAPI-1094: Resource Patient/10006053 not found, specified in path: Encounter.subject
RiskAssessment -> same
Patient, Device -> 200 OK (no outbound references)
```

So deliverable 5's "HAPI FHIR validates **every** emitted resource" holds only for standalone
resources. `PUT` with the logical id doesn't help — HAPI rejects purely-numeric client-assigned
ids (`HAPI-0960`).

**Verified fix:** a transaction Bundle with conditional references. Tested working:

```
POST /fhir  (transaction Bundle, PUT Patient?identifier=…|10006053, PUT Encounter?identifier=…)
  -> 200 OK      Patient/2/_history/1
  -> 201 Created Encounter/4/_history/1
```

The mapper already emits the right business identifiers, so this is a contained change to
`hapi_client.py`.

### F3 — Replay simulators cannot drive the live pipeline *(medium)* — **FIXED**

`simulators/sinks.py` implements `ConsoleSink` and `JSONLSink` only. Line 6 acknowledges an
ingest-gateway sink "becomes a third Sink" — planned, never built. `grep` confirms nothing in
`simulators/` posts to `:8000/observations`.

The service chain genuinely works — `eval/load/ramp.js` drives obs → window → score → alert →
notify and passes. But it is driven by k6 fixtures, not by the replays.

Deliverable 1's acceptance test — "both replays raise alerts through one engine" — is therefore
**not demonstrable as written**, and it is marked ✅ in the README. This is the single biggest gap
between claimed and demonstrable, and it is roughly a 30-line `HTTPSink`.

### F4 — No fairness analysis, and `gender` is the #3 SHAP feature *(medium)* — **FIXED**

`gender` ranks third by mean |SHAP| in the promoted model, above most vitals. In this cohort the
association is statistically real — 66.2% of male stays have an event vs 42.9% of female
(Fisher OR 2.62, p=0.007) — but that is a 100-patient sample and will not transfer.

There is **no subgroup or fairness analysis anywhere** in `ml/` or `eval/`. `docs/compliance.md`
mentions fairness; nothing measures it. For a clinical AI capstone this is both a credibility gap
and a guaranteed question from any reviewer.

**Fix:** add per-subgroup AUROC/AUPRC to `eval/report.py`, and state explicitly whether `gender`
is retained as a legitimate clinical covariate (it is one, in SAPS-II and APACHE) or dropped.

### F5 — `ui/tsconfig.tsbuildinfo` not gitignored *(trivial)*

Reappears on every UI build and dirties the tree.

---

## Not re-verified by me

Stated for completeness — each is documented with specific evidence by the builder, and I have no
reason to doubt any of them, but I did not reproduce them in this session:

1. **Live `kind` cluster: HPA scaling under load, chaos-kill survival.** This machine gives Docker
   4GB; a kind cluster plus the compose stack would not fit. Manifests and charts verified
   structurally instead.
2. **Physical Wear OS watch.** `edge/wear_os/README.md` documents an emulator run, not hardware.
3. **Notes were generated with Groq `gpt-oss-120b`, not `claude-sonnet-5`.** Both backends are
   implemented with per-provider cost tracking; the plan specified Anthropic. A documented
   substitution, not a defect — but the generated corpus reflects the Groq model.

---

## Code review assessment

Better than the plan asked for, in three specific ways.

**It found bugs the plan didn't anticipate.** The death-attribution issue is genuinely subtle —
`hospital_expire_flag` propagates from admission to every ICU stay beneath it, and a naive
implementation labels an earlier, live-discharged stay as "about to die". They caught it, fixed
it correctly, and documented it with a reproducible example. I verified it independently.

**Design rules are implemented, not just cited.** R1, R2, R5, R6 all appear in code with comments
tracing back to the EDA finding that motivated them. R6's calendar-vs-rolling-bucket reasoning is
the kind of detail that gets lost between plan and implementation.

**Failure is reported rather than hidden.** The 15.4% alert-coverage result is prominent in both
the top-level README and `eval/README.md`, framed as a real finding. The compose README explicitly
declines to claim a simultaneous nine-service `up` that wasn't exercised. Skipped tests
self-detect missing infrastructure and name where to start it.

The weaknesses cluster in one place: **claims in the deliverable-status table are more absolute
than the code supports.** F1, F2 and F3 are all cases where a ✅ is defensible in spirit but fails
the acceptance test as literally written. The per-phase READMEs are careful; the summary table is
not.

---

## Appendix — how F1 and F2 were fixed (2026-09-08)

### F1 — the escalation rule now has three limbs

`warehouse/news2.py` gained `should_escalate()` and `escalation_reason()`, defined **once** and
imported by `agent-orchestrator`'s EscalationDecider, `eval/alerting.py`'s replay,
`eval/rag_agent.py`'s agreement metric, and `risk-engine`'s response. `capstone.news2` now stores
`max_component`, `max_component_nongcs`, `red_params`, `sedated` and `gcs_drop`.

| Limb | Trigger |
|---|---|
| 1 | ICU-recalibrated aggregate tier reaches `high` (E5, unchanged) |
| 2 | Any single non-GCS parameter scores 3 (RCP 2017) |
| 3 | GCS falls ≥2 points within 4h with no sedative running |

**Why limb 3 exists, and why the "efficient" answer was rejected.** Limb 2 alone scores best on
coverage-per-alert (1.29 vs 1.25) — but it does **not** catch the patient who exposed the bug.
Stay 34617352's only red parameter was GCS, so a rule that ignores GCS leaves that death exactly
as unflagged as the original bug did. What separates that patient from a sedated one is
trajectory: GCS 7 for six hours, then 3, off sedation. Limb 3 costs ~1pp of alert burden and is
the clinically defensible answer; "we ignore GCS" is not something to tell a clinician when a
falling GCS is the textbook deterioration sign.

Verified live: that patient now returns `escalate: True`, reason `GCS fell >= 2 points within 4h
with no sedative running` — and the LLM advisory, which previously *disagreed with the policy and
was right*, now agrees.

| Metric | Before | After |
|---|---|---|
| Events with a preceding alert | 12/78 (15.4%) | **32/78 (41.0%)** |
| Median lead time | 0.75h | 0.63h |
| Patient-hours alerted | 12.8% | 32.9% |
| Alerts per patient-day | 1.59 | 3.56 |

Fixing this surfaced a further defect: `eval/rag_agent.py`'s `escalation_agreement` had a
**hand-copied second definition** of the rule. Because both copies were wrong in the same
direction, the metric reported 100% agreement for a policy that was dropping a NEWS2 trigger. It
now imports the shared predicate. This is the clearest argument in the codebase for defining a
rule once.

### F2 — FHIR publish keyed on business identifiers

`services/fhir-mapper/hapi_client.py` was rewritten. Publishing now goes through a `transaction`
Bundle: resources with a known business identifier are sent as **conditional updates**
(`PUT Patient?identifier=urn:mimic-iv:subject_id|10006053`) and outbound references are rewritten
to **conditional references**, which FHIR resolves server-side. `/fhir/_publish` accepts a list,
publishing it as one transaction so a referent and its dependants land together.

Verified against a live HAPI server: all seven mapped resource types publish, references resolve
to real server ids (`Patient/11` → `Encounter/12` → `MedicationAdministration` citing both), and
republishing is idempotent (same id, not a duplicate patient).

Non-FHIR reference strings the Observation contract legitimately carries (`ICUStay/…`,
`Subject/S05`) are deliberately left untouched — rewriting them would invent a mapping that does
not exist. That remains a separate modelling question.

**A further defect found while verifying this:** every `MedicationAdministration` for an admission
500'd, because FHIR's `code` primitive forbids consecutive whitespace and the EMAR drug name
`"Sodium Chloride 0.9%  Flush"` carries a double space. `mappers.py` now normalises the code token
and preserves the original verbatim in `display`.

### Still open

**F3** (replay simulators cannot drive the live pipeline) and **F4** (no fairness/subgroup
analysis, `gender` ranks #3 by SHAP) are unchanged. Neither was in scope for this pass.

### Verification

`324 passed, 2 skipped` with Kafka, Postgres, EMQX, HAPI FHIR and Keycloak running. `ruff check`
clean; `mypy` clean on the changed modules. New regression tests pin every limb, including
`test_the_motivating_case_now_escalates`, which names stay 34617352 hour 35 directly so this
specific patient can never silently stop escalating again.

---

## Appendix 2 — how F3 and F4 were fixed (2026-09-08)

### F3 — the replays now drive the live pipeline

F3 was larger than the missing sink it looked like. `sinks.py` had no HTTP implementation, but
`stream-processor` also never called anything downstream: it consumed and windowed, and stopped.
The only thing that had ever driven the full chain was `eval/load/ramp.js`, calling each service
itself. Four pieces were needed:

| Added | Why |
|---|---|
| `HTTPSink` (`simulators/sinks.py`) | Batched POST to ingest-gateway; both replays gain `--sink http` |
| `POST /score/live` (risk-engine) | `/score/{stay_id}/{hour}` is a warehouse lookup. A live producer — a replay, or a wearable with no `stay_id` at all — has vitals and nothing else |
| `capstone.news2_thresholds` | The ICU cut-points existed only as two locals in `main()` and two numbers in a markdown report, so live scoring could not apply the recalibrated tier |
| `EscalationLoop` (`stream-processor/escalation.py`) | Closes stream → score → alert → notify, using the same `should_escalate` predicate |

It deliberately does **not** call agent-orchestrator: the agent makes LLM calls and belongs *after*
a patient is alerting, not once per observation. Both paths share one predicate, so the cheap
deterministic path and the expensive narrative path cannot disagree.

Verified end to end against real Kafka:

```
ICU replay  (stay 34807493)  382 observations -> 49 scored -> 22 escalated -> 1 alert
   "NEWS2 3: single-parameter red flag (RCP 2017): rr scoring 3"     <- F1's limb, live
wearable replay (subject S05) 152,097 observations -> 0 alerts       <- healthy volunteer
```

**A bug I introduced and then found:** the first version throttled scoring by wall-clock time.
Under `--compress 3600` (or `--no-sleep`) a 40-hour stay arrives in under a second, so everything
after the first observation was throttled away: 336 observations in, one scored, no alert out. The
throttle is now measured in *stream* time, which means the same thing live and at 3600×.
`GET /escalation/stats` was added because nothing anywhere had made that silence visible.

**A pre-existing bug F3 exposed:** the Empatica E4's `TEMP` channel is **wrist skin temperature**
(31.5–33.9 °C on a healthy wrist) and was mapped onto `temp_c`, LOINC 8310-5 "Body temperature".
NEWS2's temperature component expects a core measurement and scores ≤35 °C as a red flag, so every
wearable subject would have raised a hypothermia alert. This was invisible before F3 because the
wearable path never reached NEWS2. Skin temperature is now a distinct device-native channel
(`temp_skin`) that no scorer can mistake for core, with a regression test pinning it.

### F4 — `gender` dropped, subgroup audit added

Two questions, answered in that order.

**Does it earn its place?** A 20-repeat grouped-CV ablation, same model and same folds:

| | AUPRC | Wins |
|---|---|---|
| with `gender` | 0.5065 | **13/20 repeats** |
| without | 0.4953 | |

A +0.011 delta inside a bootstrap CI roughly twenty times that wide, winning barely more often
than a coin flip. **Dropped.** `include_demographics` now defaults to False. Age and first care
unit stay — a validated severity covariate and clinical context, not proxies. The headline claim
survives without it: LightGBM still beats recalibrated NEWS2 in **20/20** repeats.

**Does one threshold land equally?** `ml/evaluation/fairness.py` reports per-subgroup AUROC/AUPRC,
event rate and alert rate by sex, age band and care unit, at a single shared threshold. Subgroups
below 200 rows or 10 positives are marked underpowered and their metrics withheld rather than
printed as numbers nobody should act on. Race is excluded and said so: at n=100 most categories
hold single-digit patient counts.

It found subgroup variation the cohort-average AUROC of 0.87 hides. **The first version of this
report then overstated what that variation showed — see the correction in Appendix 3.**

By sex, discrimination is essentially equal (AUROC 0.863 F / 0.803 M) and the alert-rate gap
(6.6% vs 16.6%) tracks a real in-sample event-rate difference rather than miscalibration.

**Three defects found while fixing F4:** `gbm._as_categorical`, `logistic.build_pipeline` and the
promoted-model manifest all hardcoded `gender`, so no ablation could run at all — an optional
feature was a structural requirement of three separate components. All three now follow the actual
frame.

### Verification

`324 passed, 2 skipped`. `ruff`, `black` and `mypy` pre-commit hooks pass.

One artefact is short of a full refresh: `eval/output/report.html` has current axis-1 and axis-2
numbers, with axis 3 (latency) and axis 4 (RAG/agent) marked skipped — the Groq daily token quota
(200k) was exhausted. A `--skip-agent` flag was added for exactly this, because the earlier
behaviour was worse: the run aborted on the rate limit and left a wholly superseded report on
disk. Re-run `python eval/run_eval.py` with quota to restore all four axes.

---

## Appendix 3 — correcting this report's own Trauma SICU claim (2026-09-08)

Appendix 2 reported Trauma SICU at **AUROC 0.469** and described it as *"worse than chance — the
model cannot rank this population at all."* Asked to fix that subgroup, I first tried to, and the
investigation showed the claim should never have been made.

### What the number actually is

Bootstrapping that estimate **resampled by patient** rather than by row gives:

```
Trauma SICU   AUROC 0.438   95% CI 0.13 - 0.91   (width 0.78)
```

An interval that wide is consistent with a useless model *and* with an excellent one. It supports
no claim in either direction. The count-based gate I had written — `n_rows >= 200 and
n_positives >= 10` — passed it comfortably at 453 rows and 17 positives, and published a midpoint
that was noise. Five other subgroups were in the same position (age 80+, CCU, MICU, MICU/SICU,
hour 24+).

### Four fixes attempted, and measured, before that became clear

| Hypothesis | Result |
|---|---|
| Late events are harder, and TSICU's are late | **Inverted.** Early events are *easier* (corr +0.79); the model scores 0.859 at hour 0–2 and 0.509 at hour 24+ |
| Missing personal-baseline features (deviation from the patient's own norm) | **Worse.** Hour 6–23 AUROC −0.093; 19 features against 42 late positives is overfitting |
| Early rows teach a "current severity" shortcut that inverts in trauma | **Not supported.** A late-only model scores 0.540 against the global model's 0.601 on the same rows — early rows help |
| Route to SOFA late (SOFA scores 0.927 on TSICU) | **Fixes TSICU (0.913) but costs 0.157 overall AUPRC** — 0.472 → 0.315. AUPRC is this project's headline metric by design |

A fifth, adding a SOFA limb to the *alerting* path, moved TSICU event coverage not at all (4/8
under every variant tested) while raising alert burden from 32.9% to 51.6%.

### What was actually fixed

**1. The gate now measures precision, not sample size.** `subgroup_metrics` bootstraps grouped by
patient — rows from one patient are not independent, and a row-level bootstrap reports an interval
far narrower than the data earns — and withholds any point estimate whose 95% CI exceeds 0.40.
Counts and the interval stay visible, so suppression is legible rather than a silent gap.

**2. `subgroups_of_concern` tests the CI's upper bound, not the point estimate.** A low midpoint
with a high upper bound means "not measured", not "bad". Under this test nothing in the current
cohort is confidently poor, which is the correct answer.

**3. The real, adequately-powered finding is time in stay, not a care unit.** It is now a subgroup
dimension in its own right:

| Time since ICU admission | AUROC (95% CI) | AUPRC |
|---|---|---|
| hour 0–5 | 0.899 (0.81–0.96) | **0.715** |
| hour 6–23 | 0.668 (0.51–0.88) | **0.074** |
| hour 24+ | withheld — CI 0.30–0.77 | — |

57% of training positives fall in the first two hours, so the headline AUPRC is carried almost
entirely by early-stay rows. That is a real limitation, well measured, and far more consequential
than any single care unit.

**4. The model now declares its validated scope on every prediction.** `serving.scope_for_hour`
adds `in_validated_scope`, `validated_scope_max_hour` and a `scope_note` to each ML response,
pointing a consumer to the deterministic NEWS2/SOFA endpoint — which is what the alerting engine
escalates on anyway — outside the first six hours. That is the actionable output of a subgroup
audit at this sample size: not a per-unit patch, but an honest statement of where the number
applies.

### What was not fixed

Trauma SICU's true performance is **unknown**, and this cohort cannot determine it: 453 at-risk
rows, 17 positive rows, 8 composite events. That is the honest answer, and it differs from both
"it is fine" and from the "worse than chance" this report originally asserted.
