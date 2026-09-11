# Agentic AI Platform for Remote Patient Monitoring in ICU and Post-Discharge Care

**Capstone build — ISB AMPBA.** Full build plan: [PROJECT_PLAN.md](PROJECT_PLAN.md).
One composite event's journey to an alert, in order:
[`docs/workflow_simple.md`](docs/workflow_simple.md). Full end-to-end
architecture, every component labelled real/stubbed:
[`docs/workflow.md`](docs/workflow.md).

> This platform is validated on a 100-patient demo subset of MIMIC-IV. Clinical
> narrative is LLM-generated from structured data. Post-discharge signals are real
> MIMIC physiology passed through a simulated home sensor layer, and the
> post-discharge model is a transfer from ICU data with no post-discharge labels. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice. (PROJECT_PLAN.md section 17 — this notice belongs on
> every artefact this project produces, not just here.)

> **The two arms are not equal, and the title should not imply they are.** The
> ICU arm has real labels (120 composite events), a trained model, and a full
> evaluation. The post-discharge arm has **no outcome labels at all**, and the
> reason is structural rather than fixable: **MIMIC-IV records discharge,
> readmission and death *times* but no physiology whatsoever after `dischtime`**
> (E20). The volunteer wearable dataset that once stood in for this has been
> removed — healthy 21-year-olds with zero deterioration events could never
> supply the outcome (E10 retired).
>
> So the post-discharge arm is built as an explicit **transfer with a measured
> bound**, not a second validated model.
> [`ml/evaluation/home_kit_transfer_report.md`](ml/evaluation/home_kit_transfer_report.md)
> trains the full feature set with **kit-shaped channel dropout** and scores it
> under each assumed home sensor suite. Result: a full home kit (watch + BP cuff
> + CGM) reaches **AUPRC 0.345, retaining 70%** of the ICU model's 0.493, and a
> watch alone reaches 0.317. Dropout training beats masking-at-inference in
> **9-10 of 10** paired repeats for every kit — and costs the ICU arm 0.107
> AUPRC, losing all 10. **That reverses an earlier conclusion in this repo:** one
> gracefully-degrading model is *not* as good as two fits, so the ICU arm ships
> the dropout-free model and the post-discharge arm ships the dropout-trained
> one.
>
> The floor neither arm escapes: core temperature, GCS, FiO2 and arterial-line
> presence have **no home instrument at any price**, and no channel a home kit
> *can* measure correlates with them above |r| = 0.22 (notebook 03 §5). That gap
> is information, not a modelling artefact. Both studies are honest proxies and
> explicitly **not** a readmission model — for that, see E21: 53 readmissions in
> 260 live discharges (20.4%) is the best-conditioned task in this dataset.

## The finding that motivates this

The EDA (`notebooks/01_capstone_eda.ipynb`, 71 cells, 22 figures) shows ICU
monitoring and intervention density holding near peak until roughly 55–60% of
a stay, then collapsing at step-down to the ward — while the patient's
underlying risk does not collapse with it. **The observation stream stops
well before the risk does.** Phase 7's alerting analysis later found the same
gap from a different angle, independently: of 78 real composite deterioration
events, the original alerting rule caught only **15.4%** with any preceding
alert at all. Two causes, not one. Many patients deteriorate within 1–3 hours
of ICU admission, before an hours-scale monitoring cadence can accumulate
enough signal to fire — that part is a real limit of the data. But an
independent review (finding F1, [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md))
also found the escalation policy was tiering on NEWS2's aggregate score alone
and silently dropping RCP 2017's second trigger, "a score of 3 in any single
parameter". The rule now has three limbs — aggregate tier, any red non-GCS
parameter, and a falling GCS off sedation — which raises coverage to **41.0%**
while alerting on less of the cohort than the ward-standard rule does. Both
the limit and the fix are reported rather than smoothed over: see
[`eval/README.md`](eval/README.md) and `warehouse/news2.py`'s module docstring
for the measurement of every variant considered.

## Status: all 9 phases complete

| Phase | What it built | Key evidence |
|---|---|---|
| [0](PROJECT_PLAN.md#6-phase-0--foundations) | Repo scaffolding, DVC, CI, pre-commit | `.github/`, `.dvc/`, `.pre-commit-config.yaml` |
| [1](PROJECT_PLAN.md#7-phase-1--warehouse-and-derived-concepts) | DuckDB warehouse, 65 `mimic-code` concepts, hourly grid | [`warehouse/concept_status.md`](warehouse/concept_status.md) — 65/65 built, 0 failed |
| [2](PROJECT_PLAN.md#8-phase-2--stream-contract-simulators-edge) | One Observation contract, 3 real producers, a real Wear OS app | [`edge/wear_os/README.md`](edge/wear_os/README.md) — built and run on a real Wear OS emulator |
| [3](PROJECT_PLAN.md#9-phase-3--synthetic-clinical-narrative) | LLM-generated notes with a fact ledger (every sentence traces to a source row) | [`notes_synth/README.md`](notes_synth/README.md) |
| [4](PROJECT_PLAN.md#10-phase-4--microservices-and-the-agent-engine) | 9 FastAPI services + a LangGraph agent graph, Docker- and Helm-verified | [`services/README.md`](services/README.md) |
| [5](PROJECT_PLAN.md#11-phase-5--predictive-models) | Composite-deterioration model, beats recalibrated NEWS2 | [`ml/evaluation/report.md`](ml/evaluation/report.md) — 20/20 CV repeats |
| [6](PROJECT_PLAN.md#12-phase-6--smart-hospital-connectivity-and-clinical-reporting) | Clinician dashboard, live alert push, automated PDF reports | [`ui/README.md`](ui/README.md), [`reports/README.md`](reports/README.md) |
| [7](PROJECT_PLAN.md#13-phase-7--evaluation-and-validation-framework) | One report across prediction / alerting / latency / RAG+agent | [`eval/README.md`](eval/README.md) |
| [8](PROJECT_PLAN.md#14-phase-8--infrastructure-security-compliance) | Real Kafka, K8s (HPA/chaos/NetworkPolicy), Keycloak, Postgres audit, HAPI FHIR | [`infra/k8s/README.md`](infra/k8s/README.md), [`infra/compose/README.md`](infra/compose/README.md), [`docs/compliance.md`](docs/compliance.md) |

Commit history follows the same order: `git log --oneline --reverse` shows
one (or two, where a phase needed a re-verification pass) commit per phase,
each with a detailed message covering what was built, real bugs found while
building it, and how it was verified.

## Deliverable traceability (PROJECT_PLAN.md section 3)

| # | Deliverable | Owning phase | Acceptance test | Status |
|---|---|---|---|---|
| 1 | Real-Time Patient Monitoring System | P2, P4 | Live watch + both replays raise alerts through one engine | ✅ `--sink http` + stream-processor's escalation loop; verified live (F3 fixed) |
| 2 | Agentic AI Clinical Monitoring Engine | P4 | Agent graph produces a scored, cited escalation decision with a full audit trail | ✅ |
| 3 | Early Warning & Alert System | P1, P4 | Recalibrated NEWS2 fires with measured lead time to event | ✅ all three NEWS2 limbs; 41.0% event coverage, 0.63h median lead (F1 fixed) |
| 4 | Predictive Risk Modeling Module | P5 | Beats NEWS2 on AUPRC in ≥15 of 20 CV repeats | ✅ 20/20 under patient-level CV (F6 fixed a stay-level grouping leak); subgroup fairness audit; `gender` kept on ablation (F4 reversed) |
| 4b | …the same module, for the post-discharge arm | P5 | A model that runs on home-kit-obtainable channels only | ✅ **Transfer with a measured bound**, not a second validated model. Channel-dropout training under each assumed home kit: full kit **AUPRC 0.345 (70% retained)**, watch-only 0.317, beating inference-time masking in 9–10/10 paired repeats — and costing the ICU arm 0.107, so two fits ship rather than one ([report](ml/evaluation/home_kit_transfer_report.md)). The floor is physical: core temp, GCS, FiO2 have no home sensor and no proxy above \|r\|=0.22 |
| 4c | …the same module, for readmission | P5 | An admission-level 30-day readmission model on the cohort that matches the outcome | ⚠️ **Rebuilt and honestly negative.** Cohort corrected from 113 ICU admissions / 22 positives to **252 live discharges / 53 positives (21.0%)**; features corrected from ICU physiology to care-transition facts. Beats the previous model **19/20 paired repeats** (AUPRC 0.239→0.282) — but AUROC CI **0.412–0.590 includes 0.5**, so no usable model at this n ([report](ml/evaluation/readmission_report.md)) |
| 5 | FHIR-Based Integration Layer | P4 | HAPI FHIR validates every emitted resource | ✅ all 7 mapped resource types, references resolved, against a live HAPI server (F2 fixed) |
| 6 | RAG-Powered Clinical Summarization | P3, P4 | Every claim traces to a fact-ledger entry | ✅ |
| 7 | Smart Hospital Connectivity Layer | P6 | Remote clinician sees live vitals and acknowledges an alert off-site | ✅ |
| 8 | Multi-Source Data Pipeline | P1, P2 | Four sources land in one contract | ✅ |
| 9 | Scalable Microservices Architecture | P4, P8 | HPA scales under k6 load; pods survive chaos kill | ✅ both observed live on a real `kind` cluster |
| 10 | Evaluation & Validation Framework | P7 | Single report covering all four axes | ✅ |
| 11 | Security & Compliance Layer | P8 | Audit chain verifies; NetworkPolicy denies by default | ✅ both observed live |

## Beyond the plan's 9 phases

Several more real components were added after all 9 phases and 11 deliverables
above were independently verified complete (see `VALIDATION_REPORT.md`).
PROJECT_PLAN.md doesn't call for any of them — each earned its place by
measurement, or by closing a gap the phases above had already stated honestly
rather than hidden.

| Component | What it is | Evidence |
|---|---|---|
| [`services/event-studio/`](services/event-studio/README.md) | Compose a synthetic patient event in a browser and drive the real pipeline **synchronously** — risk-engine, alert-service, notification-gateway (email + SMS) and rag-service, in that order, in one request/response. Optionally, picking a real demo patient (`GET /patients`) also calls `agent-orchestrator` for that stay's real chart — a second, independent question from the composed vitals, never conflated with them | Verified against six real running services: severity 0.95 → real alert raised, real notification with real channel routing, dry-run email + SMS text, 3 real rag-service passages; a same-bucket resubmit correctly deduped with zero further alert/notify. The agent-orchestrator picker verified live through a browser against seven services (Groq LLM configured): a low-severity composed event alongside a real critical stay produced two correctly-disagreeing escalation verdicts in one response |
| [`services/common/email.py`](services/event-studio/README.md#escalation-email-and-sms-stated-plainly) + [`sms.py`](services/common/sms.py) | Two independent guarded paging senders, both sent for real by `notification-gateway` on every high-severity alert notification (not by event-studio, which only ever composes a text preview). Email is what this project demonstrates live (an SMTP account, e.g. a free Gmail app password, vs. SMS's funded Twilio account); SMS stays wired and configurable | 34 tests total (17 each) asserting a *refusal* except the mocked live-transport path; **a real, pre-existing double-notification bug found while wiring paging in** — `alert-service` and `stream-processor`'s `EscalationLoop` each independently called `notification-gateway` on a new alert, so every real alert was already notifying twice before this change made "twice" mean two pages. Fixed by making alert-service the one caller |
| [`ml/evaluation/channel_dropout.py`](ml/evaluation/channel_dropout_report.md) | Masks channels on the *promoted* ICU model at prediction time, instead of training a second model for the post-discharge arm | Settled the "one model or two" question the two-arm gap raised — see the callout above |
| [`ml/synthetic/`](ml/synthetic/README.md) | A faithful implementation of the EMR-WGAN tutorial (Yan et al., JMIR AI 2024) — preprocessing, WGAN-GP with per-block SoftMax, both training paradigms, and the paper's full nine-metric quality battery — used to settle whether synthetic data can substitute for the patients this cohort does not have | **It cannot, and that is now measured rather than argued.** The generator is genuinely good: dimension-wise distance 1.44, inside the 0.52–1.56 band the paper reports for its own runs on 181,294 patients. It still buys nothing — across 20 paired comparisons no augmented arm beats the real-only baseline (best −0.0006 AUPRC, 10/20 wins against a 15/20 bar), and generating whole class-balanced *patients* takes positive subjects from 39 per fold to **539** while AUPRC moves −0.0337. This closes the one gap [`synthetic_ceiling_report.md`](ml/evaluation/synthetic_ceiling_report.md) admitted it could not reach. **A finding that runs opposite to the paper:** membership-inference risk reaches 0.84 against a 0.51 chance floor and 1.0 for publishing the real records, verified against a swap control — so this synthetic cohort is **not de-identified and must not be shared**, which is the one use the method exists for |
| **Disease awareness, end to end** — [`warehouse/disease.py`](warehouse/README.md#disease-context-and-per-disease-thresholds), per-chapter NEWS2 cut-points, the `DiseaseContext` and `CarePlanner` graph nodes, patient-scoped RAG, and the diagnosis + care plan on the escalation email | The platform knew *that* a patient was deteriorating but nothing about *what they had*. Now: 140 stays carry a diagnosis chapter and 17 Charlson comorbidity flags; NEWS2 is recalibrated per chapter where one earns it; a new alert invokes `agent-orchestrator`, which scores it with the learned model, retrieves **this admission's own notes** plus guidelines, and generates a grounded care plan that reaches the clinician's inbox | Whole chain verified live end to end against five real services with a real LLM — [`eval/disease_aware_chain.py`](eval/disease_aware_chain.py). Leakage measured, not assumed: `dx_chapter` alone scores AUPRC **0.0392 vs a 0.0403 base rate** (no leak) and no disease arm beats disease-blind (best 4/20 paired repeats) — both reported in [`ml/evaluation/disease_leakage_report.md`](ml/evaluation/disease_leakage_report.md). **A real bug found by running it:** `/assess` defaulted to the stay's *latest* hour, so a patient alerting at hour 3 with NEWS2 11 was assessed at hour 40 with NEWS2 6 — the email contradicted its own subject line and the care plan was suppressed |

[`ml/evaluation/export_training_data.py`](ml/evaluation/report.md) shipped
alongside these: a reproducible export of the exact holdout split
`eval/prediction.py` uses, verified column-for-column against the promoted
model's manifest, so the training data behind any number in this repo can be
regenerated rather than hand-copied. Its output is gitignored — exported rows
are patient-derived, and `docs/DATA_USE.md`'s rule is that data never enters
git.

## Repository layout

```
data/{raw,interim,processed}/   raw gitignored; symlink to the MIMIC-IV demo folder
notebooks/                      01 cohort/coverage EDA, 02 patient medical history, 03 time series
warehouse/                      DuckDB build + mimic-code concepts + hourly grid
simulators/                     real_event_replay, home_kit_stream, arrival_models
notes_synth/                    LLM note generation + fact ledger
ml/{features,models,evaluation}/  labels, feature engineering, models, Phase 5 report
ml/synthetic/                   EMR-WGAN synthetic EHR generation + the paper's quality battery
services/                       9 FastAPI microservices + shared common/contracts, plus event-studio (a local-only demo front end, not one of the 9)
edge/{wear_os,edge_agent}/      real Wear OS app + BLE-to-MQTT bridge
ui/                              clinician dashboard (React)
reports/                         automated PDF clinical reports
infra/{compose,k8s,helm,observability,ci}/  docker-compose, kind/K8s manifests, Helm charts, Prometheus/Grafana
eval/                            validation framework + k6 load tests
docs/                            compliance mapping, data-use terms, workflow diagrams (full + simple)
```

## Quick start

```bash
# 1. Warehouse
pip install duckdb
python warehouse/build_duckdb.py && python warehouse/run_concepts.py
python warehouse/hourly_grid.py && python warehouse/disease.py && python warehouse/news2.py

# 1b. EDA (all three notebooks execute against the warehouse)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# 2. Models
pytest ml/ -v && python ml/evaluation/run_all.py

# 2b. The disease-aware escalation chain, end to end, against real services
python eval/disease_aware_chain.py

# 3. Services, locally (no infra needed for the deterministic paths)
uvicorn app:app --app-dir services/risk-engine --port 8001
# ... see services/README.md for the full set and which ones need Phase 8 infra

# 4. Full local infra (Kafka, Postgres, HAPI FHIR, Keycloak, Prometheus, ...)
docker compose -f infra/compose/docker-compose.yml up -d <services you need>
# see infra/compose/README.md — an 8GB machine can't run everything at once

# 5. Everything
pytest -q   # 431 passed, 15 skipped without infra
```

## What's real vs. what's honestly scoped

Every phase's own README states, specifically, what was verified by actually
running it versus what is structurally complete behind a clean interface
pending infra this environment doesn't have running continuously (a live
Kafka broker, a Kubernetes cluster, a physical Wear OS watch). Nothing in this
repo claims a capability that wasn't exercised for real somewhere in this
project's history — where a claim couldn't be verified, the relevant README
says so plainly rather than staying silent. Start with
[`infra/compose/README.md`](infra/compose/README.md) and
[`infra/k8s/README.md`](infra/k8s/README.md) for the most recent (Phase 8)
examples of that standard, and [`docs/compliance.md`](docs/compliance.md) for
what "architecture demonstration, not a live compliance posture" means
concretely.
