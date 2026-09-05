# Agentic AI Platform for Remote Patient Monitoring in ICU and Post-Discharge Care

**Capstone build plan — ISB AMPBA**
Version 2.0 · Supersedes v1.0 · Reconciled against the executed EDA (`01_capstone_eda.ipynb`, 71 cells, 22 figures)

---

## 1. Context

Greenfield project. Three PhysioNet demo datasets are on disk. A full EDA has been completed and is
reproducible end-to-end in `01_capstone_eda.ipynb`; every number quoted in this plan traces to a cell in
that notebook.

### Decisions taken

| Decision | Choice | Consequence |
|---|---|---|
| Data access | **Demo data only** — no PhysioNet credentialing | Clinical narrative must be synthesised; all clinical metrics are pipeline-validity demonstrations |
| Build depth | **Full production-grade** | Real Kubernetes, Kafka, TLS, audit logging, load testing, CI/CD |
| Streaming | **Replay and live watch both** | One stream contract; three producers |

The production-grade choice was made with the timeline risk stated and accepted. Section 10 carries the
critical path and the descope order if schedule pressure appears.

### The motivating finding

EDA §6 shows ICU monitoring and intervention density holding near peak until roughly 55–60% of the
hospital stay, then collapsing at the step-down to the ward — while the patient's underlying risk does
not collapse with it. **The observation stream stops well before the risk does.** This gap, visible
directly in the data rather than asserted from literature, is the argument for the platform and should
open the report.

---

## 2. What the EDA established

Every design rule in this plan derives from one of these.

| # | Finding | Evidence | Design consequence |
|---|---|---|---|
| E1 | ICU vitals are charted **hourly**, not streamed | 77.7% of HR gaps fall in 55–65 min; only 3.9% under 5 min | Native cadence is hourly. "Real-time" is produced by replay + live watch, and must be labelled as such |
| E2 | Hourly grid is the only well-powered training surface | 12,004 patient-hours vs 140 stays | Primary task is hourly deterioration, not whole-stay mortality |
| E3 | Vital completeness is uneven | HR/RR/SpO2 ≥97%; SBP 62%; temp 27% | Carry-forward mandatory; `*_was_imputed` and `hours_since_last_obs` are features |
| E4 | **NEWS2 already works** | 110/140 stays reach ≥5; 68 reach ≥7; median 6 of 7 components available | Alerting deliverable ships in Phase 1. Every ML model must beat it |
| E5 | NEWS2 is ward-derived, not ICU-derived | 79% of ICU stays trip the medium threshold | Thresholds must be recalibrated to the ICU population before use as an alert trigger |
| E6 | Labels are scarce | 20 ICU deaths, 53 readmissions, 52 vasopressor stays | Methodology carries credibility, not headline metrics |
| E7 | All six SOFA organ systems computable | Vasopressors 52, vent 66, urine 137 stays, labs 100% | `mimic-code` concept SQL is reusable |
| E8 | Sepsis-3 reachable | 121 admissions have both an antibiotic order and a culture | Sepsis pathway viable |
| E9 | ECG links to the clinical cohort | 92/100 patients, 12-lead 500 Hz, 10 s | Genuine multimodal fusion; but episodic, and **unlabelled** (no `machine_measurements`) |
| E10 | Wearables are healthy volunteers | Median age ~21 vs ICU median 63; no patient link | Transport/DSP testbed only. Deterioration scenarios require documented morphing |
| E11 | No `note` module | Confirmed in dataset README | Synthetic narrative required |
| E12 | Admission burst is an **ordering** burst | Transfers 7.5×, orders 2.9×, but ICU monitoring only 1.31×; peak 33.7 ev/pt/hr vs 18.2 baseline | Simulator needs per-family arrival models. Size load tests at 33.7, not the mean |
| E13 | Discharge taper (4.7×) exceeds the admission burst (1.4×) | 16.9% of events in first 10% of stay vs 3.6% in last 10% | Stay-relative windows leak. Anchor to hours-since-ICU-admission |
| E14 | Monitoring collapses at ICU step-down | ICU families hold to ~55–60% of stay then fall off a cliff | The motivating finding. Justifies the post-discharge arm |
| E15 | Measurement intensity tracks outcome, but is confounded | Naive 13.5×; all 15 deaths were ICU patients; within-ICU it is **3.2×** | Event-rate features are legitimate but must be normalised by care setting |
| E16 | Care runs on a **4-hourly clock** | 48.4% of events on 00/04/08/12/16/20 vs 25% uniform. Labs peak 05:00, meds 08:00, orders 10:00 | Alerts cluster into six daily bursts. Dedup must align to the rhythm; overnight path required |

---

## 3. Deliverable traceability

Every deliverable has exactly one owning phase and a concrete acceptance test.

| # | Deliverable | Phase | Acceptance test |
|---|---|---|---|
| 1 | Real-Time Patient Monitoring System | P2, P4 | Live watch + both replays raise alerts through one engine |
| 2 | Agentic AI Clinical Monitoring Engine | P4 | Agent graph produces a scored, cited escalation decision with a full audit trail |
| 3 | Early Warning & Alert System | P1, P4 | Recalibrated NEWS2 fires with measured lead time to event |
| 4 | Predictive Risk Modeling Module | P5 | Beats NEWS2 on AUPRC in ≥15 of 20 CV repeats |
| 5 | FHIR-Based Integration Layer | P4 | HAPI FHIR validates every emitted resource |
| 6 | RAG-Powered Clinical Summarization | P3, P4 | Every claim traces to a fact-ledger entry |
| 7 | **Smart Hospital Connectivity Layer** | **P6** | Remote clinician sees live vitals and acknowledges an alert off-site |
| 8 | Multi-Source Data Pipeline | P1, P2 | Four sources land in one contract |
| 9 | Scalable Microservices Architecture | P4, P8 | HPA scales under k6 load; pods survive chaos kill |
| 10 | Evaluation & Validation Framework | P7 | Single report covering all four axes |
| 11 | Security & Compliance Layer | P8 | Audit chain verifies; NetworkPolicy denies by default |

**Declared outputs:** risk scores (P5) · deterioration alerts (P4) · clinical summaries (P3/P4) ·
**automated reports (P6)** · physician notifications (P4/P6).

---

## 4. Cross-cutting design rules

These are non-negotiable and derive directly from §2. They apply across every phase.

**R1 — Anchor all prediction windows to hours-since-ICU-admission, never to stay-relative position.**
(E13) A stay-relative window lets a model learn "the record has gone quiet" as a proxy for "about to be
discharged alive." Labels are evaluated only on windows ending strictly before the outcome.

**R2 — Every imputed value carries a flag.** (E3) A model that cannot distinguish a fresh measurement
from a four-hour-old carry-forward will be confidently wrong exactly when it matters.

**R3 — Absence is signal; model it, do not impute it away.** (E3, E15) Arterial-line presence (46% of
stays), NTproBNP ordering (27%), CRP ordering (19%) all encode clinician suspicion. Ordering behaviour is
a feature.

**R4 — Normalise event-rate features by care setting.** (E15) Un-normalised, the model learns "this
patient is in the ICU," which it already knows.

**R5 — Model each event family's arrival process separately.** (E12) Bursty for orders and transfers,
near-stationary with a 4-hourly comb for monitoring. One global rate is wrong in both directions.

**R6 — Align alert deduplication to the 4-hourly clock.** (E16) Otherwise clinicians see six synchronised
walls of alerts per day.

**R7 — Never present replayed or morphed data as measured.** (E1, E10) Every figure, table and slide
derived from replay or morphing carries that label inline.

---

## 5. Repository layout

```
capstone-rpm/
├── data/{raw,interim,processed}/      # raw gitignored; symlinks to the three dataset folders
├── notebooks/01_capstone_eda.ipynb    # the completed EDA (already built)
├── warehouse/                         # DuckDB build + mimic-code concepts + hourly grid
├── simulators/                        # icu_replay, wearable_replay, morphing, arrival_models
├── notes_synth/                       # LLM note generation + fact ledger
├── ml/{features,models,evaluation}/
├── services/                          # 9 FastAPI microservices
├── edge/{wear_os,edge_agent}/
├── ui/                                # clinician dashboard + mobile notification view
├── reports/                           # automated clinical report templates
├── infra/{compose,k8s,helm,observability,ci}/
├── eval/                              # validation framework + load tests
└── docs/                              # compliance mapping, architecture, honest-reporting statement
```

---

## 6. Phase 0 — Foundations

1. `git init`, pre-commit (ruff, black, mypy), `pyproject.toml`, Python 3.11 pinned.
2. `pip install duckdb` — **not currently installed**, and Phase 1 blocks on it.
3. GitHub Actions from day one: lint → unit tests → build all service images → integration smoke.
   Production-grade means CI exists before the code it guards, not after.
4. MLflow for model registry and experiment tracking; DVC for the synthetic-notes artefact.
5. `docs/DATA_USE.md` — PhysioNet DUA terms for all three datasets, and the rule that no raw data enters
   git.

---

## 7. Phase 1 — Warehouse and derived concepts

**Goal:** one queryable store plus every standard ICU severity score.

1. `warehouse/build_duckdb.py` — load all 31 CSVs into `mimic4_demo.db`, schemas `mimic_hosp` / `mimic_icu`.
   DuckDB reads `.csv.gz` natively. Prefer the `.csv.gz` originals; ignore the loose `.csv` duplicates
   (`icu/chartevents.csv`, `icu/outputevents.csv`, `icu/datetimeevents 2.csv`).
2. Vendor `MIT-LCP/mimic-code` → `mimic-iv/concepts_duckdb/`. Run in dependency order: `demographics` →
   `measurement` → `medication` → `treatment` → `organfailure` → `firstday` → `score` → `sepsis`.
3. **Expect partial failure.** `warehouse/run_concepts.py` executes each `.sql` in try/except and emits
   `concept_status.md`. Known thin areas from EDA: `rrt`/`crrt` (9 stays), `blood_differential`,
   anything note-derived. Patch or stub; do not abandon the set.
4. Must-work concepts: `icustay_detail`, `vitalsign`, `bg`, `chemistry`, `cbc`, `coagulation`,
   `ventilation`, `norepinephrine_equivalent_dose`, `urine_output`, `kdigo_stages`, `sofa`, `sapsii`,
   `oasis`, `sirs`, `charlson`, `suspicion_of_infection`, `sepsis3`.
5. `warehouse/hourly_grid.py` — port from EDA §5. One row per `(stay_id, hour_since_intime)`, hours before
   `intime` dropped, forward-filled with `*_was_imputed` and `hours_since_last_obs` per channel (**R2**).
   Reproduces 12,004 rows.
6. `warehouse/news2.py` — port from EDA §7, then **recalibrate thresholds to the ICU population** (**E5**).
   Report both ward-standard and ICU-recalibrated cut-points with their alert rates.

**Reuse, don't rebuild:** cohort-selection and time-binning logic from
`healthylaife/MIMIC-IV-Data-Pipeline` (`preprocessing/day_intervals_preproc/`, `preprocess_outcomes.py`),
reimplemented against DuckDB — their code assumes full-scale CSVs and a notebook flow that will not
survive the demo subset.

---

## 8. Phase 2 — Stream contract, simulators, edge

**Goal:** one message schema that all three producers satisfy.

1. `services/contracts/observation.py` — Pydantic, Avro-serialised. Fields: `patient_ref`, `device_id`,
   `source` (`icu_monitor|wearable|manual|lab`), `code` (LOINC where one exists), `value`, `unit`,
   `effective_time`, `ingest_time`, `quality_flags`. Deliberately FHIR `Observation`-shaped so Phase 4 is
   a projection, not a translation.
2. `simulators/arrival_models.py` — **per-family inter-arrival distributions fitted in EDA §6** (**R5**).
   Bursty for orders/transfers, near-stationary with a 4-hourly comb for monitoring.
3. `simulators/icu_replay.py` — replays `hourly_grid` at configurable time compression (1 h → 1 s), using
   the arrival models rather than a uniform tick.
4. `simulators/wearable_replay.py` — true-rate Empatica streaming (BVP 64 Hz, ACC 32 Hz, EDA/TEMP 4 Hz,
   HR 1 Hz). Honour `data_constraints.txt`: `f07` PPG/temp invalid, `S02` duplicated blocks, and
   `f14_a/b`, `S11_a/b`, `S16_a/b` as split sessions. These are useful fault fixtures, not nuisances.
5. `simulators/morphing.py` — condition wearable segments on a target NEWS2 trajectory (HR baseline shift,
   HRV suppression, SpO2 desaturation) so post-discharge deterioration scenarios exist (**E10**). Output is
   watermarked as synthetic (**R7**).
6. `edge/wear_os/` — Wear OS app sampling HR + accelerometer, batching over BLE to `edge/edge_agent/`,
   which buffers offline, computes windowed features locally, and publishes to MQTT (EMQX) with
   client-cert auth. MQTT→Kafka bridge lives in `ingest-gateway`.

---

## 9. Phase 3 — Synthetic clinical narrative

**Goal:** unblock RAG summarisation, and get evaluation ground truth as a by-product.

`notes_synth/generate.py` uses `claude-sonnet-5` to expand structured facts into discharge summaries,
daily nursing progress notes, and ECG/radiology report stubs.

**The critical design point: emit a fact ledger alongside every note.** Each generated sentence carries
the `(table, row_id, value)` tuples it derives from, written to `notes_synth/fact_ledger.parquet`. Real
MIMIC notes could not give you this. It means Phase 7 measures summarisation faithfulness mechanically —
every claim either traces to a ledger fact or is a hallucination. **The missing-notes weakness becomes
the evaluation methodology.**

Source facts: `diagnoses_icd` × `d_icd_diagnoses.long_title`, `prescriptions`, `procedures_icd`,
`microbiologyevents`, `labevents` abnormal flags, `services`, `transfers`, `discharge_location`, and the
Phase 1 severity scores.

Controls: generation runs once and is versioned via DVC; token spend is capped and logged per run; every
note is watermarked `SYNTHETIC — generated from MIMIC-IV demo structured data` in its header (**R7**).

---

## 10. Phase 4 — Microservices and the agent engine

Nine FastAPI services, one Dockerfile and one Helm chart each.

| Service | Responsibility |
|---|---|
| `ingest-gateway` | MQTT + REST ingress, schema validation, auth → Kafka `raw.*` |
| `stream-processor` | Windowing; rolling stats, trend slopes, HRV from IBI, **event-rate features normalised by care setting (R4)** |
| `fhir-mapper` | Projects onto HAPI FHIR: Patient, Encounter, Observation, Condition, MedicationAdministration, Procedure, DiagnosticReport, DocumentReference, RiskAssessment, Communication, Device |
| `risk-engine` | Serves Phase 5 models; computes recalibrated NEWS2 and SOFA deterministically |
| `rag-service` | pgvector over synthetic notes + guideline corpus; returns passages with ledger IDs |
| `agent-orchestrator` | LangGraph agent graph (below) |
| `alert-service` | Raise, dedupe, suppress, escalate, acknowledge — **dedup aligned to the 4-hourly clock (R6)** |
| `notification-gateway` | WebSocket to dashboard, FCM push to clinician phones, overnight escalation path |
| `clinician-api` | BFF for the UI; enforces SMART-on-FHIR scopes |

**Agent graph:** `VitalsMonitor` → `LabInterpreter` → `RiskScorer` → `ContextRetriever` (RAG) →
`EscalationDecider` → `Summarizer`.

Three constraints make this production-grade rather than a demo:

- **The LLM never computes a risk score.** It reads `risk-engine` output. Numeric reasoning is
  deterministic; the LLM does synthesis and explanation only.
- **`EscalationDecider` is a policy engine with LLM advisory input, not the reverse.** A recalibrated
  NEWS2 above threshold escalates regardless of what the model says.
- **Every agent step writes `{input_hash, tool_calls, output, model_id, tokens, latency_ms}` to the audit
  log.** This serves observability, cost control, and the compliance deliverable simultaneously.

---

## 11. Phase 5 — Predictive models

**Primary task: hourly deterioration.** Composite event — death, vasopressor initiation, ventilation
initiation, or unplanned ICU readmission — within the next 6 h and 12 h, from the `hourly_grid`. This is
the only task with adequate positives (**E2**: ~12k patient-hours vs 140 stays).

**Secondary:** ICU mortality and 30-day readmission, both reported with wide CIs and an explicit
underpowered caveat.

Given n=140, methodology carries the credibility:

- **Grouped splits by `subject_id`** — no patient spans train and test.
- Repeated stratified 5-fold CV, ≥20 repeats; bootstrap 95% CIs on AUROC and AUPRC.
- **AUPRC is the headline metric**, not AUROC — base rates are low.
- Calibration curves and Brier score. A miscalibrated alerting model is worse than none.
- **Baselines that must be beaten before any claim:** recalibrated NEWS2, SOFA, and logistic regression on
  age + last HR/RR/SpO2. If gradient boosting does not beat NEWS2, report that as the result.
- Models: L2 logistic regression → LightGBM → small GRU over the 24 h window. Stop at whichever wins; no
  transformer at this n.
- **ECG fusion:** derive per-study features (rate, QRS duration, QTc, rhythm class) with `neurokit2` —
  necessary because the demo has no `machine_measurements` labels (**E9**). Join on `subject_id` + nearest
  `ecg_time`; report the delta with and without.
- SHAP attribution surfaced through `risk-engine` so every alert carries a reason.
- All runs logged to MLflow; the promoted model is a registry artefact, not a pickle in a folder.

---

## 12. Phase 6 — Smart Hospital connectivity and clinical reporting

*This phase closes the two gaps found when auditing v1.0 against the deliverable list.*

**Clinician dashboard** (`ui/`) — React + WebSocket:
- Ward view: all monitored patients ranked by current risk, colour-banded by escalation tier.
- Patient view: live vitals with the NEWS2 trace, contributing SHAP factors, retrieved note passages with
  ledger citations, and the agent's escalation rationale.
- Alert inbox with acknowledge / escalate / suppress, writing back to `alert-service`.
- Mobile-responsive so an off-site clinician can act from a phone — the "physician away from the hospital"
  scenario is the deliverable, so it must be demonstrated on a phone form factor, not just claimed.

**Automated clinical reports** (`reports/`) — the missing declared output:
- Shift handover summary per ward, generated on the 4-hourly boundary (**E16**).
- Daily patient summary with trajectory, interventions and outstanding risks.
- Post-discharge weekly digest from wearable telemetry.
- All rendered from the agent `Summarizer` with fact-ledger citations, exported to PDF and FHIR
  `DocumentReference`.

---

## 13. Phase 7 — Evaluation and validation framework

`eval/` produces one HTML report across four axes:

1. **Prediction** — AUROC/AUPRC with CIs, calibration, decision-curve analysis, against all three baselines.
2. **Alerting** — sensitivity at fixed alert budget, alerts per patient-day, **median lead time to event**
   (the metric that matters clinically), false-alarm rate **stratified by hour of day** (**E16**).
3. **Latency** — p50/p95/p99 ingest → feature → score → alert → notification under k6 load at 10×, 100×,
   1000× device count. **Peak sizing uses 33.7 events/patient/hour, not the mean** (**E12**).
4. **RAG and agent** — retrieval recall@k, faithfulness against `fact_ledger.parquet`, escalation agreement
   with the rule-based policy, LLM cost per patient-day, and manual review of 50 summaries.

---

## 14. Phase 8 — Infrastructure, security, compliance

- **Local:** `docker-compose.yml` — Kafka, EMQX, Postgres+pgvector, HAPI FHIR, Keycloak, Prometheus,
  Grafana, Jaeger, MLflow, all nine services.
- **Kubernetes:** `infra/k8s/` + Helm charts. `kind` for development. HPA on `stream-processor` and
  `risk-engine`, PodDisruptionBudgets, liveness/readiness probes, NetworkPolicies denying by default.
- **Security:** TLS at ingress via cert-manager, mTLS between services, OIDC through Keycloak with
  SMART-on-FHIR scopes, sealed-secrets, encryption at rest on the Postgres PVC.
- **Audit:** append-only hash-chained table — every PHI read, agent decision, and alert acknowledgement.
  Tamper-evident by construction.
- **Compliance:** `docs/compliance.md` mapping each technical control to its HIPAA/GDPR safeguard. State
  plainly that MIMIC is already de-identified, so this is architecture demonstration, not a live
  compliance posture.

---

## 15. Verification

Run in order. Each step is independently checkable.

```bash
pip install duckdb
python warehouse/build_duckdb.py && python warehouse/run_concepts.py
```
`concept_status.md` lists ≥15 successful concepts. Assert `sofa` covers 140 stays and `hourly_grid` has
~12,004 rows.

```bash
pytest ml/ -v && python ml/evaluation/run_all.py
```
Assert the deterioration model beats recalibrated NEWS2 on AUPRC in ≥15 of 20 CV repeats.

```bash
docker compose -f infra/compose/docker-compose.yml up -d
python simulators/icu_replay.py --stay-id <a stay reaching NEWS2>=7> --compress 3600
```
An alert appears in the dashboard and a push notification fires. Confirm the payload carries a SHAP reason
and a RAG passage with a traceable ledger ID.

```bash
k6 run eval/load/ramp.js
```
Assert p95 ingest→alert latency under 2 s at 1000 simulated devices, sized at peak arrival rate.

**Live path:** pair the Wear OS watch, confirm it survives a 60 s network drop with no data loss, and
confirm its stream raises an alert through the same engine as the replay. Then acknowledge that alert from
a phone off the local network — this is deliverable 7's actual test.

---

## 16. Critical path and descope order

Sequential dependencies: **P0 → P1 → P2 → P4 → P5 → P7**. P3 can run in parallel after P1. P6 needs P4.
P8 runs alongside from P4 onward.

If schedule pressure appears, descope in this order — each step preserves a coherent, demonstrable system:

1. Kubernetes → docker-compose only (keep the manifests written, unoperated).
2. mTLS between services → TLS at ingress only.
3. GRU model → stop at LightGBM.
4. Chaos and 1000× load testing → 100× only.
5. Wear OS app → replay only.

**Never descope:** the fact ledger (it is the evaluation methodology), grouped CV (without it the metrics
are meaningless), or the audit log (it is a deliverable in its own right).

---

## 17. Honest-reporting requirement

Every artefact — report, slides, README, dashboard footer — states on first mention:

> This platform is validated on a 100-patient demo subset of MIMIC-IV. Clinical narrative is LLM-generated
> from structured data. Wearable deterioration signals are synthetically morphed from healthy-volunteer
> recordings. The engineering is real and the methodology is rigorous; the clinical performance figures
> demonstrate pipeline validity and do not transfer to clinical practice.

This belongs in the opening paragraph, not a footnote. With 20 ICU deaths and 53 readmissions, no clinical
claim from this data generalises — and saying so plainly is what makes the rest of the work credible.
