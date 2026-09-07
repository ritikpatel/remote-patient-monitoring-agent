# Agentic AI Platform for Remote Patient Monitoring in ICU and Post-Discharge Care

**Capstone build — ISB AMPBA.** Full build plan: [PROJECT_PLAN.md](PROJECT_PLAN.md).

> This platform is validated on a 100-patient demo subset of MIMIC-IV. Clinical
> narrative is LLM-generated from structured data. Wearable deterioration
> signals are synthetically morphed from healthy-volunteer recordings. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice. (PROJECT_PLAN.md section 17 — this notice belongs on
> every artefact this project produces, not just here.)

## The finding that motivates this

The EDA (`notebooks/01_capstone_eda.ipynb`, 71 cells, 22 figures) shows ICU
monitoring and intervention density holding near peak until roughly 55–60% of
a stay, then collapsing at step-down to the ward — while the patient's
underlying risk does not collapse with it. **The observation stream stops
well before the risk does.** Phase 7's alerting analysis later found the same
gap from a different angle, independently: of 78 real composite deterioration
events, alerts caught only **15.4%** with any preceding alert at all, because
many patients deteriorate within 1–3 hours of ICU admission — before an
hours-scale monitoring cadence can accumulate enough signal to fire. That is
reported as a genuine, important result, not smoothed over — see
[`eval/README.md`](eval/README.md).

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
| 1 | Real-Time Patient Monitoring System | P2, P4 | Live watch + both replays raise alerts through one engine | ✅ |
| 2 | Agentic AI Clinical Monitoring Engine | P4 | Agent graph produces a scored, cited escalation decision with a full audit trail | ✅ |
| 3 | Early Warning & Alert System | P1, P4 | Recalibrated NEWS2 fires with measured lead time to event | ✅ (lead time measured at 0.75h median — see the finding above) |
| 4 | Predictive Risk Modeling Module | P5 | Beats NEWS2 on AUPRC in ≥15 of 20 CV repeats | ✅ 20/20 |
| 5 | FHIR-Based Integration Layer | P4 | HAPI FHIR validates every emitted resource | ✅ verified against a live HAPI server (Phase 8) |
| 6 | RAG-Powered Clinical Summarization | P3, P4 | Every claim traces to a fact-ledger entry | ✅ |
| 7 | Smart Hospital Connectivity Layer | P6 | Remote clinician sees live vitals and acknowledges an alert off-site | ✅ |
| 8 | Multi-Source Data Pipeline | P1, P2 | Four sources land in one contract | ✅ |
| 9 | Scalable Microservices Architecture | P4, P8 | HPA scales under k6 load; pods survive chaos kill | ✅ both observed live on a real `kind` cluster |
| 10 | Evaluation & Validation Framework | P7 | Single report covering all four axes | ✅ |
| 11 | Security & Compliance Layer | P8 | Audit chain verifies; NetworkPolicy denies by default | ✅ both observed live |

## Repository layout

```
data/{raw,interim,processed}/   raw gitignored; symlinks to the 3 dataset folders
notebooks/                      the completed EDA
warehouse/                      DuckDB build + mimic-code concepts + hourly grid
simulators/                     icu_replay, wearable_replay, morphing, arrival_models
notes_synth/                    LLM note generation + fact ledger
ml/{features,models,evaluation}/  labels, feature engineering, models, Phase 5 report
services/                       9 FastAPI microservices + shared common/contracts
edge/{wear_os,edge_agent}/      real Wear OS app + BLE-to-MQTT bridge
ui/                              clinician dashboard (React)
reports/                         automated PDF clinical reports
infra/{compose,k8s,helm,observability,ci}/  docker-compose, kind/K8s manifests, Helm charts, Prometheus/Grafana
eval/                            validation framework + k6 load tests
docs/                            compliance mapping, data-use terms
```

## Quick start

```bash
# 1. Warehouse
pip install duckdb
python warehouse/build_duckdb.py && python warehouse/run_concepts.py

# 2. Models
pytest ml/ -v && python ml/evaluation/run_all.py

# 3. Services, locally (no infra needed for the deterministic paths)
uvicorn app:app --app-dir services/risk-engine --port 8001
# ... see services/README.md for the full set and which ones need Phase 8 infra

# 4. Full local infra (Kafka, Postgres, HAPI FHIR, Keycloak, Prometheus, ...)
docker compose -f infra/compose/docker-compose.yml up -d <services you need>
# see infra/compose/README.md — an 8GB machine can't run everything at once

# 5. Everything
pytest -q   # 274 passed, 12 skipped (skips self-detect missing optional infra)
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
