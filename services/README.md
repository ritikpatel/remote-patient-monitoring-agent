# Services

PROJECT_PLAN.md section 10: nine FastAPI microservices plus the LangGraph agent
engine. `services/common/` and `services/contracts/` are shared libraries, not
services themselves.

## The nine services

Phase 8 (`infra/compose/README.md`, `infra/k8s/README.md`) stood up and
verified most of what this table used to list as outstanding infra — the
"Real today" column below now reflects that; "Remaining gap" only lists what
is still genuinely not built, not infra that merely isn't running in any
given `pytest` session.

| Service | Port | Real today | Remaining gap |
|---|---|---|---|
| `ingest-gateway` | 8000 | REST ingress, schema validation (Pydantic), API-key auth, MQTT message handling; real `KafkaPublisher` wiring (`PUBLISHER_BACKEND=kafka`) verified end to end against a live broker (Phase 8) | A live EMQX subscriber loop — the broker is up and reachable, but nothing calls `handle_mqtt_message` from a real subscription yet |
| `stream-processor` | 8003 | Rolling stats, trend slopes, HRV (RMSSD), event-rate normalisation (R4) over HTTP; a real Kafka consumer (`kafka_consumer.py`) keeping a live rolling window per channel, verified end to end against a live broker (Phase 8) | Nothing — fully self-contained either way |
| `fhir-mapper` | 8002 | 11 real FHIR R4B resource mappers; `POST /fhir/_publish` actually posts a mapped resource to a live HAPI FHIR server and got back a HAPI-assigned id, verified for real (Phase 8) | Nothing for the core FHIR path |
| `risk-engine` | 8001 | Deterministic NEWS2 (ward + ICU-recalibrated) and SOFA; `/score/ml` serves Phase 5's real trained model + SHAP (`ml/models/serving.py`), a genuine 503 if none is exported; real HPA/PodDisruptionBudget/NetworkPolicy verified against a live `kind` cluster (Phase 8) | Nothing for the logic — see `ml/README.md`'s serving section for the one known perf limitation |
| `rag-service` | 8004 | Real TF-IDF retrieval over notes_synth's real generated notes + a small guideline corpus, returning fact-ledger IDs | Still TF-IDF, not live pgvector — the `vector` extension is created in Phase 8's Postgres (`infra/compose/postgres/init.sql`) but nothing queries it yet |
| `alert-service` | 8005 | Raise/dedupe (aligned to the real 4-hourly clock, R6)/suppress/escalate/acknowledge, SQLite-backed | Postgres for multi-instance deployment of this service's own alert store specifically (distinct from the audit log, which does have a real Postgres backend now — see `clinician-api`/`agent-orchestrator` below) |
| `notification-gateway` | 8006 | Real WebSocket broadcast (tested against real socket connections), overnight-escalation routing (E16) | FCM credentials (the real HTTP v1 call is implemented; `NoopPushSender` is what runs today) |
| `clinician-api` | 8007 | Real BFF aggregating the above over HTTP, SMART-scope enforcement (JWT), audit logging; real Keycloak-issued RS256 tokens verified against a live realm's JWKS (Phase 8); audit log can run on real Postgres (`AUDIT_DATABASE_URL`), tamper-detection verified against a live instance | Nothing left for auth or audit specifically |
| `agent-orchestrator` | 8008 | The full LangGraph graph, all three Phase 4 constraints enforced and tested, real Groq LLM integration; audit log can run on real Postgres, same as `clinician-api` | Swap `GROQ_API_KEY` for `ANTHROPIC_API_KEY` to match PROJECT_PLAN.md's claude-sonnet-5 default — otherwise fully functional |

Every service exposes `/health` and, since Phase 8, a real Prometheus
`/metrics` (`services/common/observability.py`) and OpenTelemetry tracing to
Jaeger when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Every one has a
`Dockerfile`; `risk-engine`, `agent-orchestrator`, and `rag-service` were
**built and run for real** in this environment during Phase 4 (see below),
and `ingest-gateway`, `stream-processor`, `fhir-mapper`, and `clinician-api`
were built and run for real again during Phase 8 to verify the Kafka/HAPI
FHIR/Keycloak paths above — the remaining two (`alert-service`,
`notification-gateway`) use the identical Dockerfile pattern and were not
independently container-verified, for time, not because of any known
difference. `risk-engine`'s Dockerfile changed again in Phase 5 (adds `ml/`
for `/score/ml`) and was re-verified with a real container build and run —
that pass found and fixed two real container-only bugs (a missing `libgomp1`
system package LightGBM needs, and `torch` pulling in an entire unused CUDA
toolkit on Linux); see `ml/README.md`'s Docker verification section for the
full account.

## The agent graph (`agent-orchestrator`)

`VitalsMonitor -> LabInterpreter -> RiskScorer -> ContextRetriever -> EscalationDecider -> Summarizer`
(`services/agent-orchestrator/graph.py`), and the three constraints that make it
production-grade rather than a demo (`nodes.py`'s docstring), each backed by a test
that *proves* it rather than just asserting it:

1. **The LLM never computes a risk score** — `risk_scorer` only relays
   risk-engine's HTTP response (`test_nodes.py::test_risk_scorer_relays_risk_engine_verbatim`).
2. **EscalationDecider is policy-first** — `test_nodes.py::test_escalation_decider_escalates_on_high_tier_despite_contrary_llm_advice`
   wires in an LLM double that explicitly recommends *against* escalating, and
   proves the policy escalates anyway.
3. **Every step is audited** — `services/common/audit.py`'s hash chain; a real run
   writes 6 rows (one per node) and `verify_chain()` proves none were altered
   (`test_graph.py::test_audit_chain_is_intact_after_a_full_run`).

A real end-to-end run (Groq LLM, real risk-engine/rag-service, real warehouse data)
produced a well-grounded clinical summary with zero fabricated values — see the
session transcript or re-run `notes_synth`'s pattern with `GroqBackend`.

## Real Docker verification

Docker was available in this environment, so three representative services were
actually built and run as containers (not just written and assumed to work):

```
risk-engine:        built, run standalone, real /score request against the mounted warehouse -- correct output
agent-orchestrator:  built, run on a real docker network alongside risk-engine + rag-service containers,
                     a real /run request produced a correct escalate=True decision, 3 retrieved passages,
                     and an intact 6-row audit chain -- entirely container-to-container over HTTP
rag-service:         built as part of the above
```

Two real bugs were caught by actually running things, documented where they were
fixed (not just here): a `sys.modules` collision from every service sharing the
`app.py` basename (`services/common/testing.py`), and an httpx `ASGITransport`
being async-only (`clinician-api/app.py`, `agent-orchestrator/tests/test_nodes.py`).

## Testing across service boundaries without live infra

Where one service calls another over HTTP, its tests use `httpx`'s (or Starlette
`TestClient`'s) ASGI transport pointed directly at the real downstream FastAPI `app`
object — a genuine in-process HTTP call (status codes, JSON (de)serialization,
routing) with no live socket or docker-compose needed. `services/common/testing.py`
explains why this needs a shared helper rather than a plain `import app`.

## Cross-service module loading

`services/common/testing.py::load_service_app` — every service's entry point is
named `app.py` by convention (see each Dockerfile's `--app-dir` CMD, since a
hyphenated directory name like `risk-engine` cannot be part of a dotted Python
import path). Running the whole repo's tests in one pytest process means a plain
`import app` from one service's test file silently returns a *different* service's
already-cached module. Every service's own tests, and every cross-service test,
load through this helper instead.
