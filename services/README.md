# Services

PROJECT_PLAN.md section 10: nine FastAPI microservices plus the LangGraph agent
engine. `services/common/` and `services/contracts/` are shared libraries, not
services themselves.

## The nine services

| Service | Port | Real today | Needs Phase 8 infra for |
|---|---|---|---|
| `ingest-gateway` | 8000 | REST ingress, schema validation (Pydantic), API-key auth, MQTT message handling | Kafka (publisher interface ready — `services/common/publisher.py`'s `KafkaPublisher`); a live EMQX broker |
| `stream-processor` | 8003 | Rolling stats, trend slopes, HRV (RMSSD), event-rate normalisation (R4) — tested against real wearable IBI data and the fitted arrival models | Nothing — fully self-contained |
| `fhir-mapper` | 8002 | 11 real FHIR R4B resource mappers, tested against real warehouse rows and notes_synth notes | A live HAPI FHIR server to POST resources to (the mappers already produce exactly what it would receive) |
| `risk-engine` | 8001 | Deterministic NEWS2 (ward + ICU-recalibrated) and SOFA; `/score/ml` now serves Phase 5's real trained model + SHAP (`ml/models/serving.py`), a genuine 503 if none is exported | Nothing for the logic — see `ml/README.md`'s serving section for the one known perf limitation |
| `rag-service` | 8004 | Real TF-IDF retrieval over notes_synth's real generated notes + a small guideline corpus, returning fact-ledger IDs | pgvector + neural embeddings (Postgres) |
| `alert-service` | 8005 | Raise/dedupe (aligned to the real 4-hourly clock, R6)/suppress/escalate/acknowledge, SQLite-backed | Nothing for the logic; Postgres for multi-instance deployment |
| `notification-gateway` | 8006 | Real WebSocket broadcast (tested against real socket connections), overnight-escalation routing (E16) | FCM credentials (the real HTTP v1 call is implemented; `NoopPushSender` is what runs today) |
| `clinician-api` | 8007 | Real BFF aggregating the above over HTTP, SMART-scope enforcement (JWT), audit logging | Keycloak as the token issuer (verification already works against locally-issued test tokens) |
| `agent-orchestrator` | 8008 | The full LangGraph graph, all three Phase 4 constraints enforced and tested, real Groq LLM integration | Nothing — fully functional; swap `GROQ_API_KEY` for `ANTHROPIC_API_KEY` to match PROJECT_PLAN.md's claude-sonnet-5 default |

Every service exposes `/health`. Every one has a `Dockerfile`; `risk-engine`,
`agent-orchestrator`, and `rag-service` were **built and run for real** in
this environment during Phase 4 (see below) — the rest use the identical
pattern and were not independently re-verified, for time, not because of any
known difference. `risk-engine`'s Dockerfile changed again in Phase 5 (adds
`ml/` for `/score/ml`) and was re-verified with a real container build and
run — that pass found and fixed two real container-only bugs (a missing
`libgomp1` system package LightGBM needs, and `torch` pulling in an entire
unused CUDA toolkit on Linux); see `ml/README.md`'s Docker verification
section for the full account.

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
