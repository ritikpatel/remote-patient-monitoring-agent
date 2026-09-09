# docker-compose (Phase 8)

PROJECT_PLAN.md section 14: "Kafka, EMQX, Postgres+pgvector, HAPI FHIR,
Keycloak, Prometheus, Grafana, Jaeger, MLflow, all nine services."

## Status: verified component-by-component, not as one simultaneous `up`

This machine has 8GB of RAM. Bringing up all nine service images (each
installs this monorepo's *entire* shared dependency set -- torch, lightgbm,
shap, neurokit2 -- per `services/*/Dockerfile`'s own comment on that pattern)
alongside Kafka, Postgres, HAPI FHIR (a JVM server with a genuinely slow cold
start), Keycloak, Prometheus, Grafana, Jaeger, and MLflow at once was not
something this environment could run and meaningfully verify simultaneously.
Rather than claim a full `docker compose up` succeeded when it wasn't actually
exercised end-to-end at that scale, every piece below was started, used for
real, and verified on its own (or in the small combination it actually needs),
then torn down before the next -- each result is real, the simultaneous
9-service claim just isn't made.

`docker compose -f infra/compose/docker-compose.yml config` (no `--profile` or
subset needed) does validate the *whole* file's syntax and interpolation for
real -- that check passes clean.

### Real Kafka wiring: ingest-gateway -> Kafka -> stream-processor

This is the actual new capability Phase 8 contributes on top of Phase 4-7's
direct-HTTP service chain (which Phase 7's `eval/README.md` explicitly flagged
as a scope gap): `services/common/publisher.py`'s `KafkaPublisher` and
`services/stream-processor/kafka_consumer.py`'s `KafkaConsumerThread` had
never been run against a real broker together before this phase.

Verified via `docker compose up kafka ingest-gateway stream-processor`, then a
real HTTP POST to `ingest-gateway`'s `/observations` and a `GET
/window/latest/...` against `stream-processor` over the compose network
(service DNS names, not localhost) -- the observation's real mean/std/slope
came back correctly. Also covered by
`services/stream-processor/tests/test_kafka_consumer.py`'s self-skipping
end-to-end test (skips if no broker is reachable at `localhost:9092`).

Three real bugs were found and fixed getting this to actually work end-to-end
(each documented in code where it was fixed, not just here):

1. **`auto_offset_reset="latest"` raced topic auto-creation.** A `raw.*` topic
   ingest-gateway creates on its first publish can already hold that first
   message before the consumer's periodic metadata refresh notices the new
   topic and resolves "latest" -- the seek lands *past* the message,
   permanently skipping it. Fixed by switching to `"earliest"`
   (`kafka_consumer.py`); a committed offset still governs every restart after
   the first, so this doesn't cause replay in steady state.
2. **`metadata_max_age_ms`'s 5-minute default** meant a pattern subscription
   only notices a brand-new matching topic on its next refresh, which could be
   minutes away. Lowered to 3s (`kafka_consumer.py`) -- this project's topic
   set is small and effectively static, so refreshing that often costs nothing
   real.
3. **A hardcoded `group_id="stream-processor"`** made this project's own
   pytest end-to-end test fight over the topic's one partition with an
   *actual* running docker-compose `stream-processor` instance consuming under
   the same group id on the same broker -- Kafka's group protocol assigns a
   partition exclusively to one member, so whichever won the race got the
   message. Fixed by making `group_id` a `KafkaConsumerThread` parameter;
   `stream-processor`'s own `app.py` still defaults to the real
   `"stream-processor"` group, only the test uses a unique one.
4. **Kafka's advertised-listener address is not one-size-fits-all.** A single
   `PLAINTEXT` listener advertising `kafka:9092` (correct for
   `ingest-gateway`/`stream-processor`, both on the compose network) silently
   broke every *host-side* kafka-python client -- including this project's own
   pytest suite -- with a DNS lookup failure on `kafka`. Fixed with two
   listeners (`INTERNAL://kafka:29092` for containers,
   `EXTERNAL://localhost:9092` for the host) rather than picking one audience
   over the other.

### Real Postgres-backed audit log

`services/common/audit_postgres.py`'s `PostgresAuditLog` -- the same
hash-chain schema and `compute_row_hash` formula as the SQLite `AuditLog`, so
a chain verifies identically regardless of which backend wrote it. Verified
for real against a dockerized `pgvector/pgvector:pg16` (record, tamper a row
via a second raw connection, `verify_chain()` catches it at the tampered
`seq`), and via `agent-orchestrator`'s actual `build_audit_log()` factory with
`AUDIT_DATABASE_URL` set. Self-skipping tests:
`services/common/tests/test_audit_postgres.py` (skips if no Postgres is
reachable).

### Real MQTT wiring: edge_agent -> EMQX -> ingest-gateway

The other half of the Kafka story above, for the edge device's own transport
rather than the replay simulators' HTTP one. `edge/edge_agent/mqtt_publisher.py`'s
`MqttPublisher` had run against a real broker since Phase 4 (a deliberately-closed
port proved its connect-fails path); `services/ingest-gateway/app.py`'s
`handle_mqtt_message` was always real, tested code. Nothing had ever run a live
subscription connecting the two -- this project's actual MQTT ingress path
stopped at the broker.

`services/ingest-gateway/mqtt_subscriber.py`'s `MqttSubscriberThread` closes
that gap: a `paho-mqtt` client subscribing to `capstone/observations/#`
(the same topic tree `MqttPublisher` publishes to), gated on `MQTT_HOST` the
same way `KafkaConsumerThread` above is gated on `KAFKA_BOOTSTRAP_SERVERS` --
idle in every test and any standalone run, started by `docker-compose.yml`'s
`emqx` service alone. Verified via `docker compose up emqx`, then
`edge_agent`'s real CLI (`python -m edge.edge_agent.agent run --mqtt-host
localhost --mqtt-port 1883`) publishing 24 real observations over a real MQTT
connection, and `GET /mqtt/stats` on a running `ingest-gateway` showing
`messages_received: 24, errors: 0` -- the same real `Publisher` the REST path
uses received all 24. Also covered by
`services/ingest-gateway/tests/test_mqtt_subscriber.py`'s self-skipping
end-to-end test (skips if no broker is reachable at `localhost:1883`), plus
pure-logic tests of the connect/message callbacks that need no broker at all.

### Real HAPI FHIR validation

`services/fhir-mapper/hapi_client.py`'s `POST /fhir/_publish` takes any
resource this service already mapped and actually POSTs it to a live HAPI
FHIR server -- a second, independent validation beyond `mappers.py`'s own
`fhir.resources`/Pydantic construction. Verified for real: a real warehouse
`Patient` resource, mapped and published to a dockerized
`hapiproject/hapi:v7.2.0`, came back with a HAPI-assigned `id` and
`meta.versionId`. Self-skipping test in
`services/fhir-mapper/tests/test_app.py` (skips if no HAPI server is
reachable at `localhost:8090`).

### Real Keycloak OIDC + SMART-on-FHIR scopes

`services/common/auth.py` already verified SMART scopes against locally-issued
HS256 test tokens (Phase 4); Phase 8 added real Keycloak (RS256, via
`KEYCLOAK_JWKS_URL`/`KEYCLOAK_ISSUER`) as an alternative verification path,
switched on only when those env vars are set -- the local HS256 path (every
test's default) is unchanged. `infra/compose/keycloak/realm-export.json`
defines a `capstone-rpm` realm, a `clinician-api` client, and client scopes
named exactly like this project's SMART scope strings
(`patient/Observation.read`, etc.).

Verified for real: a dockerized `quay.io/keycloak/keycloak:25.0` imported that
realm cleanly, issued a real RS256 access token via `client_credentials`
carrying exactly the requested scopes, and `services/common/auth.decode_token`
verified it against Keycloak's real JWKS and recovered the same scopes.
Self-skipping test in `services/common/tests/test_auth.py` (skips if no
Keycloak is reachable at `localhost:8180`).

### Not independently re-verified in this pass (real, but not re-run here)

- **rag-service / pgvector**: `retrieval.py` still runs real TF-IDF, not a
  live pgvector query -- the `postgres` service's `vector` extension is
  created (see `postgres/init.sql`) but nothing queries it yet. Tracked as
  future work, not claimed as done.
- **Prometheus / Grafana**: every one of the 9 services exposes a real
  `/metrics` (`services/common/observability.py`, verified directly with
  `curl http://localhost:8001/metrics` against a running `risk-engine`); the
  provisioned Grafana dashboard (`../observability/grafana/provisioning/dashboards/json/`)
  was written but not opened in a browser against a live Grafana in this pass.
- **Jaeger**: `services/common/observability.instrument_tracing` only
  activates when `OTEL_EXPORTER_OTLP_ENDPOINT` is set (every compose service
  sets it); a real trace was not independently confirmed to appear in
  Jaeger's UI in this pass -- the export path is real (OTLP/HTTP, not a stub),
  just not screenshotted end-to-end here.
- **MLflow**: `ml/evaluation/run_all.py`'s `MLFLOW_TRACKING_URI` now honours
  this env var instead of hardcoding a local sqlite file, so pointing it at
  `http://localhost:5000` (the compose service) works by construction -- the
  full training run was not re-executed against the containerized tracking
  server in this pass (it's a multi-minute run already verified against the
  local sqlite backend in Phase 5).

## Running it

```bash
set -a; source .env; set +a   # GROQ_API_KEY, for agent-orchestrator's real LLM path
docker compose -f infra/compose/docker-compose.yml up -d <services you need>
```

Given the RAM note above, bring up only what you're testing, e.g.:

```bash
docker compose -f infra/compose/docker-compose.yml up -d kafka ingest-gateway stream-processor
docker compose -f infra/compose/docker-compose.yml up -d postgres
docker compose -f infra/compose/docker-compose.yml up -d hapi-fhir fhir-mapper
docker compose -f infra/compose/docker-compose.yml up -d keycloak clinician-api
```
