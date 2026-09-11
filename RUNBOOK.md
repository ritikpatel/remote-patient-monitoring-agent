# Runbook — bring the platform up, in order

Every command below was executed in this order during validation on 2026-09-08, and
re-validated 2026-09-09 after the MQTT subscriber and event-studio's agent-orchestrator
picker were added, and works from a clean checkout. Times are from an M-series Mac with
Docker limited to 4GB.

## 0. One-time setup

```bash
cd "/Users/ritik/Documents/Captsone ISB AMPBA"
```

The virtualenv already exists and is Python 3.11. **Do not use bare `python`** — this shell
aliases it to 3.13, which has no dependencies installed. Always use `.venv/bin/python`.

```bash
.venv/bin/python -V          # expect: Python 3.11.14
```

## 1. Prove it works before showing anything (~55s)

```bash
.venv/bin/python -m pytest -q
```
Expect **402 passed, 16 skipped** with no infra running. The skips are infrastructure-gated
and self-detecting — they name exactly what is missing. This is the single best opening demo: it runs with no Docker.

## 2. The warehouse (already built — verify, don't rebuild)

`warehouse/mimic4_demo.db` is committed at 59MB. To confirm it reconciles to the EDA:

```bash
.venv/bin/python -c "
import duckdb; c=duckdb.connect('warehouse/mimic4_demo.db', read_only=True)
for n,q in [('chartevents','select count(*) from mimiciv_icu.chartevents'),
            ('hourly_grid','select count(*) from capstone.hourly_grid'),
            ('sofa stays','select count(distinct stay_id) from mimiciv_derived.sofa')]:
    print(f'{n:14} {c.sql(q).fetchone()[0]:,}')"
```
Expect 668,862 / 12,004 / 140 — the same numbers as the EDA notebook.

To rebuild from scratch (~4 min): `.venv/bin/python warehouse/build_duckdb.py && .venv/bin/python warehouse/run_concepts.py`

**Scaling past the demo.** The same scripts build a full MIMIC-IV release, with
`--cohort-subjects N` loading a seeded sample of N ICU patients rather than the
whole thing (a whole-release warehouse is tens of GB). Start with
`.venv/bin/python warehouse/fetch_mimic4.py`, which checks the landing zone and
this machine's disk and prints the credentialed-download command for you to run —
it downloads nothing and never handles a credential. How large a cohort is
answered by measurement, not guesswork:
[`ml/evaluation/reliability_report.md`](ml/evaluation/reliability_report.md).
See [`warehouse/README.md`](warehouse/README.md) for the full sequence.

## 3. Infrastructure (~40s to healthy)

```bash
docker compose -f infra/compose/docker-compose.yml up -d kafka postgres emqx
```

Wait for health, then add the JVM services (slower, ~15s more):

```bash
docker compose -f infra/compose/docker-compose.yml up -d hapi-fhir keycloak
docker compose -f infra/compose/docker-compose.yml ps
```

**Do not `up` everything.** At 4GB it will thrash. These five are what the tests need.

`hapi-fhir`'s own Docker healthcheck never reports `healthy` in this environment — its
image has no `/bin/sh` for the healthcheck's exec probe to run, found by watching `docker
compose ps` sit on `health: starting` indefinitely while `curl localhost:8090/fhir/metadata`
was already returning 200. Harmless and pre-existing (nothing here depends on the reported
status); don't wait on it — check the port instead if you want confirmation.

## 4. Re-run the tests — watch the skips disappear

```bash
.venv/bin/python -m pytest -q
```
Now **415 passed, 3 skipped**. This is a strong demo moment: the same suite, thirteen more
tests passing, because real Kafka, Postgres, MQTT, HAPI and Keycloak are now reachable —
including `test_mqtt_subscriber.py`'s real EMQX end-to-end test (edge_agent's `MqttPublisher`
in, `mqtt_subscriber.py`'s subscriber loop out). The 3 remaining skips are no longer infra:
one self-detects that no ECG dataset exists (retired on purpose — see PROJECT_PLAN.md's E9),
one self-detects whichever half of the "is a promoted model exported" pair doesn't apply in
this environment, and one wants the five services from step 5 already running locally for
`eval/tests/test_latency.py`.

## 5. Start the services

```bash
mkdir -p /tmp/rpm-logs
run() { env "$@" nohup .venv/bin/python -m uvicorn app:app --app-dir "services/$SVC" \
        --port "$PORT" > "/tmp/rpm-logs/$SVC.log" 2>&1 & }

SVC=risk-engine          PORT=8001 run
SVC=alert-service        PORT=8005 run
SVC=notification-gateway PORT=8006 run
SVC=rag-service          PORT=8004 run
SVC=clinician-api        PORT=8007 run
SVC=agent-orchestrator   PORT=8008 run

# fhir-mapper needs HAPI or publishing returns a clear 503 rather than a fake pass.
SVC=fhir-mapper PORT=8002 run HAPI_FHIR_BASE_URL=http://localhost:8090/fhir

# These carry the live pipeline. Without PUBLISHER_BACKEND the gateway keeps
# observations in memory; without RISK_ENGINE_URL/ALERT_SERVICE_URL stream-processor
# windows them and stops there. Both default to off, so the replay demo in step 6f
# silently does nothing if you skip these env vars. stream-processor does not need
# NOTIFICATION_GATEWAY_URL -- alert-service is the one caller of notification-gateway
# on a new alert now (it already defaults to http://localhost:8006, matching alert-
# service's own startup above), not EscalationLoop; see escalation.py's docstring.
# MQTT_HOST starts mqtt_subscriber.py's background loop against the emqx container
# from step 3 -- without it, ingest-gateway's real handle_mqtt_message is still
# correct and tested, just unreached by any live subscription (see step 6f-bis).
SVC=ingest-gateway PORT=8000 run \
  PUBLISHER_BACKEND=kafka KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  MQTT_HOST=localhost MQTT_PORT=1883
SVC=stream-processor PORT=8003 run \
  KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  RISK_ENGINE_URL=http://localhost:8001 \
  ALERT_SERVICE_URL=http://localhost:8005

sleep 26
curl -s http://localhost:8003/escalation/stats   # expect {"enabled": true, ...}
curl -s http://localhost:8000/mqtt/stats         # expect {"enabled": true, "connected": true, ...}
for p in 8000 8001 8002 8003 8004 8005 8006 8007 8008; do
  printf ":%s %s\n" "$p" "$(curl -s -m3 -o /dev/null -w '%{http_code}' http://localhost:$p/health)"
done
```
All nine should return 200. **`fhir-mapper` needs `HAPI_FHIR_BASE_URL`** or publishing returns a
503 — it degrades gracefully, but the demo needs it set.

## 6. The demo sequence

**a. Risk scoring, rule-based — shows the E5 recalibration**
```bash
curl -s http://localhost:8001/score/34617352/35 | .venv/bin/python -m json.tool
```
NEWS2=8, `tier_ward: high`, `tier_icu: medium`. Say this out loud: the same score means different
things on a ward and in an ICU, and the recalibration is why. Then point at `max_component_nongcs`,
`red_params` and `gcs_drop` — those are NEWS2's *second* and *third* escalation triggers, which the
aggregate tier alone cannot express (finding F1).

**b. The ML model, with reasons**
```bash
curl -s -X POST http://localhost:8001/score/ml/34617352/35 | .venv/bin/python -m json.tool
```
Returns probability, model name, the CV AUPRC it was validated at, and SHAP reasons.

**c. The agent graph — the centrepiece**
```bash
curl -s -X POST http://localhost:8008/run -H 'content-type: application/json' \
  -d '{"patient_ref":"ICUStay/34617352","stay_id":34617352,"hour":35}' \
  | .venv/bin/python -m json.tool
```
Six nodes run in order. Look at `escalate`, `escalation_reason`, `llm_advisory`, `summary`.

This patient is the demo's centrepiece. Their GCS fell 7 → 3 with no sedative running and they
died two days later. Under the original policy the system returned `escalate: False` — the
aggregate NEWS2 tier was only 'medium' — and the LLM advisory *disagreed with the policy and was
right*. That disagreement is what review finding F1 came from. The policy now escalates on the
GCS trajectory, the reason reads `GCS fell >= 2 points within 4h with no sedative running`, and
the advisory agrees. Show the before-and-after: it is the strongest story in the project, because
the architecture surfaced its own bug through the audit log.

**d. Alerting and deduplication**
```bash
curl -s -X POST http://localhost:8005/alerts -H 'content-type: application/json' \
  -d '{"patient_ref":"ICUStay/34617352","alert_type":"deterioration","severity":"high","message":"NEWS2=8"}'
# repeat the exact same command — repeat_count increments, no second alert
curl -s http://localhost:8005/alerts/active | .venv/bin/python -m json.tool | head -20
```
Point at `dedup_key`: it ends in a 4-hourly boundary. That is EDA finding E16 in production code.

**e. FHIR**
```bash
curl -s http://localhost:8002/fhir/Patient/10006053 -o /tmp/pat.json && cat /tmp/pat.json
curl -s -X POST http://localhost:8002/fhir/_publish -H 'content-type: application/json' --data @/tmp/pat.json
```
HAPI returns the stored resource with a server id and versionId.
Try `Encounter/22942076` too: it used to fail with `HAPI-1094` because a plain POST let HAPI
assign its own id and broke the reference. Publishing now uses conditional update on the business
identifier, so references resolve. Passing a **list** publishes them as one FHIR transaction:
```bash
curl -s http://localhost:8002/fhir/Encounter/22942076 -o /tmp/enc.json
.venv/bin/python -c "import json;print(json.dumps([json.load(open('/tmp/pat.json')),json.load(open('/tmp/enc.json'))]))" > /tmp/txn.json
curl -s -X POST http://localhost:8002/fhir/_publish -H 'content-type: application/json' --data @/tmp/txn.json | .venv/bin/python -m json.tool
```

**f. The replay — driving the live system**

Console first, to show the wire shape: one ICU hour per wall-clock second, LOINC-coded, with
`imputed` flags on carried-forward values.
```bash
.venv/bin/python simulators/real_event_replay.py --stay-id 34617352 --compress 3600
```

Then the same replay into the running pipeline. **This is deliverable 1's acceptance test**, and
it needs `stream-processor` started with the escalation env vars from step 5:
```bash
.venv/bin/python simulators/real_event_replay.py --stay-id 34807493 --compress 3600 --sink http --no-sleep
curl -s http://localhost:8003/escalation/stats | .venv/bin/python -m json.tool
curl -s http://localhost:8005/alerts/active | .venv/bin/python -m json.tool
```
382 observations → gateway → Kafka → stream-processor → risk-engine → alert-service. Expect ~49
scored, ~22 escalated, and **one** alert: the 4-hourly dedup (R6) collapsing the repeats.

Now the same engine, a completely different producer — the post-discharge arm:
```bash
.venv/bin/python simulators/home_kit_stream.py --list-candidates
.venv/bin/python simulators/home_kit_stream.py --stay-id 30955999 --kit full_home \
  --sink http --gateway-url http://localhost:8000
```
A **real deteriorating MIMIC patient** streamed as a home monitoring kit would see them —
`Subject/HOME-<stay_id>`, `device_id=home-kit-sim`, every observation flagged `synthetic`. The
point worth making out loud is that this reaches the *same* engine with no per-source logic: the
channels a home kit cannot measure (core temperature, GCS, FiO2) simply never arrive, and
`/score/live` scores what it is given.

This replaced `simulators/wearable_replay.py`, which streamed a healthy volunteer who by
construction never alerted. That dataset was removed from the project (E10 retired): healthy
21-year-olds with zero deterioration events could not exercise the alerting path in the direction
that matters. See `simulators/README.md` for exactly which parts of this stream are real and which
are simulated.

**f-bis. The edge device's own path — MQTT, not the HTTP replay above**

Every replay above used `--sink http`, the same REST path `event-studio` uses. The Wear OS
watch's actual bridge (`edge_agent`) instead publishes over MQTT — this proves that whole,
previously-unconnected path for real, ingest-gateway's `MQTT_HOST` from step 5 required:
```bash
.venv/bin/python -m edge.edge_agent.agent make-demo --out /tmp/watch_demo.jsonl --minutes 2
.venv/bin/python -m edge.edge_agent.agent run --transport file --in /tmp/watch_demo.jsonl \
  --sink none --no-sleep --mqtt-host localhost --mqtt-port 1883
curl -s http://localhost:8000/mqtt/stats | .venv/bin/python -m json.tool
```
`MQTT connect to localhost:1883: ok`, `24 published, 0 buffered`, and `/mqtt/stats` showing
`messages_received: 24, errors: 0` — the same real `handle_mqtt_message` the REST path calls,
this time reached by an actual subscribed MQTT connection instead of never being called by
anything, which is what `mqtt_subscriber.py` closed (`services/README.md`'s `ingest-gateway`
row). Port 1883 here, not 8883: `--mqtt-port` defaults to the mTLS port
`edge/edge_agent/mqtt_publisher.py`'s design calls for, but `infra/compose/docker-compose.yml`'s
`emqx` only exposes plain 1883 (no cert material in this compose file, same stance as
`DEFAULT_API_KEY`'s docstring) — a real deployment sets both `--mqtt-port 8883` and the
`--ca-cert`/`--client-cert`/`--client-key` flags together.

**g. Audit chain**
```bash
curl -s http://localhost:8008/audit/verify | .venv/bin/python -m json.tool
```

**h. The dashboard**
```bash
cd ui && npm install && npm run dev     # http://localhost:5173
```

**i. event-studio — compose an event and watch it clear the whole chain live**

A second front door onto the same pipeline, for when there's no time to pick a
real stay. Needs `risk-engine`, `alert-service`, `notification-gateway` and
`rag-service` from step 5 up (not Kafka — this path calls them directly); add
`agent-orchestrator` too for the "Also assess" picker below.
```bash
AGENT_ORCHESTRATOR_URL=http://localhost:8008 uvicorn app:app --app-dir services/event-studio --port 8009
open http://localhost:8009
```
Drag severity toward the high end, **Generate** (preview only — nothing sent
yet), then **Send to pipeline**. Watch the step list: risk-engine's real score
and reason, alert-service raising a real alert (or deduping, R6, if you send
the same patient again inside the same 4h bucket), notification-gateway's real
channel routing and dry-run email + SMS results each with their own composed
`[SYNTHETIC DRILL]` text, and rag-service's real retrieved passages. A
mid-severity event escalates on the single red parameter (F1's limb), not the
aggregate — the same asymmetry the wearable replay in **f** demonstrates.

**One button, every component: also pick a real patient.** The "Also assess"
dropdown lists all 140 real demo stays (`GET /patients`, proxying
risk-engine). Pick one — try `ICUStay/34617352`, the same GCS-trajectory
patient from step **c** — and **Send to pipeline** also calls
`agent-orchestrator POST /run` for that stay's real `(stay_id, hour)`, exactly
step **c**'s curl command, now folded into this same click. This is the
answer to "does one composite-event test reach every component": the
composed vitals still drive the fast NEWS2 path above (unrelated to whatever
real patient is picked), and, independently, the picked patient's real chart
drives the full six-node agent graph, reported in its own panel underneath —
two different questions, on purpose, never conflated as if the agent had
reasoned about the composed vitals (`services/event-studio/README.md`'s
"Optionally, also agent-orchestrator" section explains why they're kept
separate rather than one being faked from the other).

Both email and SMS are dry-run by default and are only ever attempted by
`notification-gateway` after a genuine alert-service escalation — never from
**Generate**, and never twice for one alert (see `services/stream-processor/
escalation.py`'s module docstring for the double-notify bug that would
otherwise have meant twice). See `services/event-studio/README.md`'s
"Turning on live email" section for the exact env vars (a free Gmail app
password is enough — no billing account needed) and
`services/common/sms.py` for the same four guards on the SMS channel, kept
wired and configurable for whenever a Twilio account exists.

## 7. Reports and evaluation (pre-generated)

```bash
open eval/output/report.html                       # all four evaluation axes
open reports/output/shift_handover_medical.pdf
open ml/evaluation/report.md
open ml/evaluation/channel_dropout_report.md        # what the model does when a sensor goes missing
open ml/evaluation/wrist_only_report.md             # the post-discharge arm's model
open notebooks/01_capstone_eda.ipynb                # the EDA everything traces back to
```

## 8. Load test

```bash
k6 run eval/load/ramp.js
```

## 9. Shut down

```bash
pkill -f "uvicorn app:app"
docker compose -f infra/compose/docker-compose.yml down
```

---

## Local infra vs. a real deployment

Step 3's five containers exist so this runbook works on a laptop with no cloud account —
none of them are hand-rolled because AWS lacks a managed equivalent; they stand in for one:

| Local (`infra/compose/docker-compose.yml`) | AWS managed equivalent |
|---|---|
| `kafka` | MSK (or MSK Serverless) |
| `emqx` | AWS IoT Core's MQTT broker, or EMQX Cloud |
| `postgres` (audit log, `pgvector` for rag-service) | RDS for PostgreSQL (with the `pgvector` extension) |
| `keycloak` | Cognito, or Keycloak run as a managed container (ECS/EKS) rather than removed |
| `hapi-fhir` | HealthLake, or HAPI FHIR run as a managed container the same way |

None of this repo's application code talks to compose directly — every service reaches
infra through one env var each (`KAFKA_BOOTSTRAP_SERVERS`, `MQTT_HOST`, `AUDIT_DATABASE_URL`,
`HAPI_FHIR_BASE_URL`, Keycloak's realm URL), the same pattern `PUBLISHER_BACKEND=kafka` and
`MQTT_HOST` already use to switch between "nothing" (unit tests), "compose" (this runbook),
and, by pointing those same env vars at a real endpoint, a managed AWS service — no code
change, only config, matching every other "infra presence turns it on" gate in this project
(`_build_mqtt_subscriber`'s and `_build_escalation_loop`'s docstrings). Nothing in this
repository builds or deploys that AWS configuration yet — this table is the mapping, not a
claim that it has been stood up.

## Presenting this: the narrative that holds together

1. **The EDA found the problem.** Monitoring density collapses at ICU step-down (notebook §6)
   while risk does not. That gap is the reason the platform exists.
2. **The data said hourly, so the system is honest about hourly.** Real-time is produced by
   replay and by the watch, and every artefact says so.
3. **NEWS2 works today** — 128 of 140 stays trip it — so alerting never depended on the ML
   succeeding. The model had to beat NEWS2 to earn its place, and does, 20/20 CV repeats.
4. **The agent never does arithmetic.** Policy decides escalation; the LLM explains and advises,
   and every step is hash-chain audited.
5. **The honest result is the interesting one.** Alert coverage started at 15.4%. Say that early,
   then show what the review found: the policy was running only one of NEWS2's escalation triggers.
   Coverage is now 41.0%, still firing on less of the cohort than the ward-standard rule. Finding
   your ceiling and knowing why it was there is a better story than a clean number — and the fix
   came from the system's own audit log recording an LLM that disagreed with it.
