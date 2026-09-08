# Runbook — bring the platform up, in order

Every command below was executed in this order during validation on 2026-09-08 and works from a
clean checkout. Times are from an M-series Mac with Docker limited to 4GB.

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
Expect **312 passed, 14 skipped** with no infra running. The skips are infrastructure-gated
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

## 4. Re-run the tests — watch the skips disappear

```bash
.venv/bin/python -m pytest -q
```
Now **324 passed, 2 skipped**. This is a strong demo moment: the same suite, twelve more
tests passing, because real Kafka, Postgres, MQTT, HAPI and Keycloak are now reachable.

## 5. Start the services

```bash
mkdir -p /tmp/rpm-logs
run() { env "$@" nohup .venv/bin/python -m uvicorn app:app --app-dir "services/$SVC" \
        --port "$PORT" > "/tmp/rpm-logs/$SVC.log" 2>&1 & }

SVC=risk-engine          PORT=8001 run
SVC=alert-service        PORT=8005 run
SVC=notification-gateway PORT=8006 run
SVC=rag-service          PORT=8004 run
SVC=agent-orchestrator   PORT=8007 run
SVC=clinician-api        PORT=8008 run

# fhir-mapper needs HAPI or publishing returns a clear 503 rather than a fake pass.
SVC=fhir-mapper PORT=8002 run HAPI_FHIR_BASE_URL=http://localhost:8090/fhir

# These two carry the live pipeline. Without PUBLISHER_BACKEND the gateway keeps
# observations in memory; without RISK_ENGINE_URL/ALERT_SERVICE_URL stream-processor
# windows them and stops there. Both default to off, so the replay demo in step 6f
# silently does nothing if you skip these env vars.
SVC=ingest-gateway PORT=8000 run \
  PUBLISHER_BACKEND=kafka KAFKA_BOOTSTRAP_SERVERS=localhost:9092
SVC=stream-processor PORT=8003 run \
  KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  RISK_ENGINE_URL=http://localhost:8001 \
  ALERT_SERVICE_URL=http://localhost:8005 \
  NOTIFICATION_GATEWAY_URL=http://localhost:8006

sleep 26
curl -s http://localhost:8003/escalation/stats   # expect {"enabled": true, ...}
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
curl -s -X POST http://localhost:8007/run -H 'content-type: application/json' \
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

Now the same engine, a completely different producer:
```bash
.venv/bin/python simulators/wearable_replay.py --activity STRESS --participant S05 \
  --duration-s 900 --sink http --no-sleep
```
152,097 observations at true device rate (BVP 64 Hz), and **zero alerts** — because this is a
healthy volunteer. That is the point worth making out loud: the sick patient alerts, the healthy
one does not, through one engine with no per-source logic. (Until F3 let the wearable path reach
the scorer, it would have raised a false hypothermia alert on every subject — E4 `TEMP` is wrist
skin temperature, not core.)

**g. Audit chain**
```bash
curl -s http://localhost:8007/audit/verify | .venv/bin/python -m json.tool
```

**h. The dashboard**
```bash
cd ui && npm install && npm run dev     # http://localhost:5173
```

## 7. Reports and evaluation (pre-generated)

```bash
open eval/output/report.html            # all four evaluation axes
open reports/output/shift_handover_medical.pdf
open ml/evaluation/report.md
open notebooks/01_capstone_eda.ipynb    # the EDA everything traces back to
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

## Presenting this: the narrative that holds together

1. **The EDA found the problem.** Monitoring density collapses at ICU step-down (notebook §6)
   while risk does not. That gap is the reason the platform exists.
2. **The data said hourly, so the system is honest about hourly.** Real-time is produced by
   replay and by the watch, and every artefact says so.
3. **NEWS2 works today** — 110 of 140 stays trip it — so alerting never depended on the ML
   succeeding. The model had to beat NEWS2 to earn its place, and does, 20/20 CV repeats.
4. **The agent never does arithmetic.** Policy decides escalation; the LLM explains and advises,
   and every step is hash-chain audited.
5. **The honest result is the interesting one.** Alert coverage started at 15.4%. Say that early,
   then show what the review found: the policy was running only one of NEWS2's escalation triggers.
   Coverage is now 41.0%, still firing on less of the cohort than the ward-standard rule. Finding
   your ceiling and knowing why it was there is a better story than a clean number — and the fix
   came from the system's own audit log recording an LLM that disagreed with it.
