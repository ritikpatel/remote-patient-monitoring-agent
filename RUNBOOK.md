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
Expect **313 passed, 3 skipped**. The 12 skips are infrastructure-gated and self-detecting —
they name exactly what is missing. This is the single best opening demo: it runs with no Docker.

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
All infra-gated skips clear. This is a strong demo moment: the same suite, ten more tests,
because real Kafka, Postgres, MQTT, HAPI and Keycloak are now reachable.

## 5. Start the services

```bash
start() { nohup .venv/bin/python -m uvicorn app:app --app-dir "services/$1" --port "$2" \
          > "/tmp/rpm-logs/$1.log" 2>&1 & }
mkdir -p /tmp/rpm-logs
start ingest-gateway 8000
start risk-engine 8001
start stream-processor 8003
start alert-service 8005
start notification-gateway 8006
start rag-service 8004
start agent-orchestrator 8007
start clinician-api 8008
HAPI_FHIR_BASE_URL=http://localhost:8090/fhir \
  nohup .venv/bin/python -m uvicorn app:app --app-dir services/fhir-mapper --port 8002 \
  > /tmp/rpm-logs/fhir-mapper.log 2>&1 &
sleep 25
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

**f. The replay**
```bash
.venv/bin/python simulators/icu_replay.py --stay-id 34617352 --compress 3600
```
One ICU hour per wall-clock second, LOINC-coded, with `imputed` flags on carried-forward values.
(This prints to console; it does **not** feed the services — finding F3.)

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
