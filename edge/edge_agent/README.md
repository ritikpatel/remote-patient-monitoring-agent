# Edge agent

PROJECT_PLAN.md section 8, item 6: the phone/gateway-side half of the Wear OS link
(`edge/wear_os/`, see its own README for honest status — unbuilt, no Android
toolchain here). This half is real, tested Python.

## Pipeline

```
Transport (BLE from a watch, or ReplayTransport for demo/test)
  -> features.summarize_batch   (windowed features -- the window is one watch batch)
  -> Observation
  -> --sink http     -> ingest-gateway /observations/batch   [reaches the pipeline]
     MqttPublisher   -> broker                               [production design]
     else Outbox.add -> SQLite buffer, retried on next connect
```

Every batch is durable from the moment it's received (SQLite outbox) and is only
removed from the outbox once a publish actually succeeds -- see
`EdgeAgent.flush_outbox`.

## Getting watch data to the risk engine and alerting

```bash
# 1. the server side (see services/README.md for the rest)
uvicorn app:app --app-dir services/ingest-gateway --port 8000

# 2. the device side -- straight into the gateway, no broker needed
python -m edge.edge_agent.agent make-demo --out watch.jsonl --minutes 10
python -m edge.edge_agent.agent run --transport file --in watch.jsonl \
    --sink http --gateway-url http://localhost:8000 --no-sleep
```

Verified: 60 batches -> **120 observations accepted, 0 failed** by a real
running gateway.

**Why `--sink http` and not MQTT.** MQTT is the production design and
`mqtt_publisher.py` is real code, but **nothing in this repo subscribes MQTT and
feeds ingest-gateway**. `services/ingest-gateway/app.py` has a real, tested
`handle_mqtt_message`, and says in its own module docstring that the
long-running `paho-mqtt` subscriber loop that would call it is not started. So
the MQTT path stops at the broker today and cannot reach risk-engine or
alert-service on its own. `--sink http` reuses the same `HTTPSink` the ICU and
wearable replays use (finding F3), so all three producers reach the pipeline by
one tested route.

**Verified through the full chain, not just the gateway.** With a real Kafka
broker up and all five services running, the watch stream reaches scoring:

```
120 batches -> 240 observations accepted -> {"scored": 2, "errors": 0}
```

Only 2 scoring round trips for 240 observations is correct, not a fault:
`EscalationLoop` throttles per patient in *stream* time, so a 20-minute watch
stream scores twice rather than 240 times. `escalated: 0` is also correct --
`make-demo` generates a healthy 72 bpm patient. Driving the same chain with a
deteriorating stream (`simulators/morphing.py --sink http`) raises a real alert:
`NEWS2 6: single-parameter red flag (RCP 2017): hr, spo2 scoring 3`.

**Found by running it:** `infra/compose/docker-compose.yml` did not set
`RISK_ENGINE_URL` or `ALERT_SERVICE_URL` on stream-processor, and the loop only
builds when both are present. The composed stack would have ingested and
windowed observations while never scoring or alerting -- the same "every piece
present, nothing joined" state finding F3 was raised to fix. Both are set now;
`/escalation/stats` reports `enabled` so this cannot fail silently again.

A watch observation only reaches NEWS2 scoring if it carries a scoring channel
(`hr`, `rr`, `spo2`, `sbp`, `temp_c`, `gcs_total`, `fio2` — see
`services/stream-processor/escalation.py`). The demo batch stream emits `hr` and
an `activity_index`, so it exercises transport and the HR limb; a full
deterioration scenario needs `simulators/morphing.py`, which synthesises the
SpO2 the Empatica hardware cannot measure.

## Try it without any hardware or broker

```bash
python -m edge.edge_agent.agent make-demo --out demo_batches.jsonl --minutes 5
python -m edge.edge_agent.agent run --transport file --in demo_batches.jsonl \
    --patient-ref Patient/10005866 --sink jsonl --out edge_observations.jsonl
```

`make-demo` synthesizes a watch-shaped batch stream (1 Hz HR with slow drift + noise,
accelerometer alternating resting/moving windows) in the exact `protocol.WatchBatch`
JSON shape a real watch would send. Point `run --mqtt-host ...` at a real broker
(local mosquitto today, EMQX once Phase 8 stands it up) to exercise the publish path;
without one, everything lands in `--outbox` and the CLI says so.

## What's real vs. not

| Component | Status |
|---|---|
| `protocol.py`, `features.py`, `outbox.py`, `mqtt_publisher.py`, `agent.py` | Real, tested Python (`edge/edge_agent/tests/`) |
| `transport.ReplayTransport` | Real, is what every test above actually runs against |
| `transport.BleTransport` | Real `bleak`-based BLE GATT central code, never run end-to-end here -- no BLE hardware or paired watch in this environment |
| `edge/wear_os/` (the watch app) | Built, installed, and run on the Wear OS emulator (Android Studio's SDK) -- three real bugs found and fixed by actually running it. GATT central/peripheral pairing itself is still unverified (emulator BLE peripheral support is unreliable). See its own README for the full account. |

## MQTT client-cert auth

`MqttPublisher` takes `--ca-cert`/`--client-cert`/`--client-key`; when set, it calls
`tls_set(...)` for mTLS. Untested against a real cert chain (no CA in this repo to
test against) -- the plain (no-TLS) path is what the test suite exercises.
