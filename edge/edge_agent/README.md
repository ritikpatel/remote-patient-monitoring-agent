# Edge agent

PROJECT_PLAN.md section 8, item 6: the phone/gateway-side half of the Wear OS link
(`edge/wear_os/`, see its own README for honest status — unbuilt, no Android
toolchain here). This half is real, tested Python.

## Pipeline

```
Transport (BLE from a watch, or ReplayTransport for demo/test)
  -> features.summarize_batch   (windowed features -- the window is one watch batch)
  -> Observation
  -> MqttPublisher.publish (if connected) or Outbox.add (buffer offline)
```

Every batch is durable from the moment it's received (SQLite outbox) and is only
removed from the outbox once a publish actually succeeds -- see
`EdgeAgent.flush_outbox`.

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
