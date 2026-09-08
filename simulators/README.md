# Simulators

PROJECT_PLAN.md section 8: one message schema (`services/contracts/observation.py`),
three producers that all satisfy it.

## Arrival models

```bash
python simulators/arrival_models.py
```

Fits per-event-family arrival-rate models straight from the warehouse (EDA section 6,
turned into a reusable sampler rather than a one-off figure) and writes
`arrival_models.json` + a diagnostic `arrival_model_fit.png`. `real_event_replay.py` consumes
the fitted JSON; re-run this first if the warehouse changes.

## ICU replay

```bash
python simulators/real_event_replay.py --stay-id 34531557 --compress 3600
```

Replays one stay's `capstone.hourly_grid` at configurable time compression, timed
within each hour by the fitted ICU-monitoring arrival model (phase-locked per stay,
not "on the hour"). `--sink jsonl --out FILE` writes newline-delimited Observations
instead of printing them; `--no-sleep` emits as fast as possible.

## Wearable replay

```bash
python simulators/wearable_replay.py --list
python simulators/wearable_replay.py --activity STRESS --participant S01 --duration-s 30
```

True-rate Empatica E4 replay (BVP 64 Hz, ACC 32 Hz, EDA/TEMP 4 Hz, HR 1 Hz), honouring
every fault documented in `data_constraints.txt`: `f07` (BVP/TEMP invalid),
`S02` (duplicated tail), and the three Bluetooth-drop split sessions (`f14_a/b`,
`S11_a/b`, `S16_a/b`), auto-detected and replayed as one continuous session.
`--list` shows every available session with `[SPLIT]`/`[FAULT FIXTURE]` tags.

## Morphing

```bash
python simulators/morphing.py --activity STRESS --participant S01 \
    --duration-s 60 --start-score 0 --end-score 8
```

Conditions a real wearable segment on a target NEWS2-proxy trajectory (HR baseline
shift + HRV suppression on the real recording; SpO2 fully fabricated, since Empatica
doesn't measure it). Every output Observation carries `quality_flags=[synthetic]`
and `device_id="morph-sim"` -- this is a testbed for the alerting pathway, not a
clinical claim (PROJECT_PLAN.md section 17).

`--sink http` streams into a running ingest-gateway, same as the other two replays.
It was missing here until it was needed: this is the only producer that generates
*deterioration*, so until it had an HTTP sink the healthy replays could reach the
alerting engine and the deteriorating one could not. Verified end-to-end against the
real chain (Kafka + ingest-gateway + stream-processor + risk-engine + alert-service):
4,386 observations accepted, 5 scored, **1 alert raised** -- `NEWS2 6:
single-parameter red flag (RCP 2017): hr, spo2 scoring 3`.

```bash
python simulators/morphing.py --activity STRESS --participant S01 \
    --duration-s 1800 --start-score 0 --end-score 8 --sink http --no-sleep
```

## Shared plumbing

- `sinks.py` -- `ConsoleSink` / `JSONLSink`, used by all three replay scripts. A
  `KafkaSink` behind the same protocol is Phase 4's `ingest-gateway`'s job.
- All scripts read/write **derived** artifacts only (`arrival_models.json`,
  `*.jsonl` replay output) -- no raw patient data is written into this directory.
