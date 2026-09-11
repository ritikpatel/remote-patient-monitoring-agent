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

## Home-kit stream (post-discharge arm)

```bash
python simulators/home_kit_stream.py --list-candidates
python simulators/home_kit_stream.py --stay-id 30955999 --kit full_home
python simulators/home_kit_stream.py --stay-id 30955999 --kit watch_only --sink http --no-sleep
```

**Replaces the retired `wearable_replay.py` + `morphing.py` pair, and inverts what is
synthetic.** Those took a healthy volunteer's real Empatica recording and *synthesised
a deterioration onto it* — invented physiology, real sensor. The PhysioNet volunteer
dataset behind them has been removed from the project entirely: median age ~21, **zero
deterioration events**, no link to the clinical cohort, so it could never contain the
outcome this platform predicts (finding **E10 retired**).

This module goes the other way. The **physiology is real** — a MIMIC ICU patient who
genuinely deteriorated, with their genuinely recorded vitals, at the hours they were
genuinely recorded. Only the **sensor layer** is simulated:

| Part | Real or simulated |
|---|---|
| Patient, diagnosis, deterioration, hourly vital values | **real** MIMIC records |
| Which channels exist at all | **real constraint** — no home sensor for core temp, GCS, FiO2 |
| Per-channel cadence (wrist 1/min, CGM 5/min, cuff 2/day) | simulated |
| Measurement noise (PPG ±5 bpm, SpO2 ±3%, cuff ±8 mmHg) | simulated, assumed from device literature |
| Non-wear gaps (block-structured, not per-sample) | simulated |
| Within-hour detail between hourly anchors | **simulated** — MIMIC holds one value per hour |

Three named kits — `watch_only`, `watch_plus_cuff`, `full_home` — are the single source
of truth for "what a home setup can see", imported by `ml/models/channel_masking.py` so
training masks and streamed channels cannot drift apart.

Two rules keep it honest. **Carried-forward hours are never emitted**: `hourly_grid`
forward-fills and flags it, and an imputed hour means no measurement was taken, so
streaming it would fabricate an observation that existed in neither domain. **Gaps
longer than 2h between real anchors are not bridged** — inventing a smooth ramp across a
real measurement gap is the one fabrication that could flip a trend feature's sign.

Every Observation carries `quality_flags=[synthetic]`, `device_id="home-kit-sim"` and a
`Subject/HOME-<stay_id>` reference (never `ICUStay/...` — a patient at home is not an ICU
stay). Every CLI run prints the section 17 watermark.

`--list-candidates` ranks stays by *escalating hours*, computed through the same shared
`should_escalate` predicate the alerting engine uses — so a demo cannot be quietly staged
on a patient who merely looked good.

## Shared plumbing

- `sinks.py` -- `ConsoleSink` / `JSONLSink`, used by both replay scripts. A
  `KafkaSink` behind the same protocol is Phase 4's `ingest-gateway`'s job.
- All scripts read/write **derived** artifacts only (`arrival_models.json`,
  `*.jsonl` replay output) -- no raw patient data is written into this directory.
