# event-studio

Compose a synthetic patient event in a browser and drive the real pipeline with it.

```bash
uvicorn app:app --app-dir services/ingest-gateway --port 8000    # the pipeline's front door
uvicorn app:app --app-dir services/event-studio  --port 8007     # this
open http://localhost:8007
```

> Every value it produces is generated, watermarked `synthetic` on the wire, and
> describes no real patient (PROJECT_PLAN.md section 17).

## Why it runs locally

It has to reach `ingest-gateway` on `localhost`, which a hosted page cannot. It
posts to the same `/observations/batch` endpoint and authenticates with the same
shared key constant the replay simulators use, so this is a new front door onto
the existing pipeline rather than a second ingestion path.

## What it generates

A **complete** event: every channel the risk model trains on (`hr`, `rr`, `spo2`,
`sbp`, `map`, `temp_c`, `gcs_total`, `fio2`, `glucose`). A partial event cannot
exercise the real scoring path, which is the same reason
`simulators/real_event_replay.py` is the project's primary test input; this is
its interactive sibling — composed to order instead of replayed.

Severity is a target NEWS2 aggregate, and `generator.py` **inverts
`warehouse/news2.py`'s own thresholds** rather than restating them, with a
start-up assertion that every breakpoint scores what it claims. That assertion
earned its place immediately: it caught two NEWS2 respiratory-rate bands entered
the wrong way round (9–11 scores 1, 21–24 scores 2) before a single event was
generated.

The severity budget is spread **unevenly** across parameters on purpose. Real
deterioration shows in one or two systems before it shows everywhere, and it is
what makes the single-red-parameter escalation limb reachable well before the
aggregate tier — the two are genuinely different alerting paths (finding F1).

## Escalation and SMS, stated plainly

`would_escalate` comes from `warehouse.news2.should_escalate` — the same
predicate the streaming path and the agent use, imported, never copied. Tier
cut-points are read from `capstone.news2_thresholds` at runtime rather than
mirrored, because `news2.py` derives them per-build from the cohort's own
percentiles and a local copy goes stale the first time the warehouse is rebuilt.

**SMS defaults to dry run and sends nothing.** The composed text is always
returned and shown in the UI, so what *would* be sent is visible whether or not
sending is enabled.

### Turning on live SMS

Set these in your environment (never in the repo -- `detect-private-key` and the
`.gitignore` are not a substitute for keeping credentials out of files):

```bash
export SMS_MODE=live
export CLINICIAN_PHONE=+447700900123        # E.164, the phone that will ring
export TWILIO_ACCOUNT_SID=AC...
export TWILIO_AUTH_TOKEN=...
export TWILIO_FROM_NUMBER=+15550001111
```

`sms.py` uses Twilio's REST API over `httpx` rather than the vendor SDK, so
going live adds no dependency.

**Four guards, each failing closed** (all covered by tests that assert a
refusal, and the live-path test injects a fake transport so the suite can never
contact a provider):

1. `SMS_MODE` must be exactly `live`. Unset, `true`, `1`, `LIVE` -- all dry run.
2. Every credential must be present, or you get a named error naming which.
3. **The body must carry `[SYNTHETIC DRILL]`.** This module refuses to transmit
   anything that could read as a real clinical alert to the person holding the
   phone. That guard applies in dry run too, so a caller bug surfaces before
   live sending is ever switched on.
4. The destination must be E.164.

The auth token is never logged and never appears in the result returned to the
browser; a provider failure reports its status and a truncated body instead.

Architecturally this is a demo shortcut and worth naming as one: in the real
system `notification-gateway` owns paging and already routes by severity and
time of day. This exists so a live drill can demonstrate the last hop.

## Verified

Against a real running `ingest-gateway`: severity 0.95 produced NEWS2 13,
ICU tier `high`, `would_escalate: true`, **HTTP 200, accepted**. A mid-severity
event produced NEWS2 9 / `medium` and still fired — on the single red SpO2
parameter, not the aggregate — which is the F1 limb doing its job.
