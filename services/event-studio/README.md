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

**The SMS block is a preview, and sends nothing.** `SMS_MODE` defaults to
`dry_run`; no provider is wired. Two reasons, both deliberate:

- Real paging belongs to `notification-gateway` downstream, which already routes
  by severity and time of day. Sending from here would put a second copy of that
  decision outside the pipeline.
- Texting a real phone needs a provider account, credentials in env vars, and a
  clinician who agreed to be contacted. A demo whose default path pages a real
  person is a bug, not a feature.

To go live you would add a provider adapter behind `send_sms`, set
`SMS_MODE=live` and `CLINICIAN_PHONE`, and keep credentials in the environment.

## Verified

Against a real running `ingest-gateway`: severity 0.95 produced NEWS2 13,
ICU tier `high`, `would_escalate: true`, **HTTP 200, accepted**. A mid-severity
event produced NEWS2 9 / `medium` and still fired — on the single red SpO2
parameter, not the aggregate — which is the F1 limb doing its job.
