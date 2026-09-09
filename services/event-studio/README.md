# event-studio

Compose a synthetic patient event in a browser and drive the real pipeline with it.

```bash
uvicorn app:app --app-dir services/ingest-gateway --port 8000    # the pipeline's front door
uvicorn app:app --app-dir services/event-studio  --port 8009     # this
open http://localhost:8009
```

Port 8009 is deliberate, not arbitrary: 8000-8008 are the nine real services
(`services/README.md`), and this is a tenth, local-only tool that should be
able to run alongside all of them, `clinician-api` (8007) included, without a
bind collision.

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

`would_escalate` (shown instantly, before you send anything) comes from
`warehouse.news2.should_escalate` — the same predicate the streaming path and
the agent use, imported, never copied. Tier cut-points are read from
`capstone.news2_thresholds` at runtime rather than mirrored, because
`news2.py` derives them per-build from the cohort's own percentiles and a
local copy goes stale the first time the warehouse is rebuilt.

**"Send to pipeline" drives the real pipeline synchronously, and reports what
actually happened.** Clicking it does two things: posts the observations to
`ingest-gateway` (the same async Kafka path every other producer uses, for the
same audit trail), and — independently, in the same request — calls
`EscalationLoop.run_now()` (`services/stream-processor/escalation.py`), the
exact same score → alert code the live streaming path runs, just invoked
directly instead of triggered by a consumed Kafka message. If that escalates,
`alert-service` raises a real alert (or dedupes against one raised in the last
4 hours, R6) and — **from inside alert-service itself, not from here** — calls
`notification-gateway`, which broadcasts to the dashboard, attempts an FCM
push, and, only for a high-severity notification, attempts a real SMS through
the guarded sender. The whole chain's result — scored / escalated / alert
raised or deduped / notification channels / the real SMS outcome / retrieved
guideline context from `rag-service` — comes back in the same response and
renders as a step list in the UI.

**Why rag-service directly and not the full agent-orchestrator graph.**
`agent-orchestrator`'s `VitalsMonitor`/`LabInterpreter`/`RiskScorer` nodes read
`capstone.hourly_grid` and `labevents` by a real `(stay_id, hour)` — a
browser-composed patient has neither, and inventing a fake `stay_id` would
make those nodes silently return empty rows rather than an honest error. This
studio calls `rag-service`'s `/search` directly instead — the same retrieval
`ContextRetriever` performs — on the one thing a composed event actually has:
its escalation reason.

**SMS itself no longer lives here.** It moved to
[`services/common/sms.py`](../common/sms.py), with the one real sender now in
`notification-gateway` — see that service's own README. This studio only ever
calls `sms.compose()` for the text preview shown before you send anything
(zero network calls, no guard evaluated, cannot page a phone), which is also
the fix for a real bug: the previous version called the guarded sender's
`send()` on every "Generate" click regardless of whether anything was ever
actually sent to the pipeline, so a stray `SMS_MODE=live` left set in the
environment could page a real phone from a click the UI explicitly promised
would send nothing anywhere. It cannot do that anymore, structurally — this
service holds no code path that calls `sms.send()` at all.

### Turning on live SMS

SMS is configured where it is sent from now: see
[`services/notification-gateway/README`](../README.md#the-nine-services) and
[`services/common/sms.py`](../common/sms.py)'s module docstring for the four
guards (dry run by default; every message forced to carry
`[SYNTHETIC DRILL]`; E.164 destination required; every credential must be
present or none of it fires).

```bash
export SMS_MODE=live
export CLINICIAN_PHONE=+447700900123        # E.164, the phone that will ring
export TWILIO_ACCOUNT_SID=AC...
export TWILIO_AUTH_TOKEN=...
export TWILIO_FROM_NUMBER=+15550001111
uvicorn app:app --app-dir services/notification-gateway --port 8006
```

With that set, a composed event that escalates for real (via "Send to
pipeline", never via "Generate") pages the configured phone once per new
alert — never twice: alert-service's own `/alerts` handler is the *only*
caller of `notification-gateway`'s `/notify` on a new alert now. It used to
also be called a second time by `EscalationLoop` after alert-service returned,
which silently double-notified (and, since the SMS fix, would have
double-paged) every real alert the streaming path ever raised — see
`services/stream-processor/escalation.py`'s module docstring for how that was
found and fixed.

## Verified

Against six real running services (`ingest-gateway`, `risk-engine`,
`alert-service`, `notification-gateway`, `rag-service`, this one — no mocks,
no Kafka needed since this path doesn't use it): severity 0.95 produced NEWS2
13, tier `high`, a real alert (`alert-service` id assigned), a real
`notification-gateway` response naming `dashboard`/`push`/`oncall_escalation`
channels (the wall-clock time of the run fell in E16's overnight window), a
dry-run SMS result with the composed drill text, and 3 real passages retrieved
from `rag-service`. Resubmitting the same patient immediately after — same
4-hour dedup bucket — produced `was_new: false` and `notification: null`:
confirmed live, not just by the unit tests, that a dedup repeat triggers
zero further alerts and zero further SMS attempts.
