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

## Escalation, email and SMS, stated plainly

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
push, and, only for a high-severity notification, attempts a real email *and*
a real SMS, independently, through their own guarded senders. The whole
chain's result — scored / escalated / alert raised or deduped / notification
channels / the real email and SMS outcomes / retrieved guideline context from
`rag-service` — comes back in the same response and renders as a step list in
the UI.

**Why rag-service directly and not the full agent-orchestrator graph, for the
composed vitals.** `agent-orchestrator`'s `VitalsMonitor`/`LabInterpreter`/
`RiskScorer` nodes read `capstone.hourly_grid` and `labevents` by a real
`(stay_id, hour)` — a browser-composed patient has neither, and inventing a
fake `stay_id` would make those nodes silently return empty rows rather than
an honest error. This studio calls `rag-service`'s `/search` directly instead
— the same retrieval `ContextRetriever` performs — on the one thing a
composed event actually has: its escalation reason.

## Optionally, also agent-orchestrator — for a real demo patient

That constraint is about the *composed vitals*, not about whether this
service can reach `agent-orchestrator` at all. The "Also assess" picker (`GET
/patients`, proxying risk-engine — the warehouse's own owner) lists the demo
cohort's 140 real stays, each with a real `(stay_id, hour)`. Pick one, and
"Send to pipeline" also calls `agent-orchestrator POST /run` for that real
stay — the exact same call `clinician-api` makes when a clinician opens that
patient's chart (`RUNBOOK.md`) — and the response comes back as a fourth
panel: the policy's own escalate/reason, plus the LLM's advisory and summary
(clearly labelled as advisory, since the policy never reads it back, the same
constraint `nodes.py` enforces everywhere else).

This is deliberately a **second, independent** question from the fast path
above, not a replacement for it: "does this composed event escalate?" and
"what does this real patient's own chart say right now?" can disagree — a
mild composed event can sit alongside a real patient whose actual chart is
critical, or the reverse — and the UI reports both, never conflating one as
if it were an explanation of the other. It also runs regardless of whether
the composed vitals escalated: an on-demand assessment isn't conditioned on
this event's own alert in the real system either
(`docs/workflow_simple.md`'s "on-demand, not in this path" note), so this
studio doesn't gate it that way either. See `_agent_assessment()` in `app.py`.

**Neither email nor SMS lives here.** Both moved to
[`services/common/`](../common/) (`email.py`, `sms.py`), with the one real
sender of each now in `notification-gateway` — see that service's own
README. This studio only ever calls `compose()` on each for the text preview
shown before you send anything (zero network calls, no guard evaluated,
cannot page anyone), which is also the fix for a real bug: the previous SMS
version called the guarded sender's `send()` on every "Generate" click
regardless of whether anything was ever actually sent to the pipeline, so a
stray `SMS_MODE=live` left set in the environment could page a real phone
from a click the UI explicitly promised would send nothing anywhere. Neither
channel can do that anymore, structurally — this service holds no code path
that calls either module's `send()` at all.

### Turning on live email (what this project actually demonstrates)

Both channels are configured where they are sent from now: see
[`services/README.md`](../README.md#the-nine-services) and
[`services/common/email.py`](../common/email.py)'s module docstring for the
four guards (dry run by default; every message forced to carry
`[SYNTHETIC DRILL]`; a real-looking destination required; every credential
must be present or none of it fires). Email was picked as the channel to
demo live because it needs only a free SMTP account — a Gmail address and an
**app password** (Google Account → Security → 2-Step Verification → App
passwords; a plain account password will not work once 2-Step Verification
is on) covers it, no billing required:

```bash
export EMAIL_MODE=live
export CLINICIAN_EMAIL=you@example.com      # the inbox that receives it
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USERNAME=you@gmail.com
export SMTP_PASSWORD='xxxx xxxx xxxx xxxx'  # the 16-character app password, not your login password
export SMTP_FROM_ADDRESS=you@gmail.com
uvicorn app:app --app-dir services/notification-gateway --port 8006
```

With that set, a composed event that escalates for real (via "Send to
pipeline", never via "Generate") emails the configured inbox once per new
alert — never twice: alert-service's own `/alerts` handler is the *only*
caller of `notification-gateway`'s `/notify` on a new alert now. It used to
also be called a second time by `EscalationLoop` after alert-service
returned, which silently double-notified (and, since paging was wired in,
would have double-paged) every real alert the streaming path ever raised —
see `services/stream-processor/escalation.py`'s module docstring for how
that was found and fixed.

### SMS stays wired, configurable for whenever a Twilio account exists

Nothing about `notification-gateway` privileges email's code path over SMS's
-- both are attempted, independently, on every high-severity notification.
Setting `SMS_MODE=live` plus a real Twilio account (`CLINICIAN_PHONE`,
`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` — see
[`services/common/sms.py`](../common/sms.py)) sends a real text exactly as it
always did; it just is not the channel this project's own demo depends on
having credentials for.

## Verified

Against six real running services (`ingest-gateway`, `risk-engine`,
`alert-service`, `notification-gateway`, `rag-service`, this one — no mocks,
no Kafka needed since this path doesn't use it): severity 0.95 produced NEWS2
13, tier `high`, a real alert (`alert-service` id assigned), a real
`notification-gateway` response naming `dashboard`/`push`/`oncall_escalation`
channels (the wall-clock time of the run fell in E16's overnight window), and
dry-run email and SMS results with the composed drill text for each, plus 3
real passages retrieved from `rag-service`. Also verified as a real container
(`docker build` + `docker run`, no mocks): a raised alert produced the
identical email+SMS shape from inside `notification-gateway`'s own image.
Resubmitting the same patient immediately after — same 4-hour dedup bucket —
produced `was_new: false` and `notification: null`: confirmed live, not just
by the unit tests, that a dedup repeat triggers zero further alerts and zero
further email or SMS attempts.

The real-patient picker was verified the same way, against seven real running
services (the six above, plus `agent-orchestrator` with a real Groq LLM
configured), driven through an actual browser, not just `curl`: the "Also
assess" picker populated with all 140 real stays; picking `ICUStay/34617352`
and sending a *low*-severity composed event (NEWS2 2, did not escalate)
produced a `pipeline.agent_assessment` for that stay's own hour 40 showing
`escalate: true`, a NEWS2 of 14, and a real, well-grounded LLM summary
(bradycardia, absent respiratory effort, SpO₂ 90%, GCS 3, SOFA 15) — the two
verdicts genuinely disagreeing, live, exactly as designed. A second run with
a high-severity composed event showed both panels escalating for their own,
different, correctly-independent reasons (composed vitals on `sbp`/`temp_c`;
the real chart on `hr`/`rr`/`spo2`).
