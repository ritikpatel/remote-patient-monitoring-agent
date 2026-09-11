# One composite event, in order, to an alert

> This platform is validated on a 100-patient demo subset of MIMIC-IV. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice (PROJECT_PLAN.md section 17).

The short version of [`docs/workflow.md`](workflow.md): one composite health
event's actual journey, in the order it actually happens, ending in an alert.
This is the real-time path `event-studio`'s "Send to pipeline" button
exercises synchronously (`EscalationLoop.run_now()`,
`services/stream-processor/escalation.py`) — the same score → alert code the
live Kafka-driven path runs, just called directly instead of triggered by a
consumed message, so a demo gets the outcome back in one response instead of
needing to poll for it.

```mermaid
%%{init: {"flowchart": {"rankSpacing": 50, "nodeSpacing": 40, "curve": "monotoneY"}}}%%
flowchart TD
    START(["Composite health event<br/>e.g. HR 145 · SpO2 89% · SBP 88<br/>(composed in event-studio, or any<br/>of the 5 real producers)"])

    S1["① ingest-gateway :8000<br/>validates the Observation contract,<br/>checks the API key"]

    S2["② risk-engine :8001<br/>POST /score/live<br/>computes NEWS2 + ICU tier from the vitals"]

    D1{"should_escalate()?<br/>warehouse/news2.py — one shared function<br/>① ICU tier = high, OR<br/>② a single parameter scores red, OR<br/>③ GCS falls ≥2 pts /4h off sedation"}

    NOFIRE(["No alert.<br/>Scored, logged, nothing further happens."])

    S3["③ alert-service :8005<br/>raises the alert; checks the 4-hour<br/>dedup window (R6) for this patient"]

    D2{"a genuinely<br/>new alert?"}

    DEDUP(["Dedup repeat.<br/>Existing alert's repeat-count increments;<br/>nothing renotified — R6"])

    S4["④ notification-gateway :8006<br/>the one place a new alert fans out from"]

    S5A["Dashboard<br/>live WebSocket broadcast<br/>to the clinician's browser"]
    S5B["Push notification<br/>FCM to the clinician's phone"]
    S5C["Email — only if severity = high<br/>services/common/email.py, guarded:<br/>dry-run unless EMAIL_MODE=live,<br/>forces [SYNTHETIC DRILL]<br/>(live-demonstrated: SMTP, free)"]
    S5D["SMS — only if severity = high<br/>services/common/sms.py, guarded:<br/>dry-run unless SMS_MODE=live,<br/>forces [SYNTHETIC DRILL]<br/>(wired, configurable: needs Twilio)"]

    ALERT(["ALERT DELIVERED"])

    SIDE["On-demand, not in this path:<br/>agent-orchestrator's 6-node LangGraph<br/>adds a cited narrative + LLM advisory<br/>when a clinician opens the patient —<br/>same should_escalate() function,<br/>never the reverse"]

    START --> S1 --> S2 --> D1
    D1 -->|no| NOFIRE
    D1 -->|yes| S3 --> D2
    D2 -->|no, deduped| DEDUP
    D2 -->|yes| S4
    S4 --> S5A & S5B & S5C & S5D
    S5A & S5B & S5C & S5D --> ALERT
    ALERT -.-> SIDE

    classDef startend fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:2px;
    classDef step fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px;
    classDef alertstep fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-width:2px;
    classDef decision fill:#fff7ed,stroke:#ea580c,color:#7c2d12,stroke-width:2px;
    classDef stop fill:#f3f4f6,stroke:#9ca3af,color:#1f2937;
    classDef channel fill:#cffafe,stroke:#0891b2,color:#164e63;
    classDef email fill:#ecfdf5,stroke:#059669,color:#064e3b,stroke-width:2px;
    classDef sms fill:#fdf4ff,stroke:#c026d3,color:#701a75,stroke-dasharray: 3 3;
    classDef side fill:#ede9fe,stroke:#7c3aed,color:#4c1d95,stroke-dasharray: 4 2;
    classDef final fill:#dc2626,stroke:#7f1d1d,color:#ffffff,stroke-width:3px;

    class START startend;
    class S1,S2 step;
    class D1,D2 decision;
    class NOFIRE,DEDUP stop;
    class S3,S4 alertstep;
    class S5A,S5B channel;
    class S5C email;
    class S5D sms;
    class ALERT final;
    class SIDE side;
```

## The two decisions that actually happen

1. **`should_escalate()` — one function, three limbs, evaluated once, in
   risk-engine.** ICU-recalibrated NEWS2 tier reaching `high` (E5); *or* any
   single non-GCS parameter scoring a red 3 (RCP 2017 — the trigger review
   finding F1 caught the original code silently dropping); *or* GCS falling
   ≥2 points within 4 hours with no sedative running. This is the exact same
   function the on-demand agent's `EscalationDecider` imports, so the two
   paths can never disagree.
2. **A genuinely new alert vs. a dedup repeat — alert-service, aligned to the
   4-hourly clock (R6).** A repeat within the same bucket increments the
   existing alert's `repeat_count` and stops there: no second dashboard
   broadcast, no second push, no second email or SMS. This used to not be
   quite true — see the fuller diagram's note on finding F7, a real
   double-notification bug found while wiring paging into this exact path.

## Two paging channels, one trigger, different credentials

Email and SMS share the identical design: attempted independently and
unconditionally whenever a notification's severity is `"high"` (every alert
this pipeline raises today, by construction — `EscalationLoop`'s
`ALERT_TYPE`), dry-run by default regardless (nothing sends anywhere until
`EMAIL_MODE` / `SMS_MODE=live` is explicitly set), every message forced to
carry `[SYNTHETIC DRILL]`. Neither is "instead of" the other in code — which
one an operator actually sees fire is a credentials question:

- **Email** ([`services/common/email.py`](../services/common/email.py)) is
  the channel this project demonstrates live. It needs only a free SMTP
  account — a Gmail address and an app password covers it, no billing.
- **SMS** ([`services/common/sms.py`](../services/common/sms.py)) stays
  wired and independently configurable for whenever a funded Twilio account
  exists — nothing in the code favours one over the other.

## The SIDE box, made clickable for a real demo patient

The diagram's on-demand box stays on-demand for a real event's own journey —
no production alert waits on an LLM call. `event-studio` (the box this
diagram's `START` node already mentions) can still reach it from the same
click, for a real demo patient specifically: picking one of the 100 real
stays additionally calls `agent-orchestrator POST /run` for that stay's real
chart, independent of whether the composed event above escalated. It is
still never in the alert path — same `should_escalate()`, never the reverse
— just no longer only reachable by hand. See
[`services/event-studio/README.md`](../services/event-studio/README.md).

For the full component inventory — every producer, the async Kafka path
alongside this synchronous one, the agentic reasoning path, FHIR export, and
what's honestly stubbed — see [`docs/workflow.md`](workflow.md).
