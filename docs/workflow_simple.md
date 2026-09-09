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
    START(["Composite health event\ne.g. HR 145 · SpO2 89% · SBP 88\n(composed in event-studio, or any\nof the 5 real producers)"])

    S1["① ingest-gateway :8000\nvalidates the Observation contract,\nchecks the API key"]

    S2["② risk-engine :8001\nPOST /score/live\ncomputes NEWS2 + ICU tier from the vitals"]

    D1{"should_escalate()?\nwarehouse/news2.py — one shared function\n① ICU tier = high, OR\n② a single parameter scores red, OR\n③ GCS falls ≥2 pts /4h off sedation"}

    NOFIRE(["No alert.\nScored, logged, nothing further happens."])

    S3["③ alert-service :8005\nraises the alert; checks the 4-hour\ndedup window (R6) for this patient"]

    D2{"a genuinely\nnew alert?"}

    DEDUP(["Dedup repeat.\nExisting alert's repeat-count increments;\nnothing renotified — R6"])

    S4["④ notification-gateway :8006\nthe one place a new alert fans out from"]

    S5A["Dashboard\nlive WebSocket broadcast\nto the clinician's browser"]
    S5B["Push notification\nFCM to the clinician's phone"]
    S5C["SMS — only if severity = high\nservices/common/sms.py, guarded:\ndry-run unless SMS_MODE=live,\nforces [SYNTHETIC DRILL]"]

    ALERT(["ALERT DELIVERED"])

    SIDE["On-demand, not in this path:\nagent-orchestrator's 6-node LangGraph\nadds a cited narrative + LLM advisory\nwhen a clinician opens the patient —\nsame should_escalate() function,\nnever the reverse"]

    START --> S1 --> S2 --> D1
    D1 -->|no| NOFIRE
    D1 -->|yes| S3 --> D2
    D2 -->|no, deduped| DEDUP
    D2 -->|yes| S4
    S4 --> S5A & S5B & S5C
    S5A & S5B & S5C --> ALERT
    ALERT -.-> SIDE

    classDef startend fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:2px;
    classDef step fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px;
    classDef alertstep fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-width:2px;
    classDef decision fill:#fff7ed,stroke:#ea580c,color:#7c2d12,stroke-width:2px;
    classDef stop fill:#f3f4f6,stroke:#9ca3af,color:#1f2937;
    classDef channel fill:#cffafe,stroke:#0891b2,color:#164e63;
    classDef sms fill:#fdf4ff,stroke:#c026d3,color:#701a75;
    classDef side fill:#ede9fe,stroke:#7c3aed,color:#4c1d95,stroke-dasharray: 4 2;
    classDef final fill:#dc2626,stroke:#7f1d1d,color:#ffffff,stroke-width:3px;

    class START startend;
    class S1,S2 step;
    class D1,D2 decision;
    class NOFIRE,DEDUP stop;
    class S3,S4 alertstep;
    class S5A,S5B channel;
    class S5C sms;
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
   broadcast, no second push, no second SMS. This used to not be quite true —
   see the fuller diagram's note on finding F7, a real double-notification
   bug found while wiring SMS into this exact path.

## Why SMS is conditional on severity, not on "an alert exists"

Every alert this pipeline raises today is severity `"high"` by construction
(`EscalationLoop`'s `ALERT_TYPE`), so in practice every real alert reaches the
SMS check — but the condition is evaluated on the notification's own severity
field, not assumed, so a future lower-severity alert type would correctly
skip paging a phone for it. It is also dry-run by default regardless: nothing
sends anywhere until `SMS_MODE=live` is set, and every message is forced to
carry `[SYNTHETIC DRILL]` — see [`services/common/sms.py`](../services/common/sms.py).

For the full component inventory — every producer, the async Kafka path
alongside this synchronous one, the agentic reasoning path, FHIR export, and
what's honestly stubbed — see [`docs/workflow.md`](workflow.md).
