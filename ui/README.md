# Clinician dashboard

PROJECT_PLAN.md section 12: "React + WebSocket" — ward view, patient view,
alert inbox, mobile-responsive. Talks only to `clinician-api` (the BFF);
never imports another service's code or reads the warehouse directly.

## Running it

```bash
# Backend, each in its own terminal (or see services/README.md for the full set):
CLINICIAN_API_DEV_MODE=1 uvicorn app:app --app-dir services/clinician-api --port 8007
uvicorn app:app --app-dir services/risk-engine --port 8001
uvicorn app:app --app-dir services/alert-service --port 8005
uvicorn app:app --app-dir services/rag-service --port 8004
uvicorn app:app --app-dir services/agent-orchestrator --port 8008
uvicorn app:app --app-dir services/notification-gateway --port 8006

cd ui && npm install && npm run dev   # http://localhost:5173
```

`CLINICIAN_API_DEV_MODE=1` enables `POST /dev/token`, a local-only stand-in
for the Keycloak login Phase 8 adds — this dashboard is the only thing that
calls it, and the route 404s (not just refuses) when the flag is unset.

## What each view does, and where its data actually comes from

- **Ward view** (`/`) — `GET /patients`, which clinician-api proxies from
  risk-engine's own `/patients` (added this phase). Colour-banded by
  ICU-recalibrated NEWS2 tier, ranked by current score, polled every 15s.
- **Patient view** (`/patients/:stayId`) — four independent panels, because
  that's how the underlying services are actually split:
  1. NEWS2 trace — `GET /patients/{stayId}/trace` (risk-engine's `/trace`),
     rendered with a small hand-rolled SVG line chart (`NewsTraceChart.tsx`)
     rather than a charting library dependency.
  2. Current deterministic score — `GET /risk/{stayId}/{hour}` (NEWS2/SOFA,
     Phase 1/4).
  3. Contributing SHAP factors — `POST /risk/{stayId}/{hour}/ml` (Phase 5's
     promoted model). Renders "no model exported" rather than an error when
     `ml/models/promoted/` is empty — the same honesty the backend enforces.
  4. Agent escalation rationale — `POST /patients/{stayId}/{hour}/assessment`
     (agent-orchestrator's `/run`): the real LangGraph result, including the
     LLM summary, the policy-first escalation decision, and cited note
     passages with their fact-ledger IDs.
- **Alert inbox** (`/alerts`) — `GET /alerts/active` (alert-service's new
  ward-wide endpoint), with acknowledge/escalate/suppress writing back
  through clinician-api to the real alert-service, then re-fetching rather
  than optimistically patching local state.
- **Live push** — a real WebSocket connection to notification-gateway's
  `/ws/dashboard` (`useDashboardSocket.ts`), shown as a toast banner and the
  navbar's live/reconnecting indicator. Reconnects on drop.

## Verified for real, not just written

Ran all six backend services locally and the Vite dev server together, and
drove the actual rendered app through the Browser tool (not a mock, not a
static screenshot of the source) against the real warehouse's 140 stays:

- Ward view listing all 140 patients, correctly NEWS2-sorted with tier badges
- Patient view's all four panels populated with real data for a real stay,
  including a genuine Groq LLM-generated summary and real fact-ledger
  citations (e.g. `[F010]`, `F082`, `F056`)
- Raised a real alert via curl and watched it arrive as a live toast over the
  actual WebSocket connection, with no page reload
- Acknowledged a real alert from the rendered UI and confirmed alert-service
  itself recorded the acknowledgement (not just that the UI stopped showing
  it)
- Resized to an actual 375×812 mobile viewport (not just narrowed the
  desktop window) and confirmed the ward table, patient panels, and alert
  cards all remain legible and correctly laid out — the "physician away from
  the hospital" scenario PROJECT_PLAN.md section 12 requires be demonstrated,
  not just claimed

## Three real bugs this testing pass found and fixed (not in the UI code)

- **A genuine SQLite race in `services/common/audit.py`.** React's
  StrictMode double-invoking an effect (its documented, intentional dev-mode
  behaviour) fired two concurrent requests at agent-orchestrator's `/run`,
  and the audit log's read-then-write crashed with `sqlite3.OperationalError:
  cannot commit - no transaction is active` -- `check_same_thread=False`
  lifts sqlite3's same-thread restriction but does not make one Connection
  safe for concurrent use. Fixed with a lock around every access; the
  identical latent bug in `alert-service/store.py` (same pattern, same
  false claim in its own docstring about already being serialized) was
  fixed the same way pre-emptively.
- **`httpx`'s default 5s timeout was too tight for two real downstream
  calls** -- risk-engine's `/score/ml` (rebuilds its whole feature frame per
  request, a documented Phase 5 latency limitation) and agent-orchestrator's
  `/run` (a real LLM call). Both routinely take longer than 5s; clinician-api
  now uses a 30s timeout for its downstream clients and a dedicated
  `httpx.TimeoutException` handler returns a real 504 instead of a bare 500.
- **A raw float's full repr overflowed the SHAP-reasons panel on the mobile
  viewport** (`gcs_total_24h_mean=6.458333333333333...`). Fixed in
  `ml/models/serving.py::_format_value` to round to 3 significant figures.

## Known gaps

- No offline/service-worker support -- a genuinely offline phone shows
  nothing, not a cached last-known state. Out of scope for this phase.
- The `/dev/token` bootstrap is exactly what its name says: Phase 8's
  Keycloak integration replaces it with a real login flow, not an additional
  layer on top of it.
- `PatientView`'s four panels fetch independently and are not retried on
  failure beyond the browser's own request lifecycle -- a transient failure
  on one panel doesn't take down the other three, but also doesn't
  auto-recover without a page reload.
