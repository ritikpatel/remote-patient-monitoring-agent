// Axis 3 -- Latency (PROJECT_PLAN.md section 13 and section 15's verification
// step): "p50/p95/p99 ingest -> feature -> score -> alert -> notification
// under k6 load at 10x, 100x, 1000x device count... Peak sizing uses 33.7
// events/patient/hour, not the mean (E12)."
//
// **Scope, stated plainly:** there is no message bus wiring these five
// services together yet (Kafka/EMQX is Phase 8 infra, PROJECT_PLAN.md
// section 14) -- so this script IS the orchestrator, calling each real
// service's real HTTP endpoint in sequence per simulated device event, and
// timing the whole chain. Every hop is real production code; what Phase 8
// changes is who calls it (an async consumer instead of this script), not
// what each hop does. risk-engine's step also has a stated limitation: it
// scores an EXISTING warehouse-backed (stay_id, hour), not a value freshly
// derived from the observation this same iteration just ingested (that live
// "ingest one point, get a fresh NEWS2" path does not exist -- risk-engine's
// /score is a lookup against the pre-computed hourly grid, Phase 1). See
// eval/README.md for the fuller account.
//
// Device-count tiers are translated into an aggregate arrival RATE using
// E12's real peak per-patient rate (33.7 events/patient/hour, computed in
// eval/latency.py from simulators/arrival_models.json and passed in via
// __ENV.EVENTS_PER_SEC) rather than literally pacing one k6 VU per device --
// k6's constant-arrival-rate executor achieves the same throughput/latency
// measurement without needing 1000 real VUs each self-pacing at ~0.009
// events/sec, which is a harder way to hit the same target rate for no
// additional realism.

import http from "k6/http";
import { Trend } from "k6/metrics";
import { check } from "k6";

const INGEST_URL = __ENV.INGEST_URL || "http://localhost:8000";
const STREAM_URL = __ENV.STREAM_URL || "http://localhost:8003";
const RISK_URL = __ENV.RISK_URL || "http://localhost:8001";
const ALERT_URL = __ENV.ALERT_URL || "http://localhost:8005";
const NOTIFY_URL = __ENV.NOTIFY_URL || "http://localhost:8006";
const EVENTS_PER_SEC = parseFloat(__ENV.EVENTS_PER_SEC || "1");
const DURATION = __ENV.DURATION || "30s";

const fixtures = JSON.parse(open("./fixtures.json"));

export const chainDuration = new Trend("chain_duration_ms", true);

export const options = {
  // k6's default summary only computes p(90)/p(95) for a Trend -- p99 is the
  // plan's own literal bar (section 13/15), so it must be requested
  // explicitly or --summary-export simply omits it rather than erroring.
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
  scenarios: {
    pipeline: {
      executor: "constant-arrival-rate",
      rate: Math.max(1, Math.round(EVENTS_PER_SEC)),
      timeUnit: "1s",
      duration: DURATION,
      preAllocatedVUs: Math.min(500, Math.max(10, Math.round(EVENTS_PER_SEC) * 2)),
      maxVUs: 1000,
    },
  },
};

function pickFixture() {
  return fixtures[Math.floor(Math.random() * fixtures.length)];
}

export default function () {
  const fixture = pickFixture();
  const patientRef = `ICUStay/${fixture.stay_id}`;
  const start = Date.now();

  // 1. Ingest: a real observation through the real API-key-checked endpoint.
  const obsRes = http.post(
    `${INGEST_URL}/observations`,
    JSON.stringify({
      patient_ref: patientRef,
      device_id: "k6-load-test",
      source: "icu_monitor",
      code: "hr",
      value: 80 + Math.random() * 40,
      unit: "/min",
      effective_time: new Date().toISOString(),
      ingest_time: new Date().toISOString(),
    }),
    { headers: { "Content-Type": "application/json", "X-API-Key": "capstone-rpm-dev-ingest-key" } }
  );
  check(obsRes, { "ingest 200": (r) => r.status === 200 });

  // 2. Feature: a real rolling-window computation over a small synthetic
  // window (the values themselves are not warehouse-backed -- this measures
  // stream-processor's real compute latency, not a specific patient's real
  // trajectory).
  const windowRes = http.post(
    `${STREAM_URL}/window/process`,
    JSON.stringify({
      patient_ref: patientRef,
      channel: "hr",
      values: [78, 82, 85, 90, 88, 91],
      timestamps_hours: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }),
    { headers: { "Content-Type": "application/json" } }
  );
  check(windowRes, { "window 200": (r) => r.status === 200 });

  // 3. Score: the real deterministic NEWS2/SOFA path against a real,
  // warehouse-backed (stay_id, hour) -- see the module docstring's stated
  // limitation on why this isn't scoring the observation from step 1.
  const scoreRes = http.get(`${RISK_URL}/score/${fixture.stay_id}/${fixture.hour}`);
  check(scoreRes, { "score 200": (r) => r.status === 200 });

  let tier = "low";
  if (scoreRes.status === 200) {
    tier = JSON.parse(scoreRes.body).news2_tier_icu || "low";
  }

  // 4. Alert: only on a real high-tier score, exactly the production rule
  // (matches agent-orchestrator's EscalationDecider and eval/alerting.py's
  // replay).
  if (tier === "high") {
    const alertRes = http.post(
      `${ALERT_URL}/alerts`,
      JSON.stringify({
        patient_ref: patientRef,
        alert_type: "news2_high",
        severity: "high",
        message: "NEWS2 ICU-recalibrated tier high",
      }),
      { headers: { "Content-Type": "application/json" } }
    );
    check(alertRes, { "alert 200": (r) => r.status === 200 });

    // 5. Notify: the real WebSocket-fanout trigger.
    const notifyRes = http.post(
      `${NOTIFY_URL}/notify`,
      JSON.stringify({ patient_ref: patientRef, severity: "high", message: "NEWS2 high" }),
      { headers: { "Content-Type": "application/json" } }
    );
    check(notifyRes, { "notify 200": (r) => r.status === 200 });
  }

  chainDuration.add(Date.now() - start);
}
