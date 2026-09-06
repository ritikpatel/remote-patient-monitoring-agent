import type { Alert, AgentAssessment, MlScore, PatientSummary, RiskScore, TracePoint } from "./types";

// clinician-api enforces SMART-on-FHIR scopes on every route (services/common/auth.py).
// Phase 8 is where a real Keycloak issues these tokens; until then, clinician-api's
// dev-only /dev/token mints one locally (only when CLINICIAN_API_DEV_MODE=1 -- see
// that route's own docstring), and this dashboard is the one thing that calls it.
const TOKEN_KEY = "capstone-rpm-dev-token";

async function getToken(): Promise<string> {
  const cached = localStorage.getItem(TOKEN_KEY);
  if (cached) return cached;
  const resp = await fetch("/api/dev/token", { method: "POST" });
  if (!resp.ok) {
    throw new Error(
      "Could not obtain a dev token from clinician-api. Is it running with " +
        "CLINICIAN_API_DEV_MODE=1? (Phase 8 replaces this with real Keycloak login.)"
    );
  }
  const body = (await resp.json()) as { token: string };
  localStorage.setItem(TOKEN_KEY, body.token);
  return body.token;
}

async function authedFetch(path: string, init: RequestInit = {}): Promise<Response> {
  let token = await getToken();
  let resp = await fetch(`/api${path}`, {
    ...init,
    headers: { ...init.headers, Authorization: `Bearer ${token}` },
  });
  if (resp.status === 401) {
    // Token expired -- mint a fresh one and retry once.
    localStorage.removeItem(TOKEN_KEY);
    token = await getToken();
    resp = await fetch(`/api${path}`, {
      ...init,
      headers: { ...init.headers, Authorization: `Bearer ${token}` },
    });
  }
  return resp;
}

async function authedJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const resp = await authedFetch(path, init);
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${init.method ?? "GET"} ${path} -> ${resp.status}: ${body}`);
  }
  return (await resp.json()) as T;
}

export const api = {
  listPatients: () => authedJson<PatientSummary[]>("/patients"),

  getTrace: (stayId: number) => authedJson<TracePoint[]>(`/patients/${stayId}/trace`),

  getRisk: (stayId: number, hour: number, patientRef: string) =>
    authedJson<RiskScore>(
      `/risk/${stayId}/${hour}?patient_ref=${encodeURIComponent(patientRef)}`
    ),

  getMlScore: (stayId: number, hour: number, patientRef: string) =>
    authedFetch(
      `/risk/${stayId}/${hour}/ml?patient_ref=${encodeURIComponent(patientRef)}`,
      { method: "POST" }
    ).then(async (resp) => {
      if (resp.status === 503) return null; // no Phase 5 model exported -- not an error
      if (!resp.ok) throw new Error(`POST ml score -> ${resp.status}`);
      return (await resp.json()) as MlScore;
    }),

  getAssessment: (stayId: number, hour: number, patientRef: string) =>
    authedJson<AgentAssessment>(
      `/patients/${stayId}/${hour}/assessment?patient_ref=${encodeURIComponent(patientRef)}`,
      { method: "POST" }
    ),

  getActiveAlerts: () => authedJson<Alert[]>("/alerts/active"),

  acknowledgeAlert: (alertId: number, patientRef: string) =>
    authedJson<Alert>(
      `/alerts/${alertId}/acknowledge?patient_ref=${encodeURIComponent(patientRef)}`,
      { method: "POST" }
    ),

  escalateAlert: (alertId: number, patientRef: string) =>
    authedJson<Alert>(
      `/alerts/${alertId}/escalate?patient_ref=${encodeURIComponent(patientRef)}`,
      { method: "POST" }
    ),

  suppressAlert: (alertId: number, patientRef: string) =>
    authedJson<Alert>(
      `/alerts/${alertId}/suppress?patient_ref=${encodeURIComponent(patientRef)}`,
      { method: "POST" }
    ),
};
