// Mirrors the response shapes clinician-api actually returns (see
// services/clinician-api/app.py and the services it proxies) -- kept in sync
// by hand since this repo has no shared schema generator between Python and
// TypeScript.

export type Tier = "low" | "medium" | "high";

export interface PatientSummary {
  stay_id: number;
  patient_ref: string;
  hour: number;
  news2: number;
  news2_tier_icu: Tier;
  sofa_24h: number | null;
}

export interface TracePoint {
  hour: number;
  news2: number;
  news2_tier_icu: Tier;
  hr: number | null;
  rr: number | null;
  spo2: number | null;
  sbp: number | null;
  temp_c: number | null;
}

export interface RiskScore {
  stay_id: number;
  hour: number;
  news2: number | null;
  news2_tier_ward: Tier | null;
  news2_tier_icu: Tier | null;
  sofa_24h: number | null;
  reason: string[];
}

export interface MlScore {
  probability: number;
  model_name: string;
  horizon_h: number;
  cv_auprc_point_estimate: number;
  reasons: string[];
}

export interface ContextPassage {
  passage_id: string;
  text: string;
  source: "note" | "guideline";
  fact_ids: string[];
  hadm_id: number | null;
  score: number;
}

export interface AgentAssessment {
  stay_id: number;
  hour: number;
  patient_ref: string;
  vitals: Record<string, number | null>;
  vitals_flags: string[];
  abnormal_labs: { label: string; value: string; valueuom: string; charttime: string }[];
  risk_score: RiskScore;
  context_passages: ContextPassage[];
  escalate: boolean;
  escalation_reason: string;
  llm_advisory: string | null;
  summary: string | null;
  audit_rows: number[];
}

export type AlertStatus = "active" | "suppressed" | "escalated" | "acknowledged";
export type Severity = "low" | "medium" | "high";

export interface Alert {
  id: number;
  patient_ref: string;
  alert_type: string;
  severity: Severity;
  message: string;
  dedup_key: string;
  raised_at: string;
  last_seen_at: string;
  repeat_count: number;
  status: AlertStatus;
  acknowledged_by: string | null;
  acknowledged_at: string | null;
}

export interface DashboardPushMessage {
  patient_ref: string;
  severity: Severity;
  message: string;
}
