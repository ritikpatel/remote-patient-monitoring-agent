import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { NewsTraceChart } from "../components/NewsTraceChart";
import { RiskBadge } from "../components/RiskBadge";
import type { AgentAssessment, MlScore, RiskScore, TracePoint } from "../types";

/** Patient view (PROJECT_PLAN.md section 12): "live vitals with the NEWS2
 * trace, contributing SHAP factors, retrieved note passages with ledger
 * citations, and the agent's escalation rationale." Four panels, each backed
 * by a real endpoint rather than one bundled response, because that's how
 * the underlying services are actually split (deterministic score vs.
 * learned model vs. agent graph) -- see ui/README.md.
 */
export function PatientView() {
  const { stayId } = useParams<{ stayId: string }>();
  const [searchParams] = useSearchParams();
  const hour = Number(searchParams.get("hour") ?? "0");
  const patientRef = searchParams.get("ref") ?? `ICUStay/${stayId}`;
  const stayIdNum = Number(stayId);

  const [trace, setTrace] = useState<TracePoint[] | null>(null);
  const [risk, setRisk] = useState<RiskScore | null>(null);
  const [mlScore, setMlScore] = useState<MlScore | null | undefined>(undefined);
  const [assessment, setAssessment] = useState<AgentAssessment | null>(null);
  const [assessmentError, setAssessmentError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.getTrace(stayIdNum), api.getRisk(stayIdNum, hour, patientRef)])
      .then(([t, r]) => {
        if (cancelled) return;
        setTrace(t);
        setRisk(r);
      })
      .catch((err) => !cancelled && setError((err as Error).message));

    api
      .getMlScore(stayIdNum, hour, patientRef)
      .then((s) => !cancelled && setMlScore(s))
      .catch(() => !cancelled && setMlScore(null));

    api
      .getAssessment(stayIdNum, hour, patientRef)
      .then((a) => !cancelled && setAssessment(a))
      .catch((err) => !cancelled && setAssessmentError((err as Error).message));

    return () => {
      cancelled = true;
    };
  }, [stayIdNum, hour, patientRef]);

  if (error) return <p className="error">Could not load this patient: {error}</p>;

  return (
    <div className="patient-view">
      <p>
        <Link to="/">&larr; Ward view</Link>
      </p>
      <h1>
        {patientRef} <RiskBadge tier={risk?.news2_tier_icu} />
      </h1>

      <section className="panel">
        <h2>NEWS2 trace</h2>
        {trace ? <NewsTraceChart points={trace} /> : <p className="muted">Loading trace…</p>}
      </section>

      <section className="panel">
        <h2>Current score (hour {hour})</h2>
        {risk ? (
          <>
            <p>
              NEWS2 <strong>{risk.news2}</strong> — ward tier {risk.news2_tier_ward}, ICU-recalibrated tier{" "}
              {risk.news2_tier_icu}. SOFA (24h) <strong>{risk.sofa_24h ?? "—"}</strong>.
            </p>
            <ul>
              {risk.reason.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </>
        ) : (
          <p className="muted">Loading…</p>
        )}
      </section>

      <section className="panel">
        <h2>Contributing SHAP factors (Phase 5 model)</h2>
        {mlScore === undefined && <p className="muted">Loading…</p>}
        {mlScore === null && (
          <p className="muted">
            No Phase 5 model is exported in this deployment (run <code>python ml/evaluation/run_all.py</code>).
          </p>
        )}
        {mlScore && (
          <>
            <p>
              Predicted 6h deterioration probability: <strong>{(mlScore.probability * 100).toFixed(1)}%</strong>{" "}
              ({mlScore.model_name}, cross-validated AUPRC {mlScore.cv_auprc_point_estimate.toFixed(3)})
            </p>
            <ul>
              {mlScore.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </>
        )}
      </section>

      <section className="panel">
        <h2>Agent escalation rationale</h2>
        {assessmentError && <p className="error">{assessmentError}</p>}
        {!assessment && !assessmentError && <p className="muted">Running the agent graph…</p>}
        {assessment && (
          <>
            <p>
              Escalate: <strong>{assessment.escalate ? "YES" : "no"}</strong> — {assessment.escalation_reason}
            </p>
            {assessment.summary && (
              <>
                <h3>Summary</h3>
                <p>{assessment.summary}</p>
              </>
            )}
            <h3>Retrieved note passages</h3>
            {assessment.context_passages.length === 0 && <p className="muted">No passages retrieved.</p>}
            <ul className="citation-list">
              {assessment.context_passages.map((p) => (
                <li key={p.passage_id}>
                  <span className="citation-source">[{p.source}]</span> {p.text}
                  {p.fact_ids.length > 0 && (
                    <span className="citation-ids"> ({p.fact_ids.join(", ")})</span>
                  )}
                </li>
              ))}
            </ul>
            <p className="muted">
              {assessment.audit_rows.length} audited agent steps for this run.
            </p>
          </>
        )}
      </section>
    </div>
  );
}
