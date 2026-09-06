import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { RiskBadge } from "../components/RiskBadge";
import type { PatientSummary } from "../types";

const POLL_INTERVAL_MS = 15_000;

/** Ward view (PROJECT_PLAN.md section 12): "all monitored patients ranked by
 * current risk, colour-banded by escalation tier." risk-engine's /patients
 * already returns them NEWS2-descending; this just renders that order. */
export function WardView() {
  const [patients, setPatients] = useState<PatientSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await api.listPatients();
        if (!cancelled) {
          setPatients(data);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError((err as Error).message);
      }
    }
    load();
    const id = setInterval(load, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (error) return <p className="error">Could not load the ward view: {error}</p>;
  if (patients === null) return <p className="muted">Loading ward…</p>;

  return (
    <div className="ward-view">
      <h1>Ward — {patients.length} monitored patients</h1>
      <table className="patient-table">
        <thead>
          <tr>
            <th>Patient</th>
            <th>NEWS2</th>
            <th>Tier</th>
            <th>SOFA (24h)</th>
            <th>As of hour</th>
          </tr>
        </thead>
        <tbody>
          {patients.map((p) => (
            <tr key={p.stay_id} className={`patient-row patient-row--${p.news2_tier_icu}`}>
              <td>
                <Link to={`/patients/${p.stay_id}?hour=${p.hour}&ref=${encodeURIComponent(p.patient_ref)}`}>
                  {p.patient_ref}
                </Link>
              </td>
              <td className="patient-table__news2">{p.news2}</td>
              <td>
                <RiskBadge tier={p.news2_tier_icu} />
              </td>
              <td>{p.sofa_24h ?? "—"}</td>
              <td>{p.hour}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
