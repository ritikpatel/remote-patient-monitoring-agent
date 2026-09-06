import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { Alert } from "../types";

const POLL_INTERVAL_MS = 10_000;

/** Alert inbox (PROJECT_PLAN.md section 12): "acknowledge / escalate /
 * suppress, writing back to alert-service" -- every action here is a real
 * write through clinician-api to the real alert-service, re-fetched from
 * GET /alerts/active afterwards rather than optimistically patched, so the
 * displayed state always matches what alert-service actually recorded. */
export function AlertInbox() {
  const [alerts, setAlerts] = useState<Alert[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<number | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api.getActiveAlerts();
      setAlerts(data);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [load]);

  async function act(action: "acknowledge" | "escalate" | "suppress", alert: Alert) {
    setPending(alert.id);
    try {
      if (action === "acknowledge") await api.acknowledgeAlert(alert.id, alert.patient_ref);
      if (action === "escalate") await api.escalateAlert(alert.id, alert.patient_ref);
      if (action === "suppress") await api.suppressAlert(alert.id, alert.patient_ref);
      await load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setPending(null);
    }
  }

  if (error) return <p className="error">Could not load alerts: {error}</p>;
  if (alerts === null) return <p className="muted">Loading alerts…</p>;

  return (
    <div className="alert-inbox">
      <h1>Alert inbox — {alerts.length} active</h1>
      {alerts.length === 0 && <p className="muted">No active alerts.</p>}
      <ul className="alert-list">
        {alerts.map((a) => (
          <li key={a.id} className={`alert-card alert-card--${a.severity}`}>
            <div className="alert-card__body">
              <Link to={`/patients/${a.patient_ref.replace("ICUStay/", "")}?ref=${encodeURIComponent(a.patient_ref)}`}>
                {a.patient_ref}
              </Link>
              <span className={`alert-severity alert-severity--${a.severity}`}>{a.severity}</span>
              <p>{a.message}</p>
              <p className="muted">
                {a.alert_type} · raised {a.raised_at} · repeated {a.repeat_count}× · status {a.status}
              </p>
            </div>
            <div className="alert-card__actions">
              <button disabled={pending === a.id} onClick={() => act("acknowledge", a)}>
                Acknowledge
              </button>
              <button disabled={pending === a.id} onClick={() => act("escalate", a)}>
                Escalate
              </button>
              <button disabled={pending === a.id} onClick={() => act("suppress", a)}>
                Suppress
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
