import type { DashboardPushMessage } from "../types";

/** A transient banner for a live-pushed alert (notification-gateway's real
 * WebSocket broadcast, see useDashboardSocket) -- the visible proof that a
 * newly-raised alert reaches the dashboard without a page reload. */
export function Toast({ message, onDismiss }: { message: DashboardPushMessage; onDismiss: () => void }) {
  return (
    <div className={`toast toast--${message.severity}`} role="alert">
      <strong>{message.patient_ref}</strong>
      <span>{message.message}</span>
      <button className="toast__dismiss" onClick={onDismiss} aria-label="Dismiss">
        ×
      </button>
    </div>
  );
}
