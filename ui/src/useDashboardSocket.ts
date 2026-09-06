import { useEffect, useRef, useState } from "react";
import type { DashboardPushMessage } from "./types";

/** Live connection to notification-gateway's real WebSocket
 * (services/notification-gateway/app.py's /ws/dashboard) -- the "React +
 * WebSocket" half of PROJECT_PLAN.md section 12's dashboard requirement.
 * Reconnects on drop (a real socket over a real network does drop) rather
 * than leaving the dashboard silently stale.
 */
export function useDashboardSocket(onMessage: (msg: DashboardPushMessage) => void) {
  const [connected, setConnected] = useState(false);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    function connect() {
      if (cancelled) return;
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/dashboard`);

      socket.onopen = () => setConnected(true);
      socket.onmessage = (event) => {
        try {
          onMessageRef.current(JSON.parse(event.data) as DashboardPushMessage);
        } catch {
          // A malformed frame must not take the whole connection down.
        }
      };
      socket.onclose = () => {
        setConnected(false);
        if (!cancelled) retryTimer = setTimeout(connect, 3000);
      };
      socket.onerror = () => socket?.close();
    }

    connect();
    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      socket?.close();
    };
  }, []);

  return { connected };
}
