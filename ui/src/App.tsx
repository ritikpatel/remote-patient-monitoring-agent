import { useCallback, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Navbar } from "./components/Navbar";
import { Toast } from "./components/Toast";
import { AlertInbox } from "./pages/AlertInbox";
import { PatientView } from "./pages/PatientView";
import { WardView } from "./pages/WardView";
import type { DashboardPushMessage } from "./types";
import { useDashboardSocket } from "./useDashboardSocket";

interface ToastItem extends DashboardPushMessage {
  id: number;
}

let nextToastId = 0;

export function App() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const handlePush = useCallback((msg: DashboardPushMessage) => {
    const id = nextToastId++;
    setToasts((prev) => [...prev, { ...msg, id }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 10_000);
  }, []);

  const { connected } = useDashboardSocket(handlePush);

  return (
    <div className="app">
      <Navbar connected={connected} />
      <div className="toast-stack">
        {toasts.map((t) => (
          <Toast key={t.id} message={t} onDismiss={() => setToasts((prev) => prev.filter((x) => x.id !== t.id))} />
        ))}
      </div>
      <main className="app__content">
        <Routes>
          <Route path="/" element={<WardView />} />
          <Route path="/patients/:stayId" element={<PatientView />} />
          <Route path="/alerts" element={<AlertInbox />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
