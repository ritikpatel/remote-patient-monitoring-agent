import { NavLink } from "react-router-dom";

export function Navbar({ connected }: { connected: boolean }) {
  return (
    <header className="navbar">
      <span className="navbar__brand">ICU Monitoring</span>
      <nav className="navbar__links">
        <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
          Ward
        </NavLink>
        <NavLink to="/alerts" className={({ isActive }) => (isActive ? "active" : "")}>
          Alerts
        </NavLink>
      </nav>
      <span
        className={`navbar__status ${connected ? "navbar__status--live" : "navbar__status--offline"}`}
        title={connected ? "Live updates connected" : "Live updates disconnected -- reconnecting"}
      >
        {connected ? "● live" : "○ reconnecting…"}
      </span>
    </header>
  );
}
