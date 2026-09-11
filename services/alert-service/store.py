"""Raise, dedupe, suppress, escalate, acknowledge -- SQLite-backed, real logic.

PROJECT_PLAN.md section 10: "dedup aligned to the 4-hourly clock (R6)." E16 found
care runs on a 4-hourly clock (00/04/08/12/16/20 -- 48.4% of all events land on
those six slots against 25% under uniformity); R6: "Otherwise clinicians see six
synchronised walls of alerts per day." Deduping within the *calendar* 4-hour bucket
a timestamp falls in (not a rolling window from the first alert) is what actually
prevents that -- a rolling window re-opens a fresh dedup period every time a new
alert arrives, which does not suppress a burst at all.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_ref TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,        -- "low" | "medium" | "high"
    message TEXT NOT NULL,
    dedup_key TEXT NOT NULL,
    raised_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    repeat_count INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',  -- active | suppressed | escalated | acknowledged
    acknowledged_by TEXT,
    acknowledged_at TEXT
)
"""

DEDUP_BUCKET_HOURS = 4
ESCALATE_AFTER_REPEATS = 3


def dedup_bucket_start(ts: datetime) -> datetime:
    """Floors `ts` to the most recent 00/04/08/12/16/20 boundary -- the calendar
    4-hourly clock E16 found, not a rolling window anchored to the first alert.
    """
    floored_hour = (ts.hour // DEDUP_BUCKET_HOURS) * DEDUP_BUCKET_HOURS
    return ts.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


def make_dedup_key(patient_ref: str, alert_type: str, ts: datetime) -> str:
    return f"{patient_ref}:{alert_type}:{dedup_bucket_start(ts).isoformat()}"


@dataclass
class Alert:
    id: int
    patient_ref: str
    alert_type: str
    severity: str
    message: str
    dedup_key: str
    raised_at: str
    last_seen_at: str
    repeat_count: int
    status: str
    acknowledged_by: str | None
    acknowledged_at: str | None


def _row_to_alert(row: tuple) -> Alert:
    return Alert(*row)


class AlertStore:
    def __init__(self, path: Path | str) -> None:
        # check_same_thread=False: FastAPI dispatches sync route handlers via a
        # worker thread pool, so a store constructed once at app startup is used
        # from a different thread on every request -- found by actually running
        # requests through TestClient, not by inspection. That alone does NOT
        # serialize concurrent access to the shared Connection object, though --
        # a genuinely concurrent pair of requests (found for real: services/
        # common/audit.py hit the identical bug, "sqlite3.OperationalError: cannot
        # commit - no transaction is active", when React StrictMode's double-
        # effect fired two concurrent requests) can interleave a read-modify-write
        # like raise_alert's dedup check. self._lock below is what actually makes
        # this a single-writer store; a multi-process deployment moves to a real
        # Postgres table instead (Phase 8). RLock, not Lock: several methods here
        # call self.get() while already holding the lock.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute(SCHEMA)
        self.conn.commit()
        self._lock = threading.RLock()

    def raise_alert(
        self,
        patient_ref: str,
        alert_type: str,
        severity: str,
        message: str,
        timestamp: datetime | None = None,
    ) -> tuple[Alert, bool]:
        """Returns (alert, was_new). If an active alert with the same dedup key
        already exists, this bumps its repeat_count (and auto-escalates past
        ESCALATE_AFTER_REPEATS) instead of raising a duplicate.
        """
        with self._lock:
            ts = timestamp or datetime.now(UTC)
            dedup_key = make_dedup_key(patient_ref, alert_type, ts)
            existing = self.conn.execute(
                "SELECT id, repeat_count, status FROM alerts "
                "WHERE dedup_key = ? AND status != 'acknowledged'",
                (dedup_key,),
            ).fetchone()
            if existing:
                alert_id, repeat_count, status = existing
                new_count = repeat_count + 1
                new_status = (
                    "escalated"
                    if new_count >= ESCALATE_AFTER_REPEATS and status == "active"
                    else status
                )
                self.conn.execute(
                    "UPDATE alerts SET repeat_count = ?, last_seen_at = ?, status = ? "
                    "WHERE id = ?",
                    (new_count, ts.isoformat(), new_status, alert_id),
                )
                self.conn.commit()
                return self.get(alert_id), False

            cur = self.conn.execute(
                "INSERT INTO alerts "
                "(patient_ref, alert_type, severity, message, dedup_key, raised_at, "
                "last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    patient_ref,
                    alert_type,
                    severity,
                    message,
                    dedup_key,
                    ts.isoformat(),
                    ts.isoformat(),
                ),
            )
            self.conn.commit()
            return self.get(cur.lastrowid), True

    def get(self, alert_id: int) -> Alert:
        with self._lock:
            row = self.conn.execute(
                "SELECT id, patient_ref, alert_type, severity, message, dedup_key, "
                "raised_at, last_seen_at, repeat_count, status, acknowledged_by, "
                "acknowledged_at FROM alerts WHERE id = ?",
                (alert_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"no alert with id {alert_id}")
        return _row_to_alert(row)

    def suppress(self, alert_id: int) -> Alert:
        with self._lock:
            self.conn.execute("UPDATE alerts SET status = 'suppressed' WHERE id = ?", (alert_id,))
            self.conn.commit()
            return self.get(alert_id)

    def escalate(self, alert_id: int) -> Alert:
        with self._lock:
            self.conn.execute("UPDATE alerts SET status = 'escalated' WHERE id = ?", (alert_id,))
            self.conn.commit()
            return self.get(alert_id)

    def regrade(self, alert_id: int, severity: str) -> Alert:
        """Set an alert's severity from the learned model's grade.

        Separate from `escalate`/`suppress` because it changes a different axis:
        those move `status` (is this alert still live), this moves `severity` (how
        bad is it). The alert was raised by the deterministic NEWS2 policy, which
        knows *that* a patient is deteriorating but grades every alert it raises
        identically -- `EscalationLoop` hardcodes "high" on all of them. The learned
        model is what distinguishes them, and it can only do so after the alert
        exists, because grading requires a scored (stay_id, hour) that the raising
        path does not have.

        The rewrite is deliberate and recorded rather than hidden: the severity a
        clinician sees on the dashboard must match the severity that decided whether
        they were paged, and leaving the row at "high" while paging on a model grade
        of "low" would make those two disagree.
        """
        with self._lock:
            self.conn.execute("UPDATE alerts SET severity = ? WHERE id = ?", (severity, alert_id))
            self.conn.commit()
            return self.get(alert_id)

    def acknowledge(
        self, alert_id: int, acknowledged_by: str, timestamp: datetime | None = None
    ) -> Alert:
        with self._lock:
            ts = timestamp or datetime.now(UTC)
            self.conn.execute(
                "UPDATE alerts SET status = 'acknowledged', acknowledged_by = ?, "
                "acknowledged_at = ? WHERE id = ?",
                (acknowledged_by, ts.isoformat(), alert_id),
            )
            self.conn.commit()
            return self.get(alert_id)

    def active_for_patient(self, patient_ref: str) -> list[Alert]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, patient_ref, alert_type, severity, message, dedup_key, "
                "raised_at, last_seen_at, repeat_count, status, acknowledged_by, "
                "acknowledged_at FROM alerts "
                "WHERE patient_ref = ? AND status IN ('active', 'escalated') "
                "ORDER BY raised_at DESC",
                (patient_ref,),
            ).fetchall()
        return [_row_to_alert(r) for r in rows]

    def all_active(self) -> list[Alert]:
        """Ward-wide alert inbox (PROJECT_PLAN.md section 12): every active or
        escalated alert across every patient, highest severity and most
        recent first -- `active_for_patient` is scoped to one patient and
        cannot serve the dashboard's inbox view.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, patient_ref, alert_type, severity, message, dedup_key, "
                "raised_at, last_seen_at, repeat_count, status, acknowledged_by, "
                "acknowledged_at FROM alerts "
                "WHERE status IN ('active', 'escalated') "
                "ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, "
                "raised_at DESC"
            ).fetchall()
        return [_row_to_alert(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
