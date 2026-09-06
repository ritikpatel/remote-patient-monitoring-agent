"""Append-only, hash-chained audit log.

PROJECT_PLAN.md section 10 (Phase 4 agent constraints): "Every agent step writes
{input_hash, tool_calls, output, model_id, tokens, latency_ms} to the audit log."
Section 14 (Phase 8) formalizes the same table as a compliance deliverable:
"append-only hash-chained table -- every PHI read, agent decision, and alert
acknowledgement. Tamper-evident by construction."

Built once, here, because agent-orchestrator needs it immediately (every graph node
writes a row) -- Phase 8 does not introduce a new table, it operates this one under
production Postgres instead of local SQLite and adds the surrounding infra (network
policy, encryption at rest) around data this module already shapes correctly.

Hash chain: each row's `row_hash` covers its own fields plus the *previous* row's
hash, so altering or deleting a historical row breaks every subsequent hash --
`verify_chain` walks the table and proves exactly that, mechanically.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,           -- e.g. "agent-orchestrator:RiskScorer", "clinician:jdoe"
    action TEXT NOT NULL,          -- e.g. "agent_step", "phi_read", "alert_ack"
    subject_ref TEXT,              -- what this action was about, e.g. "ICUStay/34547401"
    input_hash TEXT,               -- sha256 of the action's input, not the input itself
    payload TEXT NOT NULL,         -- JSON: tool_calls, output, model_id, tokens, latency_ms, ...
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
)
"""
GENESIS_HASH = "0" * 64


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_input(obj: Any) -> str:
    """A stable hash of an arbitrary JSON-able input -- audit rows store this, not
    the raw input, so the log itself never becomes a second copy of PHI."""
    return _sha256(json.dumps(obj, sort_keys=True, default=str))


@dataclass
class AuditRow:
    seq: int
    timestamp: str
    actor: str
    action: str
    subject_ref: str | None
    input_hash: str | None
    payload: dict[str, Any]
    prev_hash: str
    row_hash: str


@dataclass
class AuditLog:
    path: Path | str
    conn: sqlite3.Connection = field(init=False, repr=False)
    # check_same_thread=False (below) lifts sqlite3's same-thread restriction, but
    # does NOT make one Connection object safe for concurrent use -- two threads
    # each doing read-modify-write against the shared implicit transaction state
    # can and do interleave. Found for real: React's StrictMode double-invoking a
    # dashboard effect fired two concurrent /run requests at agent-orchestrator,
    # and record()'s _last_hash()-then-INSERT-then-commit() raced, crashing with
    # "sqlite3.OperationalError: cannot commit - no transaction is active" --
    # not a hypothetical, an actual 500 in the browser. This lock serializes every
    # access to `conn` from this object, which is exactly the scope that needs it
    # (a single-writer log, same as the note in alert-service/store.py already
    # accepts -- Phase 8 moves this to Postgres for real multi-instance access).
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def _last_hash(self) -> str:
        row = self.conn.execute(
            "SELECT row_hash FROM audit_log ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else GENESIS_HASH

    def record(
        self,
        actor: str,
        action: str,
        payload: dict[str, Any],
        subject_ref: str | None = None,
        input_hash: str | None = None,
    ) -> AuditRow:
        with self._lock:
            timestamp = datetime.now(UTC).isoformat()
            prev_hash = self._last_hash()
            payload_json = json.dumps(payload, sort_keys=True, default=str)
            row_hash = _sha256(
                "|".join(
                    [
                        timestamp,
                        actor,
                        action,
                        subject_ref or "",
                        input_hash or "",
                        payload_json,
                        prev_hash,
                    ]
                )
            )
            cur = self.conn.execute(
                "INSERT INTO audit_log (timestamp, actor, action, subject_ref, input_hash, "
                "payload, prev_hash, row_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp,
                    actor,
                    action,
                    subject_ref,
                    input_hash,
                    payload_json,
                    prev_hash,
                    row_hash,
                ),
            )
            self.conn.commit()
        assert (
            cur.lastrowid is not None
        )  # None only if no INSERT ran, which the statement above always does
        return AuditRow(
            cur.lastrowid,
            timestamp,
            actor,
            action,
            subject_ref,
            input_hash,
            payload,
            prev_hash,
            row_hash,
        )

    def record_agent_step(
        self,
        node_name: str,
        input_data: Any,
        output: Any,
        *,
        tool_calls: list[str] | None = None,
        model_id: str | None = None,
        tokens: int | None = None,
        latency_ms: float | None = None,
        subject_ref: str | None = None,
    ) -> AuditRow:
        """The exact shape PROJECT_PLAN.md section 10 requires for an agent step:
        {input_hash, tool_calls, output, model_id, tokens, latency_ms}.
        """
        return self.record(
            actor=f"agent-orchestrator:{node_name}",
            action="agent_step",
            subject_ref=subject_ref,
            input_hash=hash_input(input_data),
            payload={
                "tool_calls": tool_calls or [],
                "output": output,
                "model_id": model_id,
                "tokens": tokens,
                "latency_ms": latency_ms,
            },
        )

    def all_rows(self) -> list[AuditRow]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT seq, timestamp, actor, action, subject_ref, input_hash, payload, "
                "prev_hash, row_hash FROM audit_log ORDER BY seq"
            ).fetchall()
        return [
            AuditRow(seq, ts, actor, action, subj, ih, json.loads(payload), prev, rh)
            for seq, ts, actor, action, subj, ih, payload, prev, rh in rows
        ]

    def verify_chain(self) -> tuple[bool, int | None]:
        """Walks every row recomputing its hash from its own fields + the previous
        row's stored hash. Returns (True, None) if intact, or (False, seq) naming
        the first row where the chain breaks -- tampering with, deleting, or
        reordering any row is detectable this way, not just theoretically.
        """
        prev_hash = GENESIS_HASH
        for row in self.all_rows():
            payload_json = json.dumps(row.payload, sort_keys=True, default=str)
            expected = _sha256(
                "|".join(
                    [
                        row.timestamp,
                        row.actor,
                        row.action,
                        row.subject_ref or "",
                        row.input_hash or "",
                        payload_json,
                        prev_hash,
                    ]
                )
            )
            if row.prev_hash != prev_hash or row.row_hash != expected:
                return False, row.seq
            prev_hash = row.row_hash
        return True, None

    def close(self) -> None:
        self.conn.close()
