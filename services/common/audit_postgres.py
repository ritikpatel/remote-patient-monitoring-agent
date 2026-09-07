"""Postgres-backed AuditLog -- the exact hash-chain schema and API as this
module's sibling, services/common/audit.py's SQLite `AuditLog`, so a caller
(agent-orchestrator, clinician-api) can switch backends via one factory
function (`build_audit_log`, below) with nothing else in either service
changing. This is precisely what audit.py's own docstring describes as Phase
8's job: "Phase 8 does not introduce a new table, it operates this one under
production Postgres instead of local SQLite."

Both backends compute the row hash via `audit.compute_row_hash` -- the same
function, not a re-derivation of the same formula -- so a chain written by one
backend verifies identically under the other, and `verify_chain()` never
rejects a good row just because two independently-typed formulas drifted.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from services.common.audit import (
    GENESIS_HASH,
    AuditLogProtocol,
    AuditRow,
    compute_row_hash,
    hash_input,
)

if TYPE_CHECKING:
    import psycopg

__all__ = ["PostgresAuditLog", "build_audit_log", "hash_input"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    seq BIGSERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    subject_ref TEXT,
    input_hash TEXT,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
)
"""


@dataclass
class PostgresAuditLog:
    dsn: str
    conn: psycopg.Connection = field(init=False, repr=False)
    # Same single-writer rationale as the SQLite AuditLog's lock (see its own
    # docstring for the real double-publish race that motivated it): one
    # connection, one lock, read-modify-write (`_last_hash` then INSERT) stays
    # atomic from this process's point of view.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        import psycopg  # noqa: PLC0415 -- optional dependency, only needed for this backend

        self.conn = psycopg.connect(self.dsn, autocommit=False)
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA)
        self.conn.commit()

    def _last_hash(self) -> str:
        with self.conn.cursor() as cur:
            cur.execute("SELECT row_hash FROM audit_log ORDER BY seq DESC LIMIT 1")
            row = cur.fetchone()
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
            row_hash = compute_row_hash(
                timestamp=timestamp,
                actor=actor,
                action=action,
                subject_ref=subject_ref,
                input_hash=input_hash,
                payload_json=payload_json,
                prev_hash=prev_hash,
            )
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO audit_log (timestamp, actor, action, subject_ref, "
                    "input_hash, payload, prev_hash, row_hash) VALUES "
                    "(%s, %s, %s, %s, %s, %s, %s, %s) RETURNING seq",
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
                (seq,) = cur.fetchone()
            self.conn.commit()
        return AuditRow(
            seq, timestamp, actor, action, subject_ref, input_hash, payload, prev_hash, row_hash
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
        """Identical shape to AuditLog.record_agent_step (see its docstring for
        the PROJECT_PLAN.md requirement this satisfies) -- duplicated rather
        than shared because it is four lines of pure payload-shaping with no
        hash-chain logic in it to drift.
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
        with self._lock, self.conn.cursor() as cur:
            cur.execute(
                "SELECT seq, timestamp, actor, action, subject_ref, input_hash, payload, "
                "prev_hash, row_hash FROM audit_log ORDER BY seq"
            )
            rows = cur.fetchall()
        return [
            AuditRow(seq, ts, actor, action, subj, ih, json.loads(payload), prev, rh)
            for seq, ts, actor, action, subj, ih, payload, prev, rh in rows
        ]

    def verify_chain(self) -> tuple[bool, int | None]:
        """Identical walk to AuditLog.verify_chain() -- see that docstring."""
        prev_hash = GENESIS_HASH
        for row in self.all_rows():
            payload_json = json.dumps(row.payload, sort_keys=True, default=str)
            expected = compute_row_hash(
                timestamp=row.timestamp,
                actor=row.actor,
                action=row.action,
                subject_ref=row.subject_ref,
                input_hash=row.input_hash,
                payload_json=payload_json,
                prev_hash=prev_hash,
            )
            if row.prev_hash != prev_hash or row.row_hash != expected:
                return False, row.seq
            prev_hash = row.row_hash
        return True, None

    def close(self) -> None:
        self.conn.close()


def build_audit_log(sqlite_path: Any, *, postgres_dsn: str | None = None) -> AuditLogProtocol:
    """The one factory both agent-orchestrator and clinician-api call: returns a
    PostgresAuditLog when `postgres_dsn` is given (docker-compose sets
    AUDIT_DATABASE_URL once its postgres service exists), else today's SQLite
    AuditLog at `sqlite_path` -- unchanged default behaviour for every test and
    every environment without that infra.
    """
    if postgres_dsn:
        return PostgresAuditLog(postgres_dsn)
    from services.common.audit import AuditLog

    return AuditLog(sqlite_path)
