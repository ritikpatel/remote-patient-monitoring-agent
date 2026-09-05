"""Durable local buffer for Observations that could not be published yet.

PROJECT_PLAN.md section 8, item 6: the edge agent "buffers offline." MQTT/EMQX is not
always reachable -- no network, a broker restart, or (as of Phase 2) the fact that
Phase 8 hasn't stood up EMQX yet -- and a bedside gateway cannot drop vitals because
of it. SQLite gives durability across process restarts for free, with no broker of
its own to run.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from services.contracts.observation import Observation

SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    published INTEGER NOT NULL DEFAULT 0
)
"""


class Outbox:
    def __init__(self, path: Path | str) -> None:
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def add(self, obs: Observation) -> None:
        self.conn.execute("INSERT INTO outbox (payload) VALUES (?)", (obs.model_dump_json(),))
        self.conn.commit()

    def pending(self, limit: int = 100) -> list[tuple[int, Observation]]:
        rows = self.conn.execute(
            "SELECT id, payload FROM outbox WHERE published = 0 ORDER BY id LIMIT ?", (limit,)
        ).fetchall()
        return [(rid, Observation.model_validate_json(payload)) for rid, payload in rows]

    def mark_published(self, ids: list[int]) -> None:
        self.conn.executemany("UPDATE outbox SET published = 1 WHERE id = ?", [(i,) for i in ids])
        self.conn.commit()

    def pending_count(self) -> int:
        row = self.conn.execute("SELECT count(*) FROM outbox WHERE published = 0").fetchone()
        return int(row[0])

    def close(self) -> None:
        self.conn.close()
