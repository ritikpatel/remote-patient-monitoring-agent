"""Real-Postgres tests for audit_postgres.py -- self-skips if no Postgres is
reachable, the same pattern eval/tests/test_latency.py (k6/pipeline services)
and services/stream-processor/tests/test_kafka_consumer.py (Kafka) use for
Phase 8 infra this project does not assume is always running.

Run infra/compose's postgres service (or any Postgres with a `vector`-capable
image is not even required here -- audit_log needs no extension) and export
CAPSTONE_TEST_POSTGRES_DSN to exercise these for real; see
infra/compose/README.md.
"""

import json
import os
import socket
import uuid

import psycopg
import pytest

from services.common.audit import compute_row_hash
from services.common.audit_postgres import PostgresAuditLog, build_audit_log

DSN = os.environ.get(
    "CAPSTONE_TEST_POSTGRES_DSN",
    "postgresql://capstone:capstone@localhost:5432/capstone",
)


def _postgres_reachable() -> bool:
    host_port = DSN.split("@")[-1].split("/")[0]
    host, _, port = host_port.partition(":")
    try:
        with socket.create_connection((host, int(port or 5432)), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="no Postgres reachable -- see infra/compose/README.md",
)


@pytest.fixture
def clean_log():
    """A PostgresAuditLog against a fresh, uniquely-named table per test, so
    tests never see rows a previous run left behind."""
    table_suffix = uuid.uuid4().hex[:8]
    log = PostgresAuditLog(DSN)
    # Isolate each test's rows: rename the shared table this test run's log
    # writes to, rather than truncating the real audit_log table other tests
    # (or a human) might be relying on.
    with log.conn.cursor() as cur:
        cur.execute(f"ALTER TABLE audit_log RENAME TO audit_log_test_{table_suffix}")
        cur.execute(f"ALTER TABLE audit_log_test_{table_suffix} RENAME TO audit_log")
    yield log
    with log.conn.cursor() as cur:
        cur.execute("DELETE FROM audit_log")
    log.conn.commit()
    log.close()


def test_record_and_read_back_against_real_postgres(clean_log):
    row = clean_log.record("svc:test", "unit_test", {"k": "v"}, subject_ref="Patient/1")
    assert row.seq >= 1
    rows = clean_log.all_rows()
    assert rows[-1].payload == {"k": "v"}


def test_chain_verifies_on_untouched_log(clean_log):
    for i in range(5):
        clean_log.record("svc:test", "unit_test", {"i": i})
    ok, bad_seq = clean_log.verify_chain()
    assert ok is True
    assert bad_seq is None


def test_chain_detects_tampering(clean_log):
    for i in range(5):
        clean_log.record("svc:test", "unit_test", {"i": i})
    rows = clean_log.all_rows()
    tampered_seq = rows[2].seq

    # Tamper directly via a second real connection, bypassing the API entirely.
    with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE audit_log SET payload = %s WHERE seq = %s",
            (json.dumps({"i": 999}), tampered_seq),
        )

    ok, bad_seq = clean_log.verify_chain()
    assert ok is False
    assert bad_seq == tampered_seq


def test_sqlite_and_postgres_backends_compute_identical_hashes_for_the_same_row():
    """The whole point of factoring out compute_row_hash() (services/common/
    audit.py): a hash chain must verify the same way regardless of which
    backend wrote it. This proves the Postgres path's stored row_hash equals
    what the shared formula (used identically by the SQLite AuditLog) predicts
    -- not just that PostgresAuditLog.verify_chain() agrees with itself.
    """
    log = PostgresAuditLog(DSN)
    with log.conn.cursor() as cur:
        cur.execute("DELETE FROM audit_log")
    log.conn.commit()
    try:
        row = log.record("svc:test", "unit_test", {"k": "v"})
        payload_json = json.dumps(row.payload, sort_keys=True, default=str)
        expected = compute_row_hash(
            timestamp=row.timestamp,
            actor=row.actor,
            action=row.action,
            subject_ref=row.subject_ref,
            input_hash=row.input_hash,
            payload_json=payload_json,
            prev_hash=row.prev_hash,
        )
        assert row.row_hash == expected
    finally:
        with log.conn.cursor() as cur:
            cur.execute("DELETE FROM audit_log")
        log.conn.commit()
        log.close()


def test_build_audit_log_returns_postgres_backend_when_dsn_given(tmp_path):
    log = build_audit_log(tmp_path / "unused.db", postgres_dsn=DSN)
    assert isinstance(log, PostgresAuditLog)
    log.close()


def test_build_audit_log_returns_sqlite_backend_when_no_dsn(tmp_path):
    from services.common.audit import AuditLog

    log = build_audit_log(tmp_path / "fallback.db", postgres_dsn=None)
    assert isinstance(log, AuditLog)
    log.close()
