-- Runs once, automatically, the first time the postgres container's data
-- volume is empty (Postgres's own docker-entrypoint-initdb.d convention).
--
-- Two real, separate uses of this one Postgres instance:
--   1. pgvector: the extension rag-service's retrieval.py would query directly
--      once Phase 8 infra exists (today it runs against a real TF-IDF index --
--      see retrieval.py's module docstring -- pgvector wiring is future work
--      beyond this phase's time budget, tracked in infra/compose/README.md).
--   2. audit_log: services/common/audit.py's real Postgres-backed AuditLog
--      (audit_postgres.py) -- same hash-chained schema as the SQLite version,
--      operated here "under production Postgres instead of local SQLite"
--      exactly as that module's docstring says Phase 8 would do.
CREATE EXTENSION IF NOT EXISTS vector;

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
);
