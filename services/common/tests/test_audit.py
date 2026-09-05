import json

from services.common.audit import AuditLog, hash_input


def test_record_and_read_back(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    row = log.record("svc:test", "unit_test", {"k": "v"}, subject_ref="Patient/1")
    assert row.seq == 1
    rows = log.all_rows()
    assert len(rows) == 1
    assert rows[0].payload == {"k": "v"}
    log.close()


def test_chain_verifies_on_untouched_log(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    for i in range(5):
        log.record("svc:test", "unit_test", {"i": i})
    ok, bad_seq = log.verify_chain()
    assert ok is True
    assert bad_seq is None
    log.close()


def test_chain_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    for i in range(5):
        log.record("svc:test", "unit_test", {"i": i})
    log.close()

    # Tamper with row 3's payload directly in the database, bypassing the API.
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "audit.db"))
    conn.execute("UPDATE audit_log SET payload = ? WHERE seq = 3", (json.dumps({"i": 999}),))
    conn.commit()
    conn.close()

    log2 = AuditLog(tmp_path / "audit.db")
    ok, bad_seq = log2.verify_chain()
    assert ok is False
    assert bad_seq == 3
    log2.close()


def test_record_agent_step_shape_matches_plan_requirement(tmp_path):
    """PROJECT_PLAN.md section 10: every agent step writes
    {input_hash, tool_calls, output, model_id, tokens, latency_ms}."""
    log = AuditLog(tmp_path / "audit.db")
    row = log.record_agent_step(
        "RiskScorer",
        input_data={"stay_id": 1},
        output={"score": 7},
        tool_calls=["risk_engine.score"],
        model_id="deterministic",
        tokens=0,
        latency_ms=12.3,
        subject_ref="ICUStay/1",
    )
    assert row.actor == "agent-orchestrator:RiskScorer"
    assert row.input_hash == hash_input({"stay_id": 1})
    assert row.payload["tool_calls"] == ["risk_engine.score"]
    assert row.payload["output"] == {"score": 7}
    assert row.payload["model_id"] == "deterministic"
    assert row.payload["latency_ms"] == 12.3
    log.close()


def test_hash_input_is_stable_regardless_of_key_order():
    assert hash_input({"a": 1, "b": 2}) == hash_input({"b": 2, "a": 1})
