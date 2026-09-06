import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from store import AlertStore, dedup_bucket_start, make_dedup_key  # noqa: E402


def test_dedup_bucket_start_floors_to_4_hourly_clock():
    assert dedup_bucket_start(datetime(2110, 1, 1, 5, 30)) == datetime(2110, 1, 1, 4, 0)
    assert dedup_bucket_start(datetime(2110, 1, 1, 0, 5)) == datetime(2110, 1, 1, 0, 0)
    assert dedup_bucket_start(datetime(2110, 1, 1, 23, 59)) == datetime(2110, 1, 1, 20, 0)


def test_dedup_key_same_within_a_bucket_different_across_buckets():
    k1 = make_dedup_key("ICUStay/1", "news2_high", datetime(2110, 1, 1, 4, 1))
    k2 = make_dedup_key("ICUStay/1", "news2_high", datetime(2110, 1, 1, 7, 59))
    k3 = make_dedup_key("ICUStay/1", "news2_high", datetime(2110, 1, 1, 8, 1))
    assert k1 == k2
    assert k1 != k3


def test_raise_alert_creates_new(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    alert, was_new = store.raise_alert("ICUStay/1", "news2_high", "high", "NEWS2 = 9")
    assert was_new is True
    assert alert.status == "active"
    assert alert.repeat_count == 1
    store.close()


def test_raise_alert_dedupes_within_same_4h_bucket(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    a1, new1 = store.raise_alert(
        "ICUStay/1", "news2_high", "high", "first", datetime(2110, 1, 1, 4, 5)
    )
    a2, new2 = store.raise_alert(
        "ICUStay/1", "news2_high", "high", "second", datetime(2110, 1, 1, 5, 30)
    )
    assert new1 is True
    assert new2 is False
    assert a1.id == a2.id
    assert a2.repeat_count == 2
    store.close()


def test_raise_alert_does_not_dedupe_across_4h_buckets(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    a1, _ = store.raise_alert(
        "ICUStay/1", "news2_high", "high", "first", datetime(2110, 1, 1, 3, 55)
    )
    a2, new2 = store.raise_alert(
        "ICUStay/1", "news2_high", "high", "second", datetime(2110, 1, 1, 4, 5)
    )
    assert new2 is True
    assert a1.id != a2.id


def test_repeated_alerts_auto_escalate(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    ts = datetime(2110, 1, 1, 4, 0)
    for _ in range(3):
        alert, _ = store.raise_alert("ICUStay/1", "news2_high", "high", "repeat", ts)
    assert alert.status == "escalated"
    assert alert.repeat_count == 3
    store.close()


def test_suppress_and_acknowledge(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    alert, _ = store.raise_alert("ICUStay/1", "news2_high", "high", "msg")
    suppressed = store.suppress(alert.id)
    assert suppressed.status == "suppressed"

    alert2, _ = store.raise_alert("ICUStay/2", "news2_high", "high", "msg2")
    acked = store.acknowledge(alert2.id, "clinician:jdoe")
    assert acked.status == "acknowledged"
    assert acked.acknowledged_by == "clinician:jdoe"
    store.close()


def test_acknowledged_alert_does_not_dedupe_a_new_one(tmp_path):
    """Once acknowledged, a fresh trigger in the same bucket should raise a new
    alert rather than silently reopening the acknowledged one."""
    store = AlertStore(tmp_path / "alerts.db")
    ts = datetime(2110, 1, 1, 4, 0)
    a1, _ = store.raise_alert("ICUStay/1", "news2_high", "high", "first", ts)
    store.acknowledge(a1.id, "clinician:jdoe")
    a2, was_new = store.raise_alert(
        "ICUStay/1", "news2_high", "high", "second", ts.replace(minute=30)
    )
    assert was_new is True
    assert a2.id != a1.id
    store.close()


def test_active_for_patient_excludes_acknowledged_and_suppressed(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    a1, _ = store.raise_alert("ICUStay/1", "type_a", "high", "m", datetime(2110, 1, 1, 0, 0))
    a2, _ = store.raise_alert("ICUStay/1", "type_b", "medium", "m", datetime(2110, 1, 1, 0, 0))
    store.acknowledge(a1.id, "clinician:jdoe")
    active = store.active_for_patient("ICUStay/1")
    assert [a.id for a in active] == [a2.id]
    store.close()


def test_all_active_spans_every_patient_ordered_by_severity(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    low, _ = store.raise_alert("ICUStay/1", "type_a", "low", "m", datetime(2110, 1, 1, 0, 0))
    high, _ = store.raise_alert("ICUStay/2", "type_b", "high", "m", datetime(2110, 1, 1, 0, 0))
    medium, _ = store.raise_alert("ICUStay/3", "type_c", "medium", "m", datetime(2110, 1, 1, 0, 0))
    suppressed, _ = store.raise_alert(
        "ICUStay/4", "type_d", "high", "m", datetime(2110, 1, 1, 0, 0)
    )
    store.suppress(suppressed.id)

    active = store.all_active()
    assert [a.id for a in active] == [high.id, medium.id, low.id]
    store.close()


def test_get_unknown_alert_raises(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    try:
        store.get(999)
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    store.close()
