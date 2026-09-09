import sys
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from services.common.publisher import InMemoryPublisher  # noqa: E402
from services.common.testing import load_service_app  # noqa: E402
from services.contracts.observation import Observation, ObservationSource  # noqa: E402

_module = load_service_app("ingest-gateway", REPO_ROOT)
DEFAULT_API_KEY = _module.DEFAULT_API_KEY
app = _module.app
handle_mqtt_message = _module.handle_mqtt_message
set_publisher = _module.set_publisher

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_publisher():
    pub = InMemoryPublisher()
    set_publisher(pub)
    yield pub


def _payload(**overrides) -> dict:
    base = Observation.for_channel(
        channel="hr",
        patient_ref="ICUStay/1",
        device_id="monitor-1",
        source=ObservationSource.icu_monitor,
        value=88.0,
        effective_time=datetime(2110, 1, 1),
    ).model_dump(mode="json")
    base.update(overrides)
    return base


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_ingest_requires_api_key():
    resp = client.post("/observations", json=_payload())
    assert resp.status_code in (401, 422)  # 422 if FastAPI rejects the missing header first


def test_ingest_rejects_wrong_api_key():
    resp = client.post("/observations", json=_payload(), headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_ingest_accepts_and_publishes(fresh_publisher):
    resp = client.post("/observations", json=_payload(), headers={"X-API-Key": DEFAULT_API_KEY})
    assert resp.status_code == 200
    assert resp.json()["topic"] == "raw.icu_monitor"
    assert len(fresh_publisher.topics["raw.icu_monitor"]) == 1


def test_ingest_rejects_malformed_observation():
    bad = _payload()
    del bad["value"]
    resp = client.post("/observations", json=bad, headers={"X-API-Key": DEFAULT_API_KEY})
    assert resp.status_code == 422


def test_batch_ingest_groups_by_topic(fresh_publisher):
    payloads = [
        _payload(),
        _payload(source="wearable", code_system="urn:capstone-rpm:device-signal", code="bvp"),
    ]
    resp = client.post("/observations/batch", json=payloads, headers={"X-API-Key": DEFAULT_API_KEY})
    assert resp.status_code == 200
    counts = resp.json()["counts"]
    assert counts["raw.icu_monitor"] == 1
    assert counts["raw.wearable"] == 1


def test_handle_mqtt_message_validates_and_publishes(fresh_publisher):
    payload = Observation.for_channel(
        channel="hr",
        patient_ref="Patient/1",
        device_id="wear-os-1",
        source=ObservationSource.wearable,
        value=72.0,
        effective_time=datetime(2110, 1, 1),
    ).model_dump_json()
    obs = handle_mqtt_message("capstone/observations/wearable/8867-4", payload.encode())
    assert obs.device_id == "wear-os-1"
    assert len(fresh_publisher.topics["raw.wearable"]) == 1


def test_handle_mqtt_message_rejects_malformed_payload():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        handle_mqtt_message("some/topic", b"not json")


# --------------------------------------------------------------------------
# MQTT subscriber wiring: MQTT_HOST gates it exactly the way
# KAFKA_BOOTSTRAP_SERVERS gates stream-processor's consumer thread -- unset in
# every test here, so these check the gate itself, not a live connection.
# --------------------------------------------------------------------------


def test_mqtt_stats_disabled_by_default():
    assert client.get("/mqtt/stats").json() == {
        "enabled": False,
        "reason": "MQTT_HOST is not set",
    }


def test_build_mqtt_subscriber_returns_none_without_mqtt_host(monkeypatch):
    monkeypatch.delenv("MQTT_HOST", raising=False)
    assert _module._build_mqtt_subscriber() is None


def test_build_mqtt_subscriber_reads_host_and_port_from_env(monkeypatch):
    monkeypatch.setenv("MQTT_HOST", "emqx")
    monkeypatch.setenv("MQTT_PORT", "1883")
    sub = _module._build_mqtt_subscriber()
    assert sub is not None
    assert sub.config.host == "emqx"
    assert sub.config.port == 1883
    assert sub.config.topic_filter == "capstone/observations/#"
