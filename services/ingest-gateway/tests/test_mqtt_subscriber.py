import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "services" / "ingest-gateway"))

from mqtt_subscriber import MqttSubscriberConfig, MqttSubscriberThread  # noqa: E402

MQTT_HOST = "localhost"
MQTT_PORT = 1883


@dataclass
class _FakeMessage:
    topic: str
    payload: bytes


@dataclass
class _FakeMqttClient:
    """Stands in for the real paho client inside `_handle_connect` -- only the one
    method (`subscribe`) that method calls needs to exist."""

    subscribed: list[tuple[str, int]] = field(default_factory=list)

    def subscribe(self, topic_filter: str, qos: int = 0) -> None:
        self.subscribed.append((topic_filter, qos))


def _subscriber(on_observation=lambda topic, payload: None) -> MqttSubscriberThread:
    return MqttSubscriberThread(MqttSubscriberConfig(), on_observation=on_observation)


# --------------------------------------------------------------------------
# Pure logic: the connect/message callbacks, driven directly with fakes --
# never touches a socket, always runs.
# --------------------------------------------------------------------------


def test_handle_connect_success_subscribes_to_the_configured_topic_filter():
    sub = _subscriber()
    client = _FakeMqttClient()
    sub._handle_connect(client, None, None, 0)
    assert sub.connected is True
    assert client.subscribed == [(MqttSubscriberConfig().topic_filter, 1)]


def test_handle_connect_failure_does_not_subscribe():
    sub = _subscriber()
    client = _FakeMqttClient()
    sub._handle_connect(client, None, None, 135)  # any nonzero reason code
    assert sub.connected is False
    assert client.subscribed == []


def test_handle_message_calls_on_observation_and_counts_it():
    received = []
    sub = _subscriber(on_observation=lambda topic, payload: received.append((topic, payload)))
    sub._handle_message(None, None, _FakeMessage("capstone/observations/wearable/8867-4", b"{}"))
    assert received == [("capstone/observations/wearable/8867-4", b"{}")]
    assert sub.messages_received == 1
    assert sub.errors == 0


def test_handle_message_a_raising_handler_is_counted_as_an_error_not_a_crash():
    def _boom(topic: str, payload: bytes) -> None:
        raise ValueError("malformed")

    sub = _subscriber(on_observation=_boom)
    # Must not raise -- this is exactly the callback paho invokes from its own
    # network thread; an uncaught exception there would take the whole
    # subscriber down on the very first bad message.
    sub._handle_message(None, None, _FakeMessage("some/topic", b"not json"))
    assert sub.errors == 1
    assert sub.messages_received == 0


def test_start_returns_false_and_does_not_raise_when_the_broker_is_unreachable():
    # A deliberately-closed port: bind, then immediately close, so nothing is
    # listening -- the same pattern edge/edge_agent/mqtt_publisher.py's own
    # docstring describes for proving MqttPublisher's connect-fails path.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()

    sub = MqttSubscriberThread(
        MqttSubscriberConfig(host="127.0.0.1", port=closed_port),
        on_observation=lambda topic, payload: None,
    )
    assert sub.start() is False
    assert sub.connected is False


# --------------------------------------------------------------------------
# Real broker: self-skips if EMQX isn't actually reachable, the same pattern
# test_kafka_consumer.py uses for Kafka.
# --------------------------------------------------------------------------


def _mqtt_reachable() -> bool:
    try:
        with socket.create_connection((MQTT_HOST, MQTT_PORT), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _mqtt_reachable(),
    reason="no MQTT broker reachable at localhost:1883 -- see infra/compose/README.md",
)
def test_mqtt_publisher_to_subscriber_reaches_handle_mqtt_message_end_to_end():
    """The real, broker-dependent path: edge_agent's MqttPublisher (what a Wear OS
    watch's edge_agent process uses today) publishes a real Observation to EMQX,
    MqttSubscriberThread (what ingest-gateway starts when MQTT_HOST is set)
    receives it for real, and ingest-gateway's own `handle_mqtt_message` validates
    and publishes it into an InMemoryPublisher -- proving the previously-
    unconnected publish and subscribe halves actually work together against a
    real broker, closing the exact gap app.py's module docstring named.
    """
    from services.common.publisher import InMemoryPublisher, topic_for
    from services.common.testing import load_service_app
    from services.contracts.observation import Observation, ObservationSource

    gateway = load_service_app("ingest-gateway", REPO_ROOT)
    publisher = InMemoryPublisher()
    gateway.set_publisher(publisher)

    received: list[Observation] = []
    config = MqttSubscriberConfig(
        host=MQTT_HOST, port=MQTT_PORT, client_id=f"test-sub-{uuid.uuid4().hex[:8]}"
    )
    subscriber = MqttSubscriberThread(
        config,
        on_observation=lambda topic, payload: received.append(
            gateway.handle_mqtt_message(topic, payload)
        ),
    )
    assert subscriber.start(), "subscriber could not connect to the reachable broker"
    try:
        # Give the background loop's initial connect/subscribe a moment before
        # publishing, mirroring test_kafka_consumer.py's own pre-publish sleep.
        time.sleep(1)

        from edge.edge_agent.mqtt_publisher import MqttConfig, MqttPublisher

        edge_publisher = MqttPublisher(MqttConfig(host=MQTT_HOST, port=MQTT_PORT))
        assert edge_publisher.connect()
        obs = Observation.for_channel(
            channel="hr",
            patient_ref=f"Patient/mqtt-e2e-{int(time.time())}",
            device_id="wear-os-e2e",
            source=ObservationSource.wearable,
            value=91.0,
            effective_time=datetime.now(UTC),
        )
        assert edge_publisher.publish(obs)
        edge_publisher.close()

        deadline = time.time() + 15
        while time.time() < deadline and not received:
            time.sleep(0.5)

        assert received, "subscriber never observed the published message"
        assert received[0].device_id == "wear-os-e2e"
        assert len(publisher.topics[topic_for(ObservationSource.wearable.value)]) == 1
    finally:
        subscriber.stop()
