"""The subscriber half of edge_agent's MqttPublisher (edge/edge_agent/mqtt_publisher.py).

app.py's module docstring names the gap this closes: "the long-running `paho-mqtt`
subscriber loop that would call [handle_mqtt_message] in production is not started
in this process." `handle_mqtt_message` was always real and tested; nothing ever
opened a live connection and called it. This module is that connection -- broker-
agnostic (nothing here is EMQX-specific, same stance as MqttPublisher), subscribing
to the same `capstone/observations/#` topic tree an edge device (or any MQTT
producer) publishes to, and invoking a handler for each message.

Design mirrors services/stream-processor/kafka_consumer.py's KafkaConsumerThread
deliberately -- same shape, same "idle until infra exists" gating, same
start/stop lifecycle called from app.py's lifespan hook -- so the two ingress
paths (Kafka-consumer-style windowing, MQTT-subscriber-style ingest) read as one
pattern applied twice rather than two different ideas. One difference: paho-mqtt's
own `loop_start()` already runs the network loop on a background thread (the same
primitive MqttPublisher uses to publish), so this wraps that instead of spinning a
second thread of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# Matches MqttConfig.topic_prefix in edge/edge_agent/mqtt_publisher.py
# ("capstone/observations"), plus the "/{source}/{code}" suffix MqttPublisher.publish
# appends -- "#" is MQTT's multi-level wildcard, so this one filter covers every
# source and every channel an edge device (or any future MQTT producer) publishes.
DEFAULT_TOPIC_FILTER = "capstone/observations/#"

# Plain MQTT, not the mTLS port (8883) MqttConfig defaults to -- matches
# infra/compose/docker-compose.yml's emqx service, which only exposes 1883.
# A real deployment would set MQTT_PORT=8883 plus the ca_cert/client_cert/
# client_key fields alongside a cert-issuing step; nothing here is TLS-specific,
# same stance as MqttPublisher.
DEFAULT_PORT = 1883


@dataclass
class MqttSubscriberConfig:
    host: str = "localhost"
    port: int = DEFAULT_PORT
    topic_filter: str = DEFAULT_TOPIC_FILTER
    client_id: str = "ingest-gateway-subscriber"
    ca_cert: Path | None = None
    client_cert: Path | None = None
    client_key: Path | None = None
    keepalive_s: int = 10
    connect_timeout_s: float = 5.0

    @property
    def tls_enabled(self) -> bool:
        return self.ca_cert is not None


# What app.py actually calls per message: `handle_mqtt_message(topic, payload)`.
# Typed as a plain Callable rather than importing app.py here, so this module has
# no dependency on FastAPI or on ingest-gateway's own app module -- the same split
# MqttPublisher keeps from EdgeAgent, and what makes the tests below able to
# inject a bare list-appending fake instead of a live handler.
ObservationHandler = Callable[[str, bytes], object]


class MqttSubscriberThread:
    """Owns the paho-mqtt client's connect/subscribe/loop lifecycle.

    `start`/`stop` are idempotent-ish -- calling `stop` before a successful
    `start`, or `stop` twice, is a no-op -- so app.py's lifespan hook can call
    them unconditionally, the same contract KafkaConsumerThread gives
    stream-processor's app.py.
    """

    def __init__(self, config: MqttSubscriberConfig, on_observation: ObservationHandler) -> None:
        self.config = config
        self._on_observation = on_observation
        self.connected = False
        self.messages_received = 0
        self.errors = 0
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=config.client_id)
        self._client.on_connect = self._handle_connect
        self._client.on_message = self._handle_message
        if config.tls_enabled:
            self._client.tls_set(
                ca_certs=str(config.ca_cert),
                certfile=str(config.client_cert) if config.client_cert else None,
                keyfile=str(config.client_key) if config.client_key else None,
            )

    def _handle_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        # ReasonCode.__eq__ compares against a bare int; 0 is "Success" for every
        # MQTT protocol version this client negotiates.
        self.connected = reason_code == 0
        if self.connected:
            client.subscribe(self.config.topic_filter, qos=1)
        else:
            logger.warning("ingest-gateway: MQTT connect failed, reason_code=%s", reason_code)

    def _handle_message(self, client, userdata, message) -> None:
        try:
            self._on_observation(message.topic, message.payload)
        except Exception:
            # One malformed or unroutable message must not take the subscriber
            # down -- the same tolerance kafka_consumer.py's consume loop applies
            # to a malformed Kafka message, and for the same reason: a live
            # device population will eventually send one bad payload, and that
            # must cost one dropped message, not the whole ingest path.
            self.errors += 1
            logger.warning(
                "ingest-gateway: failed to handle MQTT message on topic %s",
                message.topic,
                exc_info=True,
            )
            return
        self.messages_received += 1

    def start(self) -> bool:
        """Connect and start paho's background network loop. Returns whether the
        connect attempt succeeded -- mirrors MqttPublisher.connect()'s return
        contract exactly, including catching the same OSError a closed port or
        unreachable host raises, so a broker that is not up yet degrades to "not
        connected" rather than crashing app.py's startup.
        """
        try:
            self._client.connect(
                self.config.host, self.config.port, keepalive=self.config.keepalive_s
            )
        except OSError:
            logger.warning(
                "ingest-gateway: could not reach MQTT broker at %s:%s",
                self.config.host,
                self.config.port,
                exc_info=True,
            )
            return False
        self._client.loop_start()
        return True

    def stop(self) -> None:
        self._client.loop_stop()
        if self.connected:
            self._client.disconnect()
        self.connected = False
