"""Publish Observations to MQTT with optional client-cert (mTLS) auth.

PROJECT_PLAN.md section 8, item 6: "publishes to MQTT (EMQX) with client-cert auth."
EMQX itself is Phase 8 infra and does not exist yet -- this publisher is broker-
agnostic (nothing here is EMQX-specific) and is exercised in tests against a
deliberately-closed port to prove the connect-fails -> buffer-instead path works;
point it at any real broker (a local mosquitto today, EMQX from Phase 8 onward) by
config alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import paho.mqtt.client as mqtt
from services.contracts.observation import Observation


@dataclass
class MqttConfig:
    host: str = "localhost"
    port: int = 8883
    topic_prefix: str = "capstone/observations"
    ca_cert: Path | None = None
    client_cert: Path | None = None
    client_key: Path | None = None
    connect_timeout_s: float = 3.0

    @property
    def tls_enabled(self) -> bool:
        return self.ca_cert is not None


class MqttPublisher:
    def __init__(self, config: MqttConfig) -> None:
        self.config = config
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if config.tls_enabled:
            self._client.tls_set(
                ca_certs=str(config.ca_cert),
                certfile=str(config.client_cert) if config.client_cert else None,
                keyfile=str(config.client_key) if config.client_key else None,
            )
        self.connected = False

    def connect(self) -> bool:
        try:
            self._client.connect(self.config.host, self.config.port, keepalive=10)
            self._client.loop_start()
            self.connected = True
        except OSError:
            self.connected = False
        return self.connected

    def publish(self, obs: Observation) -> bool:
        if not self.connected:
            return False
        topic = f"{self.config.topic_prefix}/{obs.source.value}/{obs.code}"
        try:
            info = self._client.publish(topic, obs.model_dump_json(), qos=1)
            info.wait_for_publish(timeout=self.config.connect_timeout_s)
            return bool(info.is_published())
        except (OSError, RuntimeError, ValueError):
            self.connected = False
            return False

    def close(self) -> None:
        if self.connected:
            self._client.loop_stop()
            self._client.disconnect()
            self.connected = False
