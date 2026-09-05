"""Where a validated Observation goes after ingest-gateway accepts it.

PROJECT_PLAN.md section 10: "MQTT + REST ingress, schema validation, auth ->
Kafka `raw.*`." Kafka does not exist yet (Phase 8 infra) -- same pattern as every
other not-yet-deployed dependency in this project (simulators/sinks.py,
edge_agent's transport.py): a real implementation behind the interface, functional
today via an in-process/file substitute, swapped for the live one by config alone
once the broker exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TextIO

from services.contracts.observation import Observation


class Publisher(Protocol):
    def publish(self, topic: str, obs: Observation) -> None: ...
    def close(self) -> None: ...


@dataclass
class InMemoryPublisher:
    """Default for tests and for running a service standalone: keeps every
    published Observation in memory, per topic."""

    topics: dict[str, list[Observation]] = field(default_factory=dict)

    def publish(self, topic: str, obs: Observation) -> None:
        self.topics.setdefault(topic, []).append(obs)

    def close(self) -> None:
        pass


@dataclass
class JSONLPublisher:
    """Append-only JSONL per topic -- a durable substitute with none of Kafka's
    operational surface, matching simulators/sinks.py's JSONLSink."""

    directory: Path

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, TextIO] = {}

    def _handle(self, topic: str) -> TextIO:
        if topic not in self._handles:
            safe_name = topic.replace("/", "_").replace("*", "wildcard")
            self._handles[topic] = open(self.directory / f"{safe_name}.jsonl", "a")
        return self._handles[topic]

    def publish(self, topic: str, obs: Observation) -> None:
        fh = self._handle(topic)
        fh.write(obs.model_dump_json() + "\n")
        fh.flush()

    def close(self) -> None:
        for fh in self._handles.values():
            fh.close()


@dataclass
class KafkaPublisher:
    """Real `kafka-python-ng` producer code, never run against a live broker in
    this repo -- Kafka is Phase 8 infra. Connection is lazy (on first publish), so
    constructing this object never requires a broker to exist; only using it does.
    """

    bootstrap_servers: str = "localhost:9092"

    def __post_init__(self) -> None:
        self._producer = None

    def _ensure_connected(self):
        if self._producer is None:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: v.encode("utf-8"),
            )
        return self._producer

    def publish(self, topic: str, obs: Observation) -> None:
        producer = self._ensure_connected()
        producer.send(topic, obs.model_dump_json())

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush()
            self._producer.close()


def topic_for(source: str) -> str:
    """PROJECT_PLAN.md's `raw.*` convention: one topic per source."""
    return f"raw.{source}"
