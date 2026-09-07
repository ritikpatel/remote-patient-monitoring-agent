"""ingest-gateway: MQTT + REST ingress, schema validation, auth -> Kafka `raw.*`.

PROJECT_PLAN.md section 10. Every producer this project has (simulators/icu_replay.py,
simulators/wearable_replay.py, simulators/morphing.py, edge/edge_agent/) emits
services.contracts.observation.Observation -- schema validation here is simply
"does this parse as an Observation," which FastAPI/Pydantic does automatically on
the REST path and `handle_mqtt_message` does explicitly on the MQTT path.

Auth: a shared-secret API key (`X-API-Key` header) stands in for the mTLS client-
cert auth PROJECT_PLAN.md's edge_agent design calls for (edge/edge_agent/mqtt_publisher.py
already implements the real client-cert path for the MQTT *transport*; this header
check is ingest-gateway's own service-to-service auth on the REST path, a distinct
concern). A real deployment source this from a secret store, not a module constant --
flagged in DEFAULT_API_KEY's docstring.

MQTT ingress: `handle_mqtt_message` is real, tested code; the long-running
`paho-mqtt` subscriber loop that would call it in production is not started in this
process's tests (no EMQX broker in this environment -- Phase 8 infra), matching the
transport pattern already established in edge/edge_agent/transport.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.common.observability import instrument_metrics, instrument_tracing  # noqa: E402
from services.common.publisher import (  # noqa: E402
    InMemoryPublisher,
    JSONLPublisher,
    Publisher,
    topic_for,  # noqa: E402
)
from services.contracts.observation import Observation  # noqa: E402

# A real deployment reads this from a secret store (env var backed by Vault/sealed-
# secrets in Phase 8), never a module constant -- kept simple here because there is
# no secret manager running in this environment either.
DEFAULT_API_KEY = "capstone-rpm-dev-ingest-key"


def _default_publisher() -> Publisher:
    """PUBLISHER_BACKEND selects the real KafkaPublisher once Phase 8's broker
    exists; unset (every test, and any standalone run without docker-compose)
    keeps today's InMemoryPublisher default so nothing else changes behaviour.
    """
    backend = os.environ.get("PUBLISHER_BACKEND", "memory").lower()
    if backend == "kafka":
        from services.common.publisher import KafkaPublisher

        return KafkaPublisher(
            bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        )
    if backend == "jsonl":
        return JSONLPublisher(directory=Path(os.environ.get("JSONL_DIR", "/tmp/ingest-jsonl")))
    return InMemoryPublisher()


app = FastAPI(title="ingest-gateway", version="0.1.0")
instrument_metrics(app, "ingest-gateway")
instrument_tracing(app, "ingest-gateway")
_publisher: Publisher = _default_publisher()


def set_publisher(publisher: Publisher) -> None:
    """Swap the publisher (e.g. to a KafkaPublisher once Phase 8 stands up a
    broker, or a fresh InMemoryPublisher between tests)."""
    global _publisher
    _publisher = publisher


def get_publisher() -> Publisher:
    return _publisher


def check_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != DEFAULT_API_KEY:
        raise HTTPException(401, "invalid API key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ingest-gateway"}


@app.post("/observations", dependencies=[Depends(check_api_key)])
def ingest_observation(obs: Observation) -> dict:
    topic = topic_for(obs.source.value)
    get_publisher().publish(topic, obs)
    return {"status": "accepted", "topic": topic}


@app.post("/observations/batch", dependencies=[Depends(check_api_key)])
def ingest_batch(observations: list[Observation]) -> dict:
    counts: dict[str, int] = {}
    for obs in observations:
        topic = topic_for(obs.source.value)
        get_publisher().publish(topic, obs)
        counts[topic] = counts.get(topic, 0) + 1
    return {"status": "accepted", "counts": counts}


def handle_mqtt_message(topic: str, payload: bytes) -> Observation:
    """What a paho-mqtt on_message callback calls in production. Real validation
    and publish logic; the subscriber loop that would invoke this against a live
    EMQX broker is Phase 8 infra and is not started here.
    """
    obs = Observation.model_validate_json(payload)
    get_publisher().publish(topic_for(obs.source.value), obs)
    return obs


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
