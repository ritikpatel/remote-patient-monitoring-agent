"""stream-processor: windowing over HTTP. See windowing.py for the real logic.

Also, optionally, windowing over a real Kafka stream: if KAFKA_BOOTSTRAP_SERVERS is
set, a background consumer thread (kafka_consumer.py) subscribes to ingest-gateway's
`raw.*` topics and keeps a live rolling window per (patient_ref, code), queryable via
GET /window/latest/... -- see kafka_consumer.py's docstring for why this is the
other, previously-unbuilt half of "windowing" the plan calls for.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from escalation import EscalationLoop  # noqa: E402
from kafka_consumer import KafkaConsumerThread, WindowStore  # noqa: E402
from services.common.observability import instrument_metrics, instrument_tracing  # noqa: E402
from windowing import (  # noqa: E402
    hrv_rmssd,
    normalized_event_rate,
    rolling_stats,
    trend_slope,
)

window_store = WindowStore()
_consumer_thread: KafkaConsumerThread | None = None
_escalation: EscalationLoop | None = None


def _build_escalation_loop() -> EscalationLoop | None:
    """Finding F3: closes stream -> score -> alert when both downstream services are
    configured. Unset (every test, and any standalone run) keeps the pre-F3 behaviour
    -- consume and window, nothing else -- exactly as KAFKA_BOOTSTRAP_SERVERS already
    gates the consumer itself. Infra presence turns it on, never a code change.
    """
    risk_url = os.environ.get("RISK_ENGINE_URL")
    alert_url = os.environ.get("ALERT_SERVICE_URL")
    if not (risk_url and alert_url):
        return None
    return EscalationLoop(
        risk_engine_url=risk_url,
        alert_service_url=alert_url,
        notification_url=os.environ.get("NOTIFICATION_GATEWAY_URL"),
    )


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Starts the Kafka consumer thread iff KAFKA_BOOTSTRAP_SERVERS is set -- off
    by default (no env var, no thread, tests and standalone runs are unaffected),
    set by docker-compose once a real broker exists alongside this service.
    Mirrors ingest-gateway's `PUBLISHER_BACKEND` switch: infra presence is the
    only thing that turns this on, never a code change.
    """
    global _consumer_thread, _escalation
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    if bootstrap:
        _escalation = _build_escalation_loop()
        _consumer_thread = KafkaConsumerThread(window_store, bootstrap, escalation=_escalation)
        _consumer_thread.start()
    yield
    if _consumer_thread is not None:
        _consumer_thread.stop()
    if _escalation is not None:
        _escalation.close()


app = FastAPI(title="stream-processor", version="0.1.0", lifespan=_lifespan)
instrument_metrics(app)
instrument_tracing(app, "stream-processor")


@app.get("/escalation/stats")
def escalation_stats() -> dict:
    """Observability for the F3 loop. Without this the only way to tell a silent
    escalation loop from a genuinely quiet stream was to read the process log --
    which is exactly how the wall-clock-throttle bug hid: 336 observations went in,
    one was scored, and nothing anywhere said so.
    """
    if _escalation is None:
        return {
            "enabled": False,
            "reason": "RISK_ENGINE_URL and ALERT_SERVICE_URL are not both set",
        }
    return {
        "enabled": True,
        "scored": _escalation.scored,
        "escalated": _escalation.escalated,
        "alerts_raised": _escalation.alerts_raised,
        "errors": _escalation.errors,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "stream-processor"}


@app.get("/window/latest/{patient_ref:path}/{code}")
def get_latest_window(patient_ref: str, code: str) -> dict:
    """The live result of the Kafka-fed rolling window for one channel -- returns
    404 until at least one message for this (patient_ref, code) has actually been
    consumed, rather than a zeroed/fabricated window.
    """
    latest = window_store.get(patient_ref, code)
    if latest is None:
        raise HTTPException(404, "no window computed yet for this (patient_ref, code)")
    return {
        "patient_ref": latest.patient_ref,
        "code": latest.code,
        "mean": latest.mean,
        "std": latest.std,
        "n_samples": latest.n_samples,
        "trend_slope_per_hour": latest.trend_slope_per_hour,
        "last_ingest_time": latest.last_ingest_time,
    }


class WindowRequest(BaseModel):
    patient_ref: str
    channel: str
    values: list[float]
    timestamps_hours: list[float]  # hours since window start, one per value


class WindowFeatures(BaseModel):
    patient_ref: str
    channel: str
    mean: float
    std: float
    n_samples: int
    trend_slope_per_hour: float


@app.post("/window/process", response_model=WindowFeatures)
def process_window(req: WindowRequest) -> WindowFeatures:
    if not req.values:
        raise HTTPException(422, "values must be non-empty")
    if len(req.values) != len(req.timestamps_hours):
        raise HTTPException(422, "values and timestamps_hours must be the same length")
    stats = rolling_stats(req.values)
    slope = trend_slope(req.timestamps_hours, req.values)
    return WindowFeatures(
        patient_ref=req.patient_ref,
        channel=req.channel,
        mean=stats.mean,
        std=stats.std,
        n_samples=stats.n,
        trend_slope_per_hour=slope,
    )


class HrvRequest(BaseModel):
    patient_ref: str
    ibi_seconds: list[float]


@app.post("/window/hrv")
def process_hrv(req: HrvRequest) -> dict:
    return {"patient_ref": req.patient_ref, "rmssd_ms": hrv_rmssd(req.ibi_seconds)}


class EventRateRequest(BaseModel):
    family: str
    raw_count: int
    window_hours: float


@app.post("/window/event_rate")
def process_event_rate(req: EventRateRequest) -> dict:
    try:
        rate = normalized_event_rate(req.raw_count, req.window_hours, req.family)
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"family": req.family, "normalized_rate": rate}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
