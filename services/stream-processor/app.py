"""stream-processor: windowing over HTTP. See windowing.py for the real logic."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windowing import (  # noqa: E402
    hrv_rmssd,
    normalized_event_rate,
    rolling_stats,
    trend_slope,
)

app = FastAPI(title="stream-processor", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "stream-processor"}


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
