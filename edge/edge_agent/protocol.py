"""The batch format the Wear OS app sends over BLE, and the GATT profile it uses.

PROJECT_PLAN.md section 8, item 6: "Wear OS app sampling HR + accelerometer,
batching over BLE to edge/edge_agent." This module is the contract both sides of
that link satisfy -- edge/wear_os/ (Kotlin, GATT server / peripheral) and
edge/edge_agent/transport.py's BleTransport (Python, GATT client / central via
`bleak`) both need to agree on it, and edge/wear_os's own service references these
UUIDs and this JSON shape in comments so the two stay in sync without a shared
build.

Why one JSON blob per characteristic write rather than raw sample streaming: BLE
notifications are small (usually <=512 bytes with MTU negotiation) and each radio
wakeup costs battery, so the watch accumulates SAMPLE_WINDOW_S of HR + accelerometer
locally and sends one batch. That batch interval doubles as the edge agent's feature
window (features.py) -- there is no separate windowing step, the watch's own
batching cadence *is* the window.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Custom 128-bit UUIDs (randomly generated for this project, not a standard BLE
# profile -- there is no standard GATT service for "batched HR + accelerometer JSON").
SERVICE_UUID = "7a2e1400-9c3b-4b8a-9e2a-6f5c2d9a1001"
BATCH_CHARACTERISTIC_UUID = "7a2e1401-9c3b-4b8a-9e2a-6f5c2d9a1001"

SAMPLE_WINDOW_S = 10.0  # how often the watch batches and sends


class WatchSample(BaseModel):
    offset_ms: int  # milliseconds after batch_start
    channel: str  # "hr" | "acc_x" | "acc_y" | "acc_z"
    value: float


class WatchBatch(BaseModel):
    device_id: str
    patient_ref: str
    battery_pct: float | None = None
    batch_start: datetime
    samples: list[WatchSample] = Field(default_factory=list)
