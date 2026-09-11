"""Windowing: rolling stats, trend slopes, HRV from IBI, event-rate features
normalised by care setting (R4).

PROJECT_PLAN.md section 10. Pure functions, independent of the FastAPI layer, so
they're directly testable against real data from earlier phases: real wearable IBI
recordings (Phase 2) for HRV, and the fitted arrival-rate models (Phase 2,
simulators/arrival_models.py) as the normalisation baseline for event-rate features.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DEFAULT_ARRIVAL_MODELS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "simulators" / "arrival_models.json"
)


@dataclass
class RollingStats:
    mean: float
    std: float
    n: int


def rolling_stats(values: list[float]) -> RollingStats:
    arr = np.asarray(values, dtype=float)
    return RollingStats(mean=float(arr.mean()), std=float(arr.std(ddof=0)), n=len(arr))


def trend_slope(timestamps_hours: list[float], values: list[float]) -> float:
    """Ordinary least squares slope of value vs. time (hours) -- e.g. "HR is rising
    at 2.3 bpm/hour over this window," a much more informative feature for a
    deterioration model than the raw value or even the rolling mean alone.
    """
    if len(values) < 2:
        return 0.0
    x = np.asarray(timestamps_hours, dtype=float)
    y = np.asarray(values, dtype=float)
    slope, _intercept = np.polyfit(x, y, deg=1)
    return float(slope)


def hrv_rmssd(ibi_seconds: list[float]) -> float:
    """RMSSD (root mean square of successive differences between inter-beat
    intervals), in milliseconds -- the standard time-domain HRV metric.
    Needs >=2 IBI values to define a single successive difference.

    Retained after the volunteer wearable dataset was removed, because its consumer
    is a real device rather than that dataset: `edge/edge_agent` reads beat-to-beat
    intervals off a BLE chest strap or watch and posts them to stream-processor's
    `/hrv`. MIMIC cannot feed this -- it charts heart rate hourly, not beat-to-beat --
    which is exactly why `reports/post_discharge_digest.py` now omits HRV rather than
    approximating it from hourly HR.
    """
    if len(ibi_seconds) < 2:
        return 0.0
    ibi_ms = np.asarray(ibi_seconds, dtype=float) * 1000.0
    diffs = np.diff(ibi_ms)
    return float(np.sqrt(np.mean(diffs**2)))


def load_stationary_rate(family: str, path: Path = DEFAULT_ARRIVAL_MODELS_PATH) -> float:
    """The fitted post-burst stationary rate for one event family (events/patient/
    hour), from simulators/arrival_models.py's real fit against the warehouse --
    the normalisation baseline for R4.
    """
    models = json.loads(path.read_text())
    return float(models[family]["stationary_rate"])


def normalized_event_rate(
    raw_count: int,
    window_hours: float,
    family: str,
    arrival_models_path: Path = DEFAULT_ARRIVAL_MODELS_PATH,
) -> float:
    """R4: "Normalise event-rate features by care setting... An un-normalised
    version would mostly learn 'this patient is in the ICU,' which the model
    already knows." Dividing the observed rate by that family's fitted stationary
    rate turns an absolute count into "how much busier than this family's typical
    ICU rate," which is the actually-informative signal.
    """
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    raw_rate = raw_count / window_hours
    baseline = load_stationary_rate(family, arrival_models_path)
    if baseline <= 0:
        return 0.0
    return raw_rate / baseline
