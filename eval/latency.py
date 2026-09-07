"""Axis 3 -- Latency (PROJECT_PLAN.md section 13 and 15): runs the real k6
script (eval/load/ramp.js) against the real running services and parses its
summary for p50/p95/p99 chain latency.

E12: "Peak sizing uses 33.7 events/patient/hour, not the mean." This module
computes that peak for real from simulators/arrival_models.json's fitted
admission-burst profile (summed across every event family, at whichever hour
the combined rate peaks) rather than hard-coding the figure the plan quotes --
the fitted number moves slightly if the models are ever refit, and this stays
correct either way.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ARRIVAL_MODELS_PATH = REPO_ROOT / "simulators" / "arrival_models.json"
RAMP_SCRIPT_PATH = Path(__file__).resolve().parent / "load" / "ramp.js"
DEVICE_COUNT_TIERS = (10, 100, 1000)
LATENCY_BAR_MS = 2000  # PROJECT_PLAN.md section 15: "p95 ... latency under 2s at 1000 ... devices"


def peak_events_per_patient_per_hour() -> float:
    models = json.loads(ARRIVAL_MODELS_PATH.read_text())
    total = np.zeros(len(next(iter(models.values()))["admission_profile"]))
    for family in models.values():
        total += np.array(family["admission_profile"])
    return float(total.max())


def events_per_sec_for_device_count(device_count: int) -> float:
    return device_count * peak_events_per_patient_per_hour() / 3600


def k6_available() -> bool:
    return shutil.which("k6") is not None


@dataclass
class LatencyResult:
    device_count: int
    events_per_sec: float
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    n_iterations: int
    checks_passed: int = 0
    checks_failed: int = 0
    error: str | None = None

    @property
    def meets_p95_bar(self) -> bool | None:
        if self.p95_ms is None:
            return None
        return self.p95_ms < LATENCY_BAR_MS

    @property
    def all_checks_passed(self) -> bool:
        """A p95 well under the bar is not meaningful if a chunk of requests
        were actually failing (e.g. 500s, which k6 still times) -- this is
        checked and surfaced separately rather than assumed from the latency
        numbers alone."""
        return self.checks_failed == 0 and (self.checks_passed + self.checks_failed) > 0


def run_k6_tier(
    device_count: int,
    duration: str = "20s",
    base_urls: dict[str, str] | None = None,
) -> LatencyResult:
    """Runs eval/load/ramp.js for real against already-running services (see
    eval/README.md for how to start them) at the arrival rate E12's peak
    implies for `device_count` devices, and parses k6's own JSON summary --
    not a re-derivation of k6's percentile math, k6's own numbers, read back.
    """
    if not k6_available():
        return LatencyResult(device_count, 0.0, None, None, None, 0, error="k6 not installed")

    rate = events_per_sec_for_device_count(device_count)
    base_urls = base_urls or {}
    env = {
        "EVENTS_PER_SEC": str(rate),
        "DURATION": duration,
        "INGEST_URL": base_urls.get("ingest", "http://localhost:8000"),
        "STREAM_URL": base_urls.get("stream", "http://localhost:8003"),
        "RISK_URL": base_urls.get("risk", "http://localhost:8001"),
        "ALERT_URL": base_urls.get("alert", "http://localhost:8005"),
        "NOTIFY_URL": base_urls.get("notify", "http://localhost:8006"),
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        summary_path = Path(tmp.name)

    import os

    proc_env = {**os.environ, **env}
    result = subprocess.run(
        [
            "k6",
            "run",
            f"--summary-export={summary_path}",
            "--quiet",
            str(RAMP_SCRIPT_PATH),
        ],
        cwd=RAMP_SCRIPT_PATH.parent,
        env=proc_env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return LatencyResult(
            device_count,
            rate,
            None,
            None,
            None,
            0,
            error=f"k6 produced no summary (exit {result.returncode}): {result.stderr[-2000:]}",
        )

    summary = json.loads(summary_path.read_text())
    summary_path.unlink(missing_ok=True)

    # k6's --summary-export puts each Trend's stats directly under
    # metrics.<name> (no further nesting) -- confirmed by inspecting a real
    # export, not assumed from k6's documentation prose.
    metric = summary.get("metrics", {}).get("chain_duration_ms", {})
    n_iterations = summary.get("metrics", {}).get("iterations", {}).get("count", 0)
    checks = summary.get("metrics", {}).get("checks", {})

    return LatencyResult(
        device_count=device_count,
        events_per_sec=rate,
        p50_ms=metric.get("med"),
        p95_ms=metric.get("p(95)"),
        p99_ms=metric.get("p(99)"),
        n_iterations=int(n_iterations),
        checks_passed=int(checks.get("passes", 0)),
        checks_failed=int(checks.get("fails", 0)),
    )


def run_all_tiers(
    duration: str = "20s", base_urls: dict[str, str] | None = None
) -> list[LatencyResult]:
    return [run_k6_tier(n, duration=duration, base_urls=base_urls) for n in DEVICE_COUNT_TIERS]
