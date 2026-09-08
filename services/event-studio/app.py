"""event-studio: compose a synthetic patient event in a browser and drive the real pipeline.

Runs locally because it must reach `ingest-gateway` on localhost; a hosted page
cannot. The UI composes a complete multi-channel event (every channel the risk
model trains on), shows the NEWS2 the pipeline will compute *before* sending,
and posts it through the same `HTTPSink` contract the replay simulators use, so
this adds a front door rather than a second ingestion path.

SMS is deliberately DRY-RUN by default (`SMS_MODE=dry_run`). Alerting a real
phone needs a provider account, credentials in env vars, and a real clinician
who agreed to be paged -- none of which belong in a demo's default path. Set
SMS_MODE=live plus the provider vars to actually send.

    uvicorn app:app --app-dir services/event-studio --port 8007
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import duckdb  # noqa: E402
from generator import generate  # noqa: E402
from services.contracts.observation import Observation, ObservationSource, QualityFlag  # noqa: E402
from simulators.sinks import DEFAULT_INGEST_API_KEY  # noqa: E402
from warehouse.news2 import load_thresholds, should_escalate, tier_for_score  # noqa: E402

WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"

GATEWAY_URL = os.environ.get("INGEST_GATEWAY_URL", "http://localhost:8000")
# Same constant the replay simulators authenticate with -- a private default
# here just produces a confusing 401 against a correctly-running gateway.
API_KEY = os.environ.get("INGEST_API_KEY", DEFAULT_INGEST_API_KEY)
SMS_MODE = os.environ.get("SMS_MODE", "dry_run")

app = FastAPI(title="event-studio")
STATIC = Path(__file__).resolve().parent / "static"


_thresholds = None


def thresholds():
    """ICU-recalibrated cut-points, read from the warehouse rather than mirrored.

    `warehouse/news2.py` derives them per-build from the cohort's own
    percentiles, so a copy here would be a second source of truth that silently
    goes stale the first time the warehouse is rebuilt on different data.
    """
    global _thresholds
    if _thresholds is None:
        conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
        try:
            _thresholds = load_thresholds(conn)
        finally:
            conn.close()
    return _thresholds


class EventRequest(BaseModel):
    severity: float = Field(0.5, ge=0.0, le=1.0)
    patient_ref: str = "Patient/10005866"
    seed: int | None = None
    send: bool = False


@app.get("/health")
def health() -> dict:
    th = thresholds()
    return {
        "status": "ok",
        "service": "event-studio",
        "sms_mode": SMS_MODE,
        "icu_medium": th.icu_medium,
        "icu_high": th.icu_high,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def _observations(ev, patient_ref: str) -> list[Observation]:
    now = datetime.now(UTC)
    values = {**ev.values, **ev.extras}
    return [
        Observation.for_channel(
            channel=ch,
            patient_ref=patient_ref,
            device_id="event-studio",
            source=ObservationSource.manual,
            value=float(v),
            effective_time=now,
            # R7: composed, never measured. Every consumer can tell.
            quality_flags=[QualityFlag.synthetic],
        )
        for ch, v in values.items()
    ]


@app.post("/event")
def make_event(req: EventRequest) -> dict:
    ev = generate(req.severity, seed=req.seed)
    th = thresholds()
    tier = tier_for_score(ev.news2, th.icu_medium, th.icu_high)
    # GCS is excluded from the single-parameter limb (RCP 2017 + finding F1's
    # third limb handles falling GCS separately); a composed event has no
    # trajectory, so gcs_drop is False here by construction.
    max_nongcs = max(v for k, v in ev.subscores.items() if k != "gcs_total")
    escalates = should_escalate(tier, max_nongcs, False)

    result = {
        "severity": req.severity,
        "values": {**ev.values, **ev.extras},
        "subscores": ev.subscores,
        "news2": ev.news2,
        "tier_icu": tier,
        "max_component_nongcs": max_nongcs,
        "would_escalate": bool(escalates),
        "why": (
            f"aggregate tier '{tier}'"
            if tier == "high"
            else (
                f"single red parameter (subscore 3) in "
                f"{[k for k, v in ev.subscores.items() if v >= 3 and k != 'gcs_total']}"
                if max_nongcs >= 3
                else "no limb triggered"
            )
        ),
        "sent": False,
        "sms": None,
    }
    # The SMS block is a PREVIEW of what an escalation would page, computed from
    # the shared `should_escalate` predicate rather than a second copy of the
    # rule. Real paging is notification-gateway's job downstream; this studio
    # only shows what the pipeline would do, so it is populated on generate as
    # well as on send and never actually dials out (see send_sms).
    if escalates:
        result["sms"] = send_sms(req.patient_ref, ev.news2, result["why"])

    if not req.send:
        return result

    obs = _observations(ev, req.patient_ref)
    try:
        with httpx.Client(base_url=GATEWAY_URL, headers={"X-API-Key": API_KEY}, timeout=10) as c:
            r = c.post("/observations/batch", json=[o.model_dump(mode="json") for o in obs])
        result["sent"] = r.status_code in (200, 201, 202)
        result["gateway_status"] = r.status_code
    except httpx.HTTPError as exc:
        result["gateway_error"] = f"{type(exc).__name__}: {exc}"

    return result


def send_sms(patient_ref: str, news2: int, why: str) -> dict:
    """Dry-run unless SMS_MODE=live AND a provider is configured.

    No provider is wired here on purpose. Paging a real phone is an outward
    action needing an account, credentials and a clinician who consented to be
    contacted; defaulting to a live send would make an accidental demo run text
    a real person.
    """
    text = (
        f"[SYNTHETIC DRILL] {patient_ref}: NEWS2 {news2}, escalation - {why}. Not a real patient."
    )
    if SMS_MODE != "live":
        return {"mode": "dry_run", "to": os.environ.get("CLINICIAN_PHONE", "<unset>"), "text": text}
    return {"mode": "live", "error": "no SMS provider configured; see this module's docstring"}
