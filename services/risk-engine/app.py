"""risk-engine: deterministic severity scoring, served over HTTP.

PROJECT_PLAN.md section 10: "Serves Phase 5 models; computes recalibrated NEWS2 and
SOFA deterministically." `/score` -- the deterministic path -- is fully real,
reading straight from the Phase 1 warehouse (capstone.news2, mimiciv_derived.sofa).
`/score/ml` now serves Phase 5's promoted deterioration model (``ml/models/serving.py``,
loaded from ``ml/models/promoted/`` -- the local stand-in for pulling the registered
model from the MLflow registry) when one has been trained and exported; it still
returns a real 503, not a fabricated number, if that export is absent -- e.g. a fresh
checkout that hasn't run ``python ml/evaluation/run_all.py`` yet.

**The LLM never computes a risk score** (Phase 4's first constraint): this service
is the only source of truth agent-orchestrator's RiskScorer node is allowed to read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.models import serving  # noqa: E402
from services.common.observability import instrument_metrics, instrument_tracing  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"

app = FastAPI(title="risk-engine", version="0.1.0")
instrument_metrics(app)
instrument_tracing(app, "risk-engine")


def get_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)


class RiskScoreResponse(BaseModel):
    stay_id: int
    hour: int
    news2: int | None
    news2_tier_ward: str | None
    news2_tier_icu: str | None
    sofa_24h: int | None
    reason: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "risk-engine"}


@app.get("/score/{stay_id}/{hour}", response_model=RiskScoreResponse)
def score(stay_id: int, hour: int) -> RiskScoreResponse:
    conn = get_conn()
    try:
        news2_row = conn.execute(
            "SELECT news2, tier_ward, tier_icu, hr, rr, spo2, sbp, temp_c, gcs_total, fio2 "
            "FROM capstone.news2 WHERE stay_id = ? AND hour = ?",
            [stay_id, hour],
        ).fetchone()
        if news2_row is None:
            raise HTTPException(404, f"no hourly_grid/news2 row for stay_id={stay_id} hour={hour}")
        news2, tier_ward, tier_icu, hr, rr, spo2, sbp, temp_c, gcs_total, fio2 = news2_row

        sofa_row = conn.execute(
            "SELECT sofa_24hours FROM mimiciv_derived.sofa WHERE stay_id = ? AND hr = ?",
            [stay_id, hour],
        ).fetchone()
        sofa_24h = sofa_row[0] if sofa_row else None

        reason = _explain(hr, rr, spo2, sbp, temp_c, gcs_total, fio2)
        return RiskScoreResponse(
            stay_id=stay_id,
            hour=hour,
            news2=news2,
            news2_tier_ward=tier_ward,
            news2_tier_icu=tier_icu,
            sofa_24h=sofa_24h,
            reason=reason,
        )
    finally:
        conn.close()


def _explain(hr, rr, spo2, sbp, temp_c, gcs_total, fio2) -> list[str]:
    """A plain-language trace of which vitals are driving the score -- the
    SHAP-attribution stand-in until Phase 5 has a model to attribute. Every learned
    model's explanation later plugs into this same `reason` field on the response.
    """
    from warehouse.news2 import (
        fio2_score,
        gcs_score,
        hr_score,
        rr_score,
        sbp_score,
        spo2_score,
        temp_score,
    )

    reasons = []
    checks = [
        ("HR", hr, hr_score),
        ("RR", rr, rr_score),
        ("SpO2", spo2, spo2_score),
        ("SBP", sbp, sbp_score),
        ("Temp", temp_c, temp_score),
        ("GCS", gcs_total, gcs_score),
        ("FiO2", fio2, fio2_score),
    ]
    for name, value, scorer in checks:
        if value is None:
            continue
        s = scorer(value)
        if s > 0:
            reasons.append(f"{name}={value} contributes {s} to NEWS2")
    return reasons


class PatientSummary(BaseModel):
    stay_id: int
    patient_ref: str
    hour: int  # latest hour with a recorded score -- "now" in this replay-based demo
    news2: int
    news2_tier_icu: str
    sofa_24h: int | None


@app.get("/patients", response_model=list[PatientSummary])
def list_patients() -> list[PatientSummary]:
    """Ward view (PROJECT_PLAN.md section 12): every monitored stay ranked by
    its *current* risk. Placeholder ward registry -- Phase 8's real system
    would list patients from HAPI FHIR's Patient/Encounter resources; this
    demo's data source is the warehouse's own hourly grid, which is also
    exactly the data risk-engine already owns and reads elsewhere.

    "Current" for a stay is its latest hourly_grid row -- E1: care in this
    dataset is charted hourly, not streamed, so "latest available hour" is
    the honest analogue of "now" rather than an invented live timestamp.
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT n.stay_id, n.hour, n.news2, n.tier_icu, s.sofa_24hours
            FROM capstone.news2 n
            JOIN (SELECT stay_id, MAX(hour) AS hour FROM capstone.news2 GROUP BY stay_id) latest
              ON n.stay_id = latest.stay_id AND n.hour = latest.hour
            LEFT JOIN mimiciv_derived.sofa s ON s.stay_id = n.stay_id AND s.hr = n.hour
            ORDER BY n.news2 DESC
            """
        ).fetchall()
        return [
            PatientSummary(
                stay_id=stay_id,
                patient_ref=f"ICUStay/{stay_id}",
                hour=hour,
                news2=news2,
                news2_tier_icu=tier_icu,
                sofa_24h=sofa_24h,
            )
            for stay_id, hour, news2, tier_icu, sofa_24h in rows
        ]
    finally:
        conn.close()


class TracePoint(BaseModel):
    hour: int
    news2: int
    news2_tier_icu: str
    hr: float | None
    rr: float | None
    spo2: float | None
    sbp: float | None
    temp_c: float | None


@app.get("/trace/{stay_id}", response_model=list[TracePoint])
def trace(stay_id: int) -> list[TracePoint]:
    """The full NEWS2 trace for one stay (PROJECT_PLAN.md section 12: "live
    vitals with the NEWS2 trace") -- every hour, not just the latest, so the
    dashboard's patient view can chart it.
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT hour, news2, tier_icu, hr, rr, spo2, sbp, temp_c "
            "FROM capstone.news2 WHERE stay_id = ? ORDER BY hour",
            [stay_id],
        ).fetchall()
        if not rows:
            raise HTTPException(404, f"no news2 trace for stay_id={stay_id}")
        return [
            TracePoint(
                hour=hour,
                news2=news2,
                news2_tier_icu=tier_icu,
                hr=hr,
                rr=rr,
                spo2=spo2,
                sbp=sbp,
                temp_c=temp_c,
            )
            for hour, news2, tier_icu, hr, rr, spo2, sbp, temp_c in rows
        ]
    finally:
        conn.close()


@app.post("/score/ml/{stay_id}/{hour}")
def score_ml(stay_id: int, hour: int) -> dict:
    """Phase 5's trained composite-deterioration model, when one has been
    exported (``python ml/evaluation/run_all.py``). A real 503 -- not a
    fabricated number -- if it hasn't.
    """
    if not serving.promoted_model_available():
        raise HTTPException(
            503,
            "No Phase 5 model exported yet. Run `python ml/evaluation/run_all.py`, "
            "or use /score for the deterministic NEWS2/SOFA path.",
        )
    conn = get_conn()
    try:
        return serving.score_one(conn, stay_id, hour)
    except serving.UnknownStayHour as exc:
        raise HTTPException(404, str(exc)) from exc
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
