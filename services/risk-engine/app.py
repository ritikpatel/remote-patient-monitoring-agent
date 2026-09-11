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
    # Finding F1: NEWS2's single-parameter limb. These are *facts*, not a decision --
    # the escalation policy itself stays in agent-orchestrator's EscalationDecider
    # (PROJECT_PLAN.md section 10). `escalation_recommended` is included because the
    # predicate is shared code (warehouse.news2.should_escalate), so exposing it here
    # cannot drift from what the policy node computes.
    max_component: int | None = None
    max_component_nongcs: int | None = None
    red_params: list[str] = []
    gcs_drop: bool = False
    escalation_recommended: bool = False
    # Which threshold escalated this patient. `tier_icu` is now fitted per
    # primary-diagnosis chapter where that chapter had enough stays to earn its own
    # cut-point (warehouse/news2.py), so "NEWS2 tier is high" is no longer a single
    # global statement -- a consumer that wants to say *which* threshold fired needs
    # these two, and without them a disease-specific escalation is indistinguishable
    # from a pooled one.
    dx_group: str | None = None
    threshold_is_disease_specific: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "risk-engine"}


@app.get("/score/{stay_id}/{hour}", response_model=RiskScoreResponse)
def score(stay_id: int, hour: int) -> RiskScoreResponse:
    conn = get_conn()
    try:
        news2_row = conn.execute(
            "SELECT news2, tier_ward, tier_icu, hr, rr, spo2, sbp, temp_c, gcs_total, fio2, "
            "max_component, max_component_nongcs, red_params, gcs_drop, "
            "dx_group, threshold_is_disease_specific "
            "FROM capstone.news2 WHERE stay_id = ? AND hour = ?",
            [stay_id, hour],
        ).fetchone()
        if news2_row is None:
            raise HTTPException(404, f"no hourly_grid/news2 row for stay_id={stay_id} hour={hour}")
        (
            news2,
            tier_ward,
            tier_icu,
            hr,
            rr,
            spo2,
            sbp,
            temp_c,
            gcs_total,
            fio2,
            max_component,
            max_component_nongcs,
            red_params,
            gcs_drop,
            dx_group,
            threshold_is_disease_specific,
        ) = news2_row

        sofa_row = conn.execute(
            "SELECT sofa_24hours FROM mimiciv_derived.sofa WHERE stay_id = ? AND hr = ?",
            [stay_id, hour],
        ).fetchone()
        sofa_24h = sofa_row[0] if sofa_row else None

        from warehouse.news2 import should_escalate

        reason = _explain(hr, rr, spo2, sbp, temp_c, gcs_total, fio2)
        return RiskScoreResponse(
            stay_id=stay_id,
            hour=hour,
            news2=news2,
            news2_tier_ward=tier_ward,
            news2_tier_icu=tier_icu,
            sofa_24h=sofa_24h,
            reason=reason,
            max_component=max_component,
            max_component_nongcs=max_component_nongcs,
            red_params=[p for p in (red_params or "").split(",") if p],
            gcs_drop=bool(gcs_drop),
            escalation_recommended=should_escalate(tier_icu, max_component_nongcs, gcs_drop),
            dx_group=dx_group,
            threshold_is_disease_specific=bool(threshold_is_disease_specific),
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


class LiveVitals(BaseModel):
    """The vitals a streaming consumer actually has: whatever channels have arrived
    for this patient, plus optional trajectory/medication context it may know."""

    hr: float | None = None
    rr: float | None = None
    spo2: float | None = None
    sbp: float | None = None
    temp_c: float | None = None
    gcs_total: float | None = None
    fio2: float | None = None
    # Trajectory context for NEWS2's third escalation limb. A stream consumer has a
    # rolling window and can supply the recent best GCS; it generally has no
    # medication feed, so `sedated` is genuinely unknown rather than false.
    gcs_prev_max: float | None = None
    sedated: bool | None = None


class LiveScoreRequest(BaseModel):
    patient_ref: str
    vitals: LiveVitals
    # The patient's primary-diagnosis chapter, when the caller knows it. A replay of
    # a warehouse stay does; a wearable stream genuinely does not, which is why this
    # is optional rather than required -- an unknown diagnosis falls back to the
    # pooled ICU cut-points, exactly as before per-disease thresholds existed.
    dx_group: str | None = None


class LiveScoreResponse(BaseModel):
    patient_ref: str
    news2: int
    components_available: int
    news2_tier_ward: str
    news2_tier_icu: str
    max_component: int
    max_component_nongcs: int
    red_params: list[str]
    gcs_drop: bool
    sedation_status_known: bool
    escalation_recommended: bool
    escalation_reason: str
    reason: list[str]
    dx_group: str | None = None
    threshold_is_disease_specific: bool = False


@app.post("/score/live", response_model=LiveScoreResponse)
def score_live(req: LiveScoreRequest) -> LiveScoreResponse:
    """Score an observation as it streams, with no warehouse row to look up.

    ``/score/{stay_id}/{hour}`` is a lookup against the precomputed hourly grid, which
    only exists for the 140 ICU stays already in the warehouse. A live producer -- a
    replay driving ingest-gateway, or a wearable that has no ``stay_id`` at all -- has
    vitals and nothing else, so before F3 there was no way to score the stream. This
    applies exactly the same component scoring, the same persisted ICU-recalibrated
    cut-points, and the same ``should_escalate`` predicate as the batch path.

    Sedation caveat: NEWS2's GCS-drop limb is suppressed by concurrent sedation
    (warehouse/news2.py), but a stream carries no medication feed. When ``sedated`` is
    not supplied the limb still fires -- escalating a possibly-sedated patient is the
    safer error than silently suppressing a real neurological deterioration -- and
    ``sedation_status_known`` is False so a consumer can see it was a guess.
    """
    import pandas as pd
    from warehouse.news2 import (
        GCS_DROP_POINTS,
        POOLED_GROUP,
        escalation_reason,
        load_thresholds,
        news2_row,
        red_flags,
        should_escalate,
        tier_for_score,
    )

    v = req.vitals
    row = pd.Series(
        {
            "hr": v.hr,
            "rr": v.rr,
            "spo2": v.spo2,
            "sbp": v.sbp,
            "temp_c": v.temp_c,
            "gcs_total": v.gcs_total,
            "fio2": v.fio2,
        }
    )
    scored = news2_row(row)
    flags = red_flags(row)

    gcs_drop = False
    if v.gcs_total is not None and v.gcs_prev_max is not None:
        fell = (v.gcs_prev_max - v.gcs_total) >= GCS_DROP_POINTS
        gcs_drop = bool(fell and not bool(v.sedated))

    conn = get_conn()
    try:
        thresholds = load_thresholds(conn, req.dx_group)
    finally:
        conn.close()

    news2 = int(scored["news2"])
    tier_icu = tier_for_score(news2, thresholds.icu_medium, thresholds.icu_high)
    max_nongcs = int(flags["max_component_nongcs"])
    red_params = str(flags["red_params"])
    reason = escalation_reason(tier_icu, max_nongcs, red_params, gcs_drop)
    if gcs_drop and v.sedated is None:
        reason += " (sedation status unknown to the stream)"

    return LiveScoreResponse(
        patient_ref=req.patient_ref,
        news2=news2,
        components_available=int(scored["components"]),
        news2_tier_ward=tier_for_score(news2, thresholds.ward_medium, thresholds.ward_high),
        news2_tier_icu=tier_icu,
        max_component=int(flags["max_component"]),
        max_component_nongcs=max_nongcs,
        red_params=[p for p in red_params.split(",") if p],
        gcs_drop=gcs_drop,
        sedation_status_known=v.sedated is not None,
        escalation_recommended=should_escalate(tier_icu, max_nongcs, gcs_drop),
        escalation_reason=reason,
        reason=_explain(v.hr, v.rr, v.spo2, v.sbp, v.temp_c, v.gcs_total, v.fio2),
        dx_group=thresholds.dx_group,
        threshold_is_disease_specific=not thresholds.is_fallback
        and thresholds.dx_group != POOLED_GROUP,
    )


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
        rows = conn.execute("""
            SELECT n.stay_id, n.hour, n.news2, n.tier_icu, s.sofa_24hours
            FROM capstone.news2 n
            JOIN (SELECT stay_id, MAX(hour) AS hour FROM capstone.news2 GROUP BY stay_id) latest
              ON n.stay_id = latest.stay_id AND n.hour = latest.hour
            LEFT JOIN mimiciv_derived.sofa s ON s.stay_id = n.stay_id AND s.hr = n.hour
            ORDER BY n.news2 DESC
            """).fetchall()
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
