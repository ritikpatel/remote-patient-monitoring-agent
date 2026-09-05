"""alert-service: raise/dedupe/suppress/escalate/acknowledge over HTTP.
See store.py for the real logic (R6: dedup aligned to the 4-hourly clock)."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from store import AlertStore  # noqa: E402

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "alerts.db"

app = FastAPI(title="alert-service", version="0.1.0")
_store: AlertStore | None = None


def get_store() -> AlertStore:
    global _store
    if _store is None:
        _store = AlertStore(DEFAULT_DB_PATH)
    return _store


def set_store(store: AlertStore) -> None:
    global _store
    _store = store


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "alert-service"}


class RaiseAlertRequest(BaseModel):
    patient_ref: str
    alert_type: str
    severity: str
    message: str


@app.post("/alerts")
def raise_alert(req: RaiseAlertRequest) -> dict:
    alert, was_new = get_store().raise_alert(
        req.patient_ref, req.alert_type, req.severity, req.message
    )
    return {"alert": vars(alert), "was_new": was_new}


@app.post("/alerts/{alert_id}/suppress")
def suppress(alert_id: int) -> dict:
    try:
        return vars(get_store().suppress(alert_id))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/alerts/{alert_id}/escalate")
def escalate(alert_id: int) -> dict:
    try:
        return vars(get_store().escalate(alert_id))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


class AckRequest(BaseModel):
    acknowledged_by: str


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge(alert_id: int, req: AckRequest) -> dict:
    try:
        return vars(get_store().acknowledge(alert_id, req.acknowledged_by))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/alerts")
def active_for_patient(patient_ref: str) -> list[dict]:
    return [vars(a) for a in get_store().active_for_patient(patient_ref)]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)
