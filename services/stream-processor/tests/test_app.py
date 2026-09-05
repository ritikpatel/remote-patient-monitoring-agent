import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from services.common.testing import load_service_app  # noqa: E402

app = load_service_app("stream-processor", REPO_ROOT).app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_process_window():
    resp = client.post(
        "/window/process",
        json={
            "patient_ref": "ICUStay/1",
            "channel": "hr",
            "values": [70, 72, 74, 76],
            "timestamps_hours": [0, 1, 2, 3],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mean"] == 73.0
    assert body["trend_slope_per_hour"] == pytest.approx(2.0)
    assert body["n_samples"] == 4


def test_process_window_rejects_mismatched_lengths():
    resp = client.post(
        "/window/process",
        json={
            "patient_ref": "ICUStay/1",
            "channel": "hr",
            "values": [1, 2],
            "timestamps_hours": [0],
        },
    )
    assert resp.status_code == 422


def test_process_window_rejects_empty_values():
    resp = client.post(
        "/window/process",
        json={"patient_ref": "ICUStay/1", "channel": "hr", "values": [], "timestamps_hours": []},
    )
    assert resp.status_code == 422


def test_process_hrv():
    resp = client.post(
        "/window/hrv", json={"patient_ref": "ICUStay/1", "ibi_seconds": [0.8, 0.82, 0.79]}
    )
    assert resp.status_code == 200
    assert resp.json()["rmssd_ms"] > 0


def test_process_event_rate_unknown_family():
    resp = client.post(
        "/window/event_rate",
        json={"family": "Not A Real Family", "raw_count": 5, "window_hours": 1},
    )
    assert resp.status_code == 422
