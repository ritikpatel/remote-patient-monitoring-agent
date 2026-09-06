from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from reports.shift_handover import _last_4h_window, build_ward_shift_handover, dedup_bucket_start

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"


def _busiest_ward(conn: duckdb.DuckDBPyConnection) -> str:
    row = conn.execute(
        "select first_careunit from mimiciv_icu.icustays group by 1 order by count(*) desc limit 1"
    ).fetchone()
    assert row is not None  # mimiciv_icu.icustays is never empty in this warehouse
    return row[0]


def test_last_4h_window_keeps_only_rows_in_the_latest_real_boundary_block() -> None:
    rows = pd.DataFrame(
        {
            "hour": range(10),
            "abs_time": pd.to_datetime(
                [
                    "2110-01-01 01:00",
                    "2110-01-01 02:00",
                    "2110-01-01 03:00",
                    "2110-01-01 04:00",  # new 4h bucket starts here
                    "2110-01-01 05:00",
                    "2110-01-01 06:00",
                    "2110-01-01 07:00",
                    "2110-01-01 08:00",  # another new bucket
                    "2110-01-01 08:30",
                    "2110-01-01 08:45",
                ]
            ),
        }
    )
    window = _last_4h_window(rows)
    assert list(window.hour) == [7, 8, 9]
    assert dedup_bucket_start(rows.abs_time.iloc[-1].to_pydatetime()) == datetime(2110, 1, 1, 8, 0)


class _FakeAlertStore:
    def active_for_patient(self, patient_ref: str) -> list:
        return []


@pytest.mark.skipif(not WAREHOUSE_DB.exists(), reason="real warehouse db not built")
def test_build_ward_shift_handover_against_real_warehouse() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    ward = _busiest_ward(conn)

    result = build_ward_shift_handover(conn, ward, llm=None, alert_store=_FakeAlertStore())

    assert result.ward == ward
    assert len(result.patients) > 0
    assert result.narrative.startswith("[no LLM configured]")
    for p in result.patients:
        assert p.window_end >= p.window_start
        assert p.tier_end in ("low", "medium", "high")


@pytest.mark.skipif(not WAREHOUSE_DB.exists(), reason="real warehouse db not built")
def test_build_ward_shift_handover_sorts_by_current_news2_descending() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    ward = _busiest_ward(conn)
    result = build_ward_shift_handover(conn, ward, llm=None, alert_store=_FakeAlertStore())
    scores = [p.news2_end for p in result.patients]
    assert scores == sorted(scores, reverse=True)
