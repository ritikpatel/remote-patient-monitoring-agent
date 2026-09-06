from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from reports.daily_summary import build_daily_summary

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"

pytestmark = pytest.mark.skipif(not WAREHOUSE_DB.exists(), reason="real warehouse db not built")


def _a_multi_day_stay(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute(
        "select stay_id from capstone.news2 group by stay_id having max(hour) >= 24 limit 1"
    ).fetchone()
    assert row is not None  # this cohort has several multi-day stays
    return row[0]


def test_build_daily_summary_covers_exactly_hours_0_to_23_for_day_0() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    stay_id = _a_multi_day_stay(conn)
    result = build_daily_summary(conn, stay_id, day_index=0, llm=None)
    assert result.hour_range == (0, 23)
    assert len(result.news2_trajectory) == len(result.tier_trajectory)
    assert result.patient_ref == f"ICUStay/{stay_id}"


def test_build_daily_summary_day_1_covers_hours_24_to_47() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    stay_id = _a_multi_day_stay(conn)
    result = build_daily_summary(conn, stay_id, day_index=1, llm=None)
    assert result.hour_range == (24, 47)


def test_build_daily_summary_raises_for_a_day_the_stay_never_reached() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    row = conn.execute(
        "select stay_id from capstone.news2 group by stay_id having max(hour) < 24 limit 1"
    ).fetchone()
    assert row is not None  # this cohort has several short stays
    with pytest.raises(ValueError):
        build_daily_summary(conn, row[0], day_index=5, llm=None)
