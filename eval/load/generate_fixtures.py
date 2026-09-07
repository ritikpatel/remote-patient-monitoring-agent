"""Writes eval/load/fixtures.json -- real (stay_id, hour) pairs the k6 load
test cycles through for its risk-engine calls, so the latency axis exercises
real warehouse-backed lookups rather than made-up identifiers that would
404 on every request (and therefore measure error-handling latency, not the
real scoring path).

Usage: python eval/load/generate_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
OUTPUT_PATH = Path(__file__).resolve().parent / "fixtures.json"
N_FIXTURES = 50
SEED = 0


def main() -> int:
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    conn.execute(f"select setseed({SEED / 2**31 - 1})")
    df = conn.execute(
        f"select stay_id, hour from capstone.news2 order by random() limit {N_FIXTURES}"
    ).fetchdf()
    fixtures = df.to_dict(orient="records")
    OUTPUT_PATH.write_text(json.dumps(fixtures, indent=2))
    print(f"Wrote {len(fixtures)} fixtures to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
