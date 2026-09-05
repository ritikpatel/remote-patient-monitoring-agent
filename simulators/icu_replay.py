"""Replay one ICU stay's hourly grid as a stream of Observation messages.

PROJECT_PLAN.md section 8, item 3. Two things distinguish this from "read a table and
print a row every second":

  1. **Time compression, not a uniform tick.** `--compress 3600` (the default) means
     one simulated hour of ICU time passes in one wall-clock second -- the plan's
     "1 h -> 1 s". A full stay replays in a demonstrable handful of seconds instead of
     real hours.
  2. **Arrival-model-driven timing, not "on the hour."** Which second within each
     simulated hour a vital gets charted is drawn from the fitted ICU-monitoring
     arrival model (simulators/arrival_models.py), which is phase-locked per stay --
     E1 found HR gaps cluster at 55-65 min, i.e. a given patient is charted on a fairly
     consistent rhythm, not a metronome and not uniformly random either.

Every emitted Observation carries `source=icu_monitor` and, for a carried-forward
value, `quality_flags=[imputed]` (R2, R7) -- a downstream consumer can always tell a
fresh reading from a forward-filled one, and this replay never presents itself as
anything other than what it is: hourly-charted data played back faster than it was
recorded (R7 / PROJECT_PLAN.md section 17).

Usage:
    python simulators/icu_replay.py --stay-id 34547401 --compress 3600
    python simulators/icu_replay.py --stay-id 34547401 --sink jsonl --out replay.jsonl --no-sleep
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.contracts.observation import (  # noqa: E402
    Observation,
    ObservationSource,
    QualityFlag,
)

from simulators.arrival_models import (  # noqa: E402
    DEFAULT_OUT_PATH as DEFAULT_ARRIVAL_MODELS_PATH,
)
from simulators.arrival_models import (  # noqa: E402
    ArrivalModelSet,
)
from simulators.sinks import Sink, make_sink  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
CORE = ["hr", "rr", "spo2", "sbp", "map", "temp_c", "gcs_total", "fio2", "glucose"]
MONITORING_FAMILY = "ICU monitoring"


def schedule_stay(
    conn: duckdb.DuckDBPyConnection,
    stay_id: int,
    model_set: ArrivalModelSet,
) -> list[Observation]:
    """Build the full, time-ordered Observation schedule for one stay. Every
    non-null CORE value in the hourly grid becomes exactly one Observation, timed
    within its hour by the ICU-monitoring arrival model rather than pinned to :00.
    """
    intime_row = conn.execute(
        "SELECT intime FROM mimiciv_icu.icustays WHERE stay_id = ?", [stay_id]
    ).fetchone()
    if intime_row is None:
        raise ValueError(f"stay_id {stay_id} not found in mimiciv_icu.icustays")
    intime: datetime = intime_row[0]

    cols = ", ".join(CORE + [f"{c}_was_imputed" for c in CORE])
    rows = conn.execute(
        f"SELECT hour, {cols} FROM capstone.hourly_grid WHERE stay_id = ? ORDER BY hour", [stay_id]
    ).fetchall()
    colnames = [d[0] for d in conn.description]

    monitoring = model_set[MONITORING_FAMILY]
    rng = np.random.default_rng(stay_id)
    observations: list[Observation] = []

    for row in rows:
        rec = dict(zip(colnames, row, strict=True))
        h = int(rec["hour"])
        hour_start = intime + timedelta(hours=h)
        present = [c for c in CORE if rec[c] is not None]
        if not present:
            continue
        offsets = monitoring.sample_offsets_seconds(len(present), stay_id=stay_id, rng=rng)
        for col, offset in zip(present, offsets, strict=True):
            effective_time = hour_start + timedelta(seconds=float(offset))
            flags = [QualityFlag.imputed] if rec[f"{col}_was_imputed"] else []
            observations.append(
                Observation.for_channel(
                    channel=col,
                    patient_ref=f"ICUStay/{stay_id}",
                    device_id="icu-monitor-sim",
                    source=ObservationSource.icu_monitor,
                    value=float(rec[col]),
                    effective_time=effective_time,
                    quality_flags=flags,
                )
            )
    observations.sort(key=lambda o: o.effective_time)
    return observations


def replay(observations: list[Observation], sink: Sink, compress: float, sleep: bool) -> None:
    prev_time: datetime | None = None
    for obs in observations:
        if sleep and prev_time is not None:
            sim_dt = (obs.effective_time - prev_time).total_seconds()
            time.sleep(max(sim_dt / compress, 0.0))
        sink.emit(obs)
        prev_time = obs.effective_time


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stay-id", type=int, required=True)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--arrival-models", type=Path, default=DEFAULT_ARRIVAL_MODELS_PATH)
    ap.add_argument(
        "--compress", type=float, default=3600.0, help="simulated seconds per wall-clock second"
    )
    ap.add_argument("--sink", choices=["console", "jsonl"], default="console")
    ap.add_argument("--out", type=Path, default=Path("icu_replay.jsonl"))
    ap.add_argument(
        "--no-sleep",
        action="store_true",
        help="emit as fast as possible, ignoring --compress timing",
    )
    args = ap.parse_args()

    if not args.arrival_models.exists():
        print(
            f"ERROR: {args.arrival_models} not found. Run simulators/arrival_models.py first.",
            file=sys.stderr,
        )
        return 1

    model_set = ArrivalModelSet.from_json(args.arrival_models)
    conn = duckdb.connect(str(args.db), read_only=True)
    observations = schedule_stay(conn, args.stay_id, model_set)
    conn.close()

    if not observations:
        print(f"No hourly_grid rows for stay_id {args.stay_id}.", file=sys.stderr)
        return 1

    span_h = (
        observations[-1].effective_time - observations[0].effective_time
    ).total_seconds() / 3600
    wall_s = span_h * 3600 / args.compress
    print(
        f"Replaying stay {args.stay_id}: {len(observations)} observations over {span_h:.1f}h "
        f"of ICU time, compress={args.compress:.0f}x -> ~{wall_s:.1f}s wall clock"
        + (" (--no-sleep: as fast as possible)" if args.no_sleep else ""),
        file=sys.stderr,
    )

    sink: Sink = make_sink(args.sink, args.out)
    try:
        replay(observations, sink, compress=args.compress, sleep=not args.no_sleep)
    finally:
        sink.close()

    if args.sink == "jsonl":
        print(f"Wrote {len(observations)} observations to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
