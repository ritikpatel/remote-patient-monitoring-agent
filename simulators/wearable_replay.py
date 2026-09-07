"""True-rate replay of one Empatica E4 wearable session.

PROJECT_PLAN.md section 8, item 4. Streams at the device's actual sample rates --
BVP 64 Hz, ACC 32 Hz (3-axis), EDA/TEMP 4 Hz, HR 1 Hz -- rather than the hourly
cadence of the ICU replay: this is the one genuinely high-frequency source in the
whole platform (E1's "real-time must be produced, not assumed" cuts the other way
here -- this data really is real-time).

Honours data_constraints.txt exactly as PROJECT_PLAN.md directs, as fault fixtures
rather than nuisances to clean away:

  - f07 (STRESS): the wristband dock was never removed, so BVP and TEMP are invalid
    for this session. Still emitted (a consumer needs something to filter), every
    sample tagged `quality_flags=[device_fault]`.
  - S02 (STRESS): ACC/BVP/EDA/TEMP each have a documented row at which E4 Connect's
    export duplicates the rest of the signal. Everything from that row onward is
    tagged `quality_flags=[duplicate]` rather than silently dropped.
  - f14_a/b (STRESS), S11_a/b (AEROBIC), S16_a/b (ANAEROBIC): Bluetooth dropped
    mid-registration, splitting one logical session across two directories. Detected
    generically (any `<id>_a` / `<id>_b` pair) and replayed as one continuous session,
    concatenated in order.

Every emitted Observation carries `source=wearable` and `patient_ref="Subject/<id>"`
-- never `Patient/...` or `ICUStay/...` -- because these volunteers have no link to
the MIMIC-IV clinical cohort (E10): the naming itself keeps that boundary visible
downstream, not just in a comment.

Usage:
    python simulators/wearable_replay.py --list
    python simulators/wearable_replay.py --activity STRESS --participant S01 --duration-s 30
    python simulators/wearable_replay.py --activity STRESS --participant f07   # fault fixture
    python simulators/wearable_replay.py --activity STRESS --participant S02   # fault fixture
    python simulators/wearable_replay.py --activity STRESS --participant f14   # split session
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.contracts.observation import (  # noqa: E402
    Observation,
    ObservationSource,
    QualityFlag,
)

from simulators.sinks import Sink, make_sink  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = (
    REPO_ROOT
    / "data"
    / "raw"
    / "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"
    / "Wearable_Dataset"
)
ACTIVITIES = ["STRESS", "AEROBIC", "ANAEROBIC"]

# channel name -> (filename, n_value_columns). ACC is one file, three columns, three
# channels. IBI is event-timed (no fixed rate) so it is handled separately.
FIXED_RATE_FILES: dict[str, tuple[str, int]] = {
    "bvp": ("BVP.csv", 1),
    "eda": ("EDA.csv", 1),
    "temp_c": ("TEMP.csv", 1),
    "hr": ("HR.csv", 1),
    "acc": ("ACC.csv", 3),
}

# data_constraints.txt, applied exactly as documented -- see module docstring.
INVALID_CHANNELS: dict[tuple[str, str], set[str]] = {
    ("STRESS", "f07"): {"bvp", "temp_c"},
}
# "Duplicated raw values start in: ACC.csv: row 49,545; BVP.csv: row 99,091;
# EDA.csv and TEMP.csv: row 6,195." Read as the 1-indexed line number in the raw
# file (including the two header lines) -- converted to a 0-indexed sample position
# by subtracting 2 in _load_fixed_rate_channel.
DUPLICATE_START_LINE: dict[tuple[str, str], dict[str, int]] = {
    ("STRESS", "S02"): {"acc": 49_545, "bvp": 99_091, "eda": 6_195, "temp_c": 6_195},
}

SPLIT_PATTERN = re.compile(r"^(.+)_([ab])$")


@dataclass
class WearableSession:
    activity: str
    participant: str  # canonical id, e.g. "f14" for the f14_a/f14_b split
    parts: list[str]  # directories to concatenate in order, e.g. ["f14_a", "f14_b"]
    invalid_channels: set[str] = field(default_factory=set)
    duplicate_start_line: dict[str, int] = field(default_factory=dict)

    @property
    def is_split(self) -> bool:
        return len(self.parts) > 1

    @property
    def is_fault_fixture(self) -> bool:
        return bool(self.invalid_channels or self.duplicate_start_line)


def build_session_index(root: Path = DEFAULT_ROOT) -> list[WearableSession]:
    sessions = []
    for activity in ACTIVITIES:
        adir = root / activity
        if not adir.is_dir():
            continue
        groups: dict[str, dict[str, str]] = {}
        for d in sorted(p.name for p in adir.iterdir() if p.is_dir()):
            m = SPLIT_PATTERN.match(d)
            if m:
                groups.setdefault(m.group(1), {})[m.group(2)] = d
            else:
                groups.setdefault(d, {})["_"] = d
        for base, parts_map in groups.items():
            parts = (
                [parts_map["_"]] if "_" in parts_map else [parts_map[k] for k in sorted(parts_map)]
            )
            sessions.append(
                WearableSession(
                    activity=activity,
                    participant=base,
                    parts=parts,
                    invalid_channels=set(INVALID_CHANNELS.get((activity, base), set())),
                    duplicate_start_line=dict(DUPLICATE_START_LINE.get((activity, base), {})),
                )
            )
    return sorted(sessions, key=lambda s: (s.activity, s.participant))


def find_session(root: Path, activity: str, participant: str) -> WearableSession:
    for s in build_session_index(root):
        if s.activity == activity and s.participant == participant:
            return s
    raise ValueError(f"No session {activity}/{participant} under {root}")


@dataclass
class ChannelSamples:
    channel: str
    times: np.ndarray  # datetime64[ns, UTC] per sample
    values: np.ndarray  # float, one column
    quality_flags: list[list[QualityFlag]]  # per-sample, usually []


def _load_fixed_rate_part(
    part_dir: Path, channel: str, dup_start_line: int | None
) -> list[ChannelSamples]:
    filename, n_cols = FIXED_RATE_FILES[channel]
    path = part_dir / filename
    if not path.exists():
        return []
    with open(path) as f:
        header0 = f.readline().strip().split(",")
        header1 = f.readline().strip().split(",")
    start = pd.Timestamp(header0[0], tz="UTC")
    hz = float(header1[0])
    data = pd.read_csv(path, skiprows=2, header=None).to_numpy(dtype=float)
    assert data.shape[1] == n_cols, (
        f"{path}: expected {n_cols} column(s) for channel {channel!r} "
        f"(FIXED_RATE_FILES), found {data.shape[1]} -- the file's own header "
        f"row was never actually checked against this before"
    )
    n = len(data)
    # tz-aware Timestamps stored in a numpy array come back dtype=object (numpy
    # datetime64 has no tz support), which breaks vectorised arithmetic downstream
    # (e.g. array - scalar_Timestamp raises TypeError). Drop to naive UTC
    # datetime64[ns] here; tz is reattached at the Observation boundary.
    times = (start + pd.to_timedelta(np.arange(n) / hz, unit="s")).tz_localize(None)

    dup_from = None
    if dup_start_line is not None:
        dup_from = max(dup_start_line - 2, 0)  # -2: subtract the two header lines

    out = []
    if channel == "acc":
        axes = ["acc_x", "acc_y", "acc_z"]
        for i, axis in enumerate(axes):
            flags: list[list[QualityFlag]] = [[]] * n
            if dup_from is not None and dup_from < n:
                flags = [[] if j < dup_from else [QualityFlag.duplicate] for j in range(n)]
            out.append(ChannelSamples(axis, times.to_numpy(), data[:, i], flags))
    else:
        flags = [[]] * n
        if dup_from is not None and dup_from < n:
            flags = [[] if j < dup_from else [QualityFlag.duplicate] for j in range(n)]
        out.append(ChannelSamples(channel, times.to_numpy(), data[:, 0], flags))
    return out


def load_session(
    session: WearableSession, root: Path = DEFAULT_ROOT, channels: list[str] | None = None
) -> dict[str, ChannelSamples]:
    """Load and concatenate every part of a session, applying invalid-channel and
    duplicate-block flags. Returns one ChannelSamples per requested channel
    (acc expands to acc_x/acc_y/acc_z).
    """
    wanted = channels or ["bvp", "acc", "eda", "temp_c", "hr"]
    per_channel: dict[str, list[ChannelSamples]] = {}

    for part in session.parts:
        part_dir = root / session.activity / part
        for ch in wanted:
            dup_line = session.duplicate_start_line.get(ch)
            for cs in _load_fixed_rate_part(part_dir, ch, dup_line):
                per_channel.setdefault(cs.channel, []).append(cs)

    merged: dict[str, ChannelSamples] = {}
    for ch, parts in per_channel.items():
        times = np.concatenate([p.times for p in parts])
        values = np.concatenate([p.values for p in parts])
        flags: list[list[QualityFlag]] = sum((p.quality_flags for p in parts), [])
        merged[ch] = ChannelSamples(ch, times, values, flags)

    # invalid-channel fault (f07): tag every sample device_fault rather than dropping it.
    # acc_x/acc_y/acc_z are derived from the "acc" entry in session.invalid_channels.
    for ch_name, cs in merged.items():
        base = "acc" if ch_name.startswith("acc_") else ch_name
        if base in session.invalid_channels:
            cs.quality_flags = [list({*f, QualityFlag.device_fault}) for f in cs.quality_flags] or [
                [QualityFlag.device_fault]
            ] * len(cs.times)
    return merged


def to_observations(
    channel_samples: dict[str, ChannelSamples],
    participant: str,
    start_offset_s: float = 0.0,
    duration_s: float | None = None,
) -> list[Observation]:
    observations: list[Observation] = []
    nonempty = [cs.times for cs in channel_samples.values() if len(cs.times)]
    if not nonempty:
        return observations
    session_start = min(
        t[0] for t in nonempty
    )  # one shared anchor -- channels start at different times

    for ch, cs in channel_samples.items():
        if len(cs.times) == 0:
            continue
        t0 = session_start + np.timedelta64(int(start_offset_s * 1e9), "ns")
        t1 = (
            t0 + np.timedelta64(int(duration_s * 1e9), "ns")
            if duration_s is not None
            else cs.times[-1]
        )
        mask = (cs.times >= t0) & (cs.times <= t1)
        # plain zip+filter, not np.array(quality_flags)[mask]: when every sample's
        # flag list happens to be the same length, numpy collapses the ragged list
        # into a 2D array instead of a 1D array of lists.
        kept_flags = [f for f, keep in zip(cs.quality_flags, mask, strict=True) if keep]
        for t, v, flags in zip(cs.times[mask], cs.values[mask], kept_flags, strict=True):
            observations.append(
                Observation.for_channel(
                    channel=ch,
                    patient_ref=f"Subject/{participant}",
                    device_id="empatica-e4",
                    source=ObservationSource.wearable,
                    value=float(v),
                    effective_time=pd.Timestamp(t).to_pydatetime(),
                    quality_flags=list(flags),
                )
            )
    observations.sort(key=lambda o: o.effective_time)
    return observations


def replay(observations: list[Observation], sink: Sink, compress: float, sleep: bool) -> None:
    prev: datetime | None = None
    for obs in observations:
        if sleep and prev is not None:
            time.sleep(max((obs.effective_time - prev).total_seconds() / compress, 0.0))
        sink.emit(obs)
        prev = obs.effective_time


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--activity", choices=ACTIVITIES)
    ap.add_argument("--participant")
    ap.add_argument("--channels", default="bvp,acc,eda,temp_c,hr")
    ap.add_argument("--start-offset-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=30.0, help="0 = whole session")
    ap.add_argument("--compress", type=float, default=1.0, help="1.0 = true device rate")
    ap.add_argument("--sink", choices=["console", "jsonl"], default="console")
    ap.add_argument("--out", type=Path, default=Path("wearable_replay.jsonl"))
    ap.add_argument("--no-sleep", action="store_true")
    ap.add_argument("--list", action="store_true", help="list available sessions and exit")
    args = ap.parse_args()

    if args.list or not args.activity or not args.participant:
        for s in build_session_index(args.root):
            tag = " [SPLIT]" if s.is_split else ""
            tag += " [FAULT FIXTURE]" if s.is_fault_fixture else ""
            print(f"{s.activity:<10} {s.participant:<6} parts={s.parts}{tag}")
        return 0

    session = find_session(args.root, args.activity, args.participant)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    channel_samples = load_session(session, args.root, channels)
    duration = None if args.duration_s <= 0 else args.duration_s
    observations = to_observations(
        channel_samples, session.participant, args.start_offset_s, duration
    )

    if not observations:
        print(f"No samples for {session.activity}/{session.participant}.", file=sys.stderr)
        return 1

    span_s = (observations[-1].effective_time - observations[0].effective_time).total_seconds()
    print(
        f"Replaying {session.activity}/{session.participant} (parts={session.parts}): "
        f"{len(observations)} samples over {span_s:.1f}s, compress={args.compress:.0f}x"
        + (" -- FAULT FIXTURE" if session.is_fault_fixture else ""),
        file=sys.stderr,
    )

    sink = make_sink(args.sink, args.out)
    try:
        replay(observations, sink, compress=args.compress, sleep=not args.no_sleep)
    finally:
        sink.close()
    if args.sink == "jsonl":
        print(f"Wrote {len(observations)} observations to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
