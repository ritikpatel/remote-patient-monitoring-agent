"""Condition a real wearable segment on a target NEWS2 trajectory.

PROJECT_PLAN.md section 8, item 5 (E10): the wearable cohort is healthy volunteers
with no link to the ICU cohort, so a post-discharge *deterioration* scenario cannot
come from real data -- it has to be synthesised, deliberately and visibly (R7).

Three transforms, driven by one target trajectory (a ramp in a two-component NEWS2
proxy -- HR subscore + SpO2 subscore, reusing warehouse/news2.py's exact scoring
thresholds rather than re-deriving them):

  - **HR baseline shift**: the real HR trace's shape/variability is kept, its mean is
    re-targeted hour by hour to the HR that would produce the trajectory's HR
    subscore. Deliberately models only the tachycardic/hypoxic deterioration
    direction (HR rising), not bradycardic arrest -- documented, not hidden.
  - **HRV suppression**: real IBI (inter-beat interval) deviations from a local
    rolling mean are shrunk over the trajectory -- reduced beat-to-beat variability
    is a recognised autonomic-dysfunction marker of deterioration.
  - **SpO2 desaturation**: Empatica E4 does not measure SpO2 at all, so this channel
    does not exist in any real segment. It is synthesised at 1 Hz from the same
    trajectory's SpO2 subscore, entirely fabricated -- not "morphed."

Every Observation this module emits carries `quality_flags=[synthetic]` and
`device_id="morph-sim"` (never "empatica-e4"), and every CLI run prints the
PROJECT_PLAN.md section 17 watermark. This is a transport/DSP and alerting-pathway
testbed, not a clinical claim about any real patient.

Usage:
    python simulators/morphing.py --activity STRESS --participant S01 --duration-s 60 \
        --start-score 0 --end-score 8 --sink jsonl --out morphed.jsonl
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.contracts.observation import (  # noqa: E402
    Observation,
    ObservationSource,
    QualityFlag,
)
from warehouse.news2 import hr_score, spo2_score  # noqa: E402

from simulators.sinks import Sink, make_sink  # noqa: E402
from simulators.wearable_replay import (  # noqa: E402
    ACTIVITIES,
    DEFAULT_ROOT,
    ChannelSamples,
    build_session_index,
    find_session,
    load_session,
    replay,
)

SYNTHETIC_WATERMARK = (
    "SYNTHETIC -- HR is a re-targeted real recording, SpO2 is fabricated from "
    "scratch (Empatica E4 does not measure it), and HRV is artificially suppressed. "
    "No real patient deteriorated. See PROJECT_PLAN.md section 17."
)

# Representative (subscore -> value) breakpoints, inverting warehouse/news2.py's
# thresholds. Only the tachycardic / hypoxic direction is modelled (see docstring).
HR_BREAKPOINTS = {0: 75.0, 1: 100.0, 2: 120.0, 3: 145.0}
SPO2_BREAKPOINTS = {0: 98.0, 1: 95.0, 2: 93.0, 3: 89.0}


def _interp_breakpoints(breakpoints: dict[int, float], subscore: np.ndarray) -> np.ndarray:
    xs = sorted(breakpoints)
    ys = [breakpoints[x] for x in xs]
    return np.interp(subscore, xs, ys)


def _assert_breakpoints_match_scoring() -> None:
    """The breakpoints must actually round-trip through news2.py's real thresholds --
    otherwise a target "subscore" wouldn't correspond to what news2.py would compute
    from the morphed value, silently breaking the honesty of the trajectory.
    """
    for subscore, hr in HR_BREAKPOINTS.items():
        assert hr_score(hr) == subscore, f"HR breakpoint {hr} scores {hr_score(hr)}, not {subscore}"
    for subscore, spo2 in SPO2_BREAKPOINTS.items():
        assert (
            spo2_score(spo2) == subscore
        ), f"SpO2 breakpoint {spo2} scores {spo2_score(spo2)}, not {subscore}"


_assert_breakpoints_match_scoring()


@dataclass
class MorphConfig:
    start_score: float = 0.0  # partial NEWS2 (HR subscore + SpO2 subscore) at t=0, range [0, 6]
    end_score: float = 6.0  # partial NEWS2 at the end of the segment
    gamma: float = 1.0  # >1 = deterioration accelerates late; <1 = front-loaded
    hrv_suppression_start: float = 1.0  # 1.0 = no suppression
    hrv_suppression_end: float = 0.3  # fraction of real IBI variability retained at end
    spo2_noise_std: float = 0.3

    def trajectory(self, frac: np.ndarray) -> np.ndarray:
        frac = np.clip(frac, 0.0, 1.0)
        return self.start_score + (self.end_score - self.start_score) * frac**self.gamma

    def hr_target(self, frac: np.ndarray) -> np.ndarray:
        hr_subscore = np.clip(self.trajectory(frac) / 2, 0, 3)
        return _interp_breakpoints(HR_BREAKPOINTS, hr_subscore)

    def spo2_target(self, frac: np.ndarray) -> np.ndarray:
        spo2_subscore = np.clip(self.trajectory(frac) / 2, 0, 3)
        return _interp_breakpoints(SPO2_BREAKPOINTS, spo2_subscore)

    def hrv_shrink(self, frac: np.ndarray) -> np.ndarray:
        frac = np.clip(frac, 0.0, 1.0)
        return (
            self.hrv_suppression_start
            + (self.hrv_suppression_end - self.hrv_suppression_start) * frac
        )


def _frac_of(
    times: np.ndarray, session_start: np.datetime64, session_end: np.datetime64
) -> np.ndarray:
    span = (session_end - session_start) / np.timedelta64(1, "s")
    elapsed = (times - session_start) / np.timedelta64(1, "s")
    return elapsed / span if span > 0 else np.zeros_like(elapsed, dtype=float)


def morph_hr(cs: ChannelSamples, config: MorphConfig, session_start, session_end) -> ChannelSamples:
    frac = _frac_of(cs.times, session_start, session_end)
    real_mean = float(np.mean(cs.values))
    target = config.hr_target(frac)
    new_values = np.clip(cs.values - real_mean + target, 20.0, 220.0)
    flags = [[QualityFlag.synthetic] for _ in cs.times]
    return ChannelSamples("hr", cs.times, new_values, flags)


def morph_ibi(
    times: np.ndarray, ibi_values: np.ndarray, config: MorphConfig, session_start, session_end
) -> ChannelSamples:
    frac = _frac_of(times, session_start, session_end)
    local_mean = pd.Series(ibi_values).rolling(20, center=True, min_periods=1).mean().to_numpy()
    deviation = ibi_values - local_mean
    shrink = config.hrv_shrink(frac)
    new_values = np.clip(local_mean + deviation * shrink, 0.3, 2.0)
    flags = [[QualityFlag.synthetic] for _ in times]
    return ChannelSamples("ibi", times, new_values, flags)


def synthesize_spo2(
    session_start, session_end, config: MorphConfig, rng: np.random.Generator, hz: float = 1.0
) -> ChannelSamples:
    span_s = (session_end - session_start) / np.timedelta64(1, "s")
    n = max(int(span_s * hz), 1)
    times = session_start + (np.arange(n) / hz) * np.timedelta64(1, "s")
    frac = np.arange(n) / max(n - 1, 1)
    target = config.spo2_target(frac)
    values = np.clip(target + rng.normal(0, config.spo2_noise_std, size=n), 70.0, 100.0)
    flags = [[QualityFlag.synthetic] for _ in range(n)]
    return ChannelSamples("spo2", times, values, flags)


def _load_ibi(session, root: Path) -> tuple[np.ndarray, np.ndarray]:
    times_parts, ibi_parts = [], []
    for part in session.parts:
        path = root / session.activity / part / "IBI.csv"
        if not path.exists():
            continue
        with open(path) as f:
            header = f.readline().strip().split(",")
        start = pd.Timestamp(header[0], tz="UTC")
        data = pd.read_csv(path, skiprows=1, header=None, names=["offset_s", "ibi_s"])
        if data.empty:
            continue
        # naive UTC datetime64[ns] -- see the matching comment in wearable_replay.py.
        times_parts.append(
            (start + pd.to_timedelta(data.offset_s, unit="s")).dt.tz_localize(None).to_numpy()
        )
        ibi_parts.append(data.ibi_s.to_numpy())
    if not times_parts:
        return np.array([]), np.array([])
    return np.concatenate(times_parts), np.concatenate(ibi_parts)


def morph_session(
    activity: str,
    participant: str,
    config: MorphConfig,
    root: Path = DEFAULT_ROOT,
    rng: np.random.Generator | None = None,
    start_offset_s: float = 0.0,
    duration_s: float | None = None,
) -> dict[str, ChannelSamples]:
    """The trajectory spans exactly the replayed window (start_offset_s .. +duration_s),
    not the whole raw recording -- clipping happens to the RAW input before morphing,
    not to the morphed output, so `--end-score 8 --duration-s 20` actually reaches
    score 8 by second 20, rather than by the end of a 30+ minute source recording.
    """
    rng = rng or np.random.default_rng(0)
    session = find_session(root, activity, participant)
    channels = load_session(session, root, ["hr"])
    ibi_times, ibi_values = _load_ibi(session, root)

    starts = [cs.times[0] for cs in channels.values() if len(cs.times)]
    ends = [cs.times[-1] for cs in channels.values() if len(cs.times)]
    if len(ibi_times):
        starts.append(ibi_times[0])
        ends.append(ibi_times[-1])
    raw_start = min(starts)

    window_start = raw_start + np.timedelta64(int(start_offset_s * 1e9), "ns")
    window_end = (
        window_start + np.timedelta64(int(duration_s * 1e9), "ns")
        if duration_s is not None
        else max(ends)
    )

    def _clip(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = (times >= window_start) & (times <= window_end)
        return times[mask], values[mask]

    morphed: dict[str, ChannelSamples] = {}
    if "hr" in channels and len(channels["hr"].times):
        t, v = _clip(channels["hr"].times, channels["hr"].values)
        if len(t):
            morphed["hr"] = morph_hr(
                ChannelSamples("hr", t, v, [[]] * len(t)), config, window_start, window_end
            )
    if len(ibi_times):
        t, v = _clip(ibi_times, ibi_values)
        if len(t):
            morphed["ibi"] = morph_ibi(t, v, config, window_start, window_end)
    morphed["spo2"] = synthesize_spo2(window_start, window_end, config, rng)
    return morphed


def to_observations(channels: dict[str, ChannelSamples], participant: str) -> list[Observation]:
    observations = []
    for ch, cs in channels.items():
        for t, v, flags in zip(cs.times, cs.values, cs.quality_flags, strict=True):
            observations.append(
                Observation.for_channel(
                    channel=ch,
                    patient_ref=f"Subject/{participant}",
                    device_id="morph-sim",
                    source=ObservationSource.wearable,
                    value=float(v),
                    effective_time=pd.Timestamp(t).to_pydatetime(),
                    quality_flags=list(flags),
                )
            )
    observations.sort(key=lambda o: o.effective_time)
    return observations


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--activity", choices=ACTIVITIES, default="STRESS")
    ap.add_argument("--participant", default="S01")
    ap.add_argument("--start-offset-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=60.0, help="0 = whole session")
    ap.add_argument("--start-score", type=float, default=0.0)
    ap.add_argument("--end-score", type=float, default=6.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--compress", type=float, default=1.0)
    ap.add_argument("--sink", choices=["console", "jsonl"], default="console")
    ap.add_argument("--out", type=Path, default=Path("morphed.jsonl"))
    ap.add_argument("--no-sleep", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for s in build_session_index(args.root):
            print(f"{s.activity:<10} {s.participant}")
        return 0

    print(f"*** {SYNTHETIC_WATERMARK} ***", file=sys.stderr)

    config = MorphConfig(start_score=args.start_score, end_score=args.end_score, gamma=args.gamma)
    duration = None if args.duration_s <= 0 else args.duration_s
    channels = morph_session(
        args.activity,
        args.participant,
        config,
        args.root,
        start_offset_s=args.start_offset_s,
        duration_s=duration,
    )
    observations = to_observations(channels, args.participant)

    if not observations:
        print("No observations produced.", file=sys.stderr)
        return 1

    assert all(
        QualityFlag.synthetic in o.quality_flags for o in observations
    ), "every morphed Observation must be watermarked synthetic"

    span_s = (observations[-1].effective_time - observations[0].effective_time).total_seconds()
    print(
        f"Morphed {args.activity}/{args.participant}: {len(observations)} observations "
        f"over {span_s:.1f}s, target partial-NEWS2 {args.start_score:.0f} -> {args.end_score:.0f}",
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
