"""Post-discharge weekly digest from home-kit telemetry (PROJECT_PLAN.md
section 12).

**Rewritten when the volunteer wearable dataset was removed from the project.**
The previous version built this digest from `simulators/morphing.py`: a healthy
21-year-old's Empatica session with a deterioration *synthesised onto it*, then a
tens-of-minutes recording divided into seven buckets labelled "day 1".."day 7" to
stand in for a week. Two compressions of reality stacked on each other -- an
invented deterioration and an invented week.

This version needs neither. `simulators/home_kit_stream.py` streams a **real
deteriorating MIMIC ICU patient** through a simulated home sensor kit, and those
stays are genuinely long: the candidate list runs to 200-500 recorded hours, i.e.
8-20 real days. So the daily buckets here are **real elapsed days of a real
patient's real physiology**, and the only synthetic layer is the instrument --
device cadence, measurement noise and non-wear gaps (R7; see that module's
docstring for exactly which parts are simulated and which are recorded).

**What was lost, and not faked to cover it.** The old digest reported HRV (RMSSD)
from the Empatica's beat-to-beat inter-beat intervals. MIMIC charts heart rate
hourly, not beat-to-beat, so RMSSD is **not computable** from this source. It is
therefore absent rather than approximated from hourly HR, which would be a
fabricated number wearing a real metric's name. The digest reports what a home kit
on this patient would actually have: HR, SpO2, respiratory rate, cuff blood
pressure and CGM glucose, per real day.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from simulators.home_kit_stream import (  # noqa: E402
    DEFAULT_DB_PATH,
    DEFAULT_KIT,
    HOME_KITS,
    NO_HOME_SENSOR,
    SYNTHETIC_WATERMARK,
    build_stream,
)

from reports.narrative import (  # noqa: E402
    ANTI_FABRICATION_INSTRUCTION,
    LLMBackend,
    generate_narrative,
)

# A week's digest. Unlike the previous version this is a *window* over a longer
# real record, not a compression of a short one into a fictional week.
N_DIGEST_DAYS = 7

DIGEST_ANTI_FABRICATION_SYSTEM = (
    "You write a post-discharge weekly home-monitoring digest for a remote "
    "monitoring nurse from the structured per-day facts given. The patient and "
    "their physiology are real de-identified ICU records; the home device layer "
    "(sampling cadence, measurement noise, non-wear gaps) is simulated -- say so "
    "explicitly in your response, in your first sentence. Note that HRV is not "
    "reported because it is not computable from this source; do not infer it. "
    + ANTI_FABRICATION_INSTRUCTION
)

# Channel -> the unit to render it in, for the fact lines handed to the LLM.
_UNITS = {
    "hr": "bpm",
    "rr": "/min",
    "spo2": "%",
    "sbp": "mmHg",
    "map": "mmHg",
    "glucose": "mg/dL",
}


@dataclass
class DailyHomeKitBucket:
    """One real elapsed day. ``means`` holds only the channels that actually had a
    sample that day -- a channel absent from the dict was not measured, which is a
    different statement from a channel measured as zero, and a nurse reading the
    digest needs to be able to tell those apart."""

    day_index: int
    means: dict[str, float]
    n_samples: int


@dataclass
class PostDischargeDigest:
    stay_id: int
    subject_ref: str
    kit: str
    daily_buckets: list[DailyHomeKitBucket]
    narrative: str
    provenance: dict


def build_post_discharge_digest(
    stay_id: int,
    llm: LLMBackend | None,
    kit_name: str = DEFAULT_KIT,
    db_path: Path = DEFAULT_DB_PATH,
    n_days: int = N_DIGEST_DAYS,
    seed: int = 0,
) -> PostDischargeDigest:
    """Digest the first ``n_days`` real days of one stay's simulated home stream."""
    kit = HOME_KITS[kit_name]
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        observations, provenance = build_stream(conn, stay_id, kit, seed=seed)
    finally:
        conn.close()
    if not observations:
        raise ValueError(f"no home-kit observations produced for stay_id={stay_id}")

    # Bucket by real elapsed day from the first observation.
    start = observations[0].effective_time
    code_to_channel = {
        code: channel
        for channel, code in ((c, _channel_code(c)) for c in kit.channels)
        if code is not None
    }

    buckets: list[DailyHomeKitBucket] = []
    for day in range(1, n_days + 1):
        lo = start + timedelta(days=day - 1)
        hi = start + timedelta(days=day)
        in_day = [o for o in observations if lo <= o.effective_time < hi]
        by_channel: dict[str, list[float]] = {}
        for obs in in_day:
            channel = code_to_channel.get(obs.code)
            if channel is not None:
                by_channel.setdefault(channel, []).append(obs.value)
        buckets.append(
            DailyHomeKitBucket(
                day_index=day,
                means={c: float(np.mean(v)) for c, v in by_channel.items()},
                n_samples=len(in_day),
            )
        )

    fact_lines = []
    for b in buckets:
        if not b.means:
            fact_lines.append(f"- Day {b.day_index}: no readings (device not worn)")
            continue
        parts = ", ".join(
            f"{c} {b.means[c]:.1f} {_UNITS.get(c, '')}".strip()
            for c in kit.channels
            if c in b.means
        )
        fact_lines.append(f"- Day {b.day_index}: {parts} ({b.n_samples:,} samples)")

    user = (
        f"Patient: {provenance['subject_ref']} (real MIMIC ICU stay {stay_id}, "
        f"physiology recorded; home device layer simulated)\n"
        f"Assumed home kit '{kit.name}': {', '.join(kit.channels)}\n"
        f"Channels with no home sensor, therefore absent: "
        f"{', '.join(NO_HOME_SENSOR)}\n"
        f"HRV (RMSSD): not computable from hourly-charted HR -- omitted, do not infer\n"
        f"Non-wear hours in the record: {provenance['nonwear_hours']}\n" + "\n".join(fact_lines)
    )
    narrative = generate_narrative(llm, DIGEST_ANTI_FABRICATION_SYSTEM, user)

    return PostDischargeDigest(
        stay_id=stay_id,
        subject_ref=provenance["subject_ref"],
        kit=kit.name,
        daily_buckets=buckets,
        narrative=narrative,
        provenance={**provenance, "watermark": SYNTHETIC_WATERMARK},
    )


def _channel_code(channel: str) -> str | None:
    from services.contracts.observation import CHANNELS

    ch = CHANNELS.get(channel)
    return ch.code if ch else None
