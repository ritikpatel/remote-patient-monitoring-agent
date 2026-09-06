"""Post-discharge weekly digest from wearable telemetry (PROJECT_PLAN.md
section 12).

**R7, stated as plainly as the module docstring it comes from
(simulators/morphing.py): the wearable cohort is healthy volunteers with no
ICU link (E10), so a genuine week of post-discharge deterioration cannot come
from real data.** This digest is built from `simulators/morphing.py`'s real
morphing transforms conditioned on a synthetic deterioration trajectory,
compressed into that session's actual recorded duration (tens of minutes,
not seven days) and divided evenly into 7 buckets labelled "day 1".."day 7"
to stand in for a week -- an explicit demo compression, not a claim that a
week was actually recorded. Every value in this report is watermarked
accordingly, and the report text says so before it says anything else.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.common.testing import load_module  # noqa: E402
from simulators.morphing import MorphConfig, morph_session  # noqa: E402

from reports.narrative import (  # noqa: E402
    ANTI_FABRICATION_INSTRUCTION,
    LLMBackend,
    generate_narrative,
)

_windowing = load_module(
    REPO_ROOT / "services" / "stream-processor" / "windowing.py",
    "reports_post_discharge_windowing",
)
hrv_rmssd = _windowing.hrv_rmssd

N_DEMO_DAYS = 7

DIGEST_ANTI_FABRICATION_SYSTEM = (
    "You write a post-discharge weekly wearable-telemetry digest for a remote "
    "monitoring nurse from the structured per-day facts given. Every value "
    "you were given is from morphed (synthetically conditioned) wearable "
    "data compressed to stand in for a week -- say so explicitly in your "
    "response, in your first sentence. " + ANTI_FABRICATION_INSTRUCTION
)


@dataclass
class DailyWearableBucket:
    day_index: int  # 1-7, standing in for a calendar day
    mean_hr: float
    mean_spo2: float
    hrv_rmssd_ms: float


@dataclass
class PostDischargeDigest:
    participant: str
    activity: str
    start_score: float
    end_score: float
    daily_buckets: list[DailyWearableBucket]
    narrative: str


def build_post_discharge_digest(
    activity: str,
    participant: str,
    llm: LLMBackend | None,
    start_score: float = 0.0,
    end_score: float = 6.0,
    duration_s: float = 600.0,
) -> PostDischargeDigest:
    config = MorphConfig(start_score=start_score, end_score=end_score)
    channels = morph_session(activity, participant, config, duration_s=duration_s)

    hr_times = channels["hr"].times if "hr" in channels else np.array([])
    hr_values = channels["hr"].values if "hr" in channels else np.array([])
    spo2_times = channels["spo2"].times
    spo2_values = channels["spo2"].values
    ibi_times = channels["ibi"].times if "ibi" in channels else np.array([])
    ibi_values = channels["ibi"].values if "ibi" in channels else np.array([])

    session_start = min((t[0] for t in (hr_times, spo2_times, ibi_times) if len(t)), default=None)
    session_end = max((t[-1] for t in (hr_times, spo2_times, ibi_times) if len(t)), default=None)
    if session_start is None or session_end is None:
        raise ValueError(f"no channels produced for {activity}/{participant}")
    span = (session_end - session_start) / np.timedelta64(1, "s")

    buckets = []
    for day in range(1, N_DEMO_DAYS + 1):
        bucket_start = session_start + np.timedelta64(
            int((day - 1) / N_DEMO_DAYS * span * 1e9), "ns"
        )
        bucket_end = session_start + np.timedelta64(int(day / N_DEMO_DAYS * span * 1e9), "ns")

        hr_mask = (hr_times >= bucket_start) & (hr_times < bucket_end)
        spo2_mask = (spo2_times >= bucket_start) & (spo2_times < bucket_end)
        ibi_mask = (ibi_times >= bucket_start) & (ibi_times < bucket_end)

        buckets.append(
            DailyWearableBucket(
                day_index=day,
                mean_hr=float(np.mean(hr_values[hr_mask])) if hr_mask.any() else float("nan"),
                mean_spo2=(
                    float(np.mean(spo2_values[spo2_mask])) if spo2_mask.any() else float("nan")
                ),
                hrv_rmssd_ms=hrv_rmssd(list(ibi_values[ibi_mask])) if ibi_mask.any() else 0.0,
            )
        )

    facts = "\n".join(
        f"- Day {b.day_index}: mean HR {b.mean_hr:.1f} bpm, mean SpO2 {b.mean_spo2:.1f}%, "
        f"HRV (RMSSD) {b.hrv_rmssd_ms:.1f} ms"
        for b in buckets
    )
    user = (
        f"Participant: {participant} (morphed from a real {activity.lower()}-session "
        f"wearable recording, {span:.0f}s compressed to stand in for {N_DEMO_DAYS} days)\n"
        f"Target deterioration trajectory: NEWS2 HR+SpO2 subscore {start_score} -> {end_score}\n"
        f"{facts}"
    )
    narrative = generate_narrative(llm, DIGEST_ANTI_FABRICATION_SYSTEM, user)

    return PostDischargeDigest(
        participant=participant,
        activity=activity,
        start_score=start_score,
        end_score=end_score,
        daily_buckets=buckets,
        narrative=narrative,
    )
