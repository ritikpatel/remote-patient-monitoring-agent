"""Generate a complete, internally-consistent patient event at a target severity.

Every channel the risk model trains on is produced, because a partial event
cannot exercise the real scoring path (see `simulators/real_event_replay.py` --
this module is its interactive sibling: same contract, same channel set, but
composed to order instead of replayed from a recording).

Severity is expressed as a target NEWS2 aggregate. The generator inverts
`warehouse/news2.py`'s own scoring thresholds rather than restating them, so a
requested severity and the score the pipeline computes cannot drift apart -- the
same discipline `simulators/morphing.py` uses for its two channels, extended to
all seven NEWS2 parameters.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from warehouse.news2 import (  # noqa: E402
    fio2_score,
    gcs_score,
    hr_score,
    rr_score,
    sbp_score,
    spo2_score,
    temp_score,
)

# (subscore -> representative value) per NEWS2 parameter. Only the deteriorating
# direction is modelled for rate-type channels; each value is asserted below to
# actually score what it claims, so these cannot silently drift from news2.py.
BREAKPOINTS: dict[str, dict[int, float]] = {
    "hr": {0: 75.0, 1: 100.0, 2: 120.0, 3: 145.0},
    "rr": {0: 16.0, 1: 10.0, 2: 22.0, 3: 26.0},
    "spo2": {0: 98.0, 1: 95.0, 2: 93.0, 3: 89.0},
    "sbp": {0: 120.0, 1: 105.0, 2: 95.0, 3: 88.0},
    "temp_c": {0: 37.0, 1: 38.5, 2: 39.5, 3: 34.5},
    "gcs_total": {0: 15.0, 3: 12.0},
    "fio2": {0: 21.0, 2: 40.0},
}
SCORERS = {
    "hr": hr_score,
    "rr": rr_score,
    "spo2": spo2_score,
    "sbp": sbp_score,
    "temp_c": temp_score,
    "gcs_total": gcs_score,
    "fio2": fio2_score,
}
JITTER = {
    "hr": 3.0,
    "rr": 1.0,
    "spo2": 0.6,
    "sbp": 4.0,
    "temp_c": 0.15,
    "gcs_total": 0.0,
    "fio2": 0.0,
}


def _verify_breakpoints() -> None:
    for channel, table in BREAKPOINTS.items():
        for want, value in table.items():
            got = SCORERS[channel](value)
            assert got == want, f"{channel}={value} scores {got}, not {want}"


_verify_breakpoints()

MAX_SUBSCORE = {c: max(t) for c, t in BREAKPOINTS.items()}
CHANNEL_ORDER = ["hr", "rr", "spo2", "sbp", "temp_c", "gcs_total", "fio2"]


@dataclass
class GeneratedEvent:
    values: dict[str, float]
    subscores: dict[str, int]
    news2: int = 0
    max_component: int = 0
    extras: dict[str, float] = field(default_factory=dict)


def allocate_subscores(severity: float, rng: random.Random) -> dict[str, int]:
    """Spread a 0..1 severity across the seven parameters.

    Deliberately NOT uniform: real deterioration shows up in one or two systems
    before it shows up everywhere, so at low severity the budget is concentrated
    in a few channels rather than smeared thinly across all of them -- which is
    also what makes the single-red-parameter escalation limb reachable well
    before the aggregate tier.
    """
    severity = max(0.0, min(1.0, severity))
    budget = round(severity * sum(MAX_SUBSCORE.values()))
    subscores = dict.fromkeys(CHANNEL_ORDER, 0)
    channels = CHANNEL_ORDER[:]
    rng.shuffle(channels)
    for channel in channels:
        if budget <= 0:
            break
        take = min(MAX_SUBSCORE[channel], budget, rng.choice([1, 2, 3]))
        available = [s for s in BREAKPOINTS[channel] if s <= take]
        chosen = max(available) if available else 0
        subscores[channel] = chosen
        budget -= chosen
    return subscores


def generate(severity: float, seed: int | None = None) -> GeneratedEvent:
    rng = random.Random(seed)
    subscores = allocate_subscores(severity, rng)
    values: dict[str, float] = {}
    for channel, want in subscores.items():
        base = BREAKPOINTS[channel][want]
        noise = rng.gauss(0, JITTER[channel])
        value = base + noise
        # Keep the jitter from moving the value into a different subscore band --
        # otherwise the requested severity and the computed NEWS2 disagree.
        if SCORERS[channel](value) != want:
            value = base
        values[channel] = round(value, 1)

    actual = {c: SCORERS[c](v) for c, v in values.items()}
    ev = GeneratedEvent(values=values, subscores=actual)
    ev.news2 = sum(actual.values())
    ev.max_component = max(actual.values())
    # Channels the model uses that NEWS2 does not score. map tracks sbp; glucose
    # is independent and drifts up with severity (stress hyperglycaemia).
    ev.extras = {
        "map": round(values["sbp"] * 0.72 + rng.gauss(0, 2), 1),
        "glucose": round(110 + severity * 90 + rng.gauss(0, 12), 1),
    }
    return ev
