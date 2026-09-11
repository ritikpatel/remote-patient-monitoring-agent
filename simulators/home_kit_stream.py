"""Stream a real deteriorating MIMIC patient as if a home monitoring kit were
watching them.

**This replaces `wearable_replay.py` + `morphing.py`, and inverts what is synthetic.**

Those modules took a healthy volunteer's Empatica E4 recording and *synthesised a
deterioration onto it* -- a ramp in an HR/SpO2 NEWS2 proxy, HRV artificially
suppressed, SpO2 fabricated from nothing because the E4 cannot measure it. The
physiology was invented and the sensor was real. That was the wrong way round for
this project's purpose, and it rested on a dataset (PhysioNet wearable-device
v1.0.1, median age ~21, zero deterioration events -- old finding E10) that could
never contain the outcome the platform exists to predict.

This module goes the other way. The **physiology is real**: a MIMIC ICU patient who
genuinely deteriorated, with their genuinely recorded vitals, at the hours they were
genuinely recorded. Only the **sensor layer** is synthetic -- what a home kit would
and would not see of that same patient. Nothing invents a deterioration; the
deterioration is in the data, and the simulation is of the instrument.

## What the sensor model does, and which parts are honest

1. **Channel availability -- a physical fact, not a modelling choice.** A home kit
   cannot measure core temperature (a wrist device reads *skin* temperature, a
   different quantity -- see `services/contracts/observation.py`'s `temp_skin`
   note), cannot measure GCS (that needs a person performing a neurological exam),
   and cannot measure FiO2 (a ventilator setting; a patient at home is not
   ventilated). `HOME_KITS` below states which channels each assumed kit has. This
   is the one part of the model that is not an approximation -- it is a statement
   about what sensors exist.

2. **Per-channel cadence.** MIMIC is charted hourly for everything (E1). A home kit
   is nothing like uniform: a watch samples HR continuously, a CGM every few
   minutes, and a blood-pressure cuff twice a day if the patient remembers. Emitting
   home SBP hourly would make the stream far richer than any real deployment.

3. **Measurement noise.** Wrist PPG heart rate is not ECG heart rate; a cuff is not
   an arterial line. Each channel gets Gaussian noise at a magnitude drawn from
   device-validation literature. **These magnitudes are assumed, not measured here**
   -- they are order-of-magnitude realistic and are not a validated device model.

4. **Non-wear gaps.** Watches come off to charge. Modelled as contiguous blocks, not
   independent per-sample dropout, because that is how non-wear actually occurs and
   it is much harsher on a trend feature.

5. **Carried-forward values are never emitted as fresh readings.** `capstone.hourly_grid`
   forward-fills (R2) and flags it with `{channel}_was_imputed`. An imputed hour means
   *no measurement was taken*, so emitting one would manufacture an observation that
   never existed in either domain. Those hours are skipped, which is also what makes
   "the physiology is real" a defensible claim rather than a slogan.

## What remains genuinely synthetic, stated plainly

**Within-hour detail is invented.** MIMIC holds one HR per hour; a watch reports one
per minute. The intra-hour samples are interpolated between real hourly anchors with
noise added, so the minute-level texture is a plausible fabrication consistent with
the real hourly values -- it is not a recording of anything. Every Observation
carries `quality_flags=[synthetic]` and `device_id="home-kit-sim"`, and every CLI run
prints the section 17 watermark. R7: never present this as measured.

Usage:
    python simulators/home_kit_stream.py --list-candidates
    python simulators/home_kit_stream.py --stay-id 36558922 --kit full_home --sink console
    python simulators/home_kit_stream.py --stay-id 36558922 --kit watch_only \\
        --sink http --url http://localhost:8000/observations
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.contracts.observation import (  # noqa: E402
    Observation,
    ObservationSource,
    QualityFlag,
)

from simulators.sinks import DEFAULT_INGEST_API_KEY, Sink, make_sink  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"

SYNTHETIC_WATERMARK = (
    "SYNTHETIC SENSOR LAYER -- the patient, their deterioration and their hourly "
    "vitals are real MIMIC-IV records; the device cadence, measurement noise, "
    "non-wear gaps and all within-hour detail are simulated. No home kit recorded "
    "this. See PROJECT_PLAN.md section 17."
)

# Channels with no home sensor at any price. Not a restriction imposed on the model
# -- a statement about which instruments exist outside a hospital.
NO_HOME_SENSOR = ("temp_c", "gcs_total", "fio2")


@dataclass(frozen=True)
class ChannelSensor:
    """One channel's home-device model.

    ``interval_min`` is the sampling interval in minutes; ``noise_sd`` the Gaussian
    measurement-error SD in the channel's own unit; ``device`` names the instrument
    so a reader can see what is being assumed rather than inferring it from numbers.
    """

    interval_min: float
    noise_sd: float
    device: str


# Assumed from device-validation literature, order-of-magnitude realistic, NOT a
# validated device model (see the docstring's point 3). Each entry is a claim about
# a specific instrument, which is why the instrument is named alongside it.
SENSORS: dict[str, ChannelSensor] = {
    # Wrist PPG heart rate: MAE around 5 bpm at rest, considerably worse in motion.
    "hr": ChannelSensor(interval_min=1.0, noise_sd=5.0, device="wrist PPG"),
    # Respiratory rate derived from the PPG waveform -- a derivation, not a
    # measurement, and the noisiest thing in the kit.
    "rr": ChannelSensor(interval_min=5.0, noise_sd=3.0, device="wrist PPG (derived)"),
    # Reflectance pulse oximetry at the wrist: +/-3-4% against arterial blood gas.
    "spo2": ChannelSensor(interval_min=5.0, noise_sd=3.0, device="wrist reflectance SpO2"),
    # Home oscillometric cuff, twice daily. The cadence matters more than the noise:
    # two readings a day cannot support a 4-hour trend feature at all.
    "sbp": ChannelSensor(interval_min=720.0, noise_sd=8.0, device="home BP cuff"),
    "map": ChannelSensor(interval_min=720.0, noise_sd=8.0, device="home BP cuff (derived)"),
    # Continuous glucose monitor: MARD in the 9-10% range.
    "glucose": ChannelSensor(interval_min=5.0, noise_sd=10.0, device="CGM"),
}


@dataclass(frozen=True)
class HomeKit:
    """A named sensor suite. The point of making this configurable is that "what a
    home kit can see" is a procurement decision, not a property of the model -- a
    richer kit loses less signal, and `ml/evaluation/channel_dropout.py` measures
    exactly how much for each of these."""

    name: str
    channels: tuple[str, ...]
    description: str


HOME_KITS: dict[str, HomeKit] = {
    "watch_only": HomeKit(
        name="watch_only",
        channels=("hr", "spo2"),
        description="A consumer smartwatch and nothing else.",
    ),
    "watch_plus_cuff": HomeKit(
        name="watch_plus_cuff",
        channels=("hr", "rr", "spo2", "sbp", "map"),
        description="Smartwatch plus a home blood-pressure cuff.",
    ),
    "full_home": HomeKit(
        name="full_home",
        channels=("hr", "rr", "spo2", "sbp", "map", "glucose"),
        description=(
            "Smartwatch, BP cuff and CGM -- every channel a home setup can obtain. "
            "The remaining gap against the ICU feature set (core temperature, GCS, "
            "FiO2, arterial line) has no home instrument."
        ),
    ),
}
DEFAULT_KIT = "full_home"

# Fraction of hours the device is not worn, and how long a typical non-wear block
# lasts. Charging overnight is the common case, which is why this is block-structured.
DEFAULT_NONWEAR_FRACTION = 0.08
DEFAULT_NONWEAR_BLOCK_H = 4

# The longest gap between two real hourly anchors that may be interpolated across.
# Beyond this the patient genuinely went unmeasured, and a smooth ramp over the gap
# is the one fabrication capable of flipping the sign of a trend feature -- so the
# stream carries a hole instead, which is also what a real device gap looks like.
MAX_BRIDGE_H = 2.0


@dataclass
class StayVitals:
    """One MIMIC stay's real hourly vitals, with imputed hours already removed
    per channel. ``observed[channel]`` maps ICU hour -> genuinely recorded value."""

    stay_id: int
    observed: dict[str, dict[int, float]] = field(default_factory=dict)
    max_hour: int = 0


def deterioration_candidates(
    conn: duckdb.DuckDBPyConnection, limit: int = 20
) -> list[tuple[int, int, str, int]]:
    """Stays that really did deteriorate, most-escalating first.

    A home-kit simulation is only interesting over a patient whose physiology
    actually turned, and picking them by the shared escalation predicate rather than
    by hand keeps the demo honest -- these are stays the alerting engine independently
    considers escalating, not stays chosen because they looked good.
    """
    return conn.execute(
        """
        SELECT n.stay_id,
               COUNT(*) AS escalating_hours,
               ANY_VALUE(d.dx_title) AS dx_title,
               MAX(n.hour) AS last_hour
        FROM capstone.news2 n
        JOIN capstone.disease_context d USING (stay_id)
        WHERE n.tier_icu = 'high' OR n.max_component_nongcs >= 3
        GROUP BY n.stay_id
        ORDER BY escalating_hours DESC
        LIMIT ?
        """,
        [limit],
    ).fetchall()


def load_stay_vitals(conn: duckdb.DuckDBPyConnection, stay_id: int, kit: HomeKit) -> StayVitals:
    """Real recorded values only -- carried-forward hours are dropped.

    `capstone.hourly_grid` forward-fills every channel and marks it with
    `{channel}_was_imputed` (R2/E3). An imputed hour means no measurement was taken
    that hour, so streaming it would fabricate an observation that existed in neither
    the hospital nor the home. Dropping them is what makes "the physiology is real"
    checkable rather than rhetorical.
    """
    cols = ["hour"]
    for ch in kit.channels:
        cols += [ch, f"{ch}_was_imputed"]
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM capstone.hourly_grid WHERE stay_id = ? ORDER BY hour",
        [stay_id],
    ).fetchdf()
    if rows.empty:
        raise ValueError(f"no hourly_grid rows for stay_id={stay_id}")

    out = StayVitals(stay_id=stay_id, max_hour=int(rows.hour.max()))
    for ch in kit.channels:
        observed: dict[int, float] = {}
        for r in rows.itertuples():
            value = getattr(r, ch)
            imputed = getattr(r, f"{ch}_was_imputed")
            if value is None or np.isnan(value) or bool(imputed):
                continue
            observed[int(r.hour)] = float(value)  # type: ignore[arg-type]
        out.observed[ch] = observed
    return out


def _nonwear_hours(
    max_hour: int, fraction: float, block_h: int, rng: np.random.Generator
) -> set[int]:
    """Contiguous non-wear blocks, not independent per-sample dropout.

    The distinction is not cosmetic: 8% of samples missing at random barely dents a
    24-hour rolling mean, while two 4-hour blocks in the same window removes a third
    of it. Block structure is both what really happens and the harder test.
    """
    if fraction <= 0 or max_hour <= 0:
        return set()
    n_blocks = max(1, int(round(max_hour * fraction / block_h)))
    out: set[int] = set()
    for start in rng.integers(0, max(1, max_hour), size=n_blocks):
        out.update(range(int(start), min(int(start) + block_h, max_hour + 1)))
    return out


def synthesize_channel(
    channel: str,
    observed: dict[int, float],
    icu_intime: datetime,
    sensor: ChannelSensor,
    nonwear: set[int],
    rng: np.random.Generator,
) -> list[tuple[datetime, float]]:
    """One channel's simulated home stream: (timestamp, value) at device cadence.

    The sample timeline is built **globally** at the device's own cadence, then each
    sample time is interpolated from the real hourly anchors around it. Doing it in
    that order is the whole point, and getting it wrong the other way round was a
    real bug here: iterating anchor-pairs and emitting at the device cadence *within
    each pair* emits at least one sample per anchor regardless of cadence, so a
    twice-daily BP cuff produced 218 readings over 315 hours instead of ~26. That
    made the cuff look eight times richer than any real one and silently destroyed
    the thing this cadence model exists to represent -- that two readings a day
    cannot support a 4-hour trend feature at all.

    Two rules keep the interpolation honest:

    * **A gap longer than ``MAX_BRIDGE_H`` between real anchors is not bridged.** The
      patient genuinely went unmeasured there, and drawing a smooth ramp across it is
      the one fabrication that could flip a trend feature's sign.
    * **Non-wear hours emit nothing**, even where anchors exist either side.
    """
    if not observed:
        return []
    hours = np.array(sorted(observed), dtype=float)
    values = np.array([observed[int(h)] for h in hours], dtype=float)

    # Global cadence timeline, in hours since ICU admission.
    step_h = sensor.interval_min / 60.0
    n_steps = int(hours[-1] / step_h) + 1
    sample_hours = np.arange(n_steps, dtype=float) * step_h
    sample_hours = sample_hours[(sample_hours >= hours[0]) & (sample_hours <= hours[-1])]
    if sample_hours.size == 0:
        return []

    # Interpolate onto the real anchors, then blank any sample whose surrounding
    # anchors are too far apart to interpolate between honestly.
    interpolated = np.interp(sample_hours, hours, values)
    right = np.searchsorted(hours, sample_hours, side="left").clip(1, len(hours) - 1)
    anchor_gap = hours[right] - hours[right - 1]
    keep = anchor_gap <= MAX_BRIDGE_H

    noise = rng.normal(0, sensor.noise_sd, size=sample_hours.size)
    samples: list[tuple[datetime, float]] = []
    for h, value, noise_i, keep_i in zip(sample_hours, interpolated, noise, keep, strict=True):
        if not keep_i or int(h) in nonwear:
            continue
        samples.append((icu_intime + timedelta(hours=float(h)), float(value + noise_i)))
    return samples


def to_observations(
    samples_by_channel: dict[str, list[tuple[datetime, float]]], subject_ref: str
) -> list[Observation]:
    """Every Observation is flagged synthetic and carries `home-kit-sim` as its
    device. `source=wearable` is correct -- this is the post-discharge arm's transport
    path -- and `Subject/<id>` is the right reference shape: a patient at home is not
    an ICU stay, which is exactly why that reference form exists in the contract."""
    out: list[Observation] = []
    for channel, samples in samples_by_channel.items():
        for effective_time, value in samples:
            out.append(
                Observation.for_channel(
                    channel=channel,
                    patient_ref=subject_ref,
                    device_id="home-kit-sim",
                    source=ObservationSource.wearable,
                    value=round(value, 2),
                    effective_time=effective_time,
                    quality_flags=[QualityFlag.synthetic],
                )
            )
    out.sort(key=lambda o: o.effective_time)
    return out


def build_stream(
    conn: duckdb.DuckDBPyConnection,
    stay_id: int,
    kit: HomeKit,
    seed: int = 0,
    nonwear_fraction: float = DEFAULT_NONWEAR_FRACTION,
    nonwear_block_h: int = DEFAULT_NONWEAR_BLOCK_H,
) -> tuple[list[Observation], dict]:
    """The whole pipeline for one stay. Returns (observations, provenance)."""
    rng = np.random.default_rng(seed)
    vitals = load_stay_vitals(conn, stay_id, kit)
    intime_row = conn.execute(
        "SELECT icu_intime FROM mimiciv_derived.icustay_detail WHERE stay_id = ?", [stay_id]
    ).fetchone()
    icu_intime = (
        intime_row[0].replace(tzinfo=UTC)
        if intime_row and intime_row[0] is not None
        else datetime.now(UTC)
    )
    nonwear = _nonwear_hours(vitals.max_hour, nonwear_fraction, nonwear_block_h, rng)

    samples_by_channel = {
        ch: synthesize_channel(ch, vitals.observed[ch], icu_intime, SENSORS[ch], nonwear, rng)
        for ch in kit.channels
    }
    subject_ref = f"Subject/HOME-{stay_id}"
    observations = to_observations(samples_by_channel, subject_ref)

    provenance = {
        "stay_id": stay_id,
        "subject_ref": subject_ref,
        "kit": kit.name,
        "kit_channels": list(kit.channels),
        "channels_with_no_home_sensor": list(NO_HOME_SENSOR),
        "real_observed_hours_per_channel": {ch: len(vitals.observed[ch]) for ch in kit.channels},
        "simulated_samples_per_channel": {ch: len(s) for ch, s in samples_by_channel.items()},
        "nonwear_hours": len(nonwear),
        "icu_hours_in_record": vitals.max_hour,
        "watermark": SYNTHETIC_WATERMARK,
    }
    return observations, provenance


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--stay-id", type=int, help="a MIMIC stay that really deteriorated")
    ap.add_argument("--kit", choices=sorted(HOME_KITS), default=DEFAULT_KIT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nonwear-fraction", type=float, default=DEFAULT_NONWEAR_FRACTION)
    ap.add_argument("--limit", type=int, default=0, help="cap observations emitted (0 = all)")
    ap.add_argument("--list-candidates", action="store_true")
    ap.add_argument("--sink", choices=["console", "jsonl", "http"], default="console")
    ap.add_argument("--out", type=Path, default=Path("home_kit_stream.jsonl"))
    ap.add_argument("--gateway-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default=DEFAULT_INGEST_API_KEY)
    args = ap.parse_args()

    if not args.db.exists():
        print(f"No warehouse at {args.db} -- run warehouse/build_duckdb.py first")
        return 1
    conn = duckdb.connect(str(args.db), read_only=True)

    if args.list_candidates or args.stay_id is None:
        print("Stays that really deteriorated (escalating hours, descending):\n")
        print(f"{'stay_id':>10}  {'esc.hrs':>7}  {'hours':>5}  diagnosis")
        for stay_id, esc, dx, last_hour in deterioration_candidates(conn):
            print(f"{stay_id:>10}  {esc:>7}  {last_hour:>5}  {str(dx)[:58]}")
        if args.stay_id is None:
            print("\nPick one with --stay-id.")
            conn.close()
            return 0

    kit = HOME_KITS[args.kit]
    observations, provenance = build_stream(
        conn, args.stay_id, kit, seed=args.seed, nonwear_fraction=args.nonwear_fraction
    )
    conn.close()

    print(f"\n*** {SYNTHETIC_WATERMARK}\n")
    print(f"Kit '{kit.name}': {kit.description}")
    print(f"  channels        : {', '.join(kit.channels)}")
    print(f"  no home sensor  : {', '.join(NO_HOME_SENSOR)} (+ arterial line)")
    print(f"  real hours used : {provenance['real_observed_hours_per_channel']}")
    print(f"  samples emitted : {provenance['simulated_samples_per_channel']}")
    print(
        f"  non-wear hours  : {provenance['nonwear_hours']} of {provenance['icu_hours_in_record']}"
    )
    print(f"  patient_ref     : {provenance['subject_ref']}\n")

    if args.limit:
        observations = observations[: args.limit]
    sink: Sink = make_sink(args.sink, args.out, gateway_url=args.gateway_url, api_key=args.api_key)
    for obs in observations:
        sink.emit(obs)
    sink.close()
    print(f"\nEmitted {len(observations):,} observations via '{args.sink}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
