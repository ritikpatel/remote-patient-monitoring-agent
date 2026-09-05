"""Per-event-family arrival-rate models, fitted from the warehouse.

PROJECT_PLAN.md section 8, item 2 (R5): "model each event family's arrival process
separately -- bursty for orders and transfers, near-stationary with a 4-hourly comb
for monitoring. One global rate is wrong in both directions."

This is EDA section 6 (notebooks/01_capstone_eda.ipynb cells 29-33), turned into a
reusable, sampling model instead of a one-off figure. Two axes are fitted per family,
straight from the warehouse (not re-read from CSV):

  - admission_profile(h): events/patient/hour at h hours since ICU admission, capturing
    the admission burst and discharge taper (E12, E13). Pooled across patients, whose
    admission clock-time differs, this curve is already phase-averaged over time-of-day
    -- see diurnal_multiplier below.
  - diurnal_multiplier(hour_of_day): a 24-length multiplicative weight, mean 1.0,
    capturing the 4-hourly comb (E16) -- labs peak ~05:00, meds ~08:00, ICU monitoring
    on a visible 4-hourly comb, orders ~10:00.

The combined intensity for a specific, real stay replayed at a specific hour is
modelled as separable: rate(h, clock_hour) = admission_profile(h) * diurnal_multiplier(clock_hour).
This is a standard seasonal-decomposition simplification -- it lets a replay of one
real ICU stay (anchored to that stay's real, date-shifted-but-time-of-day-preserving
`intime`) show both effects at once, which pooled EDA figures cannot demonstrate on
their own.

Fitted parameters are cached to simulators/arrival_models.json (small, derived, safe
to commit -- no patient-level rows, only cohort-aggregate rates). Re-fit with:

    python simulators/arrival_models.py [--db PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
DEFAULT_OUT_PATH = REPO_ROOT / "simulators" / "arrival_models.json"
FIGURE_PATH = REPO_ROOT / "simulators" / "arrival_model_fit.png"

# group -> (table, timestamp column). Identical pooling to EDA section 6 cell 29,
# read from the warehouse instead of raw CSVs.
EVENT_SOURCES: list[tuple[str, str, str]] = [
    ("ICU monitoring", "mimiciv_icu.chartevents", "charttime"),
    ("ICU monitoring", "mimiciv_icu.datetimeevents", "charttime"),
    ("Labs", "mimiciv_hosp.labevents", "charttime"),
    ("Medications", "mimiciv_hosp.emar", "charttime"),
    ("Medications", "mimiciv_hosp.prescriptions", "starttime"),
    ("Provider orders", "mimiciv_hosp.poe", "ordertime"),
    ("ICU interventions", "mimiciv_icu.inputevents", "starttime"),
    ("ICU interventions", "mimiciv_icu.outputevents", "charttime"),
    ("ICU interventions", "mimiciv_icu.procedureevents", "starttime"),
    ("Microbiology", "mimiciv_hosp.microbiologyevents", "charttime"),
    ("Transfers", "mimiciv_hosp.transfers", "intime"),
]
GROUPS = [
    "ICU monitoring",
    "Labs",
    "Provider orders",
    "Medications",
    "ICU interventions",
    "Microbiology",
    "Transfers",
]

# Families that keep a consistent per-stay charting phase (E1: HR gaps cluster at
# 55-65 min, i.e. one nurse charts a given patient on a fairly fixed hourly rhythm)
# rather than firing at a uniformly random moment within the hour.
PHASE_LOCKED_FAMILIES = {"ICU monitoring"}

MAX_H = 168  # 7 days -- EDA's own caveat: the at-risk denominator is too thin past this
MIN_AT_RISK = 10  # below this, treat the rate as unreliable and fall back to the mean
SMOOTHING_WINDOW = 3  # hours, centred moving average on the raw rate curve


@dataclass
class FamilyArrivalModel:
    family: str
    admission_profile: list[float]  # length MAX_H, events/patient/hour at hour h
    stationary_rate: float  # asymptotic rate used for h >= MAX_H
    diurnal_multiplier: list[float]  # length 24, mean 1.0
    phase_locked: bool

    def rate(self, hours_since_admit: int, clock_hour: int) -> float:
        base = (
            self.admission_profile[hours_since_admit]
            if 0 <= hours_since_admit < MAX_H
            else self.stationary_rate
        )
        return max(base * self.diurnal_multiplier[clock_hour % 24], 0.0)

    def sample_count(
        self, hours_since_admit: int, clock_hour: int, rng: np.random.Generator
    ) -> int:
        return int(rng.poisson(self.rate(hours_since_admit, clock_hour)))

    def sample_offsets_seconds(self, n: int, stay_id: int, rng: np.random.Generator) -> np.ndarray:
        """Where within the 3600s hour each of n events falls. Phase-locked families
        get a fixed-per-stay phase (seeded by stay_id, so it's reproducible) with a
        couple of minutes of jitter; everything else is uniform (bursty, no rhythm).
        """
        if n == 0:
            return np.array([])
        if self.phase_locked:
            phase = np.random.default_rng(stay_id).uniform(0, 3600)
            jitter = rng.normal(0, 120, size=n)  # +/- ~2 min
            return np.clip(phase + jitter, 0, 3599)
        return rng.uniform(0, 3600, size=n)


@dataclass
class ArrivalModelSet:
    models: dict[str, FamilyArrivalModel] = field(default_factory=dict)

    def __getitem__(self, family: str) -> FamilyArrivalModel:
        return self.models[family]

    def to_json(self, path: Path) -> None:
        payload = {
            fam: {
                "admission_profile": m.admission_profile,
                "stationary_rate": m.stationary_rate,
                "diurnal_multiplier": m.diurnal_multiplier,
                "phase_locked": m.phase_locked,
            }
            for fam, m in self.models.items()
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> ArrivalModelSet:
        payload = json.loads(path.read_text())
        return cls(
            {
                fam: FamilyArrivalModel(
                    family=fam,
                    admission_profile=v["admission_profile"],
                    stationary_rate=v["stationary_rate"],
                    diurnal_multiplier=v["diurnal_multiplier"],
                    phase_locked=v["phase_locked"],
                )
                for fam, v in payload.items()
            }
        )


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def _load_events(conn: duckdb.DuckDBPyConnection) -> list[tuple]:
    frames: list[tuple] = []
    for group, table, tcol in EVENT_SOURCES:
        rows = conn.execute(
            f"SELECT hadm_id, {tcol} AS event_time FROM {table} "
            f"WHERE hadm_id IS NOT NULL AND {tcol} IS NOT NULL"
        ).fetchall()
        frames.extend((group, hadm_id, event_time) for hadm_id, event_time in rows)
    return frames


def fit(conn: duckdb.DuckDBPyConnection) -> ArrivalModelSet:
    import pandas as pd

    events = pd.DataFrame(_load_events(conn), columns=["group", "hadm_id", "event_time"])
    admissions = conn.execute(
        "SELECT hadm_id, admittime, dischtime FROM mimiciv_hosp.admissions "
        "WHERE admittime IS NOT NULL AND dischtime IS NOT NULL"
    ).fetchdf()

    ev = events.merge(admissions, on="hadm_id", how="inner")
    ev["hours_since_admit"] = (ev.event_time - ev.admittime).dt.total_seconds() / 3600
    ev["stay_hours"] = (ev.dischtime - ev.admittime).dt.total_seconds() / 3600
    iw = ev[(ev.hours_since_admit >= 0) & (ev.hours_since_admit <= ev.stay_hours)].copy()
    iw["hour_bin"] = iw.hours_since_admit.astype(int)
    iw["hour_of_day"] = iw.event_time.dt.hour

    bins = np.arange(0, MAX_H + 1, 1)
    stay_hours_all = (admissions.dischtime - admissions.admittime).dt.total_seconds() / 3600
    at_risk = np.array([int((stay_hours_all > h).sum()) for h in bins[:-1]])
    at_risk_safe = np.maximum(at_risk, 1)

    models: dict[str, FamilyArrivalModel] = {}
    for group in GROUPS:
        sub = iw[(iw.group == group) & (iw.hours_since_admit < MAX_H)]
        counts, _ = np.histogram(sub.hours_since_admit, bins=bins)
        raw_rate = counts / at_risk_safe
        raw_rate = np.where(
            at_risk >= MIN_AT_RISK, raw_rate, raw_rate[at_risk >= MIN_AT_RISK].mean()
        )
        smoothed = _smooth(raw_rate, SMOOTHING_WINDOW)

        stable_mask = (bins[:-1] >= 24) & (bins[:-1] < MAX_H) & (at_risk >= MIN_AT_RISK)
        stationary_rate = (
            float(raw_rate[stable_mask].mean()) if stable_mask.any() else float(raw_rate.mean())
        )

        hod_counts = (
            iw[iw.group == group].groupby("hour_of_day").size().reindex(range(24), fill_value=0)
        )
        total = hod_counts.sum()
        diurnal = (hod_counts / total * 24).to_numpy() if total > 0 else np.ones(24)

        models[group] = FamilyArrivalModel(
            family=group,
            admission_profile=smoothed.tolist(),
            stationary_rate=stationary_rate,
            diurnal_multiplier=diurnal.tolist(),
            phase_locked=group in PHASE_LOCKED_FAMILIES,
        )
    return ArrivalModelSet(models)


def plot_fit(model_set: ArrivalModelSet, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for fam, m in model_set.models.items():
        ax.plot(m.admission_profile, lw=1.2, label=fam)
    ax.set_xlabel("hours since ICU admission")
    ax.set_ylabel("fitted rate (events/patient/hour)")
    ax.set_title("Fitted admission-burst profile per family")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for fam, m in model_set.models.items():
        ax.plot(range(24), m.diurnal_multiplier, lw=1.2, marker="o", ms=3, label=fam)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("diurnal multiplier (1.0 = uniform)")
    ax.set_title("Fitted 4-hourly comb per family")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db), read_only=True)
    model_set = fit(conn)
    conn.close()

    model_set.to_json(args.out)
    print(f"Wrote {len(model_set.models)} family models to {args.out.relative_to(REPO_ROOT)}")
    for fam, m in model_set.models.items():
        peak_h = int(np.argmax(m.admission_profile))
        peak_hod = int(np.argmax(m.diurnal_multiplier))
        print(
            f"  {fam:<18} peak admission-hour rate {m.admission_profile[peak_h]:.1f} "
            f"at h={peak_h}, stationary {m.stationary_rate:.2f}, "
            f"diurnal peak {peak_hod:02d}:00 (x{m.diurnal_multiplier[peak_hod]:.2f})"
        )

    if not args.no_plot:
        plot_fit(model_set, FIGURE_PATH)
        print(f"Wrote {FIGURE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
