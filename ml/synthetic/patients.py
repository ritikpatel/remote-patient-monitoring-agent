"""Whole synthetic *patients*, generated class-balanced, with real trajectories.

`run_tutorial.py` generates patient-*hours*: free-standing rows with no patient
identity, which is the tutorial's snapshot framing and its stated limitation.
This module generates patients -- each with a length of stay, an hourly
trajectory across nine vitals, static attributes, an event time if it
deteriorates, and a synthetic `subject_id` that subject-grouped CV can hold out
as a unit.

**Why not a GAN over the raw sequence.** The obvious approach -- flatten each
stay to a 48h x 9-vital tensor and train a GAN on that -- is roughly 460
dimensions fitted on ~100 training stays. `report.md` already shows this
architecture memorising at 89 dimensions with 1,995 rows; at four dimensions per
sample it would do nothing but reproduce its training set, and the "new patients"
would be the old ones with noise.

So the patient is parameterised low-rank instead, and the split is deliberate:

1. **What the GAN generates** (~70 dimensions): per vital, the level, linear
   trend, within-stay variability, lag-1 autocorrelation and measurement
   frequency; plus the static attributes, the length of stay, and where in the
   stay the event falls. Conditioned on class, so the caller fixes the balance.

2. **What a calibrated process generates** (the hourly trajectory): an AR(1)
   expansion realising those parameters. Lag-1 autocorrelation in this cohort
   runs 0.41-0.69 across vitals, so AR(1) is a fair description of how these
   signals actually move. This stage is *not* learned adversarially and therefore
   cannot memorise a real patient -- it only knows the parameters handed to it.

Derived features are **recomputed, never generated**: the expanded trajectory
goes through `engineer.add_rolling_features`, the same function the real pipeline
uses, so a synthetic patient's `hr_4h_std` genuinely describes that patient's own
generated heart rate. That is the coherence failure `evaluate.temporal_coherence`
measures in the snapshot generator, removed here by construction rather than
learned.

Labels follow `labels.build_labels` exactly: rows at or after the event are
censored (R1), and `label_{h}h` marks the rows within h hours before it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.features import engineer

VITALS = engineer.CORE_VITALS
# Per vital: level, slope per hour, within-stay SD, lag-1 autocorrelation, and
# the share of hours that were carried forward rather than measured.
VITAL_PARAMS = ("level", "slope", "sd", "rho", "imputed_rate")
# Stay lengths run from 1 to 169 hours with a median of 16. Capped so one
# 169-hour outlier cannot dominate a 125-stay training set, and floored at 2 so
# a synthetic stay always has a trajectory rather than a single point.
MIN_STAY_HOURS = 2
MAX_STAY_HOURS = 72
# Per-hour but not free-varying: an arterial line, once placed, stays in.
# Encoded as the share of the stay it is present for and decoded as a step,
# which reproduces both the rate and the persistence. Generating it as an
# independent per-hour coin flip would give patients whose line flickers in
# and out hourly, which no ICU chart contains.
PERSISTENT_FLAG = "has_arterial_line"


@dataclass
class StayTable:
    """Per-stay parameters, the class label, and the columns needed to decode."""

    params: pd.DataFrame
    has_event: np.ndarray
    static_columns: list[str]

    def __len__(self) -> int:
        return len(self.params)


def _safe_autocorr(series: pd.Series) -> float:
    """Lag-1 autocorrelation, defaulting to a mild positive when undefined.

    A stay of one or two hours, or one where a vital never moves, has no
    definable autocorrelation. Returning 0 there would make the decoder emit
    white noise for exactly the stays where it has least information, so it falls
    back to the cohort's typical persistence instead.
    """
    if len(series) < 3 or series.std(ddof=0) == 0:
        return 0.5
    value = series.autocorr(1)
    return 0.5 if not np.isfinite(value) else float(np.clip(value, -0.95, 0.95))


def encode_stays(
    grid: pd.DataFrame, features: pd.DataFrame, labels_frame: pd.DataFrame
) -> StayTable:
    """Reduce each real stay to the parameter vector the generator learns.

    `grid` supplies the raw hourly vitals, `features` the static attributes, and
    `labels_frame` the event time that decides the class.
    """
    label_column = next(c for c in labels_frame.columns if c.startswith("label_"))
    at_risk = labels_frame[["stay_id", "hour", label_column, "composite_event_time"]]
    stays = at_risk["stay_id"].unique()

    # Per-stay-constant columns are the ones a patient "has" rather than
    # accumulates hour by hour. Identifiers are constant per stay too and must be
    # excluded by name: carrying `subject_id` through as a feature would hand the
    # generator a column whose only content is which real patient a row came
    # from, and hand the model a free lookup key.
    identifiers = {"stay_id", "hour", "subject_id", "hadm_id"}
    static_columns = [
        c
        for c in features.columns
        if c not in identifiers and features.groupby("stay_id")[c].nunique(dropna=False).max() == 1
    ]

    rows: list[dict[str, object]] = []
    for stay_id in stays:
        block = grid.loc[grid["stay_id"] == stay_id].sort_values("hour")
        risk_rows = at_risk.loc[at_risk["stay_id"] == stay_id]
        if block.empty or risk_rows.empty:
            continue

        record: dict[str, object] = {"stay_id": stay_id}
        length = int(np.clip(len(risk_rows), MIN_STAY_HOURS, MAX_STAY_HOURS))
        record["length"] = float(length)

        has_event = bool(risk_rows[label_column].max() > 0)
        record["has_event"] = float(has_event)
        # Where the event falls, as a fraction of the stay, so the decoder can
        # place one in a synthetic stay of any length. Censoring means the event
        # is one hour past the last at-risk row.
        record["event_fraction"] = 1.0 if has_event else 0.0

        hours = block["hour"].to_numpy(dtype=float)
        for vital in VITALS:
            series = block[vital]
            observed = series.dropna()
            if observed.empty:
                record.update(
                    {
                        f"{vital}_level": np.nan,
                        f"{vital}_slope": 0.0,
                        f"{vital}_sd": 0.0,
                        f"{vital}_rho": 0.5,
                        f"{vital}_imputed_rate": 1.0,
                    }
                )
                continue
            filled = series.ffill().bfill()
            record[f"{vital}_level"] = float(filled.mean())
            record[f"{vital}_sd"] = float(filled.std(ddof=0))
            record[f"{vital}_rho"] = _safe_autocorr(filled)
            record[f"{vital}_imputed_rate"] = float(series.isna().mean())
            if len(filled) >= 2 and np.ptp(hours) > 0:
                record[f"{vital}_slope"] = float(
                    np.polyfit(hours, filled.to_numpy(dtype=float), 1)[0]
                )
            else:
                record[f"{vital}_slope"] = 0.0

        statics = features.loc[features["stay_id"] == stay_id]
        if statics.empty:
            continue
        if PERSISTENT_FLAG in features.columns:
            record[f"{PERSISTENT_FLAG}_rate"] = float(
                pd.to_numeric(statics[PERSISTENT_FLAG], errors="coerce").fillna(0).mean()
            )
        for column in static_columns:
            record[column] = statics.iloc[0][column]

        rows.append(record)

    params = pd.DataFrame(rows)
    # A distinct name from the loop-local `has_event` above, which is a plain
    # bool: reusing it pins the name to `bool` for the whole function and the
    # array assignment then fails to type-check.
    event_flags = np.asarray(params["has_event"].to_numpy(dtype=float) > 0.5)
    return StayTable(params=params, has_event=event_flags, static_columns=static_columns)


def _ar1(
    length: int,
    level: float,
    slope: float,
    sd: float,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """An AR(1) path around a linear trend, moment-matched to the request.

    The innovation SD is scaled by sqrt(1 - rho^2) so the *stationary* variance
    is `sd`; using `sd` directly would inflate the realised variability by
    1/sqrt(1-rho^2), which at rho=0.7 is a 40% overshoot.

    Stationary variance is not the same thing as *realised sample* variance,
    though, and the difference is large enough here to matter. The sample SD of a
    short, strongly autocorrelated series is biased well below the process SD --
    the sample mean absorbs the low-frequency movement -- and this cohort's stays
    have a median of 16 hours with a quarter of them at the 2-hour floor. Left
    uncorrected the decoder realised about two thirds of the variability it was
    asked for, which would hand every synthetic patient flatter vitals than the
    real patient it was parameterised from, and flatter vitals mean a smaller
    `hr_4h_std` -- one of the model's top features.

    So the deviations are rescaled to hit the requested SD and recentred on the
    requested level. The AR(1) supplies the autocorrelation structure; these two
    moments are then exact rather than approximate.
    """
    rho = float(np.clip(rho, -0.95, 0.95))
    sd = max(float(sd), 0.0)
    # Centred on the midpoint of the stay, not on hour 0. `level` is the real
    # series' *mean*, and a trend anchored at hour 0 would shift the realised
    # mean by slope * (length - 1) / 2 -- over a 72-hour stay with a modest
    # trend that is a large systematic offset in the wrong direction.
    hours = np.arange(length, dtype=float)
    centred = hours - hours.mean()

    # `sd` is the SD of the whole real series, trend included, so the trend's own
    # variance has to come out of the budget rather than be added on top of it.
    # Skipping this apportionment inflates the realised SD whenever the slope is
    # non-zero -- which is exactly the deteriorating patients, whose vitals trend
    # hardest.
    trend_var = float(np.var(centred)) * slope**2
    if trend_var > sd**2 and trend_var > 0:
        # An incoherent parameter draw: a series cannot have more variance in its
        # trend than it has in total. Real encodings never do -- the decomposition
        # forbids it -- but a generator sampling `slope` and `sd` from separate
        # output units has nothing stopping it, so the slope is scaled back to fit
        # the budget rather than allowed to overshoot the requested SD.
        slope = float(np.sign(slope) * sd / np.sqrt(float(np.var(centred))))
        trend_var = sd**2
    trend = level + slope * centred
    residual_sd = float(np.sqrt(max(sd**2 - trend_var, 0.0)))
    innovation = residual_sd * np.sqrt(max(1.0 - rho**2, 1e-6))

    deviation = np.zeros(length, dtype=float)
    deviation[0] = rng.normal(0.0, residual_sd) if residual_sd > 0 else 0.0
    for t in range(1, length):
        deviation[t] = rho * deviation[t - 1] + rng.normal(0.0, innovation)

    # Rescale the finished series, not the deviation alone. Var(trend +
    # deviation) is Var(trend) + Var(deviation) only if the two are uncorrelated,
    # and in a finite sample they are not: an AR(1) with rho near 0.8 carries
    # low-frequency content that lines up with a straight line by chance, and that
    # leftover covariance is the dominant error -- a 38-hour stay asked for SD
    # 4.27 and realised 5.13 without this.
    #
    # Rescaling the whole series fixes it exactly, and does so without touching
    # the autocorrelation, because a constant factor cannot change a correlation.
    # Projecting the trend out of the deviation would also pin the SD, but it
    # strips the low-frequency content that *is* the autocorrelation -- measured
    # at 0.365 realised against ~0.55 requested -- and the rolling-window
    # features this feeds care about that structure. What the rescale does cost
    # is the slope, multiplied by the same factor; the variance apportionment
    # above keeps that factor near 1, so the trade is a few percent on the slope
    # for an exact SD and an untouched rho.
    series = trend + deviation
    if length >= 2 and sd > 0:
        realised = float(series.std(ddof=0))
        if realised > 1e-9:
            series = level + (series - series.mean()) * (sd / realised)
    return series


def decode_stays(
    params: pd.DataFrame,
    static_columns: list[str],
    rng: np.random.Generator,
    horizon: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand parameter vectors into an hourly grid plus the matching labels.

    Returns `(grid, labels)` in the same shape the real pipeline produces, so the
    caller can run `engineer.add_rolling_features` over the grid and join.
    """
    grid_rows: list[pd.DataFrame] = []
    label_rows: list[pd.DataFrame] = []

    for position, record in params.reset_index(drop=True).iterrows():
        stay_id = int(record["stay_id"])
        length = int(np.clip(round(float(record["length"])), MIN_STAY_HOURS, MAX_STAY_HOURS))
        hours = np.arange(length)

        frame = pd.DataFrame({"stay_id": stay_id, "hour": hours})
        for vital in VITALS:
            level = record.get(f"{vital}_level", np.nan)
            if not np.isfinite(level):
                frame[vital] = np.nan
                continue
            path = _ar1(
                length,
                float(level),
                float(record.get(f"{vital}_slope", 0.0)),
                float(record.get(f"{vital}_sd", 0.0)),
                float(record.get(f"{vital}_rho", 0.5)),
                rng,
            )
            low, high = engineer_range(vital)
            frame[vital] = np.clip(path, low, high)

            # Measurement cadence: a carried-forward hour repeats the previous
            # value and is flagged, exactly as the real grid encodes it.
            rate = float(np.clip(record.get(f"{vital}_imputed_rate", 0.0), 0.0, 0.95))
            imputed = rng.random(length) < rate
            imputed[0] = False
            values = frame[vital].to_numpy()
            since = np.zeros(length)
            for t in range(1, length):
                if imputed[t]:
                    values[t] = values[t - 1]
                    since[t] = since[t - 1] + 1
            frame[vital] = values
            frame[f"{vital}_was_imputed"] = imputed
            frame[f"{vital}_hours_since_last_obs"] = since

        for column in static_columns:
            frame[column] = record[column]

        rate = float(np.clip(record.get(f"{PERSISTENT_FLAG}_rate", 0.0), 0.0, 1.0))
        placed_at = int(round((1.0 - rate) * length))
        frame[PERSISTENT_FLAG] = hours >= placed_at if rate > 0 else np.zeros(length, dtype=bool)

        has_event = bool(record["has_event"] > 0.5)
        labels = pd.DataFrame({"stay_id": stay_id, "hour": hours})
        if has_event:
            # Censoring puts the event one hour past the last at-risk row, so
            # the positives are the final `horizon` hours of the stay.
            labels[f"label_{horizon}h"] = (length - hours <= horizon).astype(int)
        else:
            labels[f"label_{horizon}h"] = 0

        grid_rows.append(frame)
        label_rows.append(labels)
        del position

    if not grid_rows:
        return pd.DataFrame(), pd.DataFrame()
    return (
        pd.concat(grid_rows, ignore_index=True),
        pd.concat(label_rows, ignore_index=True),
    )


def engineer_range(vital: str) -> tuple[float, float]:
    """Plausible physiological bounds, shared with the snapshot preprocessor."""
    from ml.synthetic.preprocess import PLAUSIBLE_RANGES

    return PLAUSIBLE_RANGES.get(vital, (-np.inf, np.inf))
