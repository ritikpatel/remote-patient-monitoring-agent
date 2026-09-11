"""Data preprocessing, per the tutorial's Figure 3 box 1 and its Methods section.

Four steps, in the tutorial's own order:

1. **Remove outliers.** The paper deletes patients whose BMI is below 10 or
   above 60 -- values "unreasonable for adult patients" -- and names clipping to
   a pre-established range as the alternative that "preserves the amount of data
   available for training". With 140 stays, deletion is not affordable here, so
   this module clips to clinically plausible physiological ranges and *counts*
   what it clipped, because a clip rate is the diagnostic that tells you whether
   the range or the data is wrong.

2. **Handle missing values.** The paper eliminates variables missing more than
   50% and imputes the rest by "random sampling ... based on the marginal
   distribution of each variable". Both are implemented here, and the second is
   deliberately *not* mean imputation: sampling the marginal preserves the
   variance a GAN then has to learn, where mean-filling would hand it a spike at
   the mean that it would faithfully reproduce.

3. **Normalize continuous variables** to (0,1) by min-max, the paper's Equation
   1, so that no variable dominates training through its range alone. The fitted
   min/max are kept on the spec so generation can invert them exactly.

4. **Handle concepts with low prevalence.** The paper drops phecodes with
   prevalence below 5e-5 because "ML-based generative models ... cannot
   accurately capture the statistical properties of these variables". That
   threshold is calibrated to 181k patients; at this cohort's scale the
   equivalent is stated as a minimum *count*, since a prevalence floor of 5e-5
   over 12,004 rows is less than one row and would drop nothing at all.

The output is a `Matrix`: a float array in [0,1] plus a `MatrixSpec` that knows
which columns are continuous, which are binary, and which form one-hot blocks.
That spec is what lets the generator attach a SoftMax per categorical block
(tutorial, Model Training) and what lets postprocessing invert every transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Clinically plausible ranges for the vitals this cohort carries. These are
# artefact bounds, not reference ranges: the job is to remove values that cannot
# be true of a living patient (a heart rate of 0 from a disconnected lead, an
# SpO2 of 300 from a probe error), never to remove values that are merely
# abnormal. A deterioration model lives on the abnormal ones, so a range tight
# enough to look tidy would delete exactly the signal being modelled.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "hr": (20.0, 250.0),
    "rr": (4.0, 60.0),
    "spo2": (50.0, 100.0),
    "sbp": (40.0, 250.0),
    "map": (20.0, 200.0),
    "temp_c": (30.0, 43.0),
    "gcs_total": (3.0, 15.0),
    "fio2": (21.0, 100.0),
    "glucose": (20.0, 800.0),
    "admission_age": (18.0, 95.0),
}

# The tutorial's 5e-5 prevalence floor, restated as a count. Below this many
# occurrences a binary column is noise the generator will either ignore or
# memorise, and neither is useful.
MIN_CONCEPT_COUNT = 20

# The paper's own missingness cutoff: "it is generally recommended to eliminate
# variables with a high missing rate (eg, more than 50%)".
MAX_MISSING_RATE = 0.50


@dataclass
class MatrixSpec:
    """Column layout of the GAN-ready matrix, and everything needed to invert it.

    `continuous`/`binary` are column names; `categorical` maps an original column
    to the ordered list of its one-hot columns, which is the block a SoftMax gets
    attached to. `minimum`/`maximum` are the fitted min-max bounds from Equation 1.
    """

    continuous: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)
    categorical: dict[str, list[str]] = field(default_factory=dict)
    minimum: dict[str, float] = field(default_factory=dict)
    maximum: dict[str, float] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def block_slices(self) -> list[tuple[int, int]]:
        """Start/stop index of each one-hot block, for the generator's SoftMax."""
        index = {c: i for i, c in enumerate(self.columns)}
        out = []
        for members in self.categorical.values():
            positions = sorted(index[m] for m in members)
            # One-hot members are emitted contiguously by `build_matrix`, so a
            # block is fully described by its endpoints. Asserted rather than
            # assumed: a non-contiguous block would silently give the SoftMax
            # the wrong columns, and the resulting one-hot violation would only
            # surface much later as an evaluation artefact.
            if positions != list(range(positions[0], positions[-1] + 1)):
                raise ValueError("one-hot block is not contiguous -- build_matrix changed?")
            out.append((positions[0], positions[-1] + 1))
        return out


@dataclass
class Matrix:
    """A GAN-ready matrix and the report of what preprocessing did to build it."""

    values: np.ndarray
    spec: MatrixSpec
    report: pd.DataFrame


def clip_outliers(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Step 1. Clip to `PLAUSIBLE_RANGES`, returning per-column clip counts."""
    out = frame.copy()
    clipped: dict[str, int] = {}
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        # `notna` guards the comparison: a NaN is missing, not an outlier, and
        # counting it here would double-charge it against step 2's report.
        hits = int(((values < low) | (values > high)).fillna(False).sum())
        if hits:
            clipped[column] = hits
        out[column] = values.clip(low, high)
    return out, clipped


def drop_high_missingness(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Step 2a. Eliminate variables above the paper's 50% missing rate."""
    rates = frame.isna().mean()
    doomed = {c: float(rates[c]) for c in frame.columns if rates[c] > MAX_MISSING_RATE}
    return frame.drop(columns=list(doomed)), doomed


def impute_by_marginal_sampling(
    frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Step 2b. The paper's strategy: sample observed values of the same variable.

    Returns the imputed frame and the missing rate that was filled per column,
    which the report carries so a reader can see how much of any column is
    imputation rather than measurement.
    """
    out = frame.copy()
    filled: dict[str, float] = {}
    for column in out.columns:
        missing = out[column].isna()
        n_missing = int(missing.sum())
        if n_missing == 0:
            continue
        observed = out.loc[~missing, column].to_numpy()
        if observed.size == 0:
            # Nothing to sample from. Zero-fill and let the report say so rather
            # than raising: a fully-missing column is a data problem to surface,
            # not a reason to abort the whole preprocessing run.
            out[column] = 0.0
            filled[column] = 1.0
            continue
        out.loc[missing, column] = rng.choice(observed, size=n_missing, replace=True)
        filled[column] = float(n_missing) / len(out)
    return out, filled


def drop_low_prevalence(
    frame: pd.DataFrame, binary: list[str], min_count: int = MIN_CONCEPT_COUNT
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Step 4. Drop binary concepts too rare for a generative model to learn.

    Prevalence is counted on the *minority* class: a flag true in 3 of 12,004
    rows and one true in 12,001 are equally unlearnable, and only checking the
    positive count would keep the second.
    """
    dropped: dict[str, int] = {}
    for column in binary:
        if column not in frame.columns:
            continue
        positives = int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())
        minority = min(positives, len(frame) - positives)
        if minority < min_count:
            dropped[column] = minority
    return frame.drop(columns=list(dropped)), dropped


def _classify_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Split columns into continuous / binary / categorical.

    The tutorial's own rule of thumb (EHR Data Types and Matrix Representation):
    discrete variables with a broad range of values "can be approximated as
    continuous", categorical ones have a limited unchanging set of options, and
    a categorical with only 2 options is a single binary column rather than a
    one-hot pair.
    """
    continuous, binary, categorical = [], [], []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            binary.append(column)
        elif isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):
            categorical.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            uniques = series.dropna().unique()
            if len(uniques) <= 2 and set(np.asarray(uniques, dtype=float)) <= {0.0, 1.0}:
                binary.append(column)
            else:
                continuous.append(column)
        else:
            categorical.append(column)
    return continuous, binary, categorical


def build_matrix(
    frame: pd.DataFrame,
    rng: np.random.Generator | None = None,
    min_concept_count: int = MIN_CONCEPT_COUNT,
) -> Matrix:
    """Run all four preprocessing steps and emit the GAN-ready matrix.

    Every column of the result lies in [0,1]: continuous by Equation 1, binary
    already there, categorical one-hot. That uniformity is what lets a single
    sigmoid/SoftMax output layer serve the whole matrix (tutorial, Synthetic Data
    Generation and Postprocessing).
    """
    rng = rng or np.random.default_rng(0)
    notes: list[dict[str, object]] = []

    working, clipped = clip_outliers(frame)
    for column, count in clipped.items():
        notes.append({"step": "clip_outliers", "column": column, "detail": float(count)})

    working, doomed = drop_high_missingness(working)
    for column, rate in doomed.items():
        notes.append({"step": "drop_missing>50%", "column": column, "detail": rate})

    continuous, binary, categorical = _classify_columns(working)

    working, dropped = drop_low_prevalence(working, binary, min_count=min_concept_count)
    for column, count in dropped.items():
        notes.append({"step": "drop_low_prevalence", "column": column, "detail": float(count)})
    binary = [c for c in binary if c not in dropped]

    # Impute before one-hot expansion so a missing categorical becomes a sampled
    # real level rather than an all-zero row, which would break the one-hot
    # constraint the generator is about to be asked to preserve.
    working, filled = impute_by_marginal_sampling(working, rng)
    for column, rate in filled.items():
        notes.append({"step": "impute_marginal", "column": column, "detail": rate})

    spec = MatrixSpec()
    blocks: list[pd.DataFrame] = []

    for column in continuous:
        values = pd.to_numeric(working[column], errors="coerce").astype(float)
        low, high = float(values.min()), float(values.max())
        spec.minimum[column], spec.maximum[column] = low, high
        span = high - low
        # A constant column has span 0. Equation 1 divides by that span, so it is
        # mapped to a constant 0.5 instead: it carries no information either way,
        # and 0.5 inverts back to the constant exactly.
        scaled = (values - low) / span if span > 0 else pd.Series(0.5, index=values.index)
        blocks.append(scaled.rename(column).to_frame())
        spec.continuous.append(column)

    for column in binary:
        values = pd.to_numeric(working[column], errors="coerce").fillna(0).astype(float)
        blocks.append(values.rename(column).to_frame())
        spec.binary.append(column)

    for column in categorical:
        dummies = pd.get_dummies(working[column].astype(str), prefix=column).astype(float)
        # Sorted so the block order is reproducible across runs; `block_slices`
        # relies on contiguity, and pandas orders dummies by level anyway.
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        blocks.append(dummies)
        spec.categorical[column] = list(dummies.columns)

    matrix = pd.concat(blocks, axis=1)
    spec.columns = list(matrix.columns)

    report = pd.DataFrame(notes, columns=pd.Index(["step", "column", "detail"]))
    return Matrix(values=matrix.to_numpy(dtype=np.float32), spec=spec, report=report)


def invert_matrix(values: np.ndarray, spec: MatrixSpec) -> pd.DataFrame:
    """Postprocessing: undo Equation 1 and collapse one-hot blocks back to levels.

    The tutorial's postprocessing step -- "rounding the values is necessary [for
    noncontinuous variables], whereas the values of continuous variables require
    rescaling to their original range by applying the inverse version of Equation
    1". Continuous columns are additionally clipped to [0,1] before inversion so
    a generator that overshoots cannot produce a heart rate outside the range the
    real data spanned.
    """
    frame = pd.DataFrame(values, columns=pd.Index(spec.columns))
    out: dict[str, pd.Series] = {}

    for column in spec.continuous:
        low, high = spec.minimum[column], spec.maximum[column]
        scaled = frame[column].clip(0.0, 1.0)
        out[column] = scaled * (high - low) + low

    for column in spec.binary:
        out[column] = frame[column].round().clip(0, 1)

    for column, members in spec.categorical.items():
        # argmax rather than a threshold: the SoftMax guarantees the block sums
        # to 1 but not that any single entry clears 0.5, and a k-way block with
        # no entry above 0.5 is entirely normal.
        out[column] = pd.Series(
            np.asarray(members)[frame[members].to_numpy().argmax(axis=1)]
        ).str.removeprefix(f"{column}_")

    return pd.DataFrame(out)
