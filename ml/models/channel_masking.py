"""Channel masks, and training-time channel dropout.

`ml/evaluation/channel_dropout.py` measured what happens when the ICU model is
scored with home-unavailable channels masked **at prediction time only**, and closed
with the obvious objection to its own method:

> *"Every channel here is 74-100% present in training, so the trees had very little
> opportunity to learn a sensible default direction for its absence. These numbers
> therefore measure an artefact of training-time availability as much as the clinical
> value of each signal. A model intended to run with channels routinely missing
> should be trained that way -- with dropout applied during training, not only
> measured at inference."*

This module is that. It is also the answer to a fair complaint about the earlier
framing: nothing here *restricts* the model. A patient at home does not have an
arterial line or a ventilator, and no amount of model design conjures one. What is
in our control is whether the model has ever seen a patient without them -- and
before this, it had not, which is why it looked so much worse in that regime.

**The kit definitions are imported, not redeclared.** `HOME_KITS` lives in
`simulators/home_kit_stream.py` because that is what physically emits the streams;
if training masks and simulator kits were two separate lists they would drift, and
the drift would look like a modelling result rather than a bookkeeping error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from simulators.home_kit_stream import HOME_KITS, NO_HOME_SENSOR  # noqa: E402

from ml.features import engineer  # noqa: E402

# Columns that are meaningless outside a hospital regardless of sensor suite:
# arterial-line presence is always false at home, and lab-order intensity is a
# readout of hospital workflow, not of the patient.
HOSPITAL_ONLY_COLUMNS = ("has_arterial_line",)


def channel_columns(channel: str, cols: list[str]) -> list[str]:
    """Every engineered column derived from one raw channel: the value, its
    imputation flag, its recency, and its rolling std/slope at both windows."""
    return [c for c in cols if c == channel or c.startswith(f"{channel}_")]


def kit_mask(kit_name: str, cols: list[str]) -> list[str]:
    """Columns to blank so the frame looks like what ``kit_name`` can observe.

    Derived by subtraction from the kit's channel list rather than enumerated, so a
    channel added to a kit in `home_kit_stream.py` is automatically unmasked here.
    """
    kit = HOME_KITS[kit_name]
    absent = [c for c in engineer.CORE_VITALS if c not in kit.channels]
    masked = [col for ch in absent for col in channel_columns(ch, cols)]
    masked += [c for c in HOSPITAL_ONLY_COLUMNS if c in cols]
    return masked


def all_kit_masks(cols: list[str]) -> dict[str, list[str]]:
    """Every named kit, plus the unmasked ICU case, keyed by name."""
    # Annotated because the literal empty list makes mypy infer dict[str, list[?]]
    # and it cannot resolve the element type from the first entry alone.
    masks: dict[str, list[str]] = {"icu_full": []}
    for name in HOME_KITS:
        masks[name] = kit_mask(name, cols)
    return masks


def apply_mask(x: pd.DataFrame, masked_columns: list[str]) -> pd.DataFrame:
    """Blank the masked columns to NaN.

    NaN rather than zero, and that is the whole mechanism: LightGBM splits on
    missingness natively (R2/R3 -- absence is signal), so a NaN says "no sensor
    reported this" while a zero says "the sensor reported zero", which for a heart
    rate is a very different clinical claim.

    The float cast is not cosmetic. The ``*_was_imputed`` columns are boolean, and
    assigning NaN into a bool Series silently promotes it to ``object`` dtype, which
    LightGBM rejects outright ("pandas dtypes must be int, float or bool"). Casting
    first keeps every masked column numeric-with-missing, which is the only
    representation the estimator can read a mask from.
    """
    out = x.copy()
    for col in masked_columns:
        if col in out.columns:
            out[col] = out[col].astype("float64")
            out[col] = np.nan
    return out


def augment_with_channel_dropout(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    kit_names: tuple[str, ...] = tuple(HOME_KITS),
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Replicate the training set once per kit, each replica masked to that kit.

    Returns (x_augmented, y_augmented, groups_augmented). The original unmasked rows
    are always included, so the model does not trade ICU performance for home-kit
    performance -- it sees both regimes.

    **Replication rather than per-row random masking, deliberately.** Random masking
    would show the model a different arbitrary channel subset on every row, which
    teaches robustness to noise-shaped absence. Real absence is not noise-shaped: it
    is *structured*, the same channels missing for the same patient for the whole
    episode, because a kit either includes a BP cuff or it does not. Replicating per
    kit reproduces that structure.

    **`groups` is replicated alongside**, which is what keeps grouped CV correct: a
    patient's masked and unmasked replicas carry the same subject id, so they cannot
    be split across train and test. Getting this wrong would leak the same patient's
    physiology into both folds under two different masks -- a subtler version of the
    stay-vs-subject grouping leak this project already found once
    (`ml/features/engineer.feature_matrix_for_training`).
    """
    del rng  # deterministic: replication needs no randomness
    cols = list(x.columns)
    masks = {name: kit_mask(name, cols) for name in kit_names}

    # Every column any kit masks is cast to float64 in EVERY replica, the unmasked
    # original included. Without this, concatenating a bool `*_was_imputed` column
    # from the unmasked frame with the same column as float-with-NaN from a masked
    # one yields `object` dtype, which LightGBM rejects. Casting per-frame is not
    # enough -- the promotion happens at concat, so the dtypes have to agree before
    # they meet.
    maskable = sorted({c for m in masks.values() for c in m} & set(cols))
    base = x.copy()
    for col in maskable:
        base[col] = base[col].astype("float64")

    frames = [base]
    ys = [y]
    gs = [groups]
    for masked_cols in masks.values():
        frames.append(apply_mask(base, masked_cols))
        ys.append(y)
        gs.append(groups)
    return (
        pd.concat(frames, ignore_index=True),
        pd.concat(ys, ignore_index=True),
        pd.concat(gs, ignore_index=True),
    )


def describe_masks(cols: list[str]) -> pd.DataFrame:
    """One row per kit: how many engineered columns survive, and which channels are
    gone. Printed by the transfer study so the reader sees what each kit costs in
    feature terms before seeing what it costs in AUPRC."""
    rows = []
    for name, masked in all_kit_masks(cols).items():
        kit = HOME_KITS.get(name)
        rows.append(
            {
                "kit": name,
                "channels": ", ".join(kit.channels) if kit else "all",
                "columns_masked": len(masked),
                "columns_remaining": len(cols) - len(masked),
                "channels_absent": ", ".join(
                    c for c in engineer.CORE_VITALS if kit and c not in kit.channels
                )
                or "none",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "HOME_KITS",
    "HOSPITAL_ONLY_COLUMNS",
    "NO_HOME_SENSOR",
    "all_kit_masks",
    "apply_mask",
    "augment_with_channel_dropout",
    "channel_columns",
    "describe_masks",
    "kit_mask",
]
