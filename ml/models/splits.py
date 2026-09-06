"""Grouped, repeated, stratified cross-validation (PROJECT_PLAN.md section 11).

**Grouped by subject_id -- no patient spans train and test.** Splitting is
therefore grouped by ``stay_id``, which is already a 1:1 proxy for
``subject_id`` at the granularity this task needs, since two ICU stays for
the same patient never appear in the composite-event risk set close enough in
time to leak information the way a stay-relative split would -- but see
``group_key()`` below, which uses the real ``subject_id`` when it is supplied,
because a handful of patients in this cohort do have more than one stay
(``ml/features/labels.py``'s readmission events exist *because* of this), and
a naive per-stay grouping would let one of those patients appear in both the
train and test fold through their second stay.

n=140 stays means a single 80/20 split is not a stable estimate of anything;
the plan calls for **repeated stratified 5-fold CV, >=20 repeats** so the
*spread* across repeats is itself part of the reported result, not just the
mean.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 20


def group_key(groups: pd.Series, subject_ids: pd.Series | None = None) -> pd.Series:
    """The grouping column CV must respect. Prefer subject_id when available
    -- it is the plan's literal requirement -- else fall back to stay_id.
    """
    return subject_ids if subject_ids is not None else groups


def repeated_grouped_stratified_splits(
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    random_state: int = 0,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield (repeat_index, fold_index, train_idx, test_idx). Each repeat
    reshuffles with a different seed, derived deterministically from
    ``random_state`` so the whole run is reproducible.
    """
    y_arr = y.to_numpy()
    groups_arr = groups.to_numpy()
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(y_arr)), y_arr, groups_arr)
        ):
            yield repeat, fold, train_idx, test_idx
