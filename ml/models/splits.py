"""Grouped, repeated, stratified cross-validation (PROJECT_PLAN.md section 11).

**Grouped by subject_id -- no patient spans train and test.** The caller is
responsible for passing the right column, and
``ml/features/engineer.feature_matrix_for_training`` returns ``subject_id``
for exactly this reason.

This module used to claim that ``stay_id`` was "already a 1:1 proxy for
subject_id at the granularity this task needs", and offered a ``group_key()``
helper to use the real subject id "when it is supplied". Both were wrong. The
proxy claim is false in this cohort -- 21 of 93 subjects in the at-risk set
have more than one ICU stay, carrying 45% of its rows and 44% of its positives
(``ml/features/labels.py``'s readmission events exist *because* of this) -- and
the helper was never once called with a subject id, so it silently returned
``stay_id`` at every call site while looking like the safeguard. Both are gone:
the escape hatch that is never taken is worse than no escape hatch, because it
stops anyone looking. ``ml/evaluation/reliability.py`` measures what the leak
was worth: **+0.0499 AUPRC** of optimism at the 6h horizon on the current
feature set (+0.0159 before `gender` was added -- a patient-constant feature
makes a stay-level split leak more, not less).

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
