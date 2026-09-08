"""LightGBM on the full engineered feature set -- second rung of the ladder
(PROJECT_PLAN.md section 11).

Deliberately does none of logistic.py's preprocessing: LightGBM splits on
missing values natively (a NaN "not yet observed" is itself informative, per
R2/R3 -- imputing it away before the tree sees it would throw that signal out
rather than let the model use it), and takes ``category``-dtype columns
directly instead of needing one-hot encoding.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ["gender", "first_careunit"]


def _present_categoricals(x: pd.DataFrame) -> list[str]:
    """LightGBM rejects a categorical_feature name that is not in the frame, so the
    list must follow the actual columns rather than the module constant."""
    return [c for c in CATEGORICAL_COLUMNS if c in x.columns]


def _as_categorical(x: pd.DataFrame) -> pd.DataFrame:
    """Casts the known categorical columns, tolerating any that are absent.

    A hard KeyError here meant the model could not be fitted on a feature matrix with
    a categorical column removed -- which is exactly what a demographic ablation does
    (ml/evaluation/fairness.py, finding F4). An optional feature should not be a
    structural requirement of the estimator.
    """
    x = x.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in x.columns:
            x[col] = x[col].astype("category")
    return x


def build_model(scale_pos_weight: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=15,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=10,
        scale_pos_weight=scale_pos_weight,
        random_state=0,
        verbosity=-1,
        # Pinned to 1: LightGBM's macOS/arm64 OpenMP thread pool is prone to a
        # SIGSEGV after many repeated fit() calls in one process (observed
        # directly here -- a run of ~10 fits inside the repeated-CV loop
        # crashed silently with no Python traceback, exit code 139). This
        # dataset is small enough (a few thousand rows) that single-threaded
        # fitting costs a fraction of a second more per fold, which is a
        # trivial price for not segfaulting midway through a 100-fold run.
        n_jobs=1,
    )


def fit_predict_proba(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame
) -> tuple[lgb.LGBMClassifier, np.ndarray]:
    x_train = _as_categorical(x_train)
    x_test = _as_categorical(x_test)
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    scale_pos_weight = negatives / positives if positives else 1.0

    model = build_model(scale_pos_weight)
    model.fit(x_train, y_train, categorical_feature=_present_categoricals(x_train))
    proba = np.asarray(model.predict_proba(x_test))[:, 1]
    return model, proba
