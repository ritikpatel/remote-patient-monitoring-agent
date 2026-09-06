"""Tests for ml/models/splits.py -- the one property that must never break:
no group spans train and test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.models import splits


def _synthetic_data(n_groups: int = 40, rows_per_group: int = 5, seed: int = 0) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        # Every 4th group is entirely positive, to guarantee both classes.
        label = 1 if g % 4 == 0 else 0
        for _ in range(rows_per_group):
            rows.append({"group": g, "y": label})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def test_no_group_spans_train_and_test() -> None:
    df = _synthetic_data()
    for _repeat, _fold, train_idx, test_idx in splits.repeated_grouped_stratified_splits(
        df["y"], df["group"], n_splits=5, n_repeats=3
    ):
        train_groups = set(df.iloc[train_idx]["group"])
        test_groups = set(df.iloc[test_idx]["group"])
        assert train_groups.isdisjoint(test_groups)


def test_every_row_appears_in_exactly_one_test_fold_per_repeat() -> None:
    df = _synthetic_data()
    seen_per_repeat: dict[int, set[int]] = {}
    for repeat, _fold, _train_idx, test_idx in splits.repeated_grouped_stratified_splits(
        df["y"], df["group"], n_splits=5, n_repeats=2
    ):
        seen_per_repeat.setdefault(repeat, set()).update(test_idx.tolist())
    for seen in seen_per_repeat.values():
        assert seen == set(range(len(df)))


def test_reproducible_given_same_random_state() -> None:
    df = _synthetic_data()
    first = list(splits.repeated_grouped_stratified_splits(df["y"], df["group"], n_repeats=2))
    second = list(splits.repeated_grouped_stratified_splits(df["y"], df["group"], n_repeats=2))
    for (r1, f1, tr1, te1), (r2, f2, tr2, te2) in zip(first, second, strict=True):
        assert r1 == r2 and f1 == f2
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)
