"""Tests for ml/models/baselines.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.models import baselines


def test_news2_and_sofa_score_extraction() -> None:
    x = pd.DataFrame({"news2": [3, 7, 10], "sofa_24hours": [1, 5, 12]})
    assert list(baselines.news2_score(x)) == [3.0, 7.0, 10.0]
    assert list(baselines.sofa_score(x)) == [1.0, 5.0, 12.0]


def test_age_vitals_logistic_regression_fits_and_predicts_proba() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = pd.DataFrame(
        {
            "admission_age": rng.uniform(20, 90, n),
            "hr": rng.uniform(50, 150, n),
            "rr": rng.uniform(10, 40, n),
            "spo2": rng.uniform(80, 100, n),
        }
    )
    # Construct a genuinely separable signal so the fit isn't degenerate.
    y = (x.hr + x.admission_age > 180).astype(int)
    x_subset = baselines.age_vitals_subset(x)
    pipeline = baselines.build_age_vitals_logistic_regression()
    pipeline.fit(x_subset, y)
    proba = pipeline.predict_proba(x_subset)[:, 1]
    assert proba.shape == (n,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_age_vitals_subset_selects_only_the_named_columns() -> None:
    x = pd.DataFrame(
        {
            "admission_age": [1],
            "hr": [2],
            "rr": [3],
            "spo2": [4],
            "news2": [99],  # must NOT leak in -- this baseline is deliberately minimal
        }
    )
    subset = baselines.age_vitals_subset(x)
    assert list(subset.columns) == ["admission_age", "hr", "rr", "spo2"]
