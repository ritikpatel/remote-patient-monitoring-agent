"""L2 logistic regression on the full engineered feature set -- the first
rung of the "L2 logistic regression -> LightGBM -> small GRU. Stop at
whichever wins" ladder (PROJECT_PLAN.md section 11).

Unlike LightGBM, sklearn's LogisticRegression cannot consume raw NaNs or
string categoricals, so this is the one model in the ladder that needs an
explicit imputation step -- median for numeric columns, one-hot for the two
categorical columns (``gender``, ``first_careunit``). Per-column missingness
is not thrown away by this: every core vital already carries its own
``_was_imputed`` flag and ``_hours_since_last_obs`` recency counter as
*separate* feature columns (R2), so the imputed value and "this was originally
missing" remain distinguishable to the model even after the median fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS = ["gender", "first_careunit", "dx_chapter"]


def build_pipeline(feature_columns: list[str]) -> Pipeline:
    # Follow the actual columns, not the module constant: a categorical column can be
    # legitimately absent (the demographic ablation in ml/evaluation/fairness.py drops
    # `gender`), and naming a missing column in a ColumnTransformer is a hard error.
    categorical_columns = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                LogisticRegression(
                    # l2 is sklearn's default penalty; passing it explicitly is
                    # deprecated as of sklearn 1.8 in favour of C/l1_ratio.
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=0,
                ),
            ),
        ]
    )


def fit_predict_proba(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame
) -> tuple[Pipeline, np.ndarray]:
    pipeline = build_pipeline(list(x_train.columns))
    pipeline.fit(x_train, y_train)
    proba = pipeline.predict_proba(x_test)[:, 1]
    return pipeline, proba
