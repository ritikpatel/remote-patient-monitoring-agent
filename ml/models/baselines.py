"""The three baselines every learned model must beat before any claim
(PROJECT_PLAN.md section 11): recalibrated NEWS2, SOFA, and logistic
regression on age + last HR/RR/SpO2.

NEWS2 and SOFA are not "trained" -- they are read straight out of the
warehouse (``capstone.news2``, ``mimiciv_derived.sofa``) and used as-is, as
monotonic risk scores, for AUROC/AUPRC purposes: a higher score is assumed
riskier, which is exactly the ordinal claim both scores make by construction.
The age+vitals logistic regression is genuinely fit per CV fold like any
other model, just with a deliberately minimal, literal feature set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

AGE_VITALS_COLUMNS = ["admission_age", "hr", "rr", "spo2"]


def news2_score(x: pd.DataFrame) -> np.ndarray:
    return x["news2"].to_numpy(dtype=float)


def sofa_score(x: pd.DataFrame) -> np.ndarray:
    return x["sofa_24hours"].to_numpy(dtype=float)


def build_age_vitals_logistic_regression() -> Pipeline:
    """ "Logistic regression on age + last HR/RR/SpO2" -- literally that
    feature set, nothing else, so it stands as the minimal-effort learned
    baseline the plan names explicitly.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    # l2 is sklearn's default penalty; passing it explicitly is
                    # deprecated as of sklearn 1.8 in favour of C/l1_ratio.
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=0,
                ),
            ),
        ]
    )


def age_vitals_subset(x: pd.DataFrame) -> pd.DataFrame:
    return x[AGE_VITALS_COLUMNS]
