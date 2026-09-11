"""Feature matrix for the composite deterioration model (PROJECT_PLAN.md section 11).

Every feature group here traces to a specific design rule from section 4, not
just "things that seemed predictive":

* **Carried-forward vitals + imputation flags + recency** (R2, E3) come
  straight from ``capstone.hourly_grid`` -- already built in Phase 1.
* **Rolling trend over the trailing 4h and 24h** for each core vital (mean,
  std, OLS slope) reuses ``services/stream-processor/windowing.py``'s
  ``rolling_stats``/``trend_slope`` -- the same functions the live streaming
  path uses, loaded through ``services/common/testing.py`` because that
  module's directory is hyphenated and not import-able by dotted path.
* **Severity scores as features, not just baselines to beat** -- NEWS2
  (``capstone.news2``) and the rolling 24h SOFA total
  (``mimiciv_derived.sofa.sofa_24hours``) are joined in as inputs. This is
  standard "augmented early-warning score" practice, and legitimate because
  the baselines each model must still separately beat are evaluated as
  stand-alone scores, not with these engineered features attached.
* **Arterial-line presence** (R3, E3 -- "Arterial-line presence (46% of
  stays)... Ordering behaviour is a feature") -- an hour-resolved boolean from
  ``mimiciv_derived.invasive_line``, not inferred indirectly from which blood
  pressure channel happened to be charted.
* **Lab-ordering intensity, normalised to the admission's own baseline rate**
  (R3, R4, E15 -- "measurement intensity tracks outcome but is confounded...
  event-rate features are legitimate but must be normalised by care
  setting") -- trailing lab-order counts divided by that admission's own
  mean rate, so the feature reads as "busier than usual for this patient
  right now," not "in the ICU" (which every row already is).
* **Static demographics** -- age, gender, first ICU care unit -- from
  ``mimiciv_derived.icustay_detail`` / ``mimiciv_icu.icustays``.
* **Disease context** -- Charlson chronic-comorbidity burden and the primary
  diagnosis's ICD chapter, from ``capstone.disease_context``
  (``warehouse/disease.py``). This closed a real gap: before it, the model's only
  case-mix signal was ``first_careunit``, which ranked *second* by mean |SHAP| and
  was silently acting as a diagnosis proxy. The two disease sources sit on opposite
  sides of a leakage line -- Charlson describes pre-existing chronic burden, ICD
  codes are assigned by billing coders after discharge -- so they are selectable
  separately via ``DISEASE_FEATURE_SETS`` and measured against each other in
  ``ml/evaluation/disease_leakage.py``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from services.common.testing import load_module
from warehouse.disease import CHARLSON_FLAGS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_windowing = load_module(
    REPO_ROOT / "services" / "stream-processor" / "windowing.py",
    "ml_features_engineer_windowing",
)
trend_slope = _windowing.trend_slope

CORE_VITALS = ["hr", "rr", "spo2", "sbp", "map", "temp_c", "gcs_total", "fio2", "glucose"]
ROLLING_WINDOWS_H = (4, 24)


def load_hourly_grid_raw(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute("select * from capstone.hourly_grid order by stay_id, hour").fetchdf()


def add_rolling_features(
    grid: pd.DataFrame, windows: tuple[int, ...] = ROLLING_WINDOWS_H
) -> pd.DataFrame:
    """Trailing (causal, current hour included) mean/std/slope per vital per
    window, computed per stay so no window ever crosses a stay boundary.
    """
    grid = grid.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    out = grid.copy()
    for w in windows:
        grouped = grid.groupby("stay_id")
        for var in CORE_VITALS:
            roll = grouped[var].rolling(window=w, min_periods=1)
            out[f"{var}_{w}h_mean"] = roll.mean().reset_index(level=0, drop=True)
            out[f"{var}_{w}h_std"] = roll.std(ddof=0).fillna(0.0).reset_index(level=0, drop=True)

        # Slope needs (time, value) pairs, not a single series -- compute per
        # stay with the shared trend_slope() from the streaming path.
        for var in CORE_VITALS:
            out[f"{var}_{w}h_slope"] = 0.0
        for _stay_id, g in grid.groupby("stay_id"):
            hours = g["hour"].to_numpy(dtype=float)
            for var in CORE_VITALS:
                values = g[var].to_numpy(dtype=float)
                slopes = np.zeros(len(g))
                for i in range(len(g)):
                    lo = max(0, i - w + 1)
                    window_hours = hours[lo : i + 1].tolist()
                    window_values = values[lo : i + 1].tolist()
                    slopes[i] = trend_slope(window_hours, window_values)
                out.loc[g.index, f"{var}_{w}h_slope"] = slopes
    return out


def add_severity_scores(grid: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    news2 = conn.execute(
        "select stay_id, hour, news2, tier_ward, tier_icu from capstone.news2"
    ).fetchdf()
    sofa = conn.execute(
        "select stay_id, hr as hour, sofa_24hours from mimiciv_derived.sofa"
    ).fetchdf()
    out = grid.merge(news2, on=["stay_id", "hour"], how="left")
    out = out.merge(sofa, on=["stay_id", "hour"], how="left")
    out["news2"] = out["news2"].fillna(0)
    out["sofa_24hours"] = out["sofa_24hours"].fillna(0)
    for tier_col in ("tier_ward", "tier_icu"):
        out[tier_col] = out[tier_col].astype("object").fillna("low")
    return out


def add_arterial_line_feature(grid: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    lines = conn.execute("""
        select il.stay_id, il.starttime, il.endtime, d.icu_intime
        from mimiciv_derived.invasive_line il
        join mimiciv_derived.icustay_detail d using (stay_id)
        where il.line_type = 'Arterial'
        """).fetchdf()
    out = grid.copy()
    out["has_arterial_line"] = 0
    if lines.empty:
        return out
    lines["start_h"] = (lines.starttime - lines.icu_intime).dt.total_seconds() / 3600.0
    lines["end_h"] = (lines.endtime - lines.icu_intime).dt.total_seconds() / 3600.0
    for stay_id, stay_lines in lines.groupby("stay_id"):
        mask = out.stay_id == stay_id
        row_hours = out.loc[mask, "hour"]
        active = pd.Series(False, index=row_hours.index)
        for _, line in stay_lines.iterrows():
            active |= (row_hours >= line.start_h) & (row_hours <= line.end_h)
        out.loc[mask, "has_arterial_line"] = active.astype(int)
    return out


def add_lab_order_intensity(grid: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """R3/R4/R5: lab-ordering rate in the trailing 4h/24h, normalised by this
    admission's own mean rate across its ICU stay so far -- "busier than
    usual for this patient," not "in the ICU."
    """
    labs = conn.execute("""
        select l.hadm_id, l.charttime, d.stay_id, d.icu_intime
        from mimiciv_hosp.labevents l
        join mimiciv_derived.icustay_detail d using (hadm_id)
        where l.hadm_id is not null
        """).fetchdf()
    out = grid.copy()
    for w in ROLLING_WINDOWS_H:
        out[f"lab_orders_{w}h"] = 0
    out["lab_order_rate_ratio"] = 1.0
    if labs.empty:
        return out

    labs["lab_hour"] = (labs.charttime - labs.icu_intime).dt.total_seconds() / 3600.0
    for stay_id, stay_labs in labs.groupby("stay_id"):
        mask = out.stay_id == stay_id
        if not mask.any():
            continue
        row_hours = out.loc[mask, "hour"].to_numpy(dtype=float)
        lab_hours = stay_labs["lab_hour"].to_numpy(dtype=float)
        counts_by_window = {}
        for w in ROLLING_WINDOWS_H:
            counts = np.array([np.sum((lab_hours <= h) & (lab_hours > h - w)) for h in row_hours])
            counts_by_window[w] = counts
            out.loc[mask, f"lab_orders_{w}h"] = counts
        stay_duration_h = max(row_hours.max() - row_hours.min() + 1, 1.0)
        admission_mean_rate = len(stay_labs) / stay_duration_h
        current_rate_24h = counts_by_window[24] / 24.0
        ratio = np.where(admission_mean_rate > 0, current_rate_24h / admission_mean_rate, 1.0)
        out.loc[mask, "lab_order_rate_ratio"] = ratio
    return out


def add_disease_features(grid: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Joins ``capstone.disease_context`` -- what this patient is being treated for.

    Until this existed the model's only case-mix signal was ``first_careunit``, which
    ranked *second* by mean |SHAP| and was acting as an unexamined disease proxy
    ("admitted to CVICU" ~ "cardiac problem"). Making the disease axis explicit is
    what lets ``ml/evaluation/disease_leakage.py`` measure it.

    Every column this adds is patient-constant, and two of them are on opposite sides
    of a leakage line -- see ``warehouse/disease.py``'s module docstring and
    ``DISEASE_FEATURE_SETS`` below. This function adds all of them to the frame; which
    ones reach a model is ``feature_matrix_for_training``'s decision, not this one's.
    """
    cols = ", ".join(["stay_id", "dx_chapter", *CHARLSON_FLAGS, "charlson_comorbidity_index"])
    disease = conn.execute(f"select {cols} from capstone.disease_context").fetchdf()
    return grid.merge(disease, on="stay_id", how="left")


def add_static_features(grid: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    # subject_id is carried for grouping, not for modelling: it is the unit CV
    # must split on (ml/models/splits.py), and it is never added to the feature
    # column list below.
    static = conn.execute("""
        select d.stay_id, d.subject_id, d.admission_age, d.gender, i.first_careunit
        from mimiciv_derived.icustay_detail d
        join mimiciv_icu.icustays i using (stay_id)
        """).fetchdf()
    return grid.merge(static, on="stay_id", how="left")


# Columns still COMPUTED (add_severity_scores / add_lab_order_intensity keep
# producing them) but no longer given to the learned models. `news2` and
# `sofa_24hours` in particular must stay in the frame, because they are what
# `ml/models/baselines.py` reads to score the NEWS2 and SOFA baselines --
# `feature_matrix_for_training(..., include_severity_scores=True)` hands them
# over for exactly that purpose and nothing else.
SEVERITY_SCORE_COLUMNS = ["news2", "sofa_24hours"]
LAB_INTENSITY_COLUMNS = ["lab_orders_4h", "lab_orders_24h", "lab_order_rate_ratio"]

# Rolling statistics kept as features. The 4h/24h *means* were dropped: a
# carried-forward raw value is already close to the recent mean, so the 18 mean
# columns were largely a restatement of the 9 raw ones, and removing them was the
# single largest measured gain of the feature-pruning study. Std and slope survive
# because they carry variability and direction, which no raw value encodes.
ROLLING_STATS = ("std", "slope")

FEATURE_COLUMNS_BASE = [
    *CORE_VITALS,
    *[f"{v}_was_imputed" for v in CORE_VITALS],
    *[f"{v}_hours_since_last_obs" for v in CORE_VITALS],
    "has_arterial_line",
    "admission_age",
]


# --- Disease awareness ---------------------------------------------------------
# Four nested feature sets, because "make it disease-aware" has a leakage question
# buried in it that only a measurement can settle (ml/evaluation/disease_leakage.py).
#
# MIMIC's ICD codes are assigned by billing coders AFTER discharge. They describe what
# the admission turned out to be about -- which is not what a bedside model knows at
# hour 3, and for this project's composite label (death / vasopressor start / invasive
# ventilation start / ICU bounce-back) some codes describe the outcome itself. "Acute
# respiratory failure with hypoxia" as a primary diagnosis sits very close to the
# ventilation component of the label.
#
# Charlson's flags do not have that problem by construction: the index exists to score
# *pre-existing chronic* burden, so it describes the patient who walked in. That is the
# split these sets encode -- "chronic" is defensible at the bedside, "coded" needs the
# leakage study before anyone believes a number produced with it.
#
DISEASE_FEATURE_SETS: dict[str, list[str]] = {
    "none": [],
    "chronic_index_only": ["charlson_comorbidity_index"],
    "chronic": [*CHARLSON_FLAGS, "charlson_comorbidity_index"],
    "coded_only": ["dx_chapter"],
    "all": [*CHARLSON_FLAGS, "charlson_comorbidity_index", "dx_chapter"],
}

# What the study actually found (20 repeats x 5 folds, horizon 6h, identical folds):
#
#   arm                 mean AUPRC   delta    paired wins vs `none`   mean AUROC
#   none                   0.5018        --                      --      0.8198
#   chronic_index_only     0.4936   -0.0083                  3 / 20      0.8429
#   chronic                0.4879   -0.0139                  4 / 20      0.8448
#   coded_only             0.4907   -0.0111                  2 / 20      0.8042
#   all                    0.4757   -0.0262                  2 / 20      0.8273
#
# Two findings, and neither is the one you would guess.
#
# **There is no detectable leak.** `dx_chapter` on its own scores AUPRC 0.0392
# against a base rate of 0.0403 -- at, in fact fractionally below, chance. The
# discharge-coded diagnosis carries essentially no information about *which hour* a
# patient deteriorates, which is the question this model is asked. Probe 2 agrees:
# no chapter spikes on the component matching its own organ system (Respiratory x
# ventilation is 1.30, unremarkable next to Infectious x readmission at 3.46, which
# is case mix, not coding). The leakage worry that motivated the whole two-set split
# turned out to be unfounded *for this task* -- worth stating plainly, because it was
# a real risk that had to be checked rather than assumed away.
#
# **Disease features do not improve the point estimate.** No arm beats disease-blind;
# the best wins 4 of 20 paired repeats. This is the same lesson `feature_pruning.py`
# already drew -- 49 positive subjects will not support more columns -- and it is
# unsurprising that 18 patient-constant columns lose. Note the split between metrics,
# though: AUROC rises consistently (+0.025 with `chronic`) while AUPRC falls. Better
# ranking overall, slightly worse ranking at the top of the list where the positives
# are. AUPRC is the metric this project judges on, at a 4% base rate, so AUPRC wins
# the argument.
#
# **Why the default is `chronic` anyway.** The -0.0139 delta is about 5% of the width
# of AUPRC's own bootstrap CI on this cohort (0.3763-0.6319), i.e. indistinguishable
# from noise, and `chronic` is the arm the leakage probes clear unambiguously. Set
# against that: the platform is disease-aware end to end now -- per-chapter NEWS2
# cut-points (warehouse/news2.py), disease-targeted retrieval and the care-plan node
# (services/agent-orchestrator/nodes.py), and the diagnosis on the escalation email --
# and a model that alone remained blind to the diagnosis would be the odd component
# out, explaining its scores in terms no other part of the chain shares.
#
# That is a judgement, not a measurement, and it is reversible in one line: set this
# to "none" to ship the disease-blind model with the best AUPRC point estimate. The
# rest of the disease-aware chain does not depend on this constant.
DEFAULT_DISEASE_FEATURES = "chronic"

# The only disease column that is categorical rather than numeric; gbm.py needs it by
# name, and importing it from here keeps one definition rather than two lists to sync.
DISEASE_CATEGORICAL_COLUMNS = ["dx_chapter"]


def disease_feature_columns(feature_set: str = DEFAULT_DISEASE_FEATURES) -> list[str]:
    if feature_set not in DISEASE_FEATURE_SETS:
        raise ValueError(
            f"unknown disease feature set {feature_set!r} -- "
            f"expected one of {sorted(DISEASE_FEATURE_SETS)}"
        )
    return list(DISEASE_FEATURE_SETS[feature_set])


def rolling_feature_columns(
    windows: tuple[int, ...] = ROLLING_WINDOWS_H,
    stats: tuple[str, ...] = ROLLING_STATS,
) -> list[str]:
    cols = []
    for w in windows:
        for var in CORE_VITALS:
            cols += [f"{var}_{w}h_{stat}" for stat in stats]
    return cols


def build_feature_frame(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per (stay_id, hour) with the full feature set attached. Does
    NOT apply label censoring -- join against ``labels.build_labels()``'s
    output on (stay_id, hour) to get the at-risk, labelled subset.
    """
    grid = load_hourly_grid_raw(conn)
    grid = add_rolling_features(grid)
    grid = add_severity_scores(grid, conn)
    grid = add_arterial_line_feature(grid, conn)
    grid = add_lab_order_intensity(grid, conn)
    grid = add_disease_features(grid, conn)
    grid = add_static_features(grid, conn)
    return grid


def feature_matrix_for_training(
    features: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_col: str,
    include_demographics: bool = True,
    include_severity_scores: bool = False,
    disease_features: str = DEFAULT_DISEASE_FEATURES,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Inner-join the full feature frame to the at-risk labelled rows, and
    return (X, y, groups) ready for ``models.splits``. ``groups`` is
    ``subject_id`` -- grouped CV must never let one patient span train and test.

    **This used to return ``stay_id``, and that was a leak.** ``models.splits``
    has always documented subject-level grouping as the requirement, and shipped
    a ``group_key()`` helper to supply it, but nothing ever called that helper
    with a subject id -- so every cross-validated number this project produced
    was grouped by stay. In this cohort that is not a technicality: 21 of 93
    subjects in the at-risk set have more than one ICU stay, and they carry 45%
    of the rows and 44% of the positives, so nearly half the signal came from
    patients who could sit in the training and test folds at once.
    ``ml/evaluation/reliability.py`` measures the cost at **+0.0499 AUPRC** of
    optimism at the 6h horizon (it was +0.0159 before ``gender`` rejoined the
    feature set -- a patient-constant feature makes a stay-level split leak
    more). Returning the subject removes it.

    ``include_demographics`` controls ``gender`` only, and defaults to **True**.
    That default has been both values, and the history matters more than the
    current setting:

    * Originally included unconditionally, where it ranked third by mean |SHAP|,
      above most vitals, with no subgroup analysis anywhere in the project.
    * Review finding **F4** dropped it. Under the then-current CV the 20-repeat
      ablation gave AUPRC 0.5065 with against 0.4953 without, winning in only
      13 of 20 repeats -- a coin flip, and not worth the fairness objection that
      carrying a protected attribute invites.
    * Correcting the CV grouping from ``stay_id`` to ``subject_id`` (see above)
      moved that ablation to **15 of 20** repeats, 0.4715 against 0.4612, which
      clears ``fairness.AblationResult.demographics_earn_their_place``'s 75%
      bar. F4 is therefore **reversed** and the column is carried.
    * That decision was taken when an ECG-fusion variant still existed and
      `gender` changed which variant won the primary horizon. ECG has since
      been removed from the project entirely (it measurably hurt, and a
      post-discharge patient has no 12-lead ECG), so there is now one model and
      that particular tie-break no longer arises. The 15/20 ablation result
      above is the one the decision rests on; re-check it against the committed
      report rather than any figure quoted from the two-variant era.

    Read the reversal with the caution it deserves: the win count moved
    13 -> 15 because the *evaluation* was corrected, not because new evidence
    about the feature arrived, and 15/20 was exactly the threshold. The delta
    (+0.0103) remains far inside a bootstrap CI roughly twenty-five times its
    size. The association in this cohort is real -- 66.2% of male stays reach a
    composite event against 42.9% of female (Fisher OR 2.62, p=0.007) -- but in
    140 stays an effect that size is also what sampling noise looks like.
    Because the model now *uses* the attribute it is audited on,
    ``ml/evaluation/fairness.py``'s subgroup audit stops being a diagnostic and
    becomes load-bearing.

    ``ml/evaluation/fairness.py`` also passes False to run the without-gender
    arm of the ablation -- one merge path, so the audited rows and the modelled
    rows can never drift apart.
    """
    merged = labels_df[["stay_id", "hour", label_col]].merge(
        features, on=["stay_id", "hour"], how="inner"
    )

    cols = list(FEATURE_COLUMNS_BASE) + rolling_feature_columns()
    cols = cols + disease_feature_columns(disease_features)
    if include_severity_scores:
        cols = cols + list(SEVERITY_SCORE_COLUMNS)

    x = merged[cols].copy()
    if include_demographics:
        x["gender"] = merged["gender"]
    # first_careunit stays: it is clinical context (which ICU a patient is in), not a
    # protected demographic attribute. admission_age likewise -- age is a validated
    # covariate in SAPS-II and APACHE, not a proxy.
    x["first_careunit"] = merged["first_careunit"]
    y = merged[label_col]
    groups = merged["subject_id"]
    return x, y, groups
