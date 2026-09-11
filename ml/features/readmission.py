"""30-day readmission: cohort, label and features, at admission granularity.

**Why this module exists, and what it replaces.** `ml/evaluation/secondary_whole_stay.py`
already predicts 30-day readmission and reports **AUROC 0.452** -- at or below chance --
describing that as "what 'no usable signal at this n' looks like". That reading is only
half right, and the other half is fixable:

* **It asks the wrong question.** Its eight features are `admission_age`, `max_news2_24h`,
  `max_sofa_24h`, `mean_hr_24h`, `mean_rr_24h`, `mean_spo2_24h`, `ever_vasopressor`,
  `ever_ventilation` -- every one of them ICU physiology. Nothing about who the patient
  is, what they were treated for, how often they have been admitted before, or where
  they were discharged to. Readmission is a *care-transition* outcome; ICU vitals
  describe an episode that has already ended by the time the prediction is made.
* **It uses a third of the available label.** Restricting to the 140 ICU stays' own
  admissions leaves 113 rows with 22 positives. The cohort that actually matches the
  outcome -- every live discharge -- has **260 rows and 53 positives**.

So this module rebuilds the task from the cohort definition up.

## Prediction time is discharge, and that changes what is legal

For the hourly deterioration model, discharge-coded ICD is leak-suspect: the codes do
not exist at hour 3 (finding E22, `ml/evaluation/disease_leakage.py`). Here the
prediction is made **at discharge**, after coding, so the *entire* coded record is
legitimate -- full diagnoses, procedures, the discharge medication list, length of stay
and discharge disposition. Same data, opposite verdict, and the discriminator is
prediction time rather than the feature.

The one thing that remains illegal is anything derived from the *next* admission, which
is the label.

## Cohort definition (CMS-style), and the two competing risks

Start from every admission with `hospital_expire_flag = 0` (260; the flag matches
`discharge_location = 'DIED'` exactly, so deaths filter cleanly). Then exclude:

* **Hospice discharges (5).** A patient discharged to hospice is not expected to return,
  and scoring them as a negative rewards a model for predicting "no readmission" on
  patients who were never candidates. CMS's readmission measures exclude them for the
  same reason.
* **Death within 30 days without readmission (6).** A patient who dies at home cannot
  be readmitted, so this is a competing risk, not a negative outcome. `patients.dod`
  makes it identifiable, which is why it is handled rather than assumed away.

Result: **252 admissions, 53 positives (21.0%)** -- against the hourly task's 4.0%, and
with the label counted once per admission rather than once per patient-hour.

## The censoring this cannot fix, stated plainly

85 of the live discharges are the last admission recorded for that patient. MIMIC dates
are shifted per patient, so there is no global observation window to censor against, and
"no subsequent admission in the dataset" is the only signal available. Those rows are
therefore treated as negatives, which is standard practice on MIMIC and is also a known
source of **label noise in one direction**: some are genuinely never readmitted, some
were readmitted after the record ends. It cannot be resolved with this data and is
reported rather than hidden.

## Grouping is not optional here

95 patients contribute 252 admissions -- mean 2.7, and one patient contributes 20. A
row-level split would put the same patient on both sides of the fold in most draws, and
`n_prior_admissions` would then be close to a patient identifier. Every evaluation of
this task must group on `subject_id` (see `ml/models/splits.py`, and the +0.0499 AUPRC
of optimism the hourly task paid for getting this wrong).
"""

from __future__ import annotations

import duckdb
import pandas as pd

READMISSION_WINDOW_DAYS = 30
RECENT_UTILISATION_DAYS = 365

# Excluded from the cohort as competing risks -- see the module docstring.
HOSPICE_DISPOSITIONS = ("HOSPICE",)

CHARLSON_FLAGS_KEPT = [
    "congestive_heart_failure",
    "chronic_pulmonary_disease",
    "renal_disease",
    "diabetes_with_cc",
    "malignant_cancer",
]

# Feature families, so the study can ablate by family rather than by column and say
# *which kind* of information carries the task.
FEATURE_FAMILIES: dict[str, list[str]] = {
    "demographics": ["age", "gender", "insurance", "marital_status"],
    "index_stay": [
        "los_days",
        "admission_type",
        "admission_location",
        "came_via_ed",
        "had_icu_stay",
        "icu_los_days",
    ],
    "disposition": ["discharge_location"],
    "history": ["n_prior_admissions", "days_since_last_discharge", "n_prior_admissions_365d"],
    "comorbidity": ["charlson_comorbidity_index", *CHARLSON_FLAGS_KEPT],
    "diagnosis": ["dx_chapter", "n_diagnoses"],
    "treatment_intensity": ["n_procedures", "n_distinct_drugs"],
}

# A deliberately small set, for the same reason `ml/evaluation/feature_pruning.py`
# exists: 24 columns against 53 positives is over-parameterised, and this project has
# already measured twice that nearly any reduction pays. This is the **LACE index**
# (Length of stay, Acuity of admission, Comorbidity, Emergency-department use) -- the
# standard validated readmission instrument -- plus the two facts notebook 02 found
# carry the most signal here: discharge disposition and recent prior utilisation.
LACE_PLUS_FEATURES = [
    "los_days",  # L
    "admission_type",  # A -- acuity
    "charlson_comorbidity_index",  # C
    "came_via_ed",  # E
    "discharge_location",
    "n_prior_admissions_365d",
    "age",
]

# The eight features `secondary_whole_stay.py` currently uses, kept verbatim so the
# comparison against the new set is like-for-like rather than a re-description.
ICU_PHYSIOLOGY_FEATURES = [
    "age",
    "max_news2_24h",
    "max_sofa_24h",
    "mean_hr_24h",
    "mean_rr_24h",
    "mean_spo2_24h",
    "ever_vasopressor",
    "ever_ventilation",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "insurance",
    "marital_status",
    "admission_type",
    "admission_location",
    "discharge_location",
    "dx_chapter",
]


def all_features() -> list[str]:
    return [c for cols in FEATURE_FAMILIES.values() for c in cols]


def build_cohort(conn: duckdb.DuckDBPyConnection, strict: bool = True) -> pd.DataFrame:
    """One row per live discharge, with the label and the exclusion reasons attached.

    ``strict=False`` keeps the hospice and died-before-readmission rows, which is the
    naive definition (260 rows). The study reports both so the effect of the exclusion
    is visible rather than asserted.
    """
    adm = conn.execute("""
        select a.subject_id, a.hadm_id, a.admittime, a.dischtime, a.hospital_expire_flag,
               a.admission_type, a.admission_location, a.discharge_location,
               a.insurance, a.marital_status, a.edregtime, p.dod, p.gender, p.anchor_age
        from mimiciv_hosp.admissions a
        join mimiciv_hosp.patients p using (subject_id)
        order by a.subject_id, a.admittime
    """).fetchdf()
    for col in ("admittime", "dischtime", "dod", "edregtime"):
        adm[col] = pd.to_datetime(adm[col])

    # History features are computed over the FULL admission sequence, including
    # in-hospital deaths, because a prior admission counts whether or not the patient
    # later died -- filtering to live discharges first would undercount it.
    adm["n_prior_admissions"] = adm.groupby("subject_id").cumcount()
    adm["prev_dischtime"] = adm.groupby("subject_id").dischtime.shift(1)
    adm["days_since_last_discharge"] = (
        adm.admittime - adm.prev_dischtime
    ).dt.total_seconds() / 86400
    adm["n_prior_admissions_365d"] = [
        int(
            (
                (adm.subject_id == r.subject_id)
                & (adm.admittime < r.admittime)
                & (
                    adm.admittime >= r.admittime - pd.Timedelta(days=RECENT_UTILISATION_DAYS)  # type: ignore[operator]
                )
            ).sum()
        )
        for r in adm.itertuples()
    ]

    adm["next_admittime"] = adm.groupby("subject_id").admittime.shift(-1)
    adm["days_to_next"] = (adm.next_admittime - adm.dischtime).dt.total_seconds() / 86400
    adm["readmit_30d"] = (
        adm.days_to_next.notna() & adm.days_to_next.le(READMISSION_WINDOW_DAYS)
    ).astype(int)
    adm["days_to_death"] = (adm.dod - adm.dischtime).dt.total_seconds() / 86400

    cohort = adm[adm.hospital_expire_flag == 0].copy()
    cohort["excluded_hospice"] = cohort.discharge_location.isin(HOSPICE_DISPOSITIONS)
    cohort["excluded_died_before_readmit"] = cohort.days_to_death.le(
        READMISSION_WINDOW_DAYS
    ) & cohort.readmit_30d.eq(0)
    # Right-censoring flag: no later admission exists in the record at all. Kept as a
    # column rather than dropped so the study can measure how much the assumption costs.
    cohort["is_last_admission"] = cohort.next_admittime.isna()

    if strict:
        cohort = cohort[~cohort.excluded_hospice & ~cohort.excluded_died_before_readmit]
    return cohort.reset_index(drop=True)


def add_features(conn: duckdb.DuckDBPyConnection, cohort: pd.DataFrame) -> pd.DataFrame:
    """Attach every family in ``FEATURE_FAMILIES``. All of it is known at discharge."""
    out = cohort.copy()
    out["age"] = out.anchor_age.astype(float)
    out["los_days"] = (out.dischtime - out.admittime).dt.total_seconds() / 86400
    out["came_via_ed"] = out.edregtime.notna().astype(int)

    icu = conn.execute("""
        select hadm_id, count(*) as n_icu_stays, sum(los) as icu_los_days
        from mimiciv_icu.icustays group by hadm_id
    """).fetchdf()
    out = out.merge(icu, on="hadm_id", how="left")
    out["had_icu_stay"] = out.n_icu_stays.notna().astype(int)
    out["icu_los_days"] = out.icu_los_days.fillna(0.0)

    dx = conn.execute("""
        select hadm_id, count(*) as n_diagnoses from mimiciv_hosp.diagnoses_icd group by hadm_id
    """).fetchdf()
    out = out.merge(dx, on="hadm_id", how="left")
    out["n_diagnoses"] = out.n_diagnoses.fillna(0).astype(int)

    # dx_chapter comes from capstone.disease_context, which is keyed by ICU stay; an
    # admission with no ICU stay has no row there, so it is recomputed here from the
    # primary diagnosis directly. The chapter mapper is imported, never re-implemented.
    from warehouse.disease import UNKNOWN_CHAPTER, icd_chapter

    primary = conn.execute("""
        select hadm_id, icd_code, icd_version from mimiciv_hosp.diagnoses_icd where seq_num = 1
    """).fetchdf()
    primary["dx_chapter"] = [
        # itertuples() attributes are a wide Scalar union under pandas-stubs.
        icd_chapter(r.icd_code, r.icd_version)  # type: ignore[arg-type]
        for r in primary.itertuples()
    ]
    out = out.merge(primary[["hadm_id", "dx_chapter"]], on="hadm_id", how="left")
    out["dx_chapter"] = out.dx_chapter.fillna(UNKNOWN_CHAPTER)

    proc = conn.execute("""
        select hadm_id, count(*) as n_procedures from mimiciv_hosp.procedures_icd group by hadm_id
    """).fetchdf()
    out = out.merge(proc, on="hadm_id", how="left")
    out["n_procedures"] = out.n_procedures.fillna(0).astype(int)

    drugs = conn.execute("""
        select hadm_id, count(distinct drug) as n_distinct_drugs
        from mimiciv_hosp.prescriptions group by hadm_id
    """).fetchdf()
    out = out.merge(drugs, on="hadm_id", how="left")
    out["n_distinct_drugs"] = out.n_distinct_drugs.fillna(0).astype(int)

    charlson_cols = ", ".join(CHARLSON_FLAGS_KEPT)
    ch = conn.execute(f"""
        select hadm_id, {charlson_cols}, charlson_comorbidity_index from mimiciv_derived.charlson
    """).fetchdf()
    out = out.merge(ch, on="hadm_id", how="left")

    for col in CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("object").fillna("UNKNOWN").astype("category")
    return out


def add_icu_physiology_features(
    conn: duckdb.DuckDBPyConnection, frame: pd.DataFrame
) -> pd.DataFrame:
    """The eight columns `secondary_whole_stay.py` uses, so the two feature sets can be
    compared on the *same* rows and folds.

    An admission with no ICU stay has none of them, which is itself part of the
    comparison: the old model could only ever score the 113 ICU admissions, while this
    cohort has 252.
    """
    out = frame.copy()
    news2 = conn.execute("""
        select i.hadm_id, max(n.news2) as max_news2_24h
        from capstone.news2 n join mimiciv_icu.icustays i using (stay_id)
        where n.hour < 24 group by i.hadm_id
    """).fetchdf()
    sofa = conn.execute("""
        select i.hadm_id, max(s.sofa_24hours) as max_sofa_24h
        from mimiciv_derived.sofa s join mimiciv_icu.icustays i using (stay_id)
        where s.hr < 24 group by i.hadm_id
    """).fetchdf()
    vitals = conn.execute("""
        select i.hadm_id,
               avg(g.hr) as mean_hr_24h, avg(g.rr) as mean_rr_24h, avg(g.spo2) as mean_spo2_24h
        from capstone.hourly_grid g join mimiciv_icu.icustays i using (stay_id)
        where g.hour < 24 group by i.hadm_id
    """).fetchdf()
    vaso = conn.execute("""
        select distinct i.hadm_id, 1 as ever_vasopressor
        from mimiciv_derived.vasoactive_agent v join mimiciv_icu.icustays i using (stay_id)
    """).fetchdf()
    vent = conn.execute("""
        select distinct i.hadm_id, 1 as ever_ventilation
        from mimiciv_derived.ventilation v join mimiciv_icu.icustays i using (stay_id)
        where v.ventilation_status = 'InvasiveVent'
    """).fetchdf()
    for frame_to_merge in (news2, sofa, vitals, vaso, vent):
        out = out.merge(frame_to_merge, on="hadm_id", how="left")
    out["ever_vasopressor"] = out.ever_vasopressor.fillna(0).astype(int)
    out["ever_ventilation"] = out.ever_ventilation.fillna(0).astype(int)
    return out


def build_task(
    conn: duckdb.DuckDBPyConnection, strict: bool = True
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns (features, y, groups). ``groups`` is ``subject_id`` -- see the docstring."""
    cohort = build_cohort(conn, strict=strict)
    frame = add_features(conn, cohort)
    frame = add_icu_physiology_features(conn, frame)
    return frame, frame.readmit_30d, frame.subject_id


def cohort_summary(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The cohort table the report opens with: how each exclusion moves n and the rate."""
    naive = build_cohort(conn, strict=False)
    strict = build_cohort(conn, strict=True)
    icu_only = naive[
        naive.hadm_id.isin(
            conn.execute("select distinct hadm_id from mimiciv_icu.icustays").fetchdf().hadm_id
        )
    ]
    rows = [
        ("all live discharges", len(naive), int(naive.readmit_30d.sum())),
        (
            "  less hospice",
            int((~naive.excluded_hospice).sum()),
            int(naive.loc[~naive.excluded_hospice, "readmit_30d"].sum()),
        ),
        (
            "  less died <=30d before readmission (SHIPPED)",
            len(strict),
            int(strict.readmit_30d.sum()),
        ),
        ("previous cohort: ICU admissions only", len(icu_only), int(icu_only.readmit_30d.sum())),
    ]
    out = pd.DataFrame(rows, columns=["cohort", "admissions", "readmitted_30d"])
    out["rate"] = out.readmitted_30d / out.admissions
    return out


def censoring_summary(cohort: pd.DataFrame) -> dict:
    return {
        "admissions": len(cohort),
        "positives": int(cohort.readmit_30d.sum()),
        "rate": float(cohort.readmit_30d.mean()),
        "patients": int(cohort.subject_id.nunique()),
        "admissions_per_patient_mean": float(cohort.groupby("subject_id").size().mean()),
        "admissions_per_patient_max": int(cohort.groupby("subject_id").size().max()),
        "right_censored_last_admissions": int(cohort.is_last_admission.sum()),
        "right_censored_fraction": float(cohort.is_last_admission.mean()),
    }


def as_categorical(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def present_categoricals(x: pd.DataFrame) -> list[str]:
    return [c for c in CATEGORICAL_COLUMNS if c in x.columns]


__all__ = [
    "CATEGORICAL_COLUMNS",
    "FEATURE_FAMILIES",
    "ICU_PHYSIOLOGY_FEATURES",
    "LACE_PLUS_FEATURES",
    "READMISSION_WINDOW_DAYS",
    "add_features",
    "add_icu_physiology_features",
    "all_features",
    "as_categorical",
    "build_cohort",
    "build_task",
    "censoring_summary",
    "cohort_summary",
    "present_categoricals",
]
