"""Per-stay disease context: what this patient is actually being treated for.

Until this module the platform was disease-blind. `ml/features/engineer.py` carried
`first_careunit` -- which ranked *second* by mean |SHAP|, above every vital except
HR -- and that column was the only thing telling the model anything about case mix.
It was doing so implicitly and unmeasurably: "admitted to CVICU" is a proxy for
"cardiac problem", learned by the model without anyone deciding it should be, and
without any of the leakage scrutiny a diagnosis feature deserves. This module makes
the disease axis explicit so it can be measured, ablated, and argued about.

Output: table `capstone.disease_context`, one row per `stay_id`.

**Two provenance classes, and the distinction is the whole point of this module.**
MIMIC's `diagnoses_icd` codes are assigned by *billing coders after discharge*. They
describe what the admission turned out to be about, which is not information a
bedside model has at hour 3 -- and for this project's composite label (death,
vasopressor start, invasive ventilation start, ICU bounce-back) some of those codes
describe the outcome itself. "Acute respiratory failure with hypoxia" as a primary
diagnosis is very close to a label leak for the ventilation component. So every
column here is tagged:

* ``CHRONIC_FEATURES`` -- the 17 Charlson comorbidity flags plus the index. These are
  *pre-existing chronic* conditions by construction (Charlson's whole purpose is
  10-year mortality from prior burden of disease), so they describe the patient who
  walked in, not what happened to them. Defensible as model inputs.
* ``ADMISSION_CODED_FEATURES`` -- ``dx_chapter``, the ICD chapter of the primary
  diagnosis. Coarse (14 chapters, not 1,472 codes) but still discharge-coded and
  therefore leak-suspect.

``ml/evaluation/disease_leakage.py`` fits the model with each set and reports the
delta, so the choice between them rests on a measurement rather than on this
docstring's opinion. Nothing here decides which set ships; it builds both.

**Why chapter and not code.** 140 ICU stays carry 102 distinct primary ICD codes --
roughly one diagnosis per patient. Any per-code feature is a unique identifier for
its patient. Rolled up to ICD chapter the cohort is at least countable
(Circulatory 41, Infectious 18, Digestive 15, Injury 15, Respiratory 13,
Neoplasm 12, then a tail of six chapters at 6 stays or fewer), which is also what
makes the per-chapter NEWS2 recalibration in `warehouse/news2.py` expressible --
with a minimum-n guard, because most of that tail cannot support its own cut-point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "mimic4_demo.db"

# The 17 Charlson comorbidity flags, as `mimiciv_derived.charlson` names them.
CHARLSON_FLAGS = [
    "myocardial_infarct",
    "congestive_heart_failure",
    "peripheral_vascular_disease",
    "cerebrovascular_disease",
    "dementia",
    "chronic_pulmonary_disease",
    "rheumatic_disease",
    "peptic_ulcer_disease",
    "mild_liver_disease",
    "diabetes_without_cc",
    "diabetes_with_cc",
    "paraplegia",
    "renal_disease",
    "malignant_cancer",
    "severe_liver_disease",
    "metastatic_solid_tumor",
    "aids",
]

# Pre-existing chronic burden -- safe to model (see module docstring).
CHRONIC_FEATURES = [*CHARLSON_FLAGS, "charlson_comorbidity_index"]

# Discharge-coded, therefore leak-suspect -- measured separately, never assumed safe.
ADMISSION_CODED_FEATURES = ["dx_chapter"]

# The label used when a stay has no primary diagnosis row at all. A real value, not a
# null: "we do not know this patient's diagnosis" is a state the model should be able
# to split on, and LightGBM handles it as just another category level.
UNKNOWN_CHAPTER = "Unknown"

# ICD-10 chapter by leading letter.
_ICD10_CHAPTERS = {
    "A": "Infectious",
    "B": "Infectious",
    "C": "Neoplasm",
    "D": "Neoplasm/Blood",
    "E": "Endocrine/Metabolic",
    "F": "Mental",
    "G": "Nervous",
    "H": "Eye/Ear",
    "I": "Circulatory",
    "J": "Respiratory",
    "K": "Digestive",
    "L": "Skin",
    "M": "Musculoskeletal",
    "N": "Genitourinary",
    "O": "Pregnancy",
    "P": "Perinatal",
    "Q": "Congenital",
    "R": "Symptoms/Signs",
    "S": "Injury",
    "T": "Injury/Poisoning",
    "V": "External causes",
    "W": "External causes",
    "X": "External causes",
    "Y": "External causes",
    "Z": "Factors/Encounter",
}

# ICD-9 chapter by numeric range (inclusive), in the conventional ordering.
_ICD9_CHAPTERS = [
    (1, 139, "Infectious"),
    (140, 239, "Neoplasm"),
    (240, 279, "Endocrine/Metabolic"),
    (280, 289, "Blood"),
    (290, 319, "Mental"),
    (320, 389, "Nervous/Sense"),
    (390, 459, "Circulatory"),
    (460, 519, "Respiratory"),
    (520, 579, "Digestive"),
    (580, 629, "Genitourinary"),
    (630, 679, "Pregnancy"),
    (680, 709, "Skin"),
    (710, 739, "Musculoskeletal"),
    (740, 759, "Congenital"),
    (760, 779, "Perinatal"),
    (780, 799, "Symptoms/Signs"),
    (800, 999, "Injury"),
]


def icd_chapter(code: str | None, version: int | None) -> str:
    """ICD-9 or ICD-10 code -> chapter name. Never raises and never returns None:
    an unmappable code is ``UNKNOWN_CHAPTER``, because a build that dies on one
    malformed billing code in a 4,506-row table is worse than one that records the
    code as unclassified and carries on.
    """
    if code is None or version is None:
        return UNKNOWN_CHAPTER
    code = str(code).strip().upper()
    if not code:
        return UNKNOWN_CHAPTER

    if int(version) == 10:
        return _ICD10_CHAPTERS.get(code[0], UNKNOWN_CHAPTER)

    # ICD-9: V-codes (supplementary factors) and E-codes (external causes) sit
    # outside the numeric ranges and are conventionally their own chapters.
    if code[0] == "V":
        return "Factors/Encounter"
    if code[0] == "E":
        return "External causes"
    head = code[:3]
    if not head.isdigit():
        return UNKNOWN_CHAPTER
    n = int(head)
    for lo, hi, name in _ICD9_CHAPTERS:
        if lo <= n <= hi:
            return name
    return UNKNOWN_CHAPTER


def build_disease_context(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per ICU stay: primary-diagnosis chapter + Charlson chronic burden.

    Joined through ``hadm_id``, so every ICU stay under one hospitalisation shares
    that admission's diagnosis and comorbidity profile -- which is correct: Charlson
    is an admission-level construct, and a patient does not acquire a different
    chronic history between two ICU stays of the same hospitalisation.
    """
    stays = conn.execute("""
        SELECT s.stay_id, s.hadm_id, s.subject_id
        FROM mimiciv_icu.icustays s
        """).fetchdf()

    primary = conn.execute("""
        SELECT d.hadm_id, d.icd_code, d.icd_version, dd.long_title
        FROM mimiciv_hosp.diagnoses_icd d
        JOIN mimiciv_hosp.d_icd_diagnoses dd
          ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
        WHERE d.seq_num = 1
        """).fetchdf()
    primary["dx_chapter"] = [
        # itertuples() attributes are a wide Scalar union under pandas-stubs.
        icd_chapter(r.icd_code, r.icd_version)  # type: ignore[arg-type]
        for r in primary.itertuples()
    ]
    primary = primary.rename(columns={"long_title": "dx_title", "icd_code": "dx_icd_code"})
    primary = primary[["hadm_id", "dx_icd_code", "dx_title", "dx_chapter"]]

    charlson_cols = ", ".join(CHARLSON_FLAGS)
    charlson = conn.execute(f"""
        SELECT hadm_id, {charlson_cols}, charlson_comorbidity_index
        FROM mimiciv_derived.charlson
        """).fetchdf()

    out = stays.merge(primary, on="hadm_id", how="left").merge(charlson, on="hadm_id", how="left")
    out["dx_chapter"] = out["dx_chapter"].fillna(UNKNOWN_CHAPTER)
    out["dx_title"] = out["dx_title"].fillna("no primary diagnosis coded")
    out["dx_icd_code"] = out["dx_icd_code"].fillna("")
    # A stay with no Charlson row has an unknown history, not a zero one. Left as
    # NaN so LightGBM splits on "unrecorded" rather than being told the patient is
    # comorbidity-free (R2/R3: absence is signal; do not impute it away).
    return out


def chapter_counts(disease: pd.DataFrame) -> pd.DataFrame:
    """Stays and subjects per chapter, descending -- the table that decides which
    chapters can support their own NEWS2 cut-point and which must fall back."""
    return (
        disease.groupby("dx_chapter")
        .agg(
            stays=("stay_id", "nunique"),
            subjects=("subject_id", "nunique"),
            distinct_codes=("dx_icd_code", "nunique"),
        )
        .sort_values("stays", ascending=False)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    conn = duckdb.connect(str(args.db))
    disease = build_disease_context(conn)

    conn.execute("CREATE SCHEMA IF NOT EXISTS capstone")
    conn.execute("DROP TABLE IF EXISTS capstone.disease_context")
    conn.register("disease_df", disease)
    conn.execute("CREATE TABLE capstone.disease_context AS SELECT * FROM disease_df")
    conn.unregister("disease_df")

    counts = chapter_counts(disease)
    print(f"capstone.disease_context: {len(disease):,} stays")
    print(f"Primary-diagnosis chapters: {len(counts)}")
    print(counts.to_string())
    n_chronic = int(disease[CHARLSON_FLAGS].notna().any(axis=1).sum())
    print(f"\nStays with a Charlson row: {n_chronic:,}/{len(disease):,}")
    print(f"Mean Charlson index: {disease.charlson_comorbidity_index.mean():.2f}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
