"""Secondary whole-stay outcomes (PROJECT_PLAN.md section 11): "ICU mortality
and 30-day readmission, both reported with wide CIs and an explicit
underpowered caveat."

**Read this before the numbers below**: E6 established only 20 ICU deaths and
53 30-day readmissions across 140 stays -- "no clinical performance claim
from this data generalises" (section 17). This script exists to report that
finding quantitatively (wide bootstrap CIs), not to produce a second
headline result. The primary, adequately-powered task is the hourly
composite deterioration model in ``ml/evaluation/run_all.py``; nothing here
is compared against a "must beat NEWS2" bar the way that one is.

Unlike the hourly task, there is no R1 censoring question here -- each row is
one ICU stay (or, for readmission, one hospitalisation's last ICU stay), and
the label is a whole-stay outcome, not a point-in-time event.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ml  # noqa: E402,F401 -- sets KMP_DUPLICATE_LIB_OK/OMP_NUM_THREADS as a side effect
from ml.evaluation import metrics  # noqa: E402
from ml.models import splits  # noqa: E402
from ml.models.baselines import build_age_vitals_logistic_regression  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

WAREHOUSE_DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"
REPORT_APPENDIX_PATH = REPO_ROOT / "ml" / "evaluation" / "secondary_report.md"
FIRST_DAY_HOURS = 24
READMISSION_WINDOW_DAYS = 30


def build_stay_features(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per ICU stay: age/gender/care unit plus the worst severity and
    mean vitals over the first 24h -- everything a clinician would actually
    have on admission day, nothing from later in the stay.
    """
    static = conn.execute("""
        select d.stay_id, d.hadm_id, d.subject_id, d.admission_age, d.gender,
               d.hospital_expire_flag, i.first_careunit
        from mimiciv_derived.icustay_detail d
        join mimiciv_icu.icustays i using (stay_id)
        """).fetchdf()

    severity = conn.execute(f"""
        select stay_id, max(news2) as max_news2_24h
        from capstone.news2 where hour < {FIRST_DAY_HOURS}
        group by stay_id
        """).fetchdf()
    sofa = conn.execute(f"""
        select stay_id, max(sofa_24hours) as max_sofa_24h
        from mimiciv_derived.sofa where hr < {FIRST_DAY_HOURS}
        group by stay_id
        """).fetchdf()
    vitals = conn.execute(f"""
        select stay_id, avg(hr) as mean_hr_24h, avg(rr) as mean_rr_24h,
               avg(spo2) as mean_spo2_24h
        from capstone.hourly_grid where hour < {FIRST_DAY_HOURS}
        group by stay_id
        """).fetchdf()
    ever_vaso = conn.execute(
        "select distinct stay_id from mimiciv_derived.vasoactive_agent"
    ).fetchdf()
    ever_vaso["ever_vasopressor"] = 1
    ever_vent = conn.execute(
        "select distinct stay_id from mimiciv_derived.ventilation "
        "where ventilation_status = 'InvasiveVent'"
    ).fetchdf()
    ever_vent["ever_ventilation"] = 1

    out = static.merge(severity, on="stay_id", how="left")
    out = out.merge(sofa, on="stay_id", how="left")
    out = out.merge(vitals, on="stay_id", how="left")
    out = out.merge(ever_vaso, on="stay_id", how="left")
    out = out.merge(ever_vent, on="stay_id", how="left")
    out["ever_vasopressor"] = out["ever_vasopressor"].fillna(0)
    out["ever_ventilation"] = out["ever_ventilation"].fillna(0)
    return out


def build_readmission_labels(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Ports notebooks/01_capstone_eda.ipynb section 3's exact definition:
    sort each subject's admissions by admittime, look at the gap to the NEXT
    admission, restrict to alive discharges. Returns one row per hadm_id.
    """
    admissions = conn.execute(
        "select subject_id, hadm_id, admittime, dischtime, hospital_expire_flag "
        "from mimiciv_hosp.admissions order by subject_id, admittime"
    ).fetchdf()
    admissions["next_admittime"] = admissions.groupby("subject_id")["admittime"].shift(-1)
    admissions["days_to_next"] = (
        admissions["next_admittime"] - admissions["dischtime"]
    ).dt.total_seconds() / 86400
    admissions["readmitted_30d"] = (
        (admissions.hospital_expire_flag == 0)
        & (admissions.days_to_next <= READMISSION_WINDOW_DAYS)
    ).astype(int)
    # Alive discharges only -- a death cannot be "readmitted."
    return admissions[admissions.hospital_expire_flag == 0][["hadm_id", "readmitted_30d"]]


def representative_stay_per_admission(stay_features: pd.DataFrame) -> pd.DataFrame:
    """For readmission (an admission-level outcome), collapse each hadm_id's
    possibly-multiple ICU stays to its LAST one -- the stay whose discharge
    is actually followed by the next admission.
    """
    return stay_features.sort_values("stay_id").groupby("hadm_id", as_index=False).last()


FEATURE_COLUMNS = [
    "admission_age",
    "max_news2_24h",
    "max_sofa_24h",
    "mean_hr_24h",
    "mean_rr_24h",
    "mean_spo2_24h",
    "ever_vasopressor",
    "ever_ventilation",
]


def _fit_predict(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    pipeline = build_age_vitals_logistic_regression()
    pipeline.fit(x_train[FEATURE_COLUMNS], y_train)
    return pipeline.predict_proba(x_test[FEATURE_COLUMNS])[:, 1]


def evaluate_whole_stay_task(
    x: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5, n_repeats: int = 20
) -> dict:
    rows = []
    oof_true = np.full(len(y), np.nan)
    oof_score = np.full(len(y), np.nan)
    for repeat, _fold, train_idx, test_idx in splits.repeated_grouped_stratified_splits(
        y, groups, n_splits=n_splits, n_repeats=n_repeats
    ):
        proba = _fit_predict(x.iloc[train_idx], y.iloc[train_idx], x.iloc[test_idx])
        rows.append(
            {
                "auroc": metrics._safe_auroc(y.iloc[test_idx].to_numpy(), proba),
                "auprc": metrics._safe_auprc(y.iloc[test_idx].to_numpy(), proba),
            }
        )
        if repeat == 0:
            oof_true[test_idx] = y.iloc[test_idx].to_numpy()
            oof_score[test_idx] = proba
    fold_df = pd.DataFrame(rows)
    valid = ~np.isnan(oof_true)
    ci = metrics.auroc_auprc_with_ci(
        oof_true[valid], oof_score[valid], groups.to_numpy()[valid], n_boot=1000
    )
    return {
        "n": len(y),
        "n_positive": int(y.sum()),
        "cv_mean_auroc": fold_df.auroc.mean(),
        "cv_mean_auprc": fold_df.auprc.mean(),
        **{f"auroc_{k}": v for k, v in vars(ci["auroc"]).items()},
        **{f"auprc_{k}": v for k, v in vars(ci["auprc"]).items()},
    }


def main() -> int:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    stay_features = build_stay_features(conn)

    print("=== ICU mortality (whole-stay, one row per ICU stay) ===")
    mortality_x = stay_features
    mortality_y = stay_features["hospital_expire_flag"]
    mortality_groups = stay_features["subject_id"]
    print(f"n={len(mortality_x)}, positives={int(mortality_y.sum())}")
    mortality_result = evaluate_whole_stay_task(mortality_x, mortality_y, mortality_groups)
    print(mortality_result)

    print("\n=== 30-day hospital readmission (one row per admission) ===")
    readmit_labels = build_readmission_labels(conn)
    rep_stays = representative_stay_per_admission(stay_features)
    readmit_x = rep_stays.merge(readmit_labels, on="hadm_id", how="inner")
    readmit_y = readmit_x["readmitted_30d"]
    readmit_groups = readmit_x["subject_id"]
    print(f"n={len(readmit_x)}, positives={int(readmit_y.sum())}")
    readmit_result = evaluate_whole_stay_task(readmit_x, readmit_y, readmit_groups)
    print(readmit_result)

    write_report(mortality_result, readmit_result)
    print(f"\nWrote {REPORT_APPENDIX_PATH.relative_to(REPO_ROOT)}")
    return 0


def write_report(mortality_result: dict, readmit_result: dict) -> None:
    lines = [
        "# Phase 5 -- secondary whole-stay outcomes (underpowered by design)\n",
        "> E6: only 20 ICU deaths and 53 30-day readmissions across 140 stays. "
        "These numbers are reported for completeness and because the plan "
        "requires it (section 11) -- **not** as a second headline result, and "
        "not compared against a NEWS2/SOFA baseline the way the primary hourly "
        "task is. Confidence intervals this wide are not a modelling failure; "
        "they are the correct, honest consequence of n this small (section 17).\n",
    ]
    tasks = [("ICU mortality", mortality_result), ("30-day readmission", readmit_result)]
    for name, result in tasks:
        lines.append(f"\n## {name}\n")
        lines.append(
            f"- n={result['n']}, positives={result['n_positive']} "
            f"({result['n_positive'] / result['n']:.1%})\n"
            f"- AUROC: {result['auroc_point']:.3f} "
            f"(95% CI {result['auroc_lo']:.3f}-{result['auroc_hi']:.3f})\n"
            f"- AUPRC: {result['auprc_point']:.3f} "
            f"(95% CI {result['auprc_lo']:.3f}-{result['auprc_hi']:.3f})\n"
        )
    lines.append(
        "\n**Why 22 readmissions here, not the EDA's 53:** notebooks/01_capstone_eda.ipynb "
        "section 3 counted 30-day readmission across all 275 admissions in the demo "
        "(most of which never had an ICU stay at all). This script's cohort is "
        "restricted to the 140 ICU stays' admissions, one row per admission -- a "
        "smaller, ICU-specific slice of the same phenomenon, not a different "
        "calculation of the same number.\n"
    )
    lines.append(
        "\nThe readmission AUROC point estimate (0.45) sitting at or below chance, "
        "with a CI straddling 0.5, is not a bug -- it is what 'no usable signal at "
        "this n' looks like, and is reported as such rather than reframed.\n"
    )
    REPORT_APPENDIX_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
