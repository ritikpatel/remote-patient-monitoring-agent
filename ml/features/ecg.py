"""ECG fusion features via neurokit2 (PROJECT_PLAN.md section 11, E9).

E9: 92/100 cohort patients have a linked 12-lead 500Hz 10s ECG study, but the
MIMIC-IV-ECG demo carries no ``machine_measurements`` labels -- there is no
cardiologist-assigned rate/rhythm/interval to simply read off. Every feature
here is *derived* from the waveform with neurokit2, not looked up.

Joined on subject_id + nearest **preceding** ``ecg_time`` (a study taken after
the hour being scored cannot inform that hour's prediction -- that would be
future leakage), within a bounded lookback window, since ECGs are episodic
rather than continuous (E9): most patient-hours will have no recent study at
all, which is itself represented as a feature (``ecg_hours_since``) rather
than silently imputed to "normal."

The rhythm classification is a **rate/variability heuristic, not a clinical
arrhythmia diagnosis** -- stated explicitly because there is no ground truth
to validate one against in this dataset (E9). It flags rate-based tachycardia/
bradycardia and an RR-interval coefficient-of-variation threshold as an
"irregular" screen (a common non-diagnostic AFib-suspicion heuristic), and
nothing stronger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ECG_ROOT = (
    REPO_ROOT / "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1"
)
DEFAULT_RECORD_LIST = DEFAULT_ECG_ROOT / "record_list.csv"
DEFAULT_MAX_LOOKBACK_H = 72.0
LEAD = "II"


def load_record_list(
    path: Path = DEFAULT_RECORD_LIST, cohort_subject_ids: set[int] | None = None
) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ecg_time"])
    if cohort_subject_ids is not None:
        df = df[df.subject_id.isin(cohort_subject_ids)]
    return df.reset_index(drop=True)


def extract_features_for_record(record_path: Path, lead: str = LEAD) -> dict[str, float] | None:
    """Rate, QRS duration, QTc (Bazett), and a rhythm-regularity heuristic
    for one 10s study. Returns None if the lead is missing or the waveform
    is too degraded for neurokit2 to find R-peaks (a handful of studies are
    expected to fail this -- see ``build_ecg_feature_table``'s failure count).
    """
    record = wfdb.rdrecord(str(record_path))
    if lead not in record.sig_name:
        return None
    signal = record.p_signal[:, record.sig_name.index(lead)]
    fs = record.fs

    try:
        signals, info = nk.ecg_process(signal, sampling_rate=fs)
        _delin_signals, delin_info = nk.ecg_delineate(
            signal, rpeaks=info, sampling_rate=fs, method="dwt"
        )
    except Exception:  # neurokit2 raises a range of signal-specific errors
        return None

    r_onsets = np.asarray(delin_info["ECG_R_Onsets"], dtype=float)
    r_offsets = np.asarray(delin_info["ECG_R_Offsets"], dtype=float)
    t_offsets = np.asarray(delin_info["ECG_T_Offsets"], dtype=float)
    r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=float)

    qrs_ms = _paired_mean_diff_ms(r_onsets, r_offsets, fs)
    qt_ms = _paired_mean_diff_ms(r_onsets, t_offsets, fs)

    rr_intervals_s = np.diff(r_peaks) / fs if len(r_peaks) > 1 else np.array([])
    hr_mean = float(np.nanmean(signals["ECG_Rate"]))
    hr_std = float(np.nanstd(signals["ECG_Rate"]))

    qtc_ms = float("nan")
    if qt_ms is not None and len(rr_intervals_s) > 0:
        mean_rr_s = float(np.mean(rr_intervals_s))
        if mean_rr_s > 0:
            qtc_ms = qt_ms / np.sqrt(mean_rr_s)  # Bazett's formula

    rr_cv = (
        float(np.std(rr_intervals_s) / np.mean(rr_intervals_s))
        if len(rr_intervals_s) > 1 and np.mean(rr_intervals_s) > 0
        else 0.0
    )

    return {
        "ecg_hr_mean": hr_mean,
        "ecg_hr_std": hr_std,
        "ecg_qrs_ms": qrs_ms if qrs_ms is not None else float("nan"),
        "ecg_qtc_ms": qtc_ms,
        "ecg_rhythm_irregular": float(rr_cv > 0.15),
        "ecg_quality_mean": float(np.nanmean(signals["ECG_Quality"])),
    }


def _paired_mean_diff_ms(onsets: np.ndarray, offsets: np.ndarray, fs: float) -> float | None:
    n = min(len(onsets), len(offsets))
    if n == 0:
        return None
    diffs = offsets[:n] - onsets[:n]
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return None
    return float(np.mean(diffs) / fs * 1000.0)


def build_ecg_feature_table(
    record_list: pd.DataFrame, ecg_root: Path = DEFAULT_ECG_ROOT
) -> tuple[pd.DataFrame, int]:
    """Returns (features_df, n_failed). ``features_df`` has one row per study
    that a feature could be extracted from: subject_id, ecg_time, and the
    ecg_* columns.
    """
    rows = []
    n_failed = 0
    for row in record_list.itertuples(index=False):
        features = extract_features_for_record(ecg_root / str(row.path))
        if features is None:
            n_failed += 1
            continue
        record: dict[str, Any] = dict(features)
        record["subject_id"] = row.subject_id
        record["ecg_time"] = row.ecg_time
        rows.append(record)
    return pd.DataFrame(rows), n_failed


def attach_nearest_ecg(
    grid_with_abs_time: pd.DataFrame,
    ecg_features: pd.DataFrame,
    max_lookback_h: float = DEFAULT_MAX_LOOKBACK_H,
) -> pd.DataFrame:
    """``grid_with_abs_time`` needs columns stay_id, hour, subject_id,
    row_abs_time. Returns stay_id, hour, and the ecg_* columns plus
    ``ecg_hours_since`` (NaN where no study is within the lookback window).
    """
    left = grid_with_abs_time.sort_values("row_abs_time").reset_index(drop=True).copy()
    right = ecg_features.sort_values("ecg_time").reset_index(drop=True).copy()
    # DuckDB returns subject_id as int32; record_list.csv reads it as int64 --
    # merge_asof's `by` key requires matching dtypes.
    left["subject_id"] = left["subject_id"].astype("int64")
    right["subject_id"] = right["subject_id"].astype("int64")

    merged = pd.merge_asof(
        left,
        right,
        left_on="row_abs_time",
        right_on="ecg_time",
        by="subject_id",
        direction="backward",
        tolerance=pd.Timedelta(hours=max_lookback_h),
    )
    merged["ecg_hours_since"] = (
        merged["row_abs_time"] - merged["ecg_time"]
    ).dt.total_seconds() / 3600.0
    # "ecg_time" itself is the join key, not a feature -- excluded here even
    # though it also starts with "ecg_".
    ecg_cols = [c for c in ecg_features.columns if c.startswith("ecg_") and c != "ecg_time"]
    return merged[["stay_id", "hour", *ecg_cols, "ecg_hours_since"]]
