"""Tests for the 30-day readmission cohort, label and features.

The risk on this task is not a crash, it is a *plausible wrong number*. A label that
counts a death as a non-readmission, a feature computed from the next admission, or a
row-level split on a cohort where one patient contributes 20 rows all produce output
that looks entirely reasonable. These tests pin the things that would otherwise fail
silently.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from ml.features import readmission as rd

WAREHOUSE = Path(__file__).resolve().parent.parent.parent / "warehouse" / "mimic4_demo.db"
pytestmark = pytest.mark.skipif(not WAREHOUSE.exists(), reason="warehouse not built")


@pytest.fixture(scope="module")
def conn():
    c = duckdb.connect(str(WAREHOUSE), read_only=True)
    yield c
    c.close()


@pytest.fixture(scope="module")
def cohort(conn):
    return rd.build_cohort(conn, strict=True)


@pytest.fixture(scope="module")
def task(conn):
    return rd.build_task(conn)


class TestCohort:
    def test_no_in_hospital_deaths_are_in_the_cohort(self, cohort):
        """A patient who died in hospital cannot be readmitted. `hospital_expire_flag`
        matches `discharge_location = 'DIED'` exactly in this database, so this is a
        clean filter rather than an approximation."""
        assert (cohort.hospital_expire_flag == 0).all()
        assert not (cohort.discharge_location == "DIED").any()

    def test_hospice_discharges_are_excluded_as_a_competing_risk(self, cohort):
        """Scoring a hospice discharge as a negative rewards the model for predicting
        'no readmission' on a patient who was never a candidate. CMS excludes them."""
        assert not cohort.discharge_location.isin(rd.HOSPICE_DISPOSITIONS).any()

    def test_death_within_the_window_without_readmission_is_excluded(self, cohort):
        """A patient who dies at home is a competing risk, not a negative outcome."""
        died_early = cohort.days_to_death.le(rd.READMISSION_WINDOW_DAYS)
        assert not (died_early & cohort.readmit_30d.eq(0)).any()

    def test_a_patient_readmitted_before_dying_is_kept_as_a_positive(self, conn):
        """The exclusion must not swallow genuine positives: readmission first, death
        after, is a real readmission."""
        naive = rd.build_cohort(conn, strict=False)
        readmitted_then_died = naive[
            naive.readmit_30d.eq(1) & naive.days_to_death.le(rd.READMISSION_WINDOW_DAYS)
        ]
        strict = rd.build_cohort(conn, strict=True)
        assert set(readmitted_then_died.hadm_id) <= set(strict.hadm_id)

    def test_the_strict_exclusions_never_remove_a_positive(self, conn):
        naive = rd.build_cohort(conn, strict=False)
        strict = rd.build_cohort(conn, strict=True)
        assert int(strict.readmit_30d.sum()) == int(naive.readmit_30d.sum())
        assert len(strict) < len(naive), "the exclusions should remove some negatives"

    def test_the_cohort_is_larger_than_the_icu_only_one_it_replaces(self, conn, cohort):
        """The whole point of the rebuild: readmission is an outcome of a
        hospitalisation, not of an ICU stay."""
        icu_hadm = (
            conn.execute("select distinct hadm_id from mimiciv_icu.icustays").fetchdf().hadm_id
        )
        icu_only = cohort[cohort.hadm_id.isin(icu_hadm)]
        assert len(cohort) > 2 * len(icu_only) * 0.9  # ~252 vs ~113
        assert int(cohort.readmit_30d.sum()) > 2 * int(icu_only.readmit_30d.sum()) * 0.9

    def test_one_row_per_admission(self, cohort):
        assert cohort.hadm_id.is_unique


class TestLabel:
    def test_the_label_matches_a_hand_computed_window(self, cohort):
        """Recomputes the label from `days_to_next` directly, so an off-by-one in the
        window or a unit error in the timedelta cannot pass."""
        expected = cohort.days_to_next.notna() & cohort.days_to_next.le(rd.READMISSION_WINDOW_DAYS)
        assert (cohort.readmit_30d.astype(bool) == expected).all()

    def test_a_readmission_after_the_window_is_a_negative(self, cohort):
        late = cohort[cohort.days_to_next.gt(rd.READMISSION_WINDOW_DAYS)]
        assert (late.readmit_30d == 0).all()

    def test_the_next_admission_always_follows_the_discharge(self, cohort):
        """A negative `days_to_next` would mean overlapping admissions, which would make
        the label meaningless."""
        known = cohort[cohort.days_to_next.notna()]
        assert (known.days_to_next >= 0).all()

    def test_right_censored_rows_are_flagged_not_silently_dropped(self, cohort):
        """30% of the cohort is the patient's last recorded admission. Treating them as
        negatives is the only option on MIMIC, but it must be visible."""
        assert cohort.is_last_admission.any()
        assert (cohort.loc[cohort.is_last_admission, "readmit_30d"] == 0).all()


class TestFeatures:
    def test_no_feature_is_derived_from_the_next_admission(self, task):
        """The one thing that stays illegal at discharge: anything about the future.
        `days_to_next` and `next_admittime` are label machinery and must never appear
        in the modelled column set."""
        x, _, _ = task
        leaks = {"days_to_next", "next_admittime", "readmit_30d", "is_last_admission"}
        assert not leaks & set(rd.all_features())
        assert not leaks & set(rd.LACE_PLUS_FEATURES)
        assert not leaks & set(rd.ICU_PHYSIOLOGY_FEATURES)

    def test_every_declared_feature_actually_exists(self, task):
        x, _, _ = task
        for name, cols in [
            ("all_features", rd.all_features()),
            ("LACE_PLUS", rd.LACE_PLUS_FEATURES),
            ("ICU_PHYSIOLOGY", rd.ICU_PHYSIOLOGY_FEATURES),
        ]:
            missing = [c for c in cols if c not in x.columns]
            assert not missing, f"{name}: {missing}"

    def test_prior_admission_count_is_zero_for_a_first_admission(self, conn):
        cohort = rd.build_cohort(conn, strict=False)
        first = cohort.sort_values(["subject_id", "admittime"]).groupby("subject_id").first()
        assert (first.n_prior_admissions == 0).all()

    def test_prior_admission_count_increases_within_a_patient(self, conn):
        cohort = rd.build_cohort(conn, strict=False).sort_values(["subject_id", "admittime"])
        for _sid, g in cohort.groupby("subject_id"):
            assert g.n_prior_admissions.is_monotonic_increasing

    def test_recent_utilisation_never_exceeds_total_prior_admissions(self, conn):
        """A 365-day window is a subset of all prior admissions; if it ever exceeded
        the total, the window filter is wrong."""
        cohort = rd.build_cohort(conn, strict=False)
        assert (cohort.n_prior_admissions_365d <= cohort.n_prior_admissions).all()

    def test_an_admission_with_no_icu_stay_is_marked_and_not_dropped(self, task):
        """The old model could only address the 113 ICU admissions. This cohort keeps
        the rest, with `had_icu_stay = 0` rather than a missing row."""
        x, _, _ = task
        assert (x.had_icu_stay == 0).any()
        assert (x.loc[x.had_icu_stay == 0, "icu_los_days"] == 0).all()

    def test_categoricals_are_category_dtype_for_lightgbm(self, task):
        x, _, _ = task
        for col in rd.present_categoricals(x):
            assert isinstance(x[col].dtype, pd.CategoricalDtype), col

    def test_the_parsimonious_set_is_a_strict_subset_of_the_full_one(self):
        """LACE+ must be comparable to the full set, not a different experiment."""
        assert set(rd.LACE_PLUS_FEATURES) < set(rd.all_features())

    def test_feature_families_partition_the_full_set(self):
        """No column may sit in two families, or leave-one-family-out would leave it in."""
        flat = [c for cols in rd.FEATURE_FAMILIES.values() for c in cols]
        assert len(flat) == len(set(flat))
        assert set(flat) == set(rd.all_features())


class TestGrouping:
    def test_groups_are_subjects_not_admissions(self, task):
        """92 patients contribute 252 admissions, one of them 20. A row-level split
        would put the same patient on both sides of nearly every fold, and
        `n_prior_admissions` would act as a patient identifier."""
        _x, y, groups = task
        assert groups.nunique() < len(y)
        assert groups.value_counts().max() > 1

    def test_grouped_folds_never_share_a_patient(self, task):
        from ml.models import splits

        _x, y, groups = task
        for _repeat, _fold, train_idx, test_idx in splits.repeated_grouped_stratified_splits(
            y, groups, n_splits=5, n_repeats=2
        ):
            assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])


def test_base_rate_is_far_healthier_than_the_hourly_task(task):
    """The reason this task is worth building: ~21% against the hourly deterioration
    task's 4.0%."""
    _x, y, _g = task
    assert 0.15 < y.mean() < 0.30
