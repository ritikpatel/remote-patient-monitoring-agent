"""Tests for per-stay disease context and the per-disease threshold guard."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest
from warehouse import disease, news2


class TestIcdChapter:
    """The chapter mapper is the one piece of clinical coding logic this project
    writes itself rather than importing from mimic-code, so it gets direct tests
    rather than only being exercised through the build.
    """

    @pytest.mark.parametrize(
        ("code", "version", "expected"),
        [
            ("I2510", 10, "Circulatory"),  # coronary atherosclerosis, the demo's #1 dx
            ("J960", 10, "Respiratory"),
            ("A419", 10, "Infectious"),
            ("C50911", 10, "Neoplasm"),
            ("41401", 9, "Circulatory"),
            ("0389", 9, "Infectious"),
            ("99859", 9, "Injury"),
        ],
    )
    def test_maps_real_codes_to_their_chapter(self, code, version, expected):
        assert disease.icd_chapter(code, version) == expected

    def test_icd9_v_and_e_codes_get_their_own_chapters(self):
        """V- and E-codes sit outside the numeric ranges entirely. Falling through
        to UNKNOWN would silently bucket every 'encounter for chemotherapy' with
        every malformed code."""
        assert disease.icd_chapter("V5811", 9) == "Factors/Encounter"
        assert disease.icd_chapter("E8790", 9) == "External causes"

    @pytest.mark.parametrize("bad", [None, "", "   ", "!!!", "U071"])
    def test_never_raises_on_a_malformed_code(self, bad):
        """A build that dies on one bad billing code in a 4,506-row table is worse
        than one that records it unclassified and carries on. ("U" is a real ICD-10
        letter -- special purposes, e.g. U07.1 COVID-19 -- but not one of the
        chapters this project groups on, so it lands in UNKNOWN by design rather
        than being silently folded into a neighbouring chapter.)"""
        assert disease.icd_chapter(bad, 9) == disease.UNKNOWN_CHAPTER
        assert disease.icd_chapter(bad, 10) == disease.UNKNOWN_CHAPTER

    def test_icd10_z_is_a_real_chapter_not_a_parse_failure(self):
        """Z-codes are factors influencing health status (e.g. Z51.11, encounter
        for chemotherapy -- which is a real primary diagnosis in this cohort), so
        they must map to a chapter rather than to UNKNOWN."""
        assert disease.icd_chapter("Z5111", 10) == "Factors/Encounter"

    def test_a_missing_version_is_unknown_not_a_crash(self):
        assert disease.icd_chapter("I2510", None) == disease.UNKNOWN_CHAPTER


class TestDiseaseContextTable:
    def test_one_row_per_icu_stay_with_no_nulls_in_the_grouping_column(self, warehouse_conn):
        """`dx_chapter` is a LightGBM categorical and the key the per-disease
        thresholds group on. A null in it would produce a silent extra category in
        one place and a fallback in the other."""
        built = disease.build_disease_context(warehouse_conn)
        n_stays = warehouse_conn.execute("select count(*) from mimiciv_icu.icustays").fetchone()[0]

        assert len(built) == n_stays
        assert built.stay_id.is_unique
        assert built.dx_chapter.notna().all()

    def test_charlson_absence_stays_null_rather_than_becoming_zero(self, warehouse_conn):
        """R2/R3: absence is signal. A stay with no Charlson row has an *unknown*
        history, and imputing 0 would tell the model the patient is
        comorbidity-free -- a different and much stronger claim."""
        built = disease.build_disease_context(warehouse_conn)
        no_charlson = built[built.charlson_comorbidity_index.isna()]
        assert (no_charlson[disease.CHARLSON_FLAGS].isna()).all().all()

    def test_chronic_and_coded_feature_sets_do_not_overlap(self):
        """The leakage argument depends entirely on these being disjoint: Charlson
        flags are pre-existing chronic conditions, ICD chapter is discharge-coded.
        A column in both would make the two arms of disease_leakage.py meaningless."""
        assert not set(disease.CHRONIC_FEATURES) & set(disease.ADMISSION_CODED_FEATURES)


class TestGroupThresholds:
    """The min-stays guard is the load-bearing part: without it a chapter with three
    patients gets its own escalation cut-point fitted to those three patients."""

    @staticmethod
    def _grid(stay_scores: dict[int, list[int]]) -> pd.DataFrame:
        rows = [
            {"stay_id": sid, "news2": score}
            for sid, scores in stay_scores.items()
            for score in scores
        ]
        return pd.DataFrame(rows)

    def test_a_chapter_below_the_bar_does_not_get_its_own_threshold(self):
        """Three stays cannot support a percentile, however many patient-hours they
        contribute between them -- which is exactly why the guard counts stays."""
        grid = self._grid({1: [5] * 200, 2: [9] * 200, 3: [2] * 200})
        stay_groups = pd.DataFrame({"stay_id": [1, 2, 3], "dx_group": ["Rare", "Rare", "Rare"]})

        out = news2.group_thresholds(grid, stay_groups, n_bootstrap=20)

        assert len(out) == 1
        assert out.iloc[0].stays == 3
        assert out.iloc[0].patient_hours == 600
        assert not out.iloc[0].own_threshold

    def test_a_chapter_at_the_bar_does_get_its_own_threshold(self):
        n = news2.MIN_STAYS_FOR_GROUP_THRESHOLD
        grid = self._grid({i: [i % 12] * 30 for i in range(n)})
        stay_groups = pd.DataFrame({"stay_id": list(range(n)), "dx_group": ["Common"] * n})

        out = news2.group_thresholds(grid, stay_groups, n_bootstrap=20)

        assert out.iloc[0].own_threshold
        assert out.iloc[0].icu_high > out.iloc[0].icu_medium, "tiers must stay non-degenerate"

    def test_cut_points_are_non_degenerate_even_on_a_flat_distribution(self):
        """Every stay scoring the same value makes P75 == P90. Without the guard
        `high` and `medium` collapse onto one number and the medium tier vanishes."""
        n = news2.MIN_STAYS_FOR_GROUP_THRESHOLD
        grid = self._grid({i: [7] * 30 for i in range(n)})
        stay_groups = pd.DataFrame({"stay_id": list(range(n)), "dx_group": ["Flat"] * n})

        out = news2.group_thresholds(grid, stay_groups, n_bootstrap=20)

        assert out.iloc[0].icu_high == out.iloc[0].icu_medium + 1

    def test_bootstrap_resamples_stays_not_rows(self):
        """Resampling rows would treat 200 correlated hours from one patient as 200
        independent draws and report an interval several times too narrow.

        Constructed so the P90 cut-point sits exactly on a stay boundary: 18 stays
        scoring 2 and 2 stays scoring 15 puts 10% of patient-hours at 15, so whether
        the high cut-point lands at 2 or at 15 depends entirely on how many of those
        two stays a resample happens to draw. A stay-level bootstrap sees that and
        reports a wide interval; a row-level one would average it away and report a
        confidently wrong point.
        """
        grid = self._grid(
            {i: [2] * 200 for i in range(18)} | {i: [15] * 200 for i in range(18, 20)}
        )
        stay_groups = pd.DataFrame({"stay_id": list(range(20)), "dx_group": ["Skewed"] * 20})

        out = news2.group_thresholds(grid, stay_groups, n_bootstrap=300)

        assert out.iloc[0].high_ci_hi - out.iloc[0].high_ci_lo > 1.0


class TestLoadThresholds:
    def test_an_unknown_group_falls_back_to_pooled_and_says_so(self, warehouse_conn):
        """`is_fallback` is what lets a consumer distinguish "escalated on this
        disease's own threshold" from "escalated on the general one". Silently
        returning the pooled row would make those indistinguishable."""
        pooled = news2.load_thresholds(warehouse_conn)
        fallback = news2.load_thresholds(warehouse_conn, "NotARealChapter")

        assert fallback.is_fallback is True
        assert fallback.dx_group == news2.POOLED_GROUP
        assert (fallback.icu_medium, fallback.icu_high) == (pooled.icu_medium, pooled.icu_high)
        assert pooled.is_fallback is False

    def test_a_group_with_its_own_threshold_is_not_marked_fallback(self, warehouse_conn):
        own = warehouse_conn.execute(
            "select dx_group from capstone.news2_thresholds where dx_group != ?",
            [news2.POOLED_GROUP],
        ).fetchone()
        if own is None:
            pytest.skip("no chapter cleared MIN_STAYS_FOR_GROUP_THRESHOLD in this warehouse")

        loaded = news2.load_thresholds(warehouse_conn, own[0])

        assert loaded.dx_group == own[0]
        assert loaded.is_fallback is False


@pytest.fixture
def warehouse_conn():
    from warehouse.disease import DEFAULT_DB_PATH

    if not DEFAULT_DB_PATH.exists():
        pytest.skip("no warehouse built -- run warehouse/build_duckdb.py")
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    yield conn
    conn.close()
