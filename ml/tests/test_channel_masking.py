"""Tests for kit masks and training-time channel dropout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import engineer
from ml.models import channel_masking, gbm


def _frame(n: int = 40) -> pd.DataFrame:
    """A feature frame with the real column names and realistic dtypes -- the
    `*_was_imputed` columns are genuinely bool, which is what makes the dtype
    handling in `apply_mask` load-bearing rather than incidental."""
    rng = np.random.default_rng(0)
    data: dict[str, object] = {}
    for v in engineer.CORE_VITALS:
        data[v] = rng.normal(90, 10, n)
        data[f"{v}_was_imputed"] = rng.integers(0, 2, n).astype(bool)
        data[f"{v}_hours_since_last_obs"] = rng.integers(0, 5, n).astype(float)
        for w in engineer.ROLLING_WINDOWS_H:
            for stat in engineer.ROLLING_STATS:
                data[f"{v}_{w}h_{stat}"] = rng.normal(0, 1, n)
    data["has_arterial_line"] = rng.integers(0, 2, n)
    data["admission_age"] = rng.integers(40, 90, n).astype(float)
    data["charlson_comorbidity_index"] = rng.integers(0, 10, n).astype(float)
    data["gender"] = ["M"] * n
    data["first_careunit"] = ["MICU"] * n
    return pd.DataFrame(data)


class TestKitMasks:
    def test_a_kit_masks_exactly_the_channels_it_lacks(self):
        cols = list(_frame().columns)
        masked = channel_masking.kit_mask("watch_only", cols)

        # hr and spo2 survive; everything else the watch cannot see is gone.
        assert not any(c == "hr" or c.startswith("hr_") for c in masked)
        assert not any(c == "spo2" or c.startswith("spo2_") for c in masked)
        for absent in ("rr", "sbp", "map", "temp_c", "gcs_total", "fio2", "glucose"):
            assert f"{absent}" in masked
            assert f"{absent}_4h_std" in masked

    def test_the_mask_covers_a_channels_whole_feature_family(self):
        """Masking the raw value but leaving its rolling std and recency behind would
        leak the channel back in through features derived from it -- the trend of a
        signal the kit cannot measure is not available either."""
        cols = list(_frame().columns)
        masked = set(channel_masking.kit_mask("full_home", cols))

        for col in cols:
            if col.startswith("temp_c"):
                assert col in masked, col

    def test_hospital_only_columns_are_masked_for_every_kit(self):
        """Arterial-line presence is always false at home. Leaving it unmasked lets
        the model read a hospital fact off a home patient."""
        cols = list(_frame().columns)
        for kit in channel_masking.HOME_KITS:
            assert "has_arterial_line" in channel_masking.kit_mask(kit, cols)

    def test_patient_history_features_are_never_masked(self):
        """Age and comorbidity burden come from the discharge record, not a sensor --
        a home monitoring service knows its patient's history."""
        cols = list(_frame().columns)
        for kit in channel_masking.HOME_KITS:
            masked = channel_masking.kit_mask(kit, cols)
            assert "admission_age" not in masked
            assert "charlson_comorbidity_index" not in masked

    def test_richer_kits_mask_strictly_fewer_columns(self):
        cols = list(_frame().columns)
        watch = set(channel_masking.kit_mask("watch_only", cols))
        cuff = set(channel_masking.kit_mask("watch_plus_cuff", cols))
        full = set(channel_masking.kit_mask("full_home", cols))
        assert full < cuff < watch


class TestApplyMask:
    def test_masked_columns_become_nan_not_zero(self):
        """NaN says "no sensor reported this"; zero says "the sensor reported zero",
        which for a heart rate is a very different clinical claim. LightGBM splits on
        missingness natively (R2/R3), so the distinction reaches the model."""
        x = _frame()
        out = channel_masking.apply_mask(x, ["rr", "rr_4h_std"])

        assert out["rr"].isna().all()
        assert out["rr_4h_std"].isna().all()
        assert not out["hr"].isna().any()

    def test_boolean_columns_survive_masking_as_float_not_object(self):
        """Assigning NaN into a bool Series silently promotes it to `object`, which
        LightGBM rejects outright ("pandas dtypes must be int, float or bool"). This
        is a real failure this module hit."""
        x = _frame()
        out = channel_masking.apply_mask(x, ["rr_was_imputed"])

        assert out["rr_was_imputed"].dtype == np.float64
        assert out["rr_was_imputed"].isna().all()

    def test_the_input_frame_is_not_mutated(self):
        x = _frame()
        before = x["rr"].to_numpy().copy()
        channel_masking.apply_mask(x, ["rr"])
        assert np.array_equal(x["rr"].to_numpy(), before)


class TestAugmentation:
    def test_replication_covers_the_original_plus_every_kit(self):
        x = _frame(n=10)
        y = pd.Series([0, 1] * 5)
        groups = pd.Series(range(10))

        x_aug, y_aug, g_aug = channel_masking.augment_with_channel_dropout(x, y, groups)

        expected = len(x) * (1 + len(channel_masking.HOME_KITS))
        assert len(x_aug) == len(y_aug) == len(g_aug) == expected

    def test_groups_are_replicated_so_grouped_cv_cannot_leak(self):
        """A patient's masked and unmasked replicas must carry the same subject id.
        Without this, grouped CV would put the same patient's physiology in train and
        test under two different masks -- a subtler version of the stay-vs-subject
        grouping leak this project already found once."""
        x = _frame(n=6)
        y = pd.Series([0, 1, 0, 1, 0, 1])
        groups = pd.Series([100, 100, 200, 200, 300, 300])

        _x_aug, _y_aug, g_aug = channel_masking.augment_with_channel_dropout(x, y, groups)

        assert set(g_aug) == {100, 200, 300}
        for gid in (100, 200, 300):
            assert (g_aug == gid).sum() == 2 * (1 + len(channel_masking.HOME_KITS))

    def test_labels_stay_aligned_with_their_rows(self):
        x = _frame(n=8)
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        groups = pd.Series(range(8))

        x_aug, y_aug, _ = channel_masking.augment_with_channel_dropout(x, y, groups)

        n_replicas = 1 + len(channel_masking.HOME_KITS)
        assert y_aug.sum() == y.sum() * n_replicas
        # Row i of each replica must carry row i's original label.
        for r in range(n_replicas):
            block = y_aug.iloc[r * len(y) : (r + 1) * len(y)].to_numpy()
            assert np.array_equal(block, y.to_numpy())
        assert len(x_aug) == len(y_aug)

    def test_the_augmented_frame_fits_without_a_dtype_error(self):
        """The end-to-end reason the dtype handling exists: LightGBM must accept the
        concatenated frame. A dtype regression here fails the whole transfer study."""
        x = _frame(n=60)
        y = pd.Series(([0] * 25 + [1] * 5) * 2)
        groups = pd.Series(list(range(30)) * 2)

        x_aug, y_aug, _ = channel_masking.augment_with_channel_dropout(x, y, groups)
        model, proba = gbm.fit_predict_proba(x_aug, y_aug, x_aug)

        assert len(proba) == len(x_aug)
        assert np.isfinite(proba).all()

    def test_the_unmasked_replica_is_present_unchanged(self):
        """The model must see both regimes -- dropping the unmasked rows would trade
        ICU performance away entirely rather than adding home robustness."""
        x = _frame(n=5)
        y = pd.Series([0, 1, 0, 1, 0])
        groups = pd.Series(range(5))

        x_aug, _, _ = channel_masking.augment_with_channel_dropout(x, y, groups)

        first_block = x_aug.iloc[: len(x)]
        assert not first_block["temp_c"].isna().any(), "the unmasked replica was masked"


class TestMaskDescription:
    def test_describe_masks_reports_every_kit_and_the_unmasked_case(self):
        cols = list(_frame().columns)
        table = channel_masking.describe_masks(cols)

        assert set(table.kit) == {"icu_full", *channel_masking.HOME_KITS}
        icu = table[table.kit == "icu_full"].iloc[0]
        assert icu.columns_masked == 0
        assert icu.channels_absent == "none"

    def test_column_counts_are_consistent(self):
        cols = list(_frame().columns)
        table = channel_masking.describe_masks(cols)
        assert (table.columns_masked + table.columns_remaining == len(cols)).all()


def test_kit_definitions_come_from_the_simulator_not_a_copy():
    """One source of truth. If training masks and simulator kits were separate lists
    they would drift, and the drift would read as a modelling result rather than a
    bookkeeping error."""
    from simulators.home_kit_stream import HOME_KITS as SIM_KITS

    assert channel_masking.HOME_KITS is SIM_KITS


def test_no_kit_can_observe_a_channel_with_no_home_sensor():
    for name, kit in channel_masking.HOME_KITS.items():
        assert not set(kit.channels) & set(channel_masking.NO_HOME_SENSOR), name


@pytest.mark.parametrize("kit", sorted(channel_masking.HOME_KITS))
def test_every_kit_leaves_at_least_one_vital_observable(kit):
    cols = list(_frame().columns)
    remaining = set(cols) - set(channel_masking.kit_mask(kit, cols))
    assert any(v in remaining for v in engineer.CORE_VITALS), kit
