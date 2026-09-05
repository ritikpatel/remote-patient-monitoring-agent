import numpy as np
import pytest

from simulators.arrival_models import DEFAULT_DB_PATH, GROUPS, MAX_H, ArrivalModelSet, fit

pytestmark = pytest.mark.skipif(
    not DEFAULT_DB_PATH.exists(), reason="warehouse not built (run warehouse/build_duckdb.py first)"
)


@pytest.fixture(scope="module")
def model_set() -> ArrivalModelSet:
    import duckdb

    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    try:
        return fit(conn)
    finally:
        conn.close()


def test_every_group_has_a_model(model_set: ArrivalModelSet):
    assert set(model_set.models) == set(GROUPS)


def test_admission_profile_has_expected_length(model_set: ArrivalModelSet):
    for m in model_set.models.values():
        assert len(m.admission_profile) == MAX_H
        assert len(m.diurnal_multiplier) == 24


def test_diurnal_multiplier_averages_to_one(model_set: ArrivalModelSet):
    for m in model_set.models.values():
        assert np.mean(m.diurnal_multiplier) == pytest.approx(1.0, abs=1e-6)


def test_icu_monitoring_is_near_stationary_relative_to_orders(model_set: ArrivalModelSet):
    """R5: monitoring should NOT show the sharp admission spike that orders/transfers do."""
    monitoring = np.array(model_set["ICU monitoring"].admission_profile)
    orders = np.array(model_set["Provider orders"].admission_profile)
    monitoring_burst = monitoring[:4].mean() / monitoring[24:168].mean()
    orders_burst = orders[:4].mean() / orders[24:168].mean()
    assert monitoring_burst < orders_burst


def test_json_round_trip(model_set: ArrivalModelSet, tmp_path):
    path = tmp_path / "models.json"
    model_set.to_json(path)
    restored = ArrivalModelSet.from_json(path)
    for fam in GROUPS:
        assert restored[fam].admission_profile == model_set[fam].admission_profile


def test_sample_count_is_reproducible_with_seeded_rng(model_set: ArrivalModelSet):
    m = model_set["ICU monitoring"]
    a = m.sample_count(10, 8, np.random.default_rng(42))
    b = m.sample_count(10, 8, np.random.default_rng(42))
    assert a == b


def test_phase_locked_offsets_cluster_tighter_than_uniform(model_set: ArrivalModelSet):
    rng = np.random.default_rng(0)
    monitoring_offsets = model_set["ICU monitoring"].sample_offsets_seconds(50, stay_id=1, rng=rng)
    orders_offsets = model_set["Provider orders"].sample_offsets_seconds(50, stay_id=1, rng=rng)
    assert monitoring_offsets.std() < orders_offsets.std()
