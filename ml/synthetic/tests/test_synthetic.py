"""Tests for the EMR-WGAN pipeline.

The properties worth pinning here are the ones the tutorial states as
requirements and that fail *silently* when broken: the matrix must land in
(0,1), one-hot blocks must stay one-hot through generation, and the inverse of
Equation 1 must actually be an inverse. A generator that quietly emits a
two-hot categorical still trains, still samples, and still scores -- it just
produces records that assert a patient was in two ICUs at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.synthetic import emr_wgan, evaluate, patients, preprocess


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "hr": rng.normal(85, 15, n),
            "sbp": rng.normal(120, 20, n),
            "map": rng.normal(75, 10, n),
            "spo2": rng.normal(96, 3, n).clip(60, 100),
            "gcs_total": rng.integers(3, 16, n).astype(float),
            "flag": rng.integers(0, 2, n).astype(bool),
            "unit": rng.choice(["MICU", "SICU", "CCU"], n),
        }
    )


def test_matrix_is_in_unit_interval(frame: pd.DataFrame) -> None:
    """Equation 1 plus sigmoid/one-hot means every column lives in [0,1]."""
    matrix = preprocess.build_matrix(frame)
    assert matrix.values.min() >= 0.0
    assert matrix.values.max() <= 1.0
    assert matrix.values.shape[0] == len(frame)


def test_one_hot_blocks_are_contiguous_and_complete(frame: pd.DataFrame) -> None:
    matrix = preprocess.build_matrix(frame)
    blocks = matrix.spec.block_slices()
    assert len(blocks) == 1
    start, stop = blocks[0]
    assert stop - start == 3  # MICU / SICU / CCU
    np.testing.assert_allclose(matrix.values[:, start:stop].sum(axis=1), 1.0, atol=1e-6)


def test_clipping_counts_outliers_and_leaves_abnormal_values_alone() -> None:
    """A tachycardic patient is not an outlier; a heart rate of 900 is."""
    frame = pd.DataFrame({"hr": [180.0, 900.0, 45.0, -5.0]})
    clipped, counts = preprocess.clip_outliers(frame)
    assert counts["hr"] == 2
    assert clipped["hr"].tolist() == [180.0, 250.0, 45.0, 20.0]


def test_high_missingness_columns_are_dropped() -> None:
    frame = pd.DataFrame({"keep": [1.0, 2.0, 3.0, None], "drop": [1.0, None, None, None]})
    kept, doomed = preprocess.drop_high_missingness(frame)
    assert "drop" in doomed and "keep" not in doomed
    assert list(kept.columns) == ["keep"]


def test_imputation_samples_observed_values_only() -> None:
    """The paper's marginal-sampling strategy must not invent unobserved values."""
    frame = pd.DataFrame({"x": [1.0, 1.0, 5.0, None, None]})
    filled, rates = preprocess.impute_by_marginal_sampling(frame, np.random.default_rng(0))
    assert filled["x"].isna().sum() == 0
    assert set(filled["x"].unique()) <= {1.0, 5.0}
    assert rates["x"] == pytest.approx(0.4)


def test_low_prevalence_uses_the_minority_class() -> None:
    """A flag true in 2 of 100 rows and one true in 98 are equally unlearnable."""
    frame = pd.DataFrame({"rare": [1.0] * 2 + [0.0] * 98, "inverse": [0.0] * 2 + [1.0] * 98})
    kept, dropped = preprocess.drop_low_prevalence(frame, ["rare", "inverse"], min_count=20)
    assert set(dropped) == {"rare", "inverse"}
    assert kept.empty or list(kept.columns) == []


def test_inversion_round_trips_continuous_values(frame: pd.DataFrame) -> None:
    """invert_matrix must undo Equation 1 exactly on the data it was fitted to."""
    matrix = preprocess.build_matrix(frame, rng=np.random.default_rng(0))
    recovered = preprocess.invert_matrix(matrix.values, matrix.spec)
    for column in ["hr", "sbp", "map"]:
        np.testing.assert_allclose(
            recovered[column].to_numpy(), frame[column].to_numpy(), rtol=1e-4
        )
    assert set(recovered["unit"].unique()) <= {"MICU", "SICU", "CCU"}


def test_constant_column_survives_the_round_trip() -> None:
    """Equation 1 divides by (max-min); a constant column must not become NaN."""
    frame = pd.DataFrame({"flat": [7.0] * 50, "hr": np.linspace(60, 100, 50)})
    matrix = preprocess.build_matrix(frame)
    recovered = preprocess.invert_matrix(matrix.values, matrix.spec)
    np.testing.assert_allclose(recovered["flat"].to_numpy(), 7.0)


def test_generator_output_respects_the_one_hot_constraint(frame: pd.DataFrame) -> None:
    """The tutorial's SoftMax requirement, checked on actual generated rows."""
    matrix = preprocess.build_matrix(frame)
    config = emr_wgan.TrainConfig(epochs=2, checkpoint_every=1, batch_size=64)
    result = emr_wgan.train(matrix.values, matrix.spec, config)

    sample = result.generator.sample(40)
    assert sample.shape == (40, matrix.spec.n_columns)
    assert sample.min() >= 0.0 and sample.max() <= 1.0
    for start, stop in matrix.spec.block_slices():
        np.testing.assert_allclose(sample[:, start:stop].sum(axis=1), 1.0, atol=1e-5)


def test_checkpoints_are_kept_along_the_trajectory(frame: pd.DataFrame) -> None:
    """The tutorial scores checkpoints rather than trusting the final weights."""
    matrix = preprocess.build_matrix(frame)
    config = emr_wgan.TrainConfig(epochs=4, checkpoint_every=2, batch_size=64)
    result = emr_wgan.train(matrix.values, matrix.spec, config)
    assert sorted(result.checkpoints) == [2, 4]

    restored = emr_wgan.generator_from_checkpoint(result.checkpoints[2], matrix.spec, config)
    assert restored.sample(5).shape == (5, matrix.spec.n_columns)


def test_conditional_generator_honours_the_requested_composition(frame: pd.DataFrame) -> None:
    """Conditional training must accept and use a label the caller chooses."""
    matrix = preprocess.build_matrix(frame)
    labels = np.random.default_rng(0).integers(0, 2, len(frame))
    condition = np.eye(2, dtype=np.float32)[labels]

    config = emr_wgan.TrainConfig(epochs=2, checkpoint_every=2, batch_size=64)
    result = emr_wgan.train(matrix.values, matrix.spec, config, conditions=condition)

    requested = np.eye(2, dtype=np.float32)[np.r_[np.ones(10, int), np.zeros(10, int)]]
    assert result.generator.sample(20, condition=requested).shape[0] == 20

    with pytest.raises(ValueError, match="requires a `condition`"):
        result.generator.sample(5)


def test_conditions_must_align_with_rows(frame: pd.DataFrame) -> None:
    matrix = preprocess.build_matrix(frame)
    with pytest.raises(ValueError, match="rows against"):
        emr_wgan.train(
            matrix.values,
            matrix.spec,
            emr_wgan.TrainConfig(epochs=1),
            conditions=np.zeros((5, 2), dtype=np.float32),
        )


def test_identical_data_scores_as_perfect_utility(frame: pd.DataFrame) -> None:
    """Every distance metric must bottom out when synthetic == real."""
    matrix = preprocess.build_matrix(frame)
    values = matrix.values
    assert evaluate.absolute_prevalence_difference(values, values, matrix.spec) == 0.0
    assert evaluate.dimension_wise_distance(values, values, matrix.spec) == pytest.approx(0.0)
    assert evaluate.column_wise_correlation(values, values) == pytest.approx(0.0, abs=1e-9)
    assert evaluate.medical_concept_abundance(values, values, matrix.spec) == pytest.approx(0.0)


def test_clinical_knowledge_violation_catches_impossible_records() -> None:
    """MAP above SBP, SpO2 above 100, GCS off the scale -- all impossible."""
    records = pd.DataFrame(
        {
            "sbp": [120.0, 80.0, 130.0, 110.0],
            "map": [75.0, 95.0, 80.0, 70.0],  # row 1 inverted
            "spo2": [97.0, 96.0, 105.0, 98.0],  # row 2 impossible
            "gcs_total": [15.0, 10.0, 8.0, 2.0],  # row 3 below the scale
            "hr": [80.0, 90.0, 70.0, 60.0],
        }
    )
    assert evaluate.clinical_knowledge_violation(records) == pytest.approx(0.75)

    clean = records.iloc[[0]]
    assert evaluate.clinical_knowledge_violation(clean) == 0.0


def test_clinical_knowledge_violation_is_nan_without_applicable_columns() -> None:
    """No relevant column must not read as a flattering zero."""
    assert np.isnan(evaluate.clinical_knowledge_violation(pd.DataFrame({"z": [1.0]})))


def test_feature_importance_overlap_is_a_proportion() -> None:
    real = pd.Series({"a": 10.0, "b": 8.0, "c": 6.0, "d": 1.0})
    same = pd.Series({"a": 9.0, "b": 7.0, "c": 5.0, "d": 0.5})
    assert evaluate.feature_importance_overlap(real, same, top_n=3) == 1.0

    disjoint = pd.Series({"a": 0.1, "b": 0.2, "c": 0.3, "d": 99.0})
    assert evaluate.feature_importance_overlap(real, disjoint, top_n=1) == 0.0


def test_membership_inference_is_chance_level_on_indistinguishable_sets() -> None:
    """Synthetic data unrelated to the members must not identify them.

    Members and non-members are drawn from the same distribution here, so no
    attack on a synthetic set can do better than chance -- an F1 far above 0.5
    would mean the metric is reading something other than membership.
    """
    rng = np.random.default_rng(0)
    members = rng.normal(size=(200, 8))
    non_members = rng.normal(size=(200, 8))
    synthetic = rng.normal(size=(400, 8))

    score = evaluate.membership_inference_f1(members, non_members, synthetic)
    assert 0.35 <= score <= 0.65


def test_latent_cluster_analysis_prefers_overlapping_distributions() -> None:
    """A shifted synthetic set must score worse (higher) than a matched one."""
    rng = np.random.default_rng(0)
    real = rng.normal(size=(400, 6))
    matched = rng.normal(size=(400, 6))
    shifted = rng.normal(size=(400, 6)) + 6.0

    assert evaluate.latent_cluster_analysis(real, matched) < evaluate.latent_cluster_analysis(
        real, shifted
    )


def test_temporal_coherence_detects_broken_derived_relationships() -> None:
    """A generator that emits `4h_std` and `24h_std` independently must be caught.

    In real data the two windows describe the same patient's variability and move
    together; independent draws destroy that, which is exactly the failure mode a
    snapshot generator has on a 42%-derived feature matrix.
    """
    rng = np.random.default_rng(0)
    base = rng.gamma(2.0, 3.0, 500)
    real = pd.DataFrame({"hr_4h_std": base, "hr_24h_std": base * 1.4 + rng.normal(0, 0.3, 500)})
    coherent = pd.DataFrame({"hr_4h_std": base, "hr_24h_std": base * 1.4 + rng.normal(0, 0.3, 500)})
    broken = pd.DataFrame(
        {"hr_4h_std": rng.gamma(2.0, 3.0, 500), "hr_24h_std": rng.gamma(2.0, 3.0, 500)}
    )

    assert evaluate.temporal_coherence(real, coherent) < 0.05
    assert evaluate.temporal_coherence(real, broken) > 0.5


def test_temporal_coherence_is_nan_without_matching_pairs() -> None:
    frame = pd.DataFrame({"hr": [1.0, 2.0, 3.0]})
    assert np.isnan(evaluate.temporal_coherence(frame, frame))


# --- patient-level generation --------------------------------------------------


def _stay_params(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """A small parameter table of the shape `patients.decode_stays` consumes."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        record = {
            "stay_id": i + 1,
            "length": float(rng.integers(6, 40)),
            "has_event": float(i % 2),
            "event_fraction": float(i % 2),
            "admission_age": float(rng.integers(40, 90)),
            "has_arterial_line_rate": float(rng.random()),
        }
        for vital in ("hr", "sbp", "spo2"):
            record[f"{vital}_level"] = {"hr": 85.0, "sbp": 120.0, "spo2": 96.0}[vital]
            record[f"{vital}_slope"] = float(rng.normal(0, 0.1))
            record[f"{vital}_sd"] = float(rng.uniform(3, 12))
            record[f"{vital}_rho"] = float(rng.uniform(0.3, 0.8))
            record[f"{vital}_imputed_rate"] = 0.0
        rows.append(record)
    return pd.DataFrame(rows), ["admission_age"]


def test_decoded_trajectories_match_the_requested_moments() -> None:
    """The decoder must realise the level and SD it was handed, not approximate them.

    An AR(1) path's *sample* SD over a short stay is biased well below its process
    SD, so `_ar1` rescales. Without that correction every synthetic patient gets
    flatter vitals than the real one it was parameterised from, which shrinks
    `hr_4h_std` -- one of the model's top features.
    """
    params, statics = _stay_params()
    grid, _labels = patients.decode_stays(params, statics, np.random.default_rng(0))

    realised_level = grid.groupby("stay_id")["hr"].mean()
    realised_sd = grid.groupby("stay_id")["hr"].std(ddof=0)
    target_sd = params.set_index("stay_id")["hr_sd"]

    np.testing.assert_allclose(realised_level.to_numpy(), 85.0, atol=1.0)
    np.testing.assert_allclose(
        realised_sd.to_numpy(), target_sd.reindex(realised_sd.index).to_numpy(), rtol=0.05
    )


def test_decoded_trajectories_are_autocorrelated_not_white_noise() -> None:
    """A patient whose vitals jump independently each hour is not a patient."""
    params, statics = _stay_params(n=30, seed=1)
    params["length"] = 48.0
    params["hr_rho"] = 0.8
    grid, _ = patients.decode_stays(params, statics, np.random.default_rng(0))

    autocorrelations = grid.groupby("stay_id")["hr"].apply(lambda s: s.autocorr(1))
    assert autocorrelations.median() > 0.5


def test_labels_follow_the_censoring_rule() -> None:
    """Positives sit in the final `horizon` hours; a stay without an event has none."""
    params, statics = _stay_params(n=10)
    _grid, labels = patients.decode_stays(params, statics, np.random.default_rng(0), horizon=6)

    has_event = params.set_index("stay_id")["has_event"] > 0.5
    for stay_id, block in labels.groupby("stay_id"):
        positives = block.loc[block["label_6h"] == 1, "hour"]
        if not has_event.loc[stay_id]:
            assert positives.empty
        else:
            assert len(positives) == min(6, len(block))
            # Contiguous, and ending at the last at-risk hour.
            assert positives.max() == block["hour"].max()


def test_vitals_stay_inside_plausible_ranges() -> None:
    """Clipping happens on the generated path, not just on the real data."""
    params, statics = _stay_params(n=20, seed=3)
    params["hr_sd"] = 200.0  # absurd, to force the clip
    grid, _ = patients.decode_stays(params, statics, np.random.default_rng(0))
    low, high = patients.engineer_range("hr")
    assert grid["hr"].min() >= low
    assert grid["hr"].max() <= high


def test_arterial_line_is_persistent_not_flickering() -> None:
    """Once placed, a line stays in -- the flag must be a step, never a coin flip."""
    params, statics = _stay_params(n=15, seed=4)
    params["has_arterial_line_rate"] = 0.5
    params["length"] = 40.0
    grid, _ = patients.decode_stays(params, statics, np.random.default_rng(0))

    for _stay_id, block in grid.groupby("stay_id"):
        flag = block.sort_values("hour")["has_arterial_line"].to_numpy().astype(int)
        # A step function changes value at most once.
        assert int(np.abs(np.diff(flag)).sum()) <= 1
