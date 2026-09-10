"""The tutorial's data-quality battery, plus one metric of this project's own.

Implemented from the paper's Data Quality Evaluation section, in its order. For
every one of them a **lower value means higher utility** except where noted, and
that direction is carried on `LOWER_IS_BETTER` rather than left to the reader.

One metric is adapted rather than copied, and the adaptation is the sort of thing
that should be argued in the open. The paper measures **clinical knowledge
violation** as male-specific diagnoses (benign prostatic hyperplasia, prostate
cancer, erectile dysfunction) turning up on female synthetic records. This cohort
cannot support that test twice over: it has 100 patients, so no sex-specific
phecode clears a usable prevalence, and `gender` was deliberately dropped from
the feature set by the fairness audit (finding F4), so it is not in the matrix to
violate. The *purpose* of the metric -- does the generator produce records that
are individually impossible? -- transfers intact to physiological constraints
this data does have, so that is what `clinical_knowledge_violation` checks:
systolic pressure below mean arterial pressure, saturations above 100%, a
Glasgow Coma Score outside 3-15. These are violations of arithmetic and
physiology, not of a reference range, so an abnormal-but-real patient never
counts as one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from ml.synthetic.preprocess import MatrixSpec

LOWER_IS_BETTER = {
    "dimension_wise_distance": True,
    "absolute_prevalence_difference": True,
    "column_wise_correlation": True,
    "latent_cluster_analysis": True,
    "medical_concept_abundance": True,
    "clinical_knowledge_violation": True,
    "temporal_coherence": True,
    "membership_inference_f1": True,
    "attribute_inference_f1": True,
    # Utility metrics where more is better: TSTR should approach TRTR, and the
    # share of top features shared with the real model should approach 1.
    "tstr_auroc": False,
    "feature_importance_overlap": False,
}

# The paper's Figure 5F ranks the top 20 features. Kept identical.
TOP_N_FEATURES = 20
# Latent cluster analysis: the paper projects to a latent space "that covers a
# specific threshold of variance" and then clusters. 0.9 of variance and k=20
# are the values used in the authors' own benchmarking work (their reference 21).
LCA_VARIANCE = 0.90
LCA_CLUSTERS = 20


@dataclass
class UtilityScores:
    dimension_wise_distance: float
    absolute_prevalence_difference: float
    column_wise_correlation: float
    latent_cluster_analysis: float
    medical_concept_abundance: float
    clinical_knowledge_violation: float
    temporal_coherence: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def absolute_prevalence_difference(
    real: np.ndarray, synthetic: np.ndarray, spec: MatrixSpec
) -> float:
    """Mean |prevalence_real - prevalence_synthetic| over binary/one-hot columns.

    This is the quantity plotted on the paper's Figure 4, where each point is one
    categorical column and the diagonal is perfect replication.
    """
    index = {c: i for i, c in enumerate(spec.columns)}
    discrete = spec.binary + [m for members in spec.categorical.values() for m in members]
    if not discrete:
        return float("nan")
    cols = [index[c] for c in discrete]
    return float(np.mean(np.abs(real[:, cols].mean(axis=0) - synthetic[:, cols].mean(axis=0))))


def dimension_wise_distance(real: np.ndarray, synthetic: np.ndarray, spec: MatrixSpec) -> float:
    """DWD: categorical APD and continuous Wasserstein, summed then normalized.

    The paper: "It calculates the average of the absolute prevalence differences
    for categorical variables and the average of the Wasserstein distances for
    continuous variables ... we add these 2 values together and then normalize
    the sum to derive the final score."

    Normalization is by the count of variable *kinds* actually present, so a
    matrix with only continuous columns is not silently halved against one with
    both.
    """
    index = {c: i for i, c in enumerate(spec.columns)}
    parts: list[float] = []

    apd = absolute_prevalence_difference(real, synthetic, spec)
    if not np.isnan(apd):
        parts.append(apd)

    if spec.continuous:
        distances = [
            wasserstein_distance(real[:, index[c]], synthetic[:, index[c]]) for c in spec.continuous
        ]
        parts.append(float(np.mean(distances)))

    if not parts:
        return float("nan")
    # Scaled by 100 to land in the same order of magnitude as the paper's Figure
    # 4 (DWD 0.52-1.56), which reports the sum over a matrix normalized to (0,1).
    return float(np.sum(parts) * 100.0 / len(parts))


def column_wise_correlation(real: np.ndarray, synthetic: np.ndarray) -> float:
    """Mean |difference| between the two Pearson correlation matrices.

    Constant columns produce a NaN correlation in both matrices; those cells are
    dropped rather than zero-filled, since a zero would be scored as perfect
    agreement on a quantity neither matrix actually defines.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_real = np.corrcoef(real, rowvar=False)
        corr_synth = np.corrcoef(synthetic, rowvar=False)
    diff = np.abs(corr_real - corr_synth)
    finite = np.isfinite(diff)
    if not finite.any():
        return float("nan")
    return float(diff[finite].mean() * 100.0)


def latent_cluster_analysis(real: np.ndarray, synthetic: np.ndarray, seed: int = 0) -> float:
    """Log mean squared deviation of the real share across latent clusters.

    Real and synthetic rows are pooled, projected by PCA to the components
    covering `LCA_VARIANCE`, and k-means'd. If the two sets occupy the latent
    space identically, every cluster holds the same real:synthetic ratio as the
    pool, each deviation is ~0, and the log runs very negative -- which is why
    the paper's Figure 5B plots values near -15 as the *high* utility end.
    """
    n_real, n_synth = len(real), len(synthetic)
    if n_real < LCA_CLUSTERS or n_synth < LCA_CLUSTERS:
        return float("nan")

    pooled = np.vstack([real, synthetic])
    is_real = np.r_[np.ones(n_real), np.zeros(n_synth)]

    components = PCA(n_components=LCA_VARIANCE, random_state=seed).fit_transform(pooled)
    labels = KMeans(n_clusters=LCA_CLUSTERS, n_init=10, random_state=seed).fit_predict(components)

    expected = n_real / (n_real + n_synth)
    deviations = []
    for cluster in range(LCA_CLUSTERS):
        mask = labels == cluster
        if not mask.any():
            continue
        deviations.append((is_real[mask].mean() - expected) ** 2)
    if not deviations:
        return float("nan")
    mean_deviation = float(np.mean(deviations))
    # An exactly-zero deviation is possible on small clusters and log(0) is -inf,
    # which would poison every downstream average. Floored at the smallest
    # representable positive double instead, which preserves the ordering.
    return float(np.log(max(mean_deviation, np.finfo(float).tiny)))


def medical_concept_abundance(real: np.ndarray, synthetic: np.ndarray, spec: MatrixSpec) -> float:
    """Normalized Manhattan distance between the two concept-count histograms.

    The paper's record-level quantity: how many distinct concepts each record
    carries. Here a "concept" is an active binary or one-hot column, so the count
    per row is how many discrete facts that patient-record asserts.
    """
    index = {c: i for i, c in enumerate(spec.columns)}
    discrete = spec.binary + [m for members in spec.categorical.values() for m in members]
    if not discrete:
        return float("nan")
    cols = [index[c] for c in discrete]

    real_counts = (real[:, cols] > 0.5).sum(axis=1)
    synth_counts = (synthetic[:, cols] > 0.5).sum(axis=1)

    bins = np.arange(0, len(discrete) + 2)
    hist_real, _ = np.histogram(real_counts, bins=bins, density=False)
    hist_synth, _ = np.histogram(synth_counts, bins=bins, density=False)
    # Normalized to proportions first so cohorts of different sizes compare.
    p_real = hist_real / max(hist_real.sum(), 1)
    p_synth = hist_synth / max(hist_synth.sum(), 1)
    return float(np.abs(p_real - p_synth).sum() / 2.0)


def clinical_knowledge_violation(records: pd.DataFrame) -> float:
    """Share of records violating a physiological constraint. See module docstring.

    Each rule is checked only on records that carry the columns it needs, and the
    returned figure is the fraction of records breaking at least one applicable
    rule -- so a matrix missing every relevant column returns NaN rather than a
    flattering zero.
    """
    rules: list[pd.Series] = []

    if {"sbp", "map"} <= set(records.columns):
        # MAP = (SBP + 2*DBP)/3, and DBP < SBP in any living patient, so MAP is
        # strictly below SBP. A 2 mmHg tolerance absorbs the rounding introduced
        # by hourly averaging rather than waving through real inversions.
        rules.append(records["map"] > records["sbp"] + 2.0)
    if "spo2" in records.columns:
        rules.append((records["spo2"] > 100.0) | (records["spo2"] < 0.0))
    if "gcs_total" in records.columns:
        rules.append((records["gcs_total"] < 3.0) | (records["gcs_total"] > 15.0))
    if "hr" in records.columns:
        rules.append(records["hr"] <= 0.0)

    if not rules:
        return float("nan")
    violated = pd.concat(rules, axis=1).any(axis=1)
    return float(violated.mean())


def temporal_coherence(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """How badly the generator breaks the arithmetic linking derived features.

    Not a metric from the tutorial. It measures the tutorial's *stated limitation*
    as it lands on this project: "it focuses on simulating static structured EHR
    data and neglects the timestamping of medical events ... EHR data inherently
    consists of time series, where the temporal information is critical".

    42% of this feature matrix (36 of 85 columns) is rolling-window statistics --
    `hr_4h_std`, `rr_24h_slope` and so on -- computed from a patient's trajectory.
    A snapshot generator emits them as free-standing columns, so nothing forces
    `hr_4h_std` and `hr_24h_std` to describe the same patient's variability. In
    real data they are strongly related, because the 4-hour window is contained in
    the 24-hour one.

    Scored as the mean |correlation_real - correlation_synthetic| over those
    within-vital pairs. 0 means the generator preserved the relationship; a value
    approaching the real correlation itself means it destroyed it. Lower is better.
    """
    shared = [c for c in real.columns if c in synthetic.columns]
    pairs: list[tuple[str, str]] = []
    for column in shared:
        if not column.endswith("_4h_std"):
            continue
        partner = column.replace("_4h_std", "_24h_std")
        if partner in shared:
            pairs.append((column, partner))

    if not pairs:
        return float("nan")

    gaps = []
    for short, long in pairs:
        r_real = real[short].corr(real[long])
        r_synth = synthetic[short].corr(synthetic[long])
        # A constant column in either frame makes its correlation undefined; that
        # pair carries no information about coherence and is skipped rather than
        # scored as agreement.
        if np.isfinite(r_real) and np.isfinite(r_synth):
            gaps.append(abs(r_real - r_synth))

    if not gaps:
        return float("nan")
    return float(np.mean(gaps))


def utility_scores(
    real: np.ndarray,
    synthetic: np.ndarray,
    spec: MatrixSpec,
    synthetic_records: pd.DataFrame,
    real_records: pd.DataFrame | None = None,
    seed: int = 0,
) -> UtilityScores:
    """Every matrix-level utility metric in one pass.

    `real_records` is the postprocessed *real* frame, needed only by
    `temporal_coherence`, which compares a correlation on both sides. Omitting it
    leaves that one metric NaN rather than failing the whole battery.
    """
    return UtilityScores(
        dimension_wise_distance=dimension_wise_distance(real, synthetic, spec),
        absolute_prevalence_difference=absolute_prevalence_difference(real, synthetic, spec),
        column_wise_correlation=column_wise_correlation(real, synthetic),
        latent_cluster_analysis=latent_cluster_analysis(real, synthetic, seed=seed),
        medical_concept_abundance=medical_concept_abundance(real, synthetic, spec),
        clinical_knowledge_violation=clinical_knowledge_violation(synthetic_records),
        temporal_coherence=(
            float("nan")
            if real_records is None
            else temporal_coherence(real_records, synthetic_records)
        ),
    )


def feature_importance_overlap(
    real_importance: pd.Series, synthetic_importance: pd.Series, top_n: int = TOP_N_FEATURES
) -> float:
    """Proportion of the real model's top-N features that the synthetic model shares.

    The paper's Figure 5F: "the overlap proportion of the top N features with
    those identified in the TSTR scenario. The higher the proportion, the higher
    the data utility."
    """
    real_top = set(real_importance.abs().nlargest(top_n).index)
    synth_top = set(synthetic_importance.abs().nlargest(top_n).index)
    if not real_top:
        return float("nan")
    return float(len(real_top & synth_top) / len(real_top))


def _nearest_distance(targets: np.ndarray, pool: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Distance from each target row to its closest row in `pool`.

    Chunked over targets because the full pairwise matrix is n_targets x n_pool
    and the synthetic pool is deliberately large in these experiments.
    """
    out = np.empty(len(targets), dtype=float)
    for start in range(0, len(targets), chunk):
        block = targets[start : start + chunk]
        distances = np.linalg.norm(block[:, None, :] - pool[None, :, :], axis=2)
        out[start : start + chunk] = distances.min(axis=1)
    return out


def membership_inference_f1(
    members: np.ndarray, non_members: np.ndarray, synthetic: np.ndarray
) -> float:
    """Can an adversary tell training records from held-out ones, given synthetics?

    The paper: "quantified using the F1-score of the inference based on the
    distances between targeted records and all synthetic records." The adversary
    sees only the synthetic set, computes each target's nearest-synthetic
    distance, and calls the closer half members. Using the median of the observed
    distances as the threshold is the strongest version of this attack that
    needs no extra knowledge, so it is the conservative choice for a privacy
    figure -- a weaker threshold would understate the risk.

    0.5 is chance-level for a balanced target set; the paper's real-data baseline
    is 0.91 and its synthetic sets score 0.29-0.31.
    """
    if len(members) == 0 or len(non_members) == 0 or len(synthetic) == 0:
        return float("nan")
    n = min(len(members), len(non_members))
    targets = np.vstack([members[:n], non_members[:n]])
    truth = np.r_[np.ones(n), np.zeros(n)]

    distances = _nearest_distance(targets, synthetic)
    predicted = (distances <= np.median(distances)).astype(int)
    return float(f1_score(truth, predicted, zero_division=0))


def attribute_inference_f1(
    real_records: np.ndarray,
    synthetic: np.ndarray,
    known_columns: list[int],
    sensitive_columns: list[int],
) -> float:
    """Can an adversary fill in sensitive fields from partially-observed records?

    The paper's construction: the adversary matches a partially-observed real
    record against the synthetic set on the attributes it knows, then reads the
    sensitive attributes off the nearest synthetic neighbour. Scored as the
    weighted F1 of those inferences, binarized at the column median so a
    continuous sensitive attribute is scored on the same footing as a flag.
    """
    if not known_columns or not sensitive_columns or len(synthetic) == 0:
        return float("nan")

    known_real = real_records[:, known_columns]
    known_synth = synthetic[:, known_columns]

    scores: list[float] = []
    for start in range(0, len(known_real), 512):
        block = known_real[start : start + 512]
        distances = np.linalg.norm(block[:, None, :] - known_synth[None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        for column in sensitive_columns:
            truth_values = real_records[start : start + 512, column]
            guessed = synthetic[nearest, column]
            # Threshold on the *real* column's median for both sides, so the
            # comparison is against one fixed definition of "high" rather than
            # two differently-placed ones.
            cut = float(np.median(real_records[:, column]))
            scores.append(
                f1_score(
                    (truth_values > cut).astype(int),
                    (guessed > cut).astype(int),
                    average="weighted",
                    zero_division=0,
                )
            )
    if not scores:
        return float("nan")
    return float(np.mean(scores))
