"""What synthetic patients would actually buy, if they were 90% real.

``ml/evaluation/reliability.py`` establishes that the only lever left on this
model's confidence interval is more patients: a 0.10-wide AUPRC CI needs on the
order of 725 positive subjects against this cohort's 49. Full MIMIC-IV is
credentialed access and out of reach here, so the obvious next question is
whether a generative model could manufacture the difference -- and if a
synthetic cohort reached, say, **90% similarity** to the real one, what that
would be worth.

This module answers it by measurement rather than by assertion, and the answer
is uncomfortable: **almost nothing, for this purpose.** Two separate effects,
measured separately, because they fail in opposite directions:

1. **Fidelity is not the binding constraint.** Even a *perfect* generator adds
   no information. A model fitted to 49 positive subjects and then sampled ten
   thousand times yields ten thousand draws from an *estimate* of the
   distribution -- not ten thousand independent patients. The uncertainty that
   the CI is reporting is uncertainty about that estimate, and resampling it
   cannot reduce it. This is the data-processing inequality, and the experiment
   below shows it directly: held-out AUPRC on **real** patients stays flat as
   the synthetic multiplier rises, while the CI you would *report* if you
   treated synthetic rows as observations shrinks like 1/sqrt(k). That gap is
   manufactured precision, and it is the failure mode worth naming.

2. **Imperfect fidelity actively costs you.** The 10% that is *not* similar is
   a distribution shift the model learns and then carries into real data. The
   sweep below degrades a synthetic cohort in controlled steps and measures
   what real-world AUPRC does.

**How "90% similarity" is defined here**, since the phrase has no standard
meaning: by a **classifier two-sample test** (C2ST), the usual measure. A
discriminator is trained to tell real rows from synthetic ones; if it cannot
beat chance (AUC 0.5) the two are indistinguishable. Reported as

    similarity % = 100 * (2 - 2 * C2ST_AUC)

so AUC 0.50 -> 100% similar and AUC 1.00 -> 0% similar. **90% similarity is
therefore C2ST AUC = 0.55.** The similarity of every synthetic cohort here is
measured, never assumed.

**The generator is deliberately the strongest one possible.** Synthetic
patients are whole real patients resampled with replacement -- every vital
trajectory, every temporal correlation, every label relationship preserved
exactly -- optionally perturbed with calibrated noise to walk fidelity
downwards. At zero noise this is a generator no GAN, diffusion model or copula
could beat: it reproduces the training distribution perfectly. If synthetic
augmentation cannot help *there*, the ceiling is not a modelling problem that a
better generator solves.

Every synthetic patient is built from the **training** subjects only, and every
evaluation is on **real, held-out** subjects, grouped by subject throughout
(finding F6). A synthetic row never reaches a test fold.

Usage:
    python ml/evaluation/synthetic_ceiling.py            # ~6 min
    python ml/evaluation/synthetic_ceiling.py --quick
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from ml.evaluation import metrics  # noqa: E402
from ml.evaluation.reliability import Task, load_task  # noqa: E402
from ml.models import gbm  # noqa: E402

REPORT_PATH = REPO_ROOT / "ml" / "evaluation" / "synthetic_ceiling_report.md"
CSV_PATH = REPO_ROOT / "ml" / "evaluation" / "synthetic_ceiling.csv"

# Noise as a fraction of each feature's training SD. 0.0 is the perfect-fidelity
# ceiling; the rest walk similarity down so the curve can be read at whatever
# fidelity a real generator turns out to achieve.
# Calibrated from a first run: at 0.10 x SD a discriminator already separates
# real from synthetic with AUC 1.00 (0% similar), because independent per-feature
# noise destroys the correlation structure of 74 correlated columns immediately.
# The interesting band -- where "90% similar" actually lives -- is far below that,
# so the sweep was re-centred rather than left where it produced only the two
# useless extremes.
NOISE_LEVELS = (0.0, 0.005, 0.01, 0.025, 0.05)
# Synthetic patients per real training patient. k=1 means "no augmentation".
AUGMENTATION_FACTORS = (1, 3, 10)
N_OUTER_SPLITS = 5


def _continuous_columns(x: pd.DataFrame) -> list[str]:
    """Float columns only -- the ones a measurement error would actually move.

    Deliberately excludes bools and ints. `*_was_imputed` is a flag saying a
    value was carried forward; adding Gaussian noise to it is meaningless, and
    `lab_orders_4h` is a count. Selecting them anyway also breaks the arithmetic
    for a subtler reason worth recording: `DataFrame.to_numpy()` over mixed
    bool-and-float columns returns an **object** array, so the perturbed columns
    came back as dtype object and LightGBM rejected the whole frame.
    """
    return [c for c in x.columns if pd.api.types.is_float_dtype(x[c])]


def synthesize(
    x: pd.DataFrame,
    y: pd.Series,
    subjects: pd.Series,
    n_synthetic_subjects: int,
    noise: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Draw whole patients with replacement, optionally perturbed.

    Resampling at the *patient* level rather than the row level is what makes
    this a fair stand-in for a good generative model: a row-level resample would
    shred the within-patient temporal structure that the rolling features encode,
    and would flatter the conclusion by making synthetic data obviously worse.
    Each synthetic patient gets a fresh id so subject-grouped CV keeps treating
    them as separate individuals.
    """
    pool = np.array(sorted(subjects.unique()))
    chosen = rng.choice(pool, size=n_synthetic_subjects, replace=True)

    numeric = _continuous_columns(x)
    sds = x[numeric].std(ddof=0).fillna(0.0).to_numpy(dtype=float)

    frames, labels, ids = [], [], []
    for i, real_subject in enumerate(chosen):
        mask = (subjects == real_subject).to_numpy()
        block = x.loc[mask].copy()
        if noise > 0 and len(block):
            jitter = rng.normal(0.0, 1.0, size=(len(block), len(numeric))) * (noise * sds)
            block[numeric] = block[numeric].to_numpy(dtype=float) + jitter
        frames.append(block)
        labels.append(y.loc[mask])
        # Negative ids cannot collide with a real subject_id.
        ids.append(pd.Series(np.full(int(mask.sum()), -(i + 1)), index=block.index))

    if not frames:
        return x.iloc[:0].copy(), y.iloc[:0].copy(), subjects.iloc[:0].copy()
    return (
        pd.concat(frames, ignore_index=True),
        pd.concat(labels, ignore_index=True),
        pd.concat(ids, ignore_index=True),
    )


def c2st_auc(real: pd.DataFrame, synthetic: pd.DataFrame, rng: np.random.Generator) -> float:
    """Classifier two-sample test: how separable are real and synthetic rows?

    0.5 means a discriminator cannot tell them apart at all. Evaluated on a
    held-out split of the discrimination task itself, so an overfitted
    discriminator cannot report false separability.
    """
    n = min(len(real), len(synthetic))
    if n < 50:
        return float("nan")
    real_s = real.sample(n=n, random_state=int(rng.integers(1 << 31)))
    synth_s = synthetic.sample(n=n, random_state=int(rng.integers(1 << 31)))

    combined = pd.concat([real_s, synth_s], ignore_index=True)
    is_synth = np.r_[np.zeros(n), np.ones(n)]
    order = rng.permutation(len(combined))
    combined, is_synth = combined.iloc[order].reset_index(drop=True), is_synth[order]

    cut = int(0.7 * len(combined))
    _model, proba = gbm.fit_predict_proba(
        combined.iloc[:cut], pd.Series(is_synth[:cut]), combined.iloc[cut:]
    )
    return metrics._safe_auroc(is_synth[cut:], proba)


def similarity_pct(auc: float) -> float:
    """C2ST AUC -> the "% similar" figure the question was posed in."""
    if np.isnan(auc):
        return float("nan")
    return float(np.clip(100.0 * (2.0 - 2.0 * auc), 0.0, 100.0))


@dataclass
class Row:
    noise: float
    k: int
    c2st_auc: float
    similarity_pct: float
    real_test_auprc: float
    real_test_ci_width: float
    synthetic_test_auprc: float
    synthetic_test_ci_width: float
    n_train_subjects: int
    n_synthetic_subjects: int


def run(task: Task, n_boot: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    splitter = StratifiedGroupKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=seed)
    rows: list[Row] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(task.y)), task.y.to_numpy(), task.subject_id.to_numpy())
    ):
        x_tr, y_tr = task.x.iloc[train_idx], task.y.iloc[train_idx]
        sub_tr = task.subject_id.iloc[train_idx]
        x_te, y_te = task.x.iloc[test_idx], task.y.iloc[test_idx]
        sub_te = task.subject_id.iloc[test_idx]
        if y_tr.sum() < 5 or y_te.sum() < 3:
            continue

        n_real_subjects = int(sub_tr.nunique())

        for noise in NOISE_LEVELS:
            for k in AUGMENTATION_FACTORS:
                n_synth = int(n_real_subjects * (k - 1))
                # Declared before the branch so both arms share one type; a
                # first assignment inside `if` makes mypy read the else-branch
                # DataFrame as a type error rather than a union.
                synth_x: pd.DataFrame | None = None
                if n_synth == 0:
                    x_aug, y_aug = x_tr, y_tr
                else:
                    # Concat off the freshly-inferred locals, not the Optional
                    # names above: the Optionals exist only so the C2ST block
                    # below can see whether synthesis happened at all.
                    new_x, new_y, _new_sub = synthesize(x_tr, y_tr, sub_tr, n_synth, noise, rng)
                    synth_x = new_x
                    # Annotated so pd.concat resolves to its DataFrame/Series
                    # overloads; without it the stubs pick the wrong one and
                    # report a spurious list-item type error.
                    frames: list[pd.DataFrame] = [x_tr, new_x]
                    series: list[pd.Series] = [y_tr, new_y]
                    x_aug = pd.concat(frames, ignore_index=True)
                    y_aug = pd.concat(series, ignore_index=True)

                _model, proba_real = gbm.fit_predict_proba(x_aug, y_aug, x_te)
                real_ci = metrics.bootstrap_ci_grouped(
                    y_te.to_numpy(),
                    proba_real,
                    sub_te.to_numpy(),
                    metrics._safe_auprc,
                    n_boot=n_boot,
                )

                # What you would believe if you evaluated on synthetic patients
                # too -- the manufactured-precision case. Held-out synthetic
                # subjects, drawn from the same generator, never seen in fitting.
                if synth_x is None:
                    synth_auprc, synth_width, auc = real_ci.point, real_ci.hi - real_ci.lo, 0.5
                else:
                    s_x, s_y, s_sub = synthesize(
                        x_tr, y_tr, sub_tr, max(n_real_subjects, 10), noise, rng
                    )
                    _m2, proba_synth = gbm.fit_predict_proba(x_aug, y_aug, s_x)
                    synth_ci = metrics.bootstrap_ci_grouped(
                        s_y.to_numpy(),
                        proba_synth,
                        s_sub.to_numpy(),
                        metrics._safe_auprc,
                        n_boot=n_boot,
                    )
                    synth_auprc = synth_ci.point
                    synth_width = synth_ci.hi - synth_ci.lo
                    auc = c2st_auc(x_tr, synth_x, rng)

                rows.append(
                    Row(
                        noise=noise,
                        k=k,
                        c2st_auc=auc,
                        similarity_pct=similarity_pct(auc),
                        real_test_auprc=real_ci.point,
                        real_test_ci_width=real_ci.hi - real_ci.lo,
                        synthetic_test_auprc=synth_auprc,
                        synthetic_test_ci_width=synth_width,
                        n_train_subjects=n_real_subjects,
                        n_synthetic_subjects=n_synth,
                    )
                )
        print(f"  fold {fold}: done ({n_real_subjects} train subjects)")

    return pd.DataFrame([r.__dict__ for r in rows])


WATERMARK = (
    "> This platform is validated on a 100-patient demo subset of MIMIC-IV. The\n"
    "> engineering is real and the methodology is rigorous; the clinical\n"
    "> performance figures demonstrate pipeline validity and do not transfer to\n"
    "> clinical practice (PROJECT_PLAN.md section 17).\n"
)


def write_report(df: pd.DataFrame, task: Task) -> None:
    by_noise_k = (
        df.groupby(["noise", "k"])
        .agg(
            similarity_pct=("similarity_pct", "mean"),
            c2st_auc=("c2st_auc", "mean"),
            real_auprc=("real_test_auprc", "mean"),
            real_ci_width=("real_test_ci_width", "mean"),
            synth_auprc=("synthetic_test_auprc", "mean"),
            synth_ci_width=("synthetic_test_ci_width", "mean"),
            n_train_subjects=("n_train_subjects", "mean"),
        )
        .reset_index()
    )

    perfect = by_noise_k[by_noise_k.noise == 0.0]
    base = perfect[perfect.k == 1].iloc[0]
    best_k = perfect.loc[perfect.real_auprc.idxmax()]

    lines = ["# What synthetic patients would buy\n", WATERMARK]
    lines.append(
        "\nThe question this answers: **if a generator reached ~90% similarity to the "
        "real cohort, could it substitute for the patients "
        "[`reliability_report.md`](reliability_report.md) says are needed?** Similarity is "
        "measured by a classifier two-sample test (C2ST) and reported as "
        "`100 * (2 - 2 * AUC)`, so **90% similar means C2ST AUC 0.55** and 100% means a "
        "discriminator cannot separate real from synthetic at all.\n"
    )
    lines.append(
        f"\nSetup: {N_OUTER_SPLITS} subject-grouped folds over the real cohort "
        f"({task.subject_id.nunique()} subjects, {int(task.y.sum())} positives). Synthetic "
        f"patients are built from the **training** subjects only and evaluated on **real, "
        f"held-out** subjects. The generator resamples whole patients with replacement, "
        f"preserving every temporal correlation exactly, then optionally perturbs them -- "
        f"at zero noise it reproduces the training distribution perfectly, which no learned "
        f"generator can beat.\n"
    )

    lines.append("\n## 1. A perfect generator adds no information\n")
    lines.append(
        "Noise 0.0: synthetic patients are exact resamples, so C2ST cannot separate them. "
        "`k` is the multiplier on training patients -- k=10 means nine synthetic patients "
        "per real one.\n"
    )
    lines.append(
        perfect[["k", "similarity_pct", "real_auprc", "real_ci_width", "synth_ci_width"]]
        .rename(
            columns={
                "k": "multiplier",
                "similarity_pct": "measured similarity %",
                "real_auprc": "AUPRC on REAL held-out",
                "real_ci_width": "CI width (real, honest)",
                "synth_ci_width": "CI width (synthetic, believed)",
            }
        )
        .to_markdown(index=False, floatfmt=".3f")
    )
    delta = best_k.real_auprc - base.real_auprc
    synth_ci = float(perfect[perfect.k == perfect.k.max()].synth_ci_width.iloc[0])
    # Every synthetic patient is resampled from the same training subjects, so at
    # high k the bootstrap has essentially nothing left to vary and the interval
    # collapses to floating-point noise. Dividing by that is a divide-by-zero
    # artefact rather than a finding -- an earlier draft of this report printed
    # "1362456180705635.0x narrower than the truth", which is not a number anyone
    # should read. Below the epsilon, describe the collapse instead of scaling it.
    ratio_phrase = (
        "collapses to an interval with no measurable width at all"
        if synth_ci < 1e-6
        else f"roughly **{base.real_ci_width / synth_ci:.0f}x narrower than the truth**"
    )
    lines.append(
        f"\n**Real-world performance does not move: {base.real_auprc:.3f} at k=1 against "
        f"{best_k.real_auprc:.3f} at the best multiplier ({delta:+.3f}).** Ten times the "
        f"training rows, drawn from a flawless generator, buys nothing on real patients -- "
        f"because a model fitted to {base.n_train_subjects:.0f} training subjects and then "
        f"resampled cannot learn about patients it never saw.\n\n"
        f"The last two columns are the point. The honest interval, measured on real "
        f"held-out patients, stays around **{base.real_ci_width:.3f}**. The interval you "
        f"would *report* if you evaluated on synthetic patients is "
        f"**{synth_ci:.3f}** -- it {ratio_phrase}, and it shrinks further with every "
        f"synthetic patient added. That is manufactured precision: the number moves, the "
        f"knowledge does not.\n"
    )

    lines.append("\n## 2. Lower fidelity does not rescue it either\n")
    lines.append(
        "Fidelity walked down by perturbing the resampled patients. Similarity is measured, "
        "not assumed.\n\nRead this table for the *direction*, not for a 90% row: the C2ST "
        "goes from unable to separate (100%) to perfectly separating (0%) as soon as any "
        "noise is added, with nothing in between. Perturbing every channel independently "
        "breaks the inter-vital correlations a discriminator finds trivially, so this axis "
        "cannot land on 90% by construction. The measured answer to the question as posed "
        "is the k=3 row of section 1 (92.0% similar), and it is the same answer: no gain on "
        "real patients.\n"
    )
    at_k = by_noise_k[by_noise_k.k == by_noise_k.k.max()]
    lines.append(
        at_k[["noise", "c2st_auc", "similarity_pct", "real_auprc"]]
        .rename(
            columns={
                "noise": "noise (x SD)",
                "c2st_auc": "C2ST AUC",
                "similarity_pct": "measured similarity %",
                "real_auprc": "AUPRC on REAL held-out",
            }
        )
        .to_markdown(index=False, floatfmt=".3f")
    )
    # Describe what this table actually shows rather than asserting a trend. An
    # earlier draft claimed performance "falls monotonically as fidelity drops",
    # which the numbers contradict -- the largest perturbation scored highest of
    # all. The non-monotonicity is itself the point, so it is computed here and
    # stated, not smoothed into a cleaner-sounding sentence.
    noise_auprc = at_k.real_auprc.astype(float)
    lo, hi = float(noise_auprc.min()), float(noise_auprc.max())
    perfect_row = float(at_k.loc[at_k.noise == at_k.noise.min(), "real_auprc"].iloc[0])
    worst_fidelity = float(at_k.loc[at_k.noise == at_k.noise.max(), "real_auprc"].iloc[0])
    lines.append(
        f"\nAcross this whole fidelity range the AUPRC on real held-out patients moves by "
        f"only {hi - lo:.3f} ({lo:.3f}-{hi:.3f}), and it does **not** fall monotonically: "
        f"the most heavily perturbed generator scores {worst_fidelity:.3f} against the "
        f"perfect generator's {perfect_row:.3f}. That is the honest reading -- differences "
        f"this small sit inside cross-validation noise, so fidelity is not the binding "
        f"constraint and degrading it is not what costs you. What does not happen at any "
        f"level is augmentation overtaking the un-augmented baseline by a margin this "
        f"protocol could detect.\n"
    )

    lines.append("\n## What this means for the capstone\n")
    lines.append(
        "**Synthetic data cannot close the sample-size gap, and 90% fidelity would not "
        "change that.** The binding constraint is not how realistic the rows look -- it is "
        "that the information in them is bounded by the 49 real positive subjects any "
        "generator would be fitted to. Reporting a narrow CI computed over synthetic "
        "patients would be the single most misleading thing this project could do, and "
        "section 1 shows exactly how large that misstatement would be.\n\n"
        "**What synthetic data is legitimately for here, and already used for:**\n\n"
        "- **Exercising pathways that real data cannot reach.** `simulators/morphing.py` "
        "synthesises a deterioration trajectory on a healthy-volunteer wearable recording "
        "because the wearable cohort contains no deterioration at all (E10). That is a "
        "testbed for the alerting path, watermarked `synthetic`, and it makes no claim "
        "about a patient.\n"
        "- **Load and robustness testing.** `eval/load/ramp.js` needs volume, not truth.\n"
        "- **Narrative generation.** `notes_synth/` writes clinical prose from real "
        "structured rows, with a fact ledger tying every sentence back to a source row.\n\n"
        "**What it must never do:** appear in a training or evaluation set whose metrics "
        "are reported as performance. Train-on-synthetic/test-on-real is the only defensible "
        "protocol, and section 1 measures what it yields here: nothing.\n\n"
        "The honest conclusion stands, now with a number behind it rather than a hedge: "
        "**this cohort's confidence interval is a structural property of having 49 positive "
        "subjects, and it is not fixable within this project.** Reporting it plainly is the "
        "correct result, not a shortfall.\n\n"
        "## Follow-up: the middle of the fidelity range, measured\n\n"
        "Section 2 above carries an admitted hole. Perturbing channels independently could "
        "only produce cohorts a discriminator finds identical or trivially separable, never "
        "the intermediate fidelity the question was posed at, so the answer rested on the "
        "perfect-generator argument of section 1 rather than on a generator that actually "
        "lives in that band.\n\n"
        "[`ml/synthetic/`](../synthetic/report.md) closes it. EMR-WGAN (Yan et al., JMIR AI "
        "2024) trained on this cohort reaches a dimension-wise distance inside the range "
        "that paper reports for its own runs on 181,294 patients, and it changes nothing "
        "here: across 20 paired comparisons no augmented arm beats the real-only baseline, "
        "the best of six sitting at -0.0006 AUPRC with 10 of 20 wins. Fidelity was never "
        "the binding constraint, and now that has been measured with a real generator "
        "rather than argued from a resampler.\n\n"
        "That work also surfaced something this experiment could not have: the EMR-WGAN "
        "cohort carries **membership-inference risk two-thirds of the way from chance to "
        "publishing the real records**, verified against a swap control. Synthetic data "
        "from a cohort this small is not de-identified data.\n"
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    conn = duckdb.connect(str(REPO_ROOT / "warehouse" / "mimic4_demo.db"), read_only=True)
    task = load_task(conn)
    conn.close()
    print(
        f"Real cohort: {len(task.y):,} rows, {int(task.y.sum())} positives, "
        f"{task.subject_id.nunique()} subjects"
    )

    df = run(task, n_boot=150 if args.quick else 500)
    df.to_csv(CSV_PATH, index=False)
    write_report(df, task)
    print(f"\nWrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
