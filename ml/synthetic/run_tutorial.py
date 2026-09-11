"""Run the tutorial end to end on this cohort, in the two stages it separates.

**Stage 1 -- data quality**, the paper's own experiment. 70% of subjects train
the generator and 30% are held out for evaluation, exactly as the tutorial does
("70% of records of the preprocessed MIMIC-IV data set were used to train the
EMR-WGAN model and the remaining 30% were used for evaluation purposes"), except
that the split is by *subject* rather than by row -- this project's finding F6 is
that a row-level split leaks a patient across both sides. Five independent runs
are trained from scratch, because the tutorial insists on it: "we highly
recommend training the model multiple times (or multiple runs) from scratch and
testing data quality at multiple checkpoints along the training trajectory",
since "the model checkpoint that corresponds to the highest quality of the
synthetic data is not necessarily the one with the lowest training loss".

Both of the tutorial's training paradigms are run. The **nonconditional** one
"does not distinguish the label variables in the EHR matrix from the remaining
variables", so the deterioration label is generated as an ordinary column. The
**conditional** one feeds the label to both networks as extra input, which
"enables the control over the categories of the generated data". That control is
not a nicety here: the label is a 4%-prevalence concept, which the tutorial warns
generative models "cannot accurately capture", and the nonconditional generator
duly collapses it to near-zero positives.

**Stage 2 -- does any of it help the model?** This is not a question the tutorial
asks, because its premise is 181,294 real patients and a synthetic set of the
same size, generated for *sharing* rather than for augmentation. Here the
generator is fitted on ~74 training subjects, so the interesting number is what
synthetic rows do to held-out performance on real patients. Per fold:

  TRTR      train on real training rows            -- the baseline to beat
  TSTR      train on synthetic rows only           -- the tutorial's utility metric
  augmented train on real + synthetic              -- what "more data" would mean here

crossed with the tutorial's "determine composition" step (its Figure 3), which is
what decides how many of the synthetic rows are positives:

  match     the real training set's own positive rate
  balanced  50/50, the oversampling play for an imbalanced label

A synthetic row never enters a test fold in either stage. Stage 2's whole
cross-validation is repeated under four different split seeds, giving 20 paired
comparisons against the baseline -- the bar the rest of this project uses before
calling a delta real.

Usage:
    python ml/synthetic/run_tutorial.py               # full run
    python ml/synthetic/run_tutorial.py --quick       # smoke test
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
from ml.evaluation.reliability import WAREHOUSE_DB, Task, load_task  # noqa: E402
from ml.models import gbm  # noqa: E402
from ml.synthetic import emr_wgan, evaluate, preprocess  # noqa: E402

OUT_DIR = REPO_ROOT / "ml" / "synthetic"
QUALITY_CSV = OUT_DIR / "quality_by_checkpoint.csv"
AUGMENT_CSV = OUT_DIR / "augmentation.csv"

# The paper trains 5 independent models and compares them (its Runs 1-5).
N_RUNS = 5
# Four repeats x 5 folds = 20 paired comparisons, the bar the rest of this
# project uses to call a delta real (`ml/evaluation/report.md`).
N_REPEATS = 4
N_FOLDS = 5
LABEL_COLUMN = "label"
COMPOSITIONS = ("match", "balanced")


@dataclass
class Cohort:
    """The real matrix, preprocessed, plus the 70/30 subject split of stage 1."""

    matrix: preprocess.Matrix
    label: np.ndarray
    train_rows: np.ndarray
    test_rows: np.ndarray
    features: pd.DataFrame


def one_hot_label(label: np.ndarray) -> np.ndarray:
    """Label as a 2-column one-hot, the form the conditional networks take."""
    out = np.zeros((len(label), 2), dtype=np.float32)
    out[np.arange(len(label)), label.astype(int)] = 1.0
    return out


def label_composition(n: int, positive_rate: float, rng: np.random.Generator) -> np.ndarray:
    """The label vector to generate against -- the tutorial's composition step."""
    n_positive = int(round(n * positive_rate))
    label = np.zeros(n, dtype=int)
    label[:n_positive] = 1
    rng.shuffle(label)
    return label


def build_cohort(task: Task, conditional: bool, seed: int = 0) -> Cohort:
    """Preprocess the real feature matrix and split 70/30 by subject.

    Under the nonconditional paradigm the label is carried *inside* the matrix as
    an ordinary column; under the conditional one it is held out of the matrix
    and handed to the networks separately. That is the whole structural
    difference between the two paradigms.
    """
    features = task.x.copy()
    label = task.y.astype(int).to_numpy()

    frame = features if conditional else features.assign(**{LABEL_COLUMN: label.astype(float)})

    rng = np.random.default_rng(seed)
    subjects = task.subject_id.to_numpy()
    unique = np.array(sorted(set(subjects.tolist())))
    rng.shuffle(unique)
    train_subjects = set(unique[: int(0.70 * len(unique))].tolist())
    is_train = np.array([s in train_subjects for s in subjects])

    matrix = preprocess.build_matrix(frame, rng=np.random.default_rng(seed))
    return Cohort(
        matrix=matrix,
        label=label,
        train_rows=np.flatnonzero(is_train),
        test_rows=np.flatnonzero(~is_train),
        features=features,
    )


def _importance(model, columns: pd.Index) -> pd.Series:
    """LightGBM split-gain importance as a named Series, for the overlap metric."""
    return pd.Series(model.booster_.feature_importance(importance_type="gain"), index=columns)


def _synthetic_frame(
    values: np.ndarray,
    spec: preprocess.MatrixSpec,
    template: pd.DataFrame,
    label: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Postprocess generated rows back into the real feature frame's schema.

    `label` is supplied for conditional generation (the caller chose it) and left
    None for nonconditional, where the label is read back out of the generated
    matrix like any other column.

    Columns are reindexed onto the template's order and dtypes restored, because
    LightGBM compares training and prediction frames by position and dtype, and a
    synthetic frame whose `first_careunit` came back as `object` instead of
    `category` trains a different model than the real one does.
    """
    records = preprocess.invert_matrix(values, spec)

    if label is None:
        y = records[LABEL_COLUMN].round().clip(0, 1).astype(int)
        records = records.drop(columns=[LABEL_COLUMN])
    else:
        y = pd.Series(label.astype(int))

    for column in template.columns:
        if column not in records.columns:
            # Dropped in preprocessing (>50% missing, or low prevalence). Refill
            # from the real column's marginal so the frame keeps its shape rather
            # than being silently narrower than the model expects.
            records[column] = template[column].sample(len(records), replace=True).to_numpy()
        records[column] = records[column].astype(template[column].dtype, errors="ignore")

    return records[template.columns], y


def run_quality(cohort: Cohort, config: emr_wgan.TrainConfig, conditional: bool) -> pd.DataFrame:
    """Stage 1: train `N_RUNS` generators, score every checkpoint of each."""
    spec = cohort.matrix.spec
    real_train = cohort.matrix.values[cohort.train_rows]
    real_test = cohort.matrix.values[cohort.test_rows]

    template = cohort.features
    x_train = template.iloc[cohort.train_rows].reset_index(drop=True)
    x_test = template.iloc[cohort.test_rows].reset_index(drop=True)
    y_train = pd.Series(cohort.label[cohort.train_rows])
    y_test = pd.Series(cohort.label[cohort.test_rows])

    cond_train = one_hot_label(cohort.label[cohort.train_rows]) if conditional else None
    condition_dim = 2 if conditional else 0
    real_rate = float(y_train.mean())

    # TRTR is the reference every TSTR is read against, and it does not depend on
    # the generator, so it is fitted once outside the run loop.
    trtr_model, trtr_proba = gbm.fit_predict_proba(x_train, y_train, x_test)
    trtr_auroc = metrics._safe_auroc(y_test.to_numpy(), trtr_proba)
    trtr_importance = _importance(trtr_model, x_train.columns)

    # Sensitive/known split for attribute inference: the adversary is assumed to
    # know the vitals (which a home monitor would expose anyway) and to be after
    # the static attributes and the outcome.
    index = {c: i for i, c in enumerate(spec.columns)}
    sensitive = [
        index[c]
        for c in (LABEL_COLUMN, "admission_age", "charlson_comorbidity_index")
        if c in index
    ]
    known = [i for i in range(len(spec.columns)) if i not in set(sensitive)]

    # Two anchors, without which the membership-inference figure is unreadable.
    # The paper reports its own "Real" column for the same reason: an F1 means
    # nothing until you know what releasing the real data would have scored, and
    # what an unrelated synthetic set scores. Computed once -- neither depends on
    # the generator.
    noise = np.random.default_rng(config.seed).random(real_train.shape).astype(np.float32)
    baseline_real = evaluate.membership_inference_f1(real_train, real_test, real_train)
    baseline_chance = evaluate.membership_inference_f1(real_train, real_test, noise)

    rows: list[dict[str, object]] = []
    for run in range(1, N_RUNS + 1):
        run_config = emr_wgan.TrainConfig(**{**config.__dict__, "seed": config.seed + run})
        trained = emr_wgan.train(real_train, spec, run_config, conditions=cond_train)
        print(
            f"  [{'conditional' if conditional else 'nonconditional'}] "
            f"run {run}/{N_RUNS}: {len(trained.checkpoints)} checkpoints",
            flush=True,
        )
        rng = np.random.default_rng(run_config.seed)

        for epoch, state in sorted(trained.checkpoints.items()):
            generator = emr_wgan.generator_from_checkpoint(state, spec, run_config, condition_dim)
            # Same size as the real training set, as the tutorial does: "All
            # synthetic data sets produced by these models have the same size as
            # the real training data set."
            if conditional:
                drawn = label_composition(len(real_train), real_rate, rng)
                synthetic = generator.sample(len(real_train), condition=one_hot_label(drawn))
                synth_x, synth_y = _synthetic_frame(synthetic, spec, template, drawn)
            else:
                synthetic = generator.sample(len(real_train))
                synth_x, synth_y = _synthetic_frame(synthetic, spec, template)

            utility = evaluate.utility_scores(
                real_train,
                synthetic,
                spec,
                synth_x,
                real_records=x_train,
                seed=run_config.seed,
            )

            # TSTR needs both classes to fit at all; a collapsed generator that
            # emits one label is a real and reportable outcome, not a crash.
            if synth_y.nunique() < 2:
                tstr_auroc, overlap = float("nan"), float("nan")
            else:
                tstr_model, tstr_proba = gbm.fit_predict_proba(synth_x, synth_y, x_test)
                tstr_auroc = metrics._safe_auroc(y_test.to_numpy(), tstr_proba)
                overlap = evaluate.feature_importance_overlap(
                    trtr_importance, _importance(tstr_model, synth_x.columns)
                )

            rows.append(
                {
                    "paradigm": "conditional" if conditional else "nonconditional",
                    "run": run,
                    "epoch": epoch,
                    **utility.as_dict(),
                    "trtr_auroc": trtr_auroc,
                    "tstr_auroc": tstr_auroc,
                    "feature_importance_overlap": overlap,
                    "membership_inference_f1": evaluate.membership_inference_f1(
                        real_train, real_test, synthetic
                    ),
                    "attribute_inference_f1": evaluate.attribute_inference_f1(
                        real_test, synthetic, known, sensitive
                    ),
                    "membership_inference_f1_real_baseline": baseline_real,
                    "membership_inference_f1_chance": baseline_chance,
                    "synthetic_positive_rate": float(synth_y.mean()),
                    "real_positive_rate": real_rate,
                }
            )

    return pd.DataFrame(rows)


@dataclass
class Fold:
    """One fold's held-out real patients -- everything an arm is scored against."""

    index: int
    repeat: int
    x_test: pd.DataFrame
    y_test: pd.Series
    subjects: pd.Series


def score_arm(
    fold: Fold,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    arm: str,
    k: int,
    composition: str,
    n_boot: int,
) -> dict[str, object] | None:
    """Fit one arm and score it on `fold`'s real held-out subjects.

    Module-level rather than a closure inside the fold loop on purpose: a nested
    function would capture the loop variables by reference, which is correct only
    for as long as nobody defers a call, and is the kind of latent bug that
    surfaces the day someone parallelises the loop.
    """
    if y_fit.nunique() < 2:
        return None
    _model, proba = gbm.fit_predict_proba(x_fit, y_fit, fold.x_test)
    truth = fold.y_test.to_numpy()
    ci = metrics.bootstrap_ci_grouped(
        truth, proba, fold.subjects.to_numpy(), metrics._safe_auprc, n_boot=n_boot
    )
    return {
        "repeat": fold.repeat,
        "fold": fold.index,
        "arm": arm,
        "composition": composition,
        "k": k,
        "n_fit_rows": len(x_fit),
        "auprc": ci.point,
        "auprc_lo": ci.lo,
        "auprc_hi": ci.hi,
        "auroc": metrics._safe_auroc(truth, proba),
    }


def run_augmentation(
    task: Task,
    config: emr_wgan.TrainConfig,
    multipliers: tuple[int, ...],
    n_boot: int,
    repeats: int = N_REPEATS,
) -> pd.DataFrame:
    """Stage 2: TRTR / TSTR / augmented, on real held-out subjects, per fold.

    Conditional throughout, because the nonconditional generator does not produce
    enough positives to fit a classifier on at all -- stage 1 measures that.

    `repeats` re-runs the whole cross-validation under a different split seed. One
    pass gives five paired comparisons against the baseline, which is far too few
    to call a small delta either way; the rest of this project settles such
    questions over 20 paired repeats (`ml/evaluation/report.md`), and this matches
    that bar rather than inventing a weaker one for the flattering case.

    The generator is refitted inside every fold on that fold's training subjects
    only. Fitting one generator on the whole cohort and reusing it across folds
    would leak every test subject into the synthetic rows the model trains on,
    which is the single easiest way to manufacture a good result here.
    """
    template = task.x
    rows: list[dict[str, object] | None] = []

    for repeat in range(repeats):
        seed = config.seed + 1000 * repeat
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(task.y)), task.y.to_numpy(), task.subject_id.to_numpy())
        ):
            rows.extend(
                _run_one_fold(
                    task,
                    template,
                    config,
                    multipliers,
                    n_boot,
                    repeat,
                    fold,
                    seed,
                    train_idx,
                    test_idx,
                )
            )

    return pd.DataFrame([r for r in rows if r is not None])


def _run_one_fold(
    task: Task,
    template: pd.DataFrame,
    config: emr_wgan.TrainConfig,
    multipliers: tuple[int, ...],
    n_boot: int,
    repeat: int,
    fold: int,
    seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> list[dict[str, object] | None]:
    """One (repeat, fold): fit a generator on the training subjects and score every arm."""
    rows: list[dict[str, object] | None] = []
    x_tr, y_tr = task.x.iloc[train_idx], task.y.iloc[train_idx]
    x_te, y_te = task.x.iloc[test_idx], task.y.iloc[test_idx]
    sub_te = task.subject_id.iloc[test_idx]
    if y_tr.sum() < 5 or y_te.sum() < 3:
        return rows

    fold_seed = seed + fold
    fold_matrix = preprocess.build_matrix(
        x_tr.reset_index(drop=True), rng=np.random.default_rng(fold_seed)
    )
    labels_tr = y_tr.astype(int).to_numpy()
    fold_config = emr_wgan.TrainConfig(**{**config.__dict__, "seed": fold_seed})
    trained = emr_wgan.train(
        fold_matrix.values, fold_matrix.spec, fold_config, conditions=one_hot_label(labels_tr)
    )
    generator = trained.generator
    rng = np.random.default_rng(fold_seed)
    real_rate = float(labels_tr.mean())
    print(
        f"  repeat {repeat} fold {fold}: generator on {len(train_idx)} real rows "
        f"({labels_tr.sum()} positives)",
        flush=True,
    )

    this_fold = Fold(index=fold, repeat=repeat, x_test=x_te, y_test=y_te, subjects=sub_te)
    rows.append(score_arm(this_fold, x_tr, y_tr, "trtr", 1, "real", n_boot))

    for composition in COMPOSITIONS:
        rate = real_rate if composition == "match" else 0.5
        for k in multipliers:
            drawn = label_composition(len(train_idx) * k, rate, rng)
            synthetic = generator.sample(len(drawn), condition=one_hot_label(drawn))
            synth_x, synth_y = _synthetic_frame(synthetic, fold_matrix.spec, template, drawn)

            rows.append(score_arm(this_fold, synth_x, synth_y, "tstr", k, composition, n_boot))
            rows.append(
                score_arm(
                    this_fold,
                    pd.concat([x_tr, synth_x], ignore_index=True),
                    pd.concat([y_tr, synth_y], ignore_index=True),
                    "augmented",
                    k,
                    composition,
                    n_boot,
                )
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--quick", action="store_true", help="short run for smoke-testing")
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--stage", choices=["quality", "augmentation", "both"], default="both")
    args = parser.parse_args()

    epochs = 60 if args.quick else args.epochs
    config = emr_wgan.TrainConfig(epochs=epochs, checkpoint_every=max(1, epochs // 6), seed=0)
    multipliers = (1, 3) if args.quick else (1, 3, 10)
    n_boot = 60 if args.quick else args.n_boot

    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    task = load_task(conn)
    print(
        f"real cohort: {len(task.y)} rows, {int(task.y.sum())} positives, "
        f"{task.subject_id.nunique()} subjects",
        flush=True,
    )

    if args.stage in ("quality", "both"):
        print("stage 1: data quality across runs and checkpoints", flush=True)
        frames = []
        for conditional in (False, True):
            cohort = build_cohort(task, conditional=conditional)
            if not conditional:
                cohort.matrix.report.to_csv(OUT_DIR / "preprocessing_report.csv", index=False)
            frames.append(run_quality(cohort, config, conditional))
        pd.concat(frames, ignore_index=True).to_csv(QUALITY_CSV, index=False)
        print(f"wrote {QUALITY_CSV}", flush=True)

    if args.stage in ("augmentation", "both"):
        print("stage 2: augmentation against real held-out subjects", flush=True)
        repeats = 1 if args.quick else args.repeats
        run_augmentation(task, config, multipliers, n_boot, repeats).to_csv(
            AUGMENT_CSV, index=False
        )
        print(f"wrote {AUGMENT_CSV}", flush=True)

    conn.close()


if __name__ == "__main__":
    main()
