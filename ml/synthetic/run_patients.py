"""Does raising the *patient* count, class-balanced, move the real numbers?

`run_tutorial.py` answered this for synthetic patient-hours and found nothing.
The obvious objection is that patient-hours are not patients: they carry no
identity, no trajectory, and a row-level generator cannot raise the count of
positive *subjects*, which `ml/evaluation/reliability.py` identifies as the
binding constraint (49 against the ~725 a 0.10-wide AUPRC interval would need).

This module removes that objection. `patients.py` generates whole synthetic
patients with hourly trajectories and synthetic subject ids, in a caller-chosen
class balance, and this harness measures what they buy.

Protocol, unchanged from everything else in this project:

- Subject-grouped 5-fold CV, repeated under several split seeds.
- The generator is refitted **inside each fold** on that fold's training stays.
  Fitting once on the whole cohort would leak every held-out patient into the
  synthetic patients the model trains on.
- Synthetic patients enter the **training** side only. Every reported number is
  measured on real held-out patients.
- Synthetic subject ids are negative so they can never collide with a real one,
  and grouped CV treats each as its own individual.

Arms: `real` (baseline), then `balanced` at each multiplier -- k=1 means one
synthetic patient added per real training patient, half of them with events.

Usage:
    python ml/synthetic/run_patients.py            # full run
    python ml/synthetic/run_patients.py --quick
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
from ml.evaluation.reliability import WAREHOUSE_DB, load_task  # noqa: E402
from ml.features import engineer  # noqa: E402
from ml.features import labels as label_module  # noqa: E402
from ml.models import gbm  # noqa: E402
from ml.synthetic import emr_wgan, patients, preprocess  # noqa: E402

OUT_CSV = REPO_ROOT / "ml" / "synthetic" / "patient_augmentation.csv"
HORIZON = 6
N_FOLDS = 5
N_REPEATS = 4
MULTIPLIERS = (1, 3, 10)
LABEL_COLUMN = f"label_{HORIZON}h"


@dataclass
class HeldOut:
    """One fold's real held-out patients -- everything an arm is scored against."""

    repeat: int
    fold: int
    x: pd.DataFrame
    y: pd.Series
    subjects: pd.Series
    n_boot: int


def score_arm(
    held_out: HeldOut,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    arm: str,
    k: int,
    n_positive_subjects: int,
) -> dict[str, object]:
    """Fit one arm and score it on real held-out patients.

    Module-level rather than nested in the fold loop: a closure would capture the
    loop variables by reference, which is correct only while nobody defers a call.
    """
    _model, proba = gbm.fit_predict_proba(x_fit, y_fit, held_out.x)
    truth = held_out.y.to_numpy()
    ci = metrics.bootstrap_ci_grouped(
        truth, proba, held_out.subjects.to_numpy(), metrics._safe_auprc, n_boot=held_out.n_boot
    )
    return {
        "repeat": held_out.repeat,
        "fold": held_out.fold,
        "arm": arm,
        "k": k,
        "n_fit_rows": len(x_fit),
        "n_positive_subjects": n_positive_subjects,
        "auprc": ci.point,
        "auprc_lo": ci.lo,
        "auprc_hi": ci.hi,
        "auroc": metrics._safe_auroc(truth, proba),
    }


def generate_patients(
    train_params: pd.DataFrame,
    static_columns: list[str],
    n_positive: int,
    n_negative: int,
    config: emr_wgan.TrainConfig,
    rng: np.random.Generator,
    template: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Fit the parameter generator on `train_params` and decode N new patients.

    Returns `(x, y, subject_id)` aligned to `template`'s columns, ready to be
    concatenated onto a real training fold.
    """
    # `has_event` is the condition, so it is held out of the matrix the generator
    # learns -- the conditional paradigm, exactly as in run_tutorial.py.
    learn = train_params.drop(columns=["stay_id", "has_event"])
    condition = np.eye(2, dtype=np.float32)[train_params["has_event"].to_numpy().astype(int)]

    matrix = preprocess.build_matrix(learn, rng=rng, min_concept_count=1)
    trained = emr_wgan.train(matrix.values, matrix.spec, config, conditions=condition)

    wanted = np.r_[np.ones(n_positive, dtype=int), np.zeros(n_negative, dtype=int)]
    rng.shuffle(wanted)
    sampled = trained.generator.sample(len(wanted), condition=np.eye(2, dtype=np.float32)[wanted])

    params = preprocess.invert_matrix(sampled, matrix.spec)

    # Preprocessing drops columns it cannot teach a generator anything about --
    # a Charlson flag nobody in this fold has is constant, so low-prevalence
    # pruning removes it. They still have to come back, or the decoder trips over
    # a missing static and the feature matrix ends up narrower than the real one.
    # Refilled by sampling the real training marginal, which for a constant
    # column reproduces that constant exactly.
    for column in learn.columns:
        if column not in params.columns:
            params[column] = (
                learn[column].sample(len(params), replace=True, random_state=0).to_numpy()
            )

    params["has_event"] = wanted.astype(float)
    # Negative ids cannot collide with a real stay or subject.
    params["stay_id"] = -(np.arange(len(params)) + 1)

    grid, labels = patients.decode_stays(params, static_columns, rng, horizon=HORIZON)
    if grid.empty:
        empty_x = template.iloc[:0].copy()
        return empty_x, pd.Series(dtype=int), pd.Series(dtype=int)

    enriched = engineer.add_rolling_features(grid)
    joined = labels.merge(enriched, on=["stay_id", "hour"], how="inner")

    x = joined.reindex(columns=template.columns)
    for column in template.columns:
        x[column] = x[column].astype(template[column].dtype, errors="ignore")
    return x, joined[LABEL_COLUMN].astype(int), joined["stay_id"].astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-boot", type=int, default=400)
    args = parser.parse_args()

    epochs = 100 if args.quick else args.epochs
    repeats = 1 if args.quick else args.repeats
    multipliers = (1, 3) if args.quick else MULTIPLIERS
    n_boot = 60 if args.quick else args.n_boot
    # 125 stays is a very small training set for a GAN, so the batch is small and
    # the run is long; both are set here rather than left at the row-level defaults.
    config = emr_wgan.TrainConfig(epochs=epochs, checkpoint_every=epochs, batch_size=32, seed=0)

    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    task = load_task(conn)
    grid = engineer.load_hourly_grid_raw(conn)
    features = engineer.build_feature_frame(conn)
    labels_frame = label_module.build_labels(conn, grid[["stay_id", "hour"]], horizons=(HORIZON,))
    conn.close()

    encoded = patients.encode_stays(grid, features, labels_frame)
    stay_to_subject = dict(zip(task.stay_id.to_numpy(), task.subject_id.to_numpy(), strict=True))
    encoded.params["subject_id"] = encoded.params["stay_id"].map(stay_to_subject)
    encoded.params = encoded.params.dropna(subset=["subject_id"]).reset_index(drop=True)

    print(
        f"real: {len(task.y)} rows, {int(task.y.sum())} positive rows, "
        f"{task.subject_id.nunique()} subjects, {len(encoded.params)} stays "
        f"({int(encoded.params['has_event'].sum())} with an event)",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        seed = config.seed + 1000 * repeat
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(task.y)), task.y.to_numpy(), task.subject_id.to_numpy())
        ):
            x_tr, y_tr = task.x.iloc[train_idx], task.y.iloc[train_idx]
            x_te, y_te = task.x.iloc[test_idx], task.y.iloc[test_idx]
            sub_tr, sub_te = task.subject_id.iloc[train_idx], task.subject_id.iloc[test_idx]
            if y_tr.sum() < 5 or y_te.sum() < 3:
                continue

            train_subjects = set(sub_tr.unique().tolist())
            fold_params = encoded.params.loc[
                encoded.params["subject_id"].isin(train_subjects)
            ].drop(columns=["subject_id"])
            if len(fold_params) < 10 or fold_params["has_event"].nunique() < 2:
                continue

            held_out = HeldOut(
                repeat=repeat, fold=fold, x=x_te, y=y_te, subjects=sub_te, n_boot=n_boot
            )

            real_positive_subjects = int(sub_tr[y_tr.to_numpy() > 0].nunique())
            rows.append(score_arm(held_out, x_tr, y_tr, "real", 0, real_positive_subjects))

            rng = np.random.default_rng(seed + fold)
            fold_config = emr_wgan.TrainConfig(**{**config.__dict__, "seed": seed + fold})
            for k in multipliers:
                total = len(fold_params) * k
                n_pos = total // 2
                synth_x, synth_y, synth_stay = generate_patients(
                    fold_params,
                    encoded.static_columns,
                    n_pos,
                    total - n_pos,
                    fold_config,
                    rng,
                    task.x,
                )
                if synth_x.empty or synth_y.nunique() < 2:
                    continue
                # Annotated so pd.concat resolves to its DataFrame/Series
                # overloads; without it the stubs pick the wrong one and report
                # a spurious argument-type error (same idiom as
                # ml/evaluation/synthetic_ceiling.py).
                frames: list[pd.DataFrame] = [x_tr, synth_x]
                series: list[pd.Series] = [y_tr, synth_y]
                rows.append(
                    score_arm(
                        held_out,
                        pd.concat(frames, ignore_index=True),
                        pd.concat(series, ignore_index=True),
                        "balanced",
                        k,
                        real_positive_subjects + int(synth_stay[synth_y > 0].nunique()),
                    )
                )

            print(
                f"  repeat {repeat} fold {fold}: {len(fold_params)} real stays, "
                f"{real_positive_subjects} positive subjects",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    if not frame.empty:
        print()
        summary = frame.groupby(["arm", "k"])[["auprc", "auroc", "n_positive_subjects"]].mean()
        print(summary.round(4))


if __name__ == "__main__":
    main()
