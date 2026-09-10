"""A falsification test for the membership-inference finding.

Stage 1 reports that an adversary holding only the synthetic cohort can tell the
generator's training subjects from held-out ones far above chance. That is a
strong claim, and before it goes in a report it needs the obvious confound ruled
out: **the attack might be reading an intrinsic difference between the two
subject groups rather than anything the generator memorised.** The 70/30 split is
random, but with 93 subjects the two halves are not guaranteed to be alike, and a
discriminator that separates them would produce the same number for a reason that
has nothing to do with privacy.

The control is a swap. Train a second generator on the *held-out* subjects and
re-run the identical attack against the identical targets. If the metric tracks
memorisation, the verdict on a fixed group should follow which subjects the
generator saw: group A should look like the training set to a generator fitted on
A and not to one fitted on B. If instead the verdict is the same either way, the
metric is reading a property of the two cohorts and the privacy claim collapses.

One trap worth naming, because falling into it would have turned a single
measurement into apparent corroboration. The attack thresholds at the median of
the observed distances, so it always predicts exactly half the targets as
members; relabelling which group is the "member" group leaves the predictions
alone and flips only the truth vector. F1(B as members) is therefore exactly
1 - F1(A as members), an algebraic identity rather than a second observation.
This module reports one figure per generator and compares *across* generators,
which is the comparison that carries information.

Usage:
    python ml/synthetic/privacy_control.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.evaluation.reliability import WAREHOUSE_DB, load_task  # noqa: E402
from ml.synthetic import emr_wgan, evaluate, run_tutorial  # noqa: E402

CSV_PATH = REPO_ROOT / "ml" / "synthetic" / "privacy_control.csv"


def main() -> None:
    conn = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    task = load_task(conn)
    cohort = run_tutorial.build_cohort(task, conditional=True)
    conn.close()

    group_a = cohort.matrix.values[cohort.train_rows]
    group_b = cohort.matrix.values[cohort.test_rows]
    label_a = cohort.label[cohort.train_rows]
    label_b = cohort.label[cohort.test_rows]
    print(f"group A (stage 1 training): {len(group_a)} rows")
    print(f"group B (stage 1 held out): {len(group_b)} rows")

    config = emr_wgan.TrainConfig(epochs=300, checkpoint_every=300, seed=0)
    rows = []

    for name, fitted_on, values, labels in (
        ("A", "A", group_a, label_a),
        ("B", "B", group_b, label_b),
    ):
        trained = emr_wgan.train(
            values,
            cohort.matrix.spec,
            config,
            conditions=run_tutorial.one_hot_label(labels),
        )
        drawn = run_tutorial.label_composition(
            len(values), float(labels.mean()), np.random.default_rng(0)
        )
        synthetic = trained.generator.sample(
            len(drawn), condition=run_tutorial.one_hot_label(drawn)
        )

        # One number per generator, always framed the same way: does the attack
        # call group A the training set?
        #
        # Deliberately NOT also reporting the mirrored "B as members" figure.
        # `membership_inference_f1` thresholds at the median of the observed
        # distances, so it predicts exactly half the targets as members; swapping
        # which group is labelled "member" leaves those predictions untouched and
        # only flips the truth vector, which makes F1(B as members) equal to
        # 1 - F1(A as members) as an algebraic identity. Printing both would look
        # like two corroborating measurements and be one.
        a_as_members = evaluate.membership_inference_f1(group_a, group_b, synthetic)
        rows.append({"generator fitted on": fitted_on, "attack scores A as members": a_as_members})
        print(f"  generator on {name}: A as members = {a_as_members:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(CSV_PATH, index=False)
    print()
    print(frame.to_markdown(index=False))

    # The evidence is the cross-generator comparison: same targets, same framing,
    # two synthetic sets differing only in which subjects the generator saw.
    on_a = float(
        frame.loc[frame["generator fitted on"] == "A", "attack scores A as members"].iloc[0]
    )
    on_b = float(
        frame.loc[frame["generator fitted on"] == "B", "attack scores A as members"].iloc[0]
    )
    print(f"\nGroup A is called the training set at {on_a:.4f} when the generator saw A,")
    print(f"and at {on_b:.4f} when it saw B instead. Swing: {on_a - on_b:+.4f}")

    if on_a > 0.65 and on_b < 0.35:
        print(
            "\nCONTROL PASSES: the attack's verdict on a fixed set of targets follows "
            "which subjects the generator was fitted to. It is reading memorisation, "
            "not a fixed difference between the two subject groups."
        )
    else:
        print(
            "\nCONTROL FAILS: the attack's verdict does not follow the generator's "
            "training set. The membership-inference figure is confounded and must not "
            "be reported as a privacy risk."
        )
    print(f"\nwrote {CSV_PATH}")


if __name__ == "__main__":
    main()
