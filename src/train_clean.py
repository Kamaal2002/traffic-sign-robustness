"""Train the clean replicates of arm A, on `train.py`'s original recipe.

    python -m src.train_clean --seed 43
    python -m src.train_clean --seed 44

Arm A at seed 42 is the accepted baseline checkpoint. It is reused exactly as
it is, so this module exists for seeds 43 and 44, the two replicates that say
how much of any difference between the arms is training-seed variation.

Nothing about the recipe changes here, and nothing is added to it. The training
configuration, the loaders, the optimiser, the schedule, the epoch budget and
the checkpoint-selection rule all come from `run_experiment`, called with no
loaders so that it builds `data.py`'s own clean pipeline - the same call
`train.py` makes for the baseline. The training artifacts are the two files
that recipe produces, `best_checkpoint.pt` and `history.json`, beside a
`protocol.json` sidecar that names the protocol version the run was started
under. The sidecar is metadata and is not read by training; the accepted
baseline predates it and has none, which the evaluator reports rather than
treats as an error.

What this module adds is the directory. `train.py` sends a run given no
`--output-dir` to `artifacts/baseline`, which is where the accepted seed-42
checkpoint and its history live, so `python -m src.train --seed 43` would
overwrite the one artifact in this project that cannot be regenerated. Here the
seed decides the directory, and the directory is claimed through the same guard
the augmented arms use, which refuses the accepted directories and any
directory already holding a run's output. `train.py` itself stays
byte-identical, because it is the code that trained the baseline.
"""

import argparse

from .data import SEED
from .train import (
    ARTIFACT_ROOT,
    BATCH_SIZE,
    EPOCHS,
    TrainingConfig,
    run_experiment,
    select_device,
)
from .train_augmented import (
    SMOKE_BATCHES,
    SMOKE_EPOCHS,
    claim_output_dir,
    write_protocol_sidecar,
)

ACCEPTED_BASELINE = ARTIFACT_ROOT / "baseline"


def default_output_dir(seed):
    """Where a replicate of this seed goes, if no directory is given."""
    return ARTIFACT_ROOT / f"clean_seed{seed}"


def run(seed, epochs=EPOCHS, batch_size=BATCH_SIZE, device=None,
        output_dir=None, smoke_test=False):
    """Train one clean replicate into its own directory."""
    if output_dir is None:
        if seed == SEED:
            raise RuntimeError(
                f"arm A at seed {SEED} is the accepted checkpoint in "
                f"{ACCEPTED_BASELINE}, which is reused as it is rather than "
                "retrained; pass --output-dir to train a seed-42 replicate "
                "somewhere else")
        output_dir = default_output_dir(seed)
    output_dir = claim_output_dir(output_dir)

    config = TrainingConfig(
        epochs=SMOKE_EPOCHS if smoke_test else epochs,
        batch_size=batch_size,
        seed=seed,
        max_train_batches=SMOKE_BATCHES if smoke_test else None,
        max_validation_batches=SMOKE_BATCHES if smoke_test else None,
        smoke_test=smoke_test,
    )
    print(f"Clean replicate, seed {seed}, no augmentation: the recipe that "
          "trained the accepted baseline, into a new directory.")
    print(f"Output: {output_dir}")
    device = select_device(device)
    write_protocol_sidecar(output_dir, "clean", seed, device)
    summary = run_experiment(config, output_dir, device=device)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=f"{SMOKE_EPOCHS} epochs of {SMOKE_BATCHES} batches, for plumbing only",
    )
    args = parser.parse_args()

    if args.smoke_test:
        print("Smoke test: a few batches per epoch. Its metrics describe a "
              "handful of classes and are not a result.")

    run(args.seed, epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, output_dir=args.output_dir,
        smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
