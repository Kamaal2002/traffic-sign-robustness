"""Train the augmented arms of the corruption-augmentation comparison.

Run from the project root with the project virtual environment:

    python -m src.train_augmented --policy balanced --seed 42
    python -m src.train_augmented --policy targeted --seed 42
    python -m src.train_augmented --policy targeted --smoke-test \
        --output-dir artifacts/smoke_augmented/targeted

A run refuses to start if its output directory already holds a checkpoint, a
history or a policy record, and the accepted baseline's directory is refused
outright, so no run can land on another run's results.

Every training control is inherited from `train.py` unchanged - the same model,
the same optimiser, scheduler, loss, epoch budget, batch size, split, and the
same checkpoint-selection rule - so the only difference between an arm here and
the accepted clean baseline is that the training images pass through an
augmentation policy. `run_experiment` is called, not reimplemented, which is
what makes that claim checkable rather than asserted.

Arm A of the comparison is not produced here. It is the existing clean seed-42
checkpoint, reused as-is, with seeds 43 and 44 trained later by `train.py` on
its original recipe.

Validation stays clean. The augmented dataset wraps the training split only,
the validation loader carries the unmodified clean pipeline, and checkpoint
selection reads clean validation macro F1 exactly as the baseline does, so no
corrupted measurement can influence which epoch is kept.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from .augmentation import (
    EpochLoader,
    POLICIES,
    build_augmented_loaders,
    expected_counts,
    format_policy,
    PROTOCOL_ARTIFACT,
    protocol_record,
)
from .corruptions import CORRUPTIONS, SEVERITY_LEVELS, corruption_parameter
from .data import SEED, build_preprocessing
from .evaluate_robustness import repository_state
from .train import (
    ARTIFACT_ROOT,
    BATCH_SIZE,
    EPOCHS,
    TrainingConfig,
    run_experiment,
    select_device,
)

# Directories that hold accepted results. `train.py` sends a run given no
# --output-dir to artifacts/baseline, which is where the accepted clean seed-42
# checkpoint lives, and the robustness tables measured from it sit beside it.
# Nothing in this experiment writes a checkpoint into either.
PROTECTED_DIRS = (ARTIFACT_ROOT / "baseline", ARTIFACT_ROOT / "robustness")

# The files a finished or running training run leaves behind. Any of them in a
# target directory means a previous run's output is already there.
TRAINING_ARTIFACTS = ("best_checkpoint.pt", "history.json", "policy.json",
                      PROTOCOL_ARTIFACT)

# The split the accepted baseline was trained on. Asserted rather than
# recomputed into the record, because an augmented arm trained on a different
# number of images would not be comparable to the baseline at all.
EXPECTED_TRAIN_IMAGES = 21300
EXPECTED_VALIDATION_IMAGES = 5340

SMOKE_EPOCHS = 2
SMOKE_BATCHES = 5


def check_output_dir(output_dir):
    """Resolve a training run's output directory, or refuse to start.

    Thirty epochs of training end in one `best_checkpoint.pt`, and the nine runs
    of this comparison differ only by a policy name and a seed in their output
    path. A mistyped or defaulted path would therefore overwrite a finished run
    with another one, and in the worst case the accepted baseline, which is the
    one artifact here that cannot be reproduced - every published number for arm
    A at seed 42 was measured from that file.

    So the directory is settled before any data is loaded: the accepted
    directories are refused outright, and any directory already holding
    training output is refused as well, rather than being written over. The
    check is here rather than in `train.py`, which produced the accepted
    baseline and is left byte-identical; every entry point this experiment adds
    goes through this function, and nothing is created until `claim_output_dir`
    has had the answer, so the check itself leaves no empty directory behind.
    """
    output_dir = Path(output_dir).resolve()
    for protected in PROTECTED_DIRS:
        protected = Path(protected).resolve()
        if output_dir == protected or protected in output_dir.parents:
            raise RuntimeError(
                f"{output_dir} is inside {protected}, which holds accepted "
                "results; training there would overwrite them. Choose a new "
                "directory.")
        if output_dir in protected.parents:
            raise RuntimeError(
                f"{output_dir} contains the accepted results in {protected}; "
                "training runs get their own directory.")
    existing = sorted(name for name in TRAINING_ARTIFACTS
                      if (output_dir / name).exists())
    if existing:
        raise RuntimeError(
            f"{output_dir} already holds {', '.join(existing)} from an earlier "
            "run; training would overwrite that checkpoint. Delete the "
            "directory deliberately or pass a new --output-dir.")
    return output_dir


def claim_output_dir(output_dir):
    """Create a training run's output directory once it has been checked."""
    output_dir = check_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_policy(name):
    """Look up one of the two declared policies by name."""
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; expected one of "
                         f"{sorted(POLICIES)}")
    return POLICIES[name]


def _source_dataset(dataset):
    """Unwrap `Subset` and `AugmentedDataset` layers down to the base dataset."""
    seen = set()
    while True:
        if id(dataset) in seen:
            raise RuntimeError("dataset wrappers form a cycle")
        seen.add(id(dataset))
        inner = getattr(dataset, "dataset", None)
        if inner is None:
            return dataset
        dataset = inner


def _wrapper_chain(loader):
    """Every wrapper class name from the loader's dataset down to the source."""
    dataset = loader.dataset
    names = []
    while dataset is not None:
        names.append(type(dataset).__name__)
        dataset = getattr(dataset, "dataset", None)
    return names


def check_no_leakage(loaders, split_info):
    """Verify the properties that would invalidate the comparison if violated.

    Four separate failures are checked, because each one would be invisible in
    the training logs and would show up only as an implausibly good number:

    * augmentation reaching the validation split, which would make the
      checkpoint-selection metric a corrupted-data metric;
    * validation carrying anything other than the frozen clean pipeline;
    * the train and validation splits sharing images or, worse, sharing tracks,
      since two frames of one physical sign are near-duplicates;
    * the official test split being constructed at all.

    Returns the record of what was checked, so the run's metadata carries the
    evidence rather than a claim that it was verified.
    """
    train_loader, validation_loader = loaders["train"], loaders["validation"]

    train_chain = _wrapper_chain(train_loader)
    validation_chain = _wrapper_chain(validation_loader)
    if "AugmentedDataset" not in train_chain:
        raise RuntimeError(f"the train loader is not augmented: {train_chain}")
    if "AugmentedDataset" in validation_chain:
        raise RuntimeError("augmentation reaches the validation split: "
                           f"{validation_chain}")
    if not isinstance(train_loader, EpochLoader):
        raise RuntimeError("the train loader does not advance the augmentation "
                           f"epoch; got {type(train_loader).__name__}")
    if isinstance(validation_loader, EpochLoader):
        raise RuntimeError("the validation loader is wrapped in an EpochLoader")

    validation_transform = str(_source_dataset(validation_loader.dataset).transform)
    if validation_transform != str(build_preprocessing()):
        raise RuntimeError("the validation split does not carry the frozen clean "
                           f"preprocessing pipeline:\n{validation_transform}")

    splits = {}
    for role, loader in (("train", train_loader), ("validation", validation_loader)):
        source = _source_dataset(loader.dataset)
        split = getattr(source, "_split", None)
        if split != "train":
            raise RuntimeError(f"the {role} loader reads the {split!r} split; "
                               "this run must not construct the official test "
                               "split")
        splits[role] = split

    train_indices = np.asarray(split_info["train_indices"])
    validation_indices = np.asarray(split_info["validation_indices"])
    shared_images = np.intersect1d(train_indices, validation_indices)
    if shared_images.size:
        raise RuntimeError(f"{shared_images.size} images appear in both splits")

    tracks = np.asarray(split_info["tracks"])
    train_tracks = set(tracks[train_indices].tolist())
    validation_tracks = set(tracks[validation_indices].tolist())
    shared_tracks = train_tracks & validation_tracks
    if shared_tracks:
        raise RuntimeError(f"{len(shared_tracks)} tracks appear in both splits")

    if len(train_loader.dataset) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(f"expected {EXPECTED_TRAIN_IMAGES} training images; "
                           f"got {len(train_loader.dataset)}")
    if len(validation_loader.dataset) != EXPECTED_VALIDATION_IMAGES:
        raise RuntimeError(f"expected {EXPECTED_VALIDATION_IMAGES} validation "
                           f"images; got {len(validation_loader.dataset)}")

    return {
        "train_wrappers": train_chain,
        "validation_wrappers": validation_chain,
        "validation_is_augmented": False,
        "validation_pipeline_matches_clean": True,
        "dataset_splits_constructed": sorted(set(splits.values())),
        "official_test_split_constructed": False,
        "shared_images": 0,
        "shared_tracks": 0,
        "train_images": len(train_loader.dataset),
        "validation_images": len(validation_loader.dataset),
        "train_tracks": len(train_tracks),
        "validation_tracks": len(validation_tracks),
    }


def write_protocol_sidecar(output_dir, arm, seed, device):
    """Record which protocol version this run was started under.

    Written before training, so it describes the files the run actually loaded
    rather than whatever the repository looks like once the run has finished.
    A drift from the manifest is printed and recorded rather than suppressed;
    it is not fatal, because a smoke test over edited code is a normal thing to
    do, and a reported drift is what distinguishes that from a protocol run.
    """
    record = {
        "arm": arm,
        "seed": seed,
        "device": str(device),
        "protocol": protocol_record(),
        "repository": repository_state(),
    }
    path = Path(output_dir) / PROTOCOL_ARTIFACT
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Protocol record: {path}")
    differences = record["protocol"]["differs_from_manifest"]
    if differences:
        print("  warning: these protocol files differ from "
              f"{record['protocol']['manifest']}, so this run is not a run of "
              f"the frozen protocol: {', '.join(differences)}")
    return path, record


def policy_record(policy, config, device, leakage, images):
    """Everything needed to reproduce and audit one augmented run."""
    return {
        "policy": policy.describe(),
        "policy_table": format_policy(policy),
        "expected_counts_per_epoch": expected_counts(policy, images),
        "family_parameters": {
            name: {"parameter": CORRUPTIONS[name].parameter,
                   "levels": [corruption_parameter(name, severity)
                              for severity in SEVERITY_LEVELS]}
            for name, _ in policy.family_weights
        },
        "augmentation_base_seed": config.seed,
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "seed": config.seed,
            "selection_metric": config.selection_metric,
            "max_train_batches": config.max_train_batches,
            "max_validation_batches": config.max_validation_batches,
            "smoke_test": config.smoke_test,
        },
        "selection": {
            "metric": config.selection_metric,
            "measured_on": "clean validation split",
            "corrupted_validation_used": False,
        },
        "leakage_checks": leakage,
        "device": str(device),
        "repository": repository_state(),
        "environment": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def run(policy_name, seed=SEED, output_dir=None, epochs=EPOCHS, device=None,
        num_workers=0, smoke_test=False, batch_size=BATCH_SIZE):
    """Train one augmented arm and write its checkpoint, history, and recipe."""
    policy = resolve_policy(policy_name)
    device = select_device(device)
    if output_dir is None:
        output_dir = ARTIFACT_ROOT / f"augmented_{policy.name}_seed{seed}"
    output_dir = claim_output_dir(output_dir)

    config = TrainingConfig(
        epochs=SMOKE_EPOCHS if smoke_test else epochs,
        batch_size=batch_size,
        seed=seed,
        max_train_batches=SMOKE_BATCHES if smoke_test else None,
        max_validation_batches=SMOKE_BATCHES if smoke_test else None,
        smoke_test=smoke_test,
    )

    # The augmentation seed is the training seed, so the three replicates draw
    # genuinely independent augmentation streams rather than reusing one stream
    # under different weight initialisations.
    loaders, split_info, augmented = build_augmented_loaders(
        policy, base_seed=config.seed, batch_size=config.batch_size,
        num_workers=num_workers,
    )
    leakage = check_no_leakage(loaders, split_info)

    print(format_policy(policy))
    print(f"Augmentation seed namespace: base_seed={config.seed}, "
          f"resampled every epoch per image")
    print("Checkpoint selection reads clean validation macro F1; corrupted "
          "validation performance is not computed in this run.")

    write_protocol_sidecar(output_dir, policy.name, config.seed, device)

    record = policy_record(policy, config, device, leakage,
                           len(loaders["train"].dataset))
    record_path = output_dir / "policy.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    summary = run_experiment(config, output_dir, device=device, loaders=loaders)

    # The augmentation epoch advances once per `iter()` on the train loader. If
    # the training loop ever stopped calling it once per epoch, every epoch
    # after the first would silently reuse one fixed set of realisations, which
    # is exactly the failure this experiment cannot afford to have gone
    # unnoticed.
    if loaders["train"].epochs != config.epochs:
        raise RuntimeError(f"the train loader was iterated "
                           f"{loaders['train'].epochs} times over "
                           f"{config.epochs} epochs")
    if augmented.epoch != config.epochs:
        raise RuntimeError(f"the augmentation epoch ended at {augmented.epoch} "
                           f"after {config.epochs} epochs")

    record["realised"] = {
        "train_loader_iterations": loaders["train"].epochs,
        "final_augmentation_epoch": augmented.epoch,
        "best_epoch": summary["best_epoch"],
        "total_seconds": summary["total_seconds"],
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    print(f"Policy record: {record_path}")
    return summary, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=sorted(POLICIES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="dataloader workers; the augmentation epoch is set before each "
             "iter() in the parent process, so workers inherit it",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=f"{SMOKE_EPOCHS} epochs of {SMOKE_BATCHES} batches, for plumbing only",
    )
    args = parser.parse_args()

    if args.smoke_test:
        print("Smoke test: a few batches per epoch. This checks that the "
              "augmented pipeline runs and that the epoch advances; its "
              "metrics describe a handful of classes and are not a result.")

    run(args.policy, seed=args.seed, output_dir=args.output_dir,
        epochs=args.epochs, device=args.device, num_workers=args.num_workers,
        smoke_test=args.smoke_test, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
