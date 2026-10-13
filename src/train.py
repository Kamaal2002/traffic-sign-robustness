"""Clean baseline training for the custom GTSRB CNN.

Run from the project root with the project virtual environment:

    python -m src.train                 # full baseline run
    python -m src.train --smoke-test    # two short epochs, separate output dir

Training uses the track-aware training split and selects a checkpoint on the
track-aware validation split. The official test loader is never iterated or
evaluated here, so the test split influences no decision in this file.
"""

import argparse
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import IMAGE_SIZE, NUM_CLASSES, SEED, create_data_loaders
from .model import BaselineCNN

# Baseline training configuration. These are conventional defaults chosen once
# and recorded, not the result of a search: the point of this stage is a clean
# reference model, and a tuned baseline would make the later augmentation
# comparison harder to attribute.
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
SELECTION_METRIC = "validation_macro_f1"

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass
class TrainingConfig:
    """The inputs needed to reproduce a run from scratch.

    Only values that `run_experiment` actually consumes live here. The loss,
    optimizer, and scheduler used to be carried alongside them as descriptive
    strings, but nothing read those strings back, so they could drift away from
    the objects the run really built. They are now derived from the constructed
    objects instead, by `describe_training_objects`.
    """

    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    dropout: float = DROPOUT
    seed: int = SEED
    selection_metric: str = SELECTION_METRIC
    max_train_batches: int | None = None
    max_validation_batches: int | None = None
    smoke_test: bool = False


def set_seed(seed=SEED):
    """Seed every source of randomness this project actually draws from.

    Python's `random` and NumPy are seeded because scikit-learn and any later
    augmentation sampling use them; `torch.manual_seed` covers the CPU and MPS
    generators and model initialisation; the CUDA call is a no-op without a GPU.
    The training shuffle itself does not depend on the global torch generator,
    because the data pipeline owns a separate seed-42 generator.

    What this does and does not buy: a fresh run of this script on the same
    machine, device, and library versions reproduces the same numbers. It does
    not guarantee identical results across devices or backends — CPU, MPS, and
    CUDA use different kernels and reduction orders, so losses diverge in the
    low-order digits and can compound over epochs. Results are therefore
    reported together with the device they were produced on.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def select_device(requested=None):
    """Return the best available device, or the one explicitly requested."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def update_confusion(matrix, predictions, targets):
    """Accumulate a 43x43 confusion matrix, rows true and columns predicted."""
    flat = targets * NUM_CLASSES + predictions
    matrix += torch.bincount(flat, minlength=NUM_CLASSES ** 2).reshape(
        NUM_CLASSES, NUM_CLASSES
    )


def metrics_from_confusion(matrix):
    """Return accuracy and macro F1 from a confusion matrix.

    Macro F1 is the unweighted mean of the per-class F1 scores, so every one of
    the 43 classes counts equally regardless of how many images it has. The
    classes are imbalanced about 10:1 (300 validation images for the largest,
    30 for the smallest), so accuracy is dominated by the frequent classes
    while macro F1 responds to the rare ones, which is the behaviour the
    robustness study cares about.

    Policy: the mean is always taken over all NUM_CLASSES labels, because the
    matrix is allocated 43x43 regardless of which classes appear. This is
    deliberate and matters for the corruption experiments — a class that fails
    completely under a corruption contributes 0 to the mean rather than
    dropping out of it and leaving the score flattered. It matches
    `sklearn.metrics.f1_score(labels=range(NUM_CLASSES), average="macro",
    zero_division=0)` and not scikit-learn's default, which averages only over
    the labels present.

    F1 is computed as 2*TP / (predicted + actual), which is the same quantity
    as the precision/recall form without needing either as an intermediate. A
    class with no predictions and no true samples has a zero denominator and
    scores 0, consistent with the policy above. On the full validation and test
    splits every class is present, so the zero case only arises on a truncated
    subset such as the smoke test, whose macro F1 is a plumbing diagnostic and
    not a result.
    """
    true_positive = matrix.diagonal().to(torch.float64)
    predicted = matrix.sum(dim=0).to(torch.float64)
    actual = matrix.sum(dim=1).to(torch.float64)
    accuracy = (true_positive.sum() / matrix.sum()).item()
    denominator = predicted + actual
    per_class_f1 = torch.where(
        denominator > 0, 2 * true_positive / denominator, torch.zeros_like(denominator)
    )
    return accuracy, per_class_f1.mean().item()


def run_training_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    """One pass over the training split, updating parameters.

    The returned metrics come from the same forward passes used for the updates,
    so they are a running estimate taken with Dropout active, BatchNorm using
    batch statistics, and weights changing as the epoch proceeds. They are not a
    clean evaluation of the end-of-epoch model, and are reported as a training
    diagnostic only; every selection decision uses the validation numbers.
    """
    model.train()
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    loss_total = 0.0
    seen = 0
    for index, (images, labels) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item() * labels.size(0)
        seen += labels.size(0)
        update_confusion(matrix, logits.argmax(dim=1).cpu(), labels.cpu())
    accuracy, macro_f1 = metrics_from_confusion(matrix)
    return {"loss": loss_total / seen, "accuracy": accuracy, "macro_f1": macro_f1}


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    """Evaluate without touching parameters, gradients, or BatchNorm buffers.

    `model.eval()` disables Dropout and switches BatchNorm to its stored running
    statistics, which is the deployment-style inference the later corruption
    evaluation also uses. `torch.no_grad()` prevents graph construction.
    """
    model.eval()
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    loss_total = 0.0
    seen = 0
    for index, (images, labels) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_total += loss.item() * labels.size(0)
        seen += labels.size(0)
        update_confusion(matrix, logits.argmax(dim=1).cpu(), labels.cpu())
    accuracy, macro_f1 = metrics_from_confusion(matrix)
    return {"loss": loss_total / seen, "accuracy": accuracy, "macro_f1": macro_f1}


def selection_key(record, metric=SELECTION_METRIC):
    """Rank an epoch for checkpoint selection. Higher is better.

    Validation macro F1 is the primary metric: the 43 classes are imbalanced
    roughly 10:1, so accuracy is dominated by the frequent classes, while the
    research question is about per-class behaviour under corruption. Using the
    same metric to select the baseline and every later model keeps the
    before/after comparison on one footing.

    Ties are broken by validation accuracy, then by lower validation loss. The
    comparison below is strict, so if two epochs are identical on all three the
    earlier one is kept, which makes selection deterministic.

    The metric is looked up by the same name that gets recorded in the
    checkpoint, so the recorded name is what actually ranks the epochs and a
    wrong name fails loudly instead of mislabelling the result.
    """
    return (
        record[metric],
        record["validation_accuracy"],
        -record["validation_loss"],
    )


def describe_training_objects(criterion, optimizer, scheduler):
    """Describe the loss, optimizer, and scheduler that were actually built.

    Everything here is read off the live objects, so the saved metadata cannot
    claim an experiment that differs from the one that ran.

    The optimizer's own `defaults` are recorded wholesale rather than a chosen
    handful of fields: that keeps the function free of optimizer-specific
    knowledge and captures hyperparameters that would otherwise go unrecorded.
    `defaults` is used in preference to `param_groups`, because the scheduler
    rewrites the group's learning rate every epoch and the value that
    identifies the experiment is the configured starting point.
    """
    return {
        "loss": type(criterion).__name__,
        "optimizer": {
            "name": type(optimizer).__name__,
            # Tuples (betas) become lists so the block stays JSON-writable.
            **{key: list(value) if isinstance(value, tuple) else value
               for key, value in optimizer.defaults.items()},
        },
        # T_max and eta_min fully describe a cosine schedule. Reading them
        # directly means swapping the scheduler for one without them raises
        # here instead of silently recording an incomplete schedule.
        "scheduler": {
            "name": type(scheduler).__name__,
            "T_max": scheduler.T_max,
            "eta_min": scheduler.eta_min,
        },
    }


def build_checkpoint(model, config, description, record, device, split_sizes):
    """Assemble a checkpoint that can rebuild this experiment from scratch.

    `state_dict()` holds tensors only, so the class count and input size are
    stored explicitly alongside it rather than being recovered from the weights.
    The loss, optimizer, and scheduler blocks come from
    `describe_training_objects`, i.e. from the objects the run constructed.

    Optimizer *configuration* is saved, not optimizer state. Resuming an
    interrupted run bit-for-bit is not a requirement for this project: what has
    to be reproducible is the experiment from a fresh start, which the config
    and seed give. For the same reason the DataLoader shuffle generator's state
    is not saved; it is recreated from seed 42 by the data pipeline, so a fresh
    run replays the same epoch ordering from the beginning.
    """
    return {
        # Cloned, not referenced. state_dict() hands back tensors that share
        # storage with the live parameters, so holding this dict while training
        # continues would quietly leave the saved weights pointing at whatever
        # the model became by the final epoch rather than the selected one.
        "model_state": {name: tensor.detach().clone()
                        for name, tensor in model.state_dict().items()},
        "model_config": {
            "architecture": type(model).__name__,
            "num_classes": model.num_classes,
            "image_size": model.image_size,
            "dropout": config.dropout,
        },
        "loss": description["loss"],
        "optimizer_config": description["optimizer"],
        "scheduler_config": description["scheduler"],
        "training_config": asdict(config),
        "seed": config.seed,
        "epoch": record["epoch"],
        "validation_metrics": {
            "loss": record["validation_loss"],
            "accuracy": record["validation_accuracy"],
            "macro_f1": record["validation_macro_f1"],
        },
        "selection_metric": config.selection_metric,
        "split_sizes": split_sizes,
        "environment": {
            "device": str(device),
            # str(), because torch.__version__ is a TorchVersion object and
            # pickling one forces every reader onto weights_only=False.
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def run_experiment(config, output_dir, device=None, loaders=None):
    """Train the baseline and write the best checkpoint plus the history.

    The training loader is iterated only inside the epoch loop. Its shuffle
    generator advances on every `iter()` call, so a diagnostic pass over it
    before epoch 1 would silently change the first epoch's ordering; any
    pre-flight check therefore builds its own loaders instead.
    """
    if config.epochs < 1:
        raise ValueError(f"epochs must be at least 1; got {config.epochs}")

    set_seed(config.seed)
    device = device or select_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if loaders is None:
        loaders, _ = create_data_loaders(batch_size=config.batch_size, download=False)
    elif loaders["train"].batch_size != config.batch_size:
        # Injected loaders carry their own batch size, and the config's value is
        # what gets recorded, so a disagreement would put a batch size in the
        # saved metadata that the run never used.
        raise ValueError(
            f"injected train loader uses batch size {loaders['train'].batch_size}, "
            f"but the config records {config.batch_size}"
        )
    split_sizes = {
        "train_images": len(loaders["train"].dataset),
        "validation_images": len(loaders["validation"].dataset),
    }

    model = BaselineCNN(
        num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, dropout=config.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    # AdamW rather than Adam: Adam's weight_decay adds an L2 term to the
    # gradient, which then passes through the adaptive per-parameter scaling and
    # decays large-gradient weights less than small-gradient ones. AdamW applies
    # the decay directly to the weights, which is what "weight decay 1e-4" is
    # meant to mean.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # Cosine annealing over the fixed epoch budget. A constant 1e-3 leaves the
    # model oscillating around the optimum late in training, which would make
    # the selected baseline partly a matter of which epoch happened to land
    # well. Decaying to zero gives a settled final model. T_max is the epoch
    # count, so the schedule introduces no tuned parameter of its own.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    description = describe_training_objects(criterion, optimizer, scheduler)

    print(f"Device: {device} | torch {torch.__version__}")
    print(f"Train images: {split_sizes['train_images']:,}; "
          f"validation images: {split_sizes['validation_images']:,}")
    print(f"Config: {description['loss']} | {description['optimizer']['name']} "
          f"lr={description['optimizer']['lr']} "
          f"weight_decay={description['optimizer']['weight_decay']} | "
          f"{description['scheduler']['name']}(T_max={description['scheduler']['T_max']}) "
          f"| batch_size={config.batch_size} epochs={config.epochs} "
          f"seed={config.seed}")
    print(f"Checkpoint selection: {config.selection_metric} "
          f"(ties broken by accuracy, then lower loss, then earlier epoch)")

    history = []
    best_record = None
    best_checkpoint = None
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_training_epoch(
            model, loaders["train"], criterion, optimizer, device,
            max_batches=config.max_train_batches,
        )
        validation_metrics = evaluate(
            model, loaders["validation"], criterion, device,
            max_batches=config.max_validation_batches,
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)

        improved = best_record is None or (
            selection_key(record, config.selection_metric)
            > selection_key(best_record, config.selection_metric)
        )
        if improved:
            best_record = record
            best_checkpoint = build_checkpoint(
                model, config, description, record, device, split_sizes
            )
        print(
            f"epoch {epoch:>2}/{config.epochs} lr {learning_rate:.2e} | "
            f"train loss {record['train_loss']:.4f} acc {record['train_accuracy']:.4f} "
            f"f1 {record['train_macro_f1']:.4f} | "
            f"val loss {record['validation_loss']:.4f} acc "
            f"{record['validation_accuracy']:.4f} f1 "
            f"{record['validation_macro_f1']:.4f} | {record['seconds']:.1f}s"
            f"{'  <- best' if improved else ''}"
        )

    checkpoint_path = output_dir / "best_checkpoint.pt"
    torch.save(best_checkpoint, checkpoint_path)
    summary = {
        "config": asdict(config),
        "loss": description["loss"],
        "optimizer": description["optimizer"],
        "scheduler": description["scheduler"],
        "device": str(device),
        "torch": str(torch.__version__),
        "split_sizes": split_sizes,
        "selection_metric": config.selection_metric,
        "best_epoch": best_record["epoch"],
        "best_validation_accuracy": best_record["validation_accuracy"],
        "best_validation_macro_f1": best_record["validation_macro_f1"],
        "best_validation_loss": best_record["validation_loss"],
        "total_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint_path.name,
        "epochs": history,
    }
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nBest epoch {best_record['epoch']} by {config.selection_metric}: "
          f"validation accuracy {best_record['validation_accuracy']:.4f}, "
          f"macro F1 {best_record['validation_macro_f1']:.4f}, "
          f"loss {best_record['validation_loss']:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print("The official test loader was not iterated or evaluated in this run.")
    return summary


def load_checkpoint(path, device=None):
    """Rebuild the model recorded in a checkpoint and return it with the payload.

    The architecture is reconstructed from the saved `model_config` rather than
    inferred from tensor shapes, and loading is strict so a mismatch fails loudly
    instead of silently leaving parameters at their initial values.

    `weights_only=False` is needed for the accepted baseline checkpoint, which
    was written before the torch version was serialised as a plain string and
    therefore contains one non-tensor object. Checkpoints written now load
    under `weights_only=True`; the flag stays permissive so the accepted
    artifact remains readable rather than being rewritten.
    """
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = BaselineCNN(
        num_classes=config["num_classes"],
        image_size=config["image_size"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="two short epochs over a few batches, written to artifacts/smoke",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=2 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_train_batches=5 if args.smoke_test else None,
        max_validation_batches=5 if args.smoke_test else None,
        smoke_test=args.smoke_test,
    )
    default_dir = ARTIFACT_ROOT / ("smoke" if args.smoke_test else "baseline")
    output_dir = Path(args.output_dir) if args.output_dir else default_dir
    if args.smoke_test:
        print("Smoke test: a few batches per epoch, written to a separate "
              "directory. A plumbing check, not a baseline result — the "
              "truncated validation pass covers only a handful of classes, so "
              "its macro F1 is not a meaningful score.")
    run_experiment(config, output_dir, device=select_device(args.device))


if __name__ == "__main__":
    main()
