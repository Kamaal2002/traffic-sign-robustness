"""Pre-flight checks for the training pipeline.

Run from the project root before launching a baseline run:

    python -m src.verify_training

This runs a short smoke test and then establishes the properties the baseline
run depends on: that an optimizer step moves parameters, that BatchNorm
statistics accumulate during training and are frozen during validation, that
validation leaves the model untouched and runs with Dropout disabled, that the
official test loader is never iterated, that a captured checkpoint is decoupled
from the live model, that saved metadata is read from the constructed training
objects, and that generated artifacts are ignored by Git. Smoke-test output
goes to a separate directory and is not a baseline result.
"""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn

from .data import IMAGE_SIZE, NUM_CLASSES, create_data_loaders
from .model import BaselineCNN
from .train import (
    ARTIFACT_ROOT,
    SELECTION_METRIC,
    TrainingConfig,
    build_checkpoint,
    describe_training_objects,
    evaluate,
    load_checkpoint,
    metrics_from_confusion,
    run_experiment,
    selection_key,
    run_training_epoch,
    select_device,
    set_seed,
    update_confusion,
)


class CountingLoader:
    """Wrap a DataLoader to record how often it is iterated.

    The point is the test loader: this makes "the official test split was not
    touched" an observed fact rather than a claim about the source code. It also
    captures the first training batch, so the epoch-1 ordering can be compared
    against a freshly built loader.
    """

    def __init__(self, loader):
        self.loader = loader
        self.iterations = 0
        self.first_batch = None

    def __iter__(self):
        self.iterations += 1
        for index, batch in enumerate(self.loader):
            if self.iterations == 1 and index == 0:
                self.first_batch = (batch[0].clone(), batch[1].clone())
            yield batch

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def batch_size(self):
        return self.loader.batch_size


def check_metric_implementation(device):
    """Cross-check accuracy and macro F1 against scikit-learn.

    The confusion-matrix implementation in train.py is small enough to be wrong
    in a way that still looks plausible across epochs, so it is compared against
    an independent implementation.

    The comparison is driven by synthetic predictions that cover all 43 classes
    with partial agreement, because a real untrained model on the unshuffled
    validation split predicts almost nothing correctly: comparing 0.0 against
    0.0 would pass without exercising a single per-class F1 term. Real model
    predictions are checked too, but only after the synthetic case has confirmed
    the arithmetic on a non-degenerate input.
    """
    generator = torch.Generator().manual_seed(7)
    targets = torch.randint(0, NUM_CLASSES, (4000,), generator=generator)
    predictions = targets.clone()
    corrupt = torch.rand(targets.shape, generator=generator) > 0.35
    predictions[corrupt] = torch.randint(
        0, NUM_CLASSES, (int(corrupt.sum()),), generator=generator
    )
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    update_confusion(matrix, predictions, targets)
    accuracy, macro_f1 = metrics_from_confusion(matrix)
    reference_f1 = f1_score(
        targets.numpy(), predictions.numpy(),
        labels=range(NUM_CLASSES), average="macro", zero_division=0,
    )
    assert abs(accuracy - (targets == predictions).double().mean().item()) < 1e-12
    assert abs(macro_f1 - reference_f1) < 1e-12, (macro_f1, reference_f1)
    # Guard against a vacuous comparison: every class must appear on both sides
    # and the metrics must sit strictly between the trivial extremes.
    assert len(targets.unique()) == NUM_CLASSES and len(predictions.unique()) == NUM_CLASSES
    assert 0.05 < accuracy < 0.95 and 0.05 < macro_f1 < 0.95
    assert matrix.sum().item() == targets.numel()
    print(f"Metrics agree with scikit-learn on {targets.numel()} synthetic "
          f"predictions spanning all {NUM_CLASSES} classes: accuracy "
          f"{accuracy:.6f}, macro F1 {macro_f1:.6f}")

    # Real predictions from the pipeline, to confirm the same code path works on
    # loader output rather than only on hand-made tensors.
    set_seed()
    model = BaselineCNN().to(device)
    loaders, _ = create_data_loaders(batch_size=64, download=False)
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    real_targets, real_predictions = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loaders["validation"]:
            predicted = model(images.to(device)).argmax(dim=1).cpu()
            update_confusion(matrix, predicted, labels)
            real_targets.append(labels)
            real_predictions.append(predicted)
    real_targets = torch.cat(real_targets)
    real_predictions = torch.cat(real_predictions)
    accuracy, macro_f1 = metrics_from_confusion(matrix)
    reference_f1 = f1_score(
        real_targets.numpy(), real_predictions.numpy(),
        labels=range(NUM_CLASSES), average="macro", zero_division=0,
    )
    assert matrix.sum().item() == len(loaders["validation"].dataset)
    assert len(real_targets.unique()) == NUM_CLASSES
    assert abs(accuracy - (real_targets == real_predictions).double().mean().item()) < 1e-12
    assert abs(macro_f1 - reference_f1) < 1e-12, (macro_f1, reference_f1)
    print(f"  and on all {matrix.sum().item():,} real validation predictions from "
          f"an untrained model: accuracy {accuracy:.6f}, macro F1 {macro_f1:.6f} "
          f"(chance level, as expected before training)")

    check_trivial_baselines(real_targets)


def check_trivial_baselines(validation_targets):
    """Quantify, on the real validation labels, why macro F1 is the metric.

    Both reference points below are computed from the labels this split
    actually contains, not from a constructed distribution. That distinction
    matters: GTSRB is imbalanced roughly 10:1 between its largest and smallest
    classes, but no single class is anywhere near a majority of the split, so a
    constant predictor is weak on accuracy too. The argument for macro F1 is
    not that accuracy can be fooled by one dominant class; it is that accuracy
    is insensitive to the rare classes, which is what the second measurement
    below shows.
    """
    counts = torch.bincount(validation_targets, minlength=NUM_CLASSES)
    total = validation_targets.numel()

    # Always answer the most frequent class. argmax takes the lowest index on a
    # tie, and several classes share the largest count, so all of them give the
    # same accuracy.
    majority = int(counts.argmax())
    constant = torch.full_like(validation_targets, majority)
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    update_confusion(matrix, constant, validation_targets)
    constant_accuracy, constant_f1 = metrics_from_confusion(matrix)
    reference_f1 = f1_score(
        validation_targets.numpy(), constant.numpy(),
        labels=range(NUM_CLASSES), average="macro", zero_division=0,
    )
    assert abs(constant_f1 - reference_f1) < 1e-12, (constant_f1, reference_f1)
    assert abs(constant_accuracy - counts[majority].item() / total) < 1e-12
    assert constant_f1 < constant_accuracy
    print(f"  always predicting the most frequent class ({majority}, with "
          f"{counts[majority].item()} of {total} validation images, "
          f"{(counts == counts[majority]).sum().item()} classes tied at that "
          f"count) scores accuracy {constant_accuracy:.6f} and macro F1 "
          f"{constant_f1:.6f}")

    # What one completely failed class costs. A perfect predictor is degraded by
    # sending every image of the smallest class to the largest one; accuracy
    # loses that class's share of the split, macro F1 loses most of 1/43.
    smallest = int(counts.argmin())
    degraded = validation_targets.clone()
    degraded[validation_targets == smallest] = majority
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    update_confusion(matrix, degraded, validation_targets)
    degraded_accuracy, degraded_f1 = metrics_from_confusion(matrix)
    accuracy_cost = 1.0 - degraded_accuracy
    f1_cost = 1.0 - degraded_f1
    assert abs(accuracy_cost - counts[smallest].item() / total) < 1e-12
    assert f1_cost > 3 * accuracy_cost
    print(f"  failing class {smallest} entirely ({counts[smallest].item()} of "
          f"{total} images) costs {accuracy_cost:.6f} accuracy but "
          f"{f1_cost:.6f} macro F1, so macro F1 is the metric that notices a "
          f"rare class being lost")


def check_training_step(device):
    """Parameters move on an optimizer step; BatchNorm accumulates; eval does not."""
    set_seed()
    model = BaselineCNN().to(device)
    loaders, _ = create_data_loaders(batch_size=32, download=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    parameters_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers_before = {n: b.detach().clone() for n, b in model.named_buffers()}
    run_training_epoch(model, loaders["train"], criterion, optimizer, device, max_batches=2)

    unchanged = [n for n, p in model.named_parameters()
                 if torch.equal(parameters_before[n], p)]
    assert not unchanged, f"optimizer step left parameters untouched: {unchanged}"
    statistics = [n for n, b in model.named_buffers()
                  if n.endswith(("running_mean", "running_var"))
                  and not torch.equal(buffers_before[n], b)]
    batch_norm_layers = [n for n, m in model.named_modules()
                         if isinstance(m, nn.BatchNorm2d)]
    stale = [layer for layer in batch_norm_layers
             if not any(name.startswith(f"{layer}.") for name in statistics)]
    assert not stale, f"running statistics did not update: {stale}"
    print(f"Two optimizer steps moved all {len(parameters_before)} parameter tensors "
          f"and updated the running statistics of all {len(batch_norm_layers)} "
          f"BatchNorm layers")

    # Validation must be inert: no parameter, no buffer, and no gradient.
    model.zero_grad(set_to_none=True)
    parameters_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers_before = {n: b.detach().clone() for n, b in model.named_buffers()}
    first = evaluate(model, loaders["validation"], criterion, device, max_batches=4)
    changed_parameters = [n for n, p in model.named_parameters()
                          if not torch.equal(parameters_before[n], p)]
    changed_buffers = [n for n, b in model.named_buffers()
                       if not torch.equal(buffers_before[n], b)]
    gradients = [n for n, p in model.named_parameters() if p.grad is not None]
    assert not changed_parameters, changed_parameters
    assert not changed_buffers, changed_buffers
    assert not gradients, gradients
    print(f"Validation changed no parameter, no buffer "
          f"(all {len(buffers_before)} identical), and produced no gradient")

    # Dropout off under eval(): repeating validation on the same data must give
    # bit-identical metrics, and every module must be out of training mode.
    second = evaluate(model, loaders["validation"], criterion, device, max_batches=4)
    assert first == second, (first, second)
    training_modules = [n for n, m in model.named_modules() if m.training]
    assert not training_modules, training_modules
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert dropout_layers and not any(m.training for m in dropout_layers)
    print(f"Dropout disabled under eval(): two validation passes returned "
          f"identical loss {first['loss']:.10f} and no module is in training mode")


def check_checkpoint_capture(device):
    """A captured checkpoint must be decoupled from the model that produced it.

    `state_dict()` returns tensors sharing storage with the module's parameters,
    so a best-checkpoint dict held in memory while training continues would end
    up holding the final epoch's weights under the selected epoch's label. A
    smoke test cannot see this when its best epoch is also its last, so the
    decoupling is checked directly: capture a checkpoint, move every parameter
    and buffer, and require the captured tensors to stay put.
    """
    set_seed()
    model = BaselineCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    record = {
        "epoch": 1, "validation_loss": 1.0,
        "validation_accuracy": 0.5, "validation_macro_f1": 0.4,
    }
    checkpoint = build_checkpoint(
        model, TrainingConfig(),
        describe_training_objects(criterion, optimizer, scheduler),
        record, device, {"train_images": 0, "validation_images": 0},
    )
    captured = {name: tensor.clone()
                for name, tensor in checkpoint["model_state"].items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
        for _, buffer in model.named_buffers():
            buffer.add_(1)
    drifted = [name for name, tensor in checkpoint["model_state"].items()
               if not torch.equal(captured[name], tensor)]
    assert not drifted, f"checkpoint tensors followed the live model: {drifted}"
    assert set(checkpoint["model_state"]) == set(model.state_dict())
    print(f"Captured checkpoint is decoupled from the live model: all "
          f"{len(captured)} state tensors stayed fixed while every parameter "
          f"and buffer moved")


def check_recorded_metadata():
    """Saved metadata must follow the training objects, not a separate label.

    Deliberately unusual values are used so the description cannot pass by
    coincidentally matching the project's configured defaults: if the recorded
    block still said AdamW at 1e-3 it would be echoing a constant rather than
    reading the object in front of it.
    """
    model = BaselineCNN()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.07, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=7, eta_min=0.01
    )
    description = describe_training_objects(criterion, optimizer, scheduler)
    assert description["loss"] == "NLLLoss", description
    assert description["optimizer"]["name"] == "SGD", description
    assert description["optimizer"]["lr"] == 0.07
    assert description["optimizer"]["weight_decay"] == 0.03
    assert description["optimizer"]["momentum"] == 0, description
    assert description["scheduler"] == {
        "name": "CosineAnnealingLR", "T_max": 7, "eta_min": 0.01
    }, description

    # The recorded learning rate is the configured one, not whatever the
    # schedule has moved the live param group to by the time of capture.
    optimizer.param_groups[0]["lr"] = 0.001
    assert describe_training_objects(criterion, optimizer, scheduler)[
        "optimizer"]["lr"] == 0.07

    # Regression guard: no free-text description of these objects may live in
    # the configuration again, because nothing would keep it honest.
    fields = set(asdict(TrainingConfig()))
    assert not fields & {"loss", "optimizer", "scheduler"}, fields

    # An injected loader whose batch size disagrees with the config must be
    # rejected rather than recorded, for the same reason: the saved metadata has
    # to describe the run that happened.
    loaders, _ = create_data_loaders(batch_size=32, download=False)
    try:
        run_experiment(TrainingConfig(epochs=1, batch_size=64), ARTIFACT_ROOT / "smoke",
                       device=torch.device("cpu"), loaders=loaders)
    except ValueError as error:
        assert "batch size 32" in str(error), error
    else:
        raise AssertionError("a mismatched injected batch size was accepted")

    # The recorded selection metric is the one that ranks epochs.
    assert SELECTION_METRIC == TrainingConfig().selection_metric
    record = {"validation_macro_f1": 0.9, "validation_accuracy": 0.8,
              "validation_loss": 0.1}
    assert selection_key(record, SELECTION_METRIC)[0] == record[SELECTION_METRIC]
    print("Saved metadata is read from the constructed loss, optimizer, and "
          "scheduler: an SGD/NLLLoss/T_max=7 setup was described as such, the "
          "configuration carries no competing labels, and an injected loader "
          "whose batch size disagreed with the config was rejected")


def check_smoke_run(device):
    """Run the training script end to end on a few batches."""
    loaders, _ = create_data_loaders(batch_size=64, download=False)
    counted = {name: CountingLoader(loader) for name, loader in loaders.items()}
    config = TrainingConfig(
        epochs=2, max_train_batches=5, max_validation_batches=5, smoke_test=True
    )
    output_dir = ARTIFACT_ROOT / "smoke"
    print(f"\n--- smoke test ({config.epochs} epochs, "
          f"{config.max_train_batches} train batches each) ---")
    summary = run_experiment(config, output_dir, device=device, loaders=counted)
    print("--- end smoke test ---\n")

    assert counted["test"].iterations == 0, counted["test"].iterations
    assert counted["train"].iterations == config.epochs, counted["train"].iterations
    assert counted["validation"].iterations == config.epochs
    print(f"Loader iterations during the run: train {counted['train'].iterations}, "
          f"validation {counted['validation'].iterations}, "
          f"test {counted['test'].iterations}")

    # The training loader's shuffle generator advances on every iter(), so the
    # first batch of epoch 1 must match a freshly built loader's first batch.
    # If any diagnostic had consumed the loader first, these would differ.
    fresh_loaders, _ = create_data_loaders(batch_size=64, download=False)
    expected_images, expected_labels = next(iter(fresh_loaders["train"]))
    observed_images, observed_labels = counted["train"].first_batch
    assert torch.equal(expected_labels, observed_labels), (
        expected_labels[:8].tolist(), observed_labels[:8].tolist()
    )
    assert torch.equal(expected_images, observed_images)
    print("Epoch 1 started on the training loader's first shuffled batch; no "
          "diagnostic pass consumed it beforehand")

    history = json.loads((output_dir / "history.json").read_text())
    assert len(history["epochs"]) == config.epochs
    assert history["best_epoch"] in {record["epoch"] for record in history["epochs"]}
    assert history["config"]["smoke_test"] is True
    model, checkpoint = load_checkpoint(output_dir / "best_checkpoint.pt", device=device)
    assert checkpoint["model_config"]["num_classes"] == NUM_CLASSES
    assert checkpoint["model_config"]["image_size"] == IMAGE_SIZE
    assert checkpoint["epoch"] == history["best_epoch"]
    assert model.num_classes == NUM_CLASSES and model.image_size == IMAGE_SIZE
    with torch.no_grad():
        logits = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    assert logits.shape == (2, NUM_CLASSES)
    print(f"Smoke checkpoint reloaded from {output_dir.name}/: epoch "
          f"{checkpoint['epoch']}, forward pass {tuple(logits.shape)}")
    return summary


def check_artifacts_ignored():
    """Every path the run writes must be ignored by Git."""
    paths = [
        ARTIFACT_ROOT.name + "/",
        f"{ARTIFACT_ROOT.name}/baseline/best_checkpoint.pt",
        f"{ARTIFACT_ROOT.name}/baseline/history.json",
        f"{ARTIFACT_ROOT.name}/smoke/best_checkpoint.pt",
        f"{ARTIFACT_ROOT.name}/smoke/history.json",
    ]
    root = Path(__file__).resolve().parents[1]
    for path in paths:
        result = subprocess.run(
            ["git", "check-ignore", "-v", path],
            cwd=root, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"NOT ignored by Git: {path}"
        print(f"  ignored by {result.stdout.split(chr(9))[0]}: {path}")
    tracked = subprocess.run(
        ["git", "ls-files", ARTIFACT_ROOT.name],
        cwd=root, capture_output=True, text=True,
    )
    assert not tracked.stdout.strip(), tracked.stdout
    print("No artifact path is tracked by Git.")


def main():
    device = select_device()
    print(f"Verifying the training pipeline on {device}.\n")
    check_metric_implementation(device)
    check_training_step(device)
    check_checkpoint_capture(device)
    check_recorded_metadata()
    check_smoke_run(device)
    print("Artifact paths:")
    check_artifacts_ignored()
    print("\nAll training pipeline checks passed. The smoke-test model is a "
          "throwaway and is not the baseline result.")


if __name__ == "__main__":
    main()
