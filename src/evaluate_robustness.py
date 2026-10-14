"""Evaluate the frozen clean baseline under the frozen corruption suite.

    .venv/bin/python -m src.evaluate_robustness

This measures how the accepted epoch-21 checkpoint degrades as each controlled
corruption gets more severe. It is a measurement stage only: nothing here
trains, fine-tunes, augments, or adapts the model, and the official test split
is never iterated. Evaluation uses the track-aware validation split, which the
checkpoint was selected on, so these numbers describe degradation relative to a
known reference point rather than generalisation to unseen data.

Everything that defines an experiment comes from the frozen modules: the model
from `model.py` through `train.py`'s loader, the split and preprocessing from
`data.py`, the severity table and per-sample seeds from `corruptions.py`, and
the metric formulas from `train.py`. This file adds the sweep, the bookkeeping,
and the integrity checks, and defines no corruption parameter of its own.

Deployment-style inference: `model.eval()` keeps BatchNorm on the running
statistics learned during clean training, so the corrupted inputs never update
them. Adapting those statistics to corrupted data is a different experiment and
would make this one unable to answer what the deployed clean model actually
does.

One result needs a caveat carried with it wherever it is quoted, so it is
recorded in the saved metadata as well: see `OCCLUSION_NOTE` below.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .corruptions import (
    CORRUPTIONS,
    SEVERITY_LEVELS,
    VALIDATION_NAMESPACE,
    CorruptedDataset,
    build_validation_observations,
    corruption_parameter,
    format_specification,
    sample_seed,
)
from .data import NUM_CLASSES, SEED, create_data_loaders, read_training_index
from .train import (
    load_checkpoint,
    metrics_from_confusion,
    per_class_from_confusion,
    select_device,
    update_confusion,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "baseline" / "best_checkpoint.pt"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "robustness" / "baseline"

# The accepted baseline, pinned so a swapped or re-trained artifact fails here
# instead of quietly producing a different robustness curve.
EXPECTED_CHECKPOINT_MD5 = "7c6ea3f9773d7a644235463d6a02bf0f"
EXPECTED_EPOCH = 21
EXPECTED_VALIDATION_SIZE = 5340
# Recorded in the accepted checkpoint and reported in src/README.md.
ACCEPTED_CLEAN = {"loss": 0.047733, "accuracy": 0.987640, "macro_f1": 0.974451}
# Clean reproduction has to be exact on the same device and build; this slack is
# only to give a readable failure rather than a bit-comparison assert.
CLEAN_TOLERANCE = 1e-6
# The two metric paths sum the same counts in different orders, so they agree to
# float64 rounding rather than exactly.
METRIC_TOLERANCE = 1e-12

BATCH_SIZE = 64
CLEAN_CONDITION = "clean"


# Carried into metadata.json so the number cannot travel without the caveat.
OCCLUSION_NOTE = (
    "Occlusion severities are nested about one anchor sampled per image, and "
    "that anchor is constrained so the severity-5 patch fits inside the frame. "
    "Low-severity patches are therefore confined to the central region of the "
    "image (at 64x64 the severity-1 patch cannot leave the central 43x43 box), "
    "and since GTSRB crops are centred on the sign the patch lands on sign "
    "content far more often than a uniformly placed one would. These numbers "
    "are not comparable to random erasing, which draws an independent uniform "
    "position at every size; read them as the cost of occluding the middle of "
    "the sign, not as the cost of occluding x% of an arbitrary region."
)

# Ten validation classes hold a single track, so their 30 images are 30 frames
# of one physical sign. Per-class scores for those classes are observations
# about that sign.
SINGLE_TRACK_WARNING = (
    "Per-class support counts frames, not independent signs: a track is 30 "
    "near-duplicate frames of one physical sign. Classes with one validation "
    "track have an effective sample size of 1, so their per-class scores are "
    "single-sign observations rather than class-level evidence."
)


class CountingDataset:
    """Pass-through dataset wrapper that counts how many samples are read.

    Wrapping the official test split with this turns "the test split was never
    read" into a measurement rather than a claim about control flow: reads
    through the wrapped loader and direct `wrapper[i]` indexing both increment
    the counter. Items and length pass through untouched, so the data itself is
    unchanged.

    What the counter does and does not cover, stated precisely because the
    number ends up in the saved metadata. It covers reads that go through this
    wrapper, in this process, for the dataset instances `create_data_loaders`
    returned on this call. It does not intercept every possible alias of the
    underlying dataset: `loader.sampler.data_source` and this wrapper's own
    `.dataset` attribute still reference the unwrapped object, a separately
    constructed `GTSRB(split="test")` is outside the instrumentation entirely,
    and with `num_workers > 0` a worker's copy keeps its own count. So a zero
    here is evidence that the instrumented path was not used, alongside the
    separate iteration count, not a structural proof that no test image could
    have been reached by any route.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.reads = 0

    def __getitem__(self, index):
        self.reads += 1
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class CountingLoader:
    """A DataLoader that records how many times it was iterated.

    Paired with `CountingDataset`, this turns "the official test split was not
    evaluated" into two numbers in the saved metadata rather than a claim
    resting on reading the code.

    It is a lightweight wrapper, not a DataLoader: it counts iteration, forwards
    ordinary attribute reads to the loader it holds, and defines `__len__`. That
    is everything this evaluation does with a loader. Attribute writes, indexing
    and `isinstance(..., DataLoader)` are not forwarded, and nothing here needs
    them.
    """

    def __init__(self, loader):
        self.loader = loader
        self.iterations = 0

    def __getattr__(self, name):
        # Only reached for attributes this wrapper does not define. The guard
        # matters during unpickling and before __init__ binds `loader`, when
        # `self.loader` would otherwise re-enter this method forever.
        if name == "loader":
            raise AttributeError(name)
        return getattr(self.loader, name)

    def __iter__(self):
        self.iterations += 1
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def instrument_loaders(loaders):
    """Wrap every loader's iteration and every dataset's reads.

    The caller rebinds its own name to the wrapped dict, so the loaders it goes
    on to use are the instrumented ones. Replacing `.dataset` before any
    iteration is what makes the read count meaningful, because a map-style
    DataLoader builds its fetcher from that attribute when the iterator is
    created. Aliases the swap does not reach are listed in `CountingDataset`.

    `DataLoader.__setattr__` refuses to set `dataset` after construction, since
    in general a swapped dataset can desync the sampler. `object.__setattr__`
    goes around that guard, which is safe here only because the wrapper is a
    pass-through of identical length over the same samples: the sampler indices,
    the dataset kind, and every item returned are unchanged.
    """
    wrapped = {}
    for name, loader in loaders.items():
        expected = len(loader.dataset)
        object.__setattr__(loader, "dataset", CountingDataset(loader.dataset))
        # After the swap, so it tests the installed wrapper rather than the
        # object it was built from: the loader must still report the same
        # number of samples it held before instrumentation.
        if len(loader.dataset) != expected:
            raise RuntimeError(f"instrumenting the {name} loader changed its "
                               f"length from {expected} to {len(loader.dataset)}")
        wrapped[name] = CountingLoader(loader)
    return wrapped


def validation_tracks_per_class(dataset, validation_indices, labels):
    """Count distinct validation tracks per class, for per-class context.

    Support alone is misleading: every track is exactly 30 frames of one
    physical sign, so a class with 30 validation images has one sign, not 30
    independent samples. The track identifiers come from `data.py`'s own
    reader, so this is the same notion of a track the split was built on.
    """
    _, tracks = read_training_index(dataset)
    per_class = defaultdict(set)
    for index in validation_indices:
        index = int(index)
        per_class[int(labels[index])].add(tracks[index])
    return {class_id: len(per_class[class_id]) for class_id in range(NUM_CLASSES)}


def file_md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(model):
    """Hash every parameter and buffer, so any mutation shows up as a change.

    `state_dict()` covers parameters and registered buffers, which for this
    model means the BatchNorm `running_mean`, `running_var`, and
    `num_batches_tracked` as well as the weights. Hashing the bytes catches a
    change of value, dtype, or shape.
    """
    fingerprint = {}
    for name, tensor in model.state_dict().items():
        payload = tensor.detach().to("cpu").contiguous()
        fingerprint[name] = hashlib.sha256(
            payload.numpy().tobytes() + str(payload.dtype).encode()
        ).hexdigest()
    return fingerprint


def repository_state():
    """Record the actual commit and working-tree state, or nothing if unavailable.

    Metadata has to describe the code that ran. If the tree is dirty the commit
    alone would be misleading, so the modified paths are recorded alongside it.
    """
    def git(*arguments):
        try:
            return subprocess.run(
                ["git", *arguments], cwd=PROJECT_ROOT, capture_output=True,
                text=True, check=True,
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    # Porcelain lines are a two-character status field, a space, then the path.
    # Stripping the whole output first would eat the leading space of an
    # unstaged entry and truncate its path by one character.
    paths = [line[3:] for line in status.splitlines() if line] if status else []
    return {
        "head_commit": commit.strip() if commit else None,
        "dirty": bool(paths) if status is not None else None,
        "modified_paths": sorted(paths),
    }


def corruption_conditions():
    """The 31 conditions: clean, then every corruption at every severity."""
    conditions = [(CLEAN_CONDITION, 0)]
    for name in CORRUPTIONS:
        conditions.extend((name, severity) for severity in SEVERITY_LEVELS)
    return conditions


def build_condition_dataset(observations, corruption, severity):
    """Wrap the observation split for one condition, clean included.

    Clean goes through the same wrapper with `corruption=None`, so the clean and
    corrupted paths differ only by the corruption step itself - the resize,
    tensor conversion, and normalisation are literally the same objects.
    """
    if corruption == CLEAN_CONDITION:
        return CorruptedDataset(observations, None, namespace=VALIDATION_NAMESPACE)
    return CorruptedDataset(
        observations, corruption, severity, namespace=VALIDATION_NAMESPACE, base_seed=SEED
    )


def condition_loader(dataset, batch_size=BATCH_SIZE, num_workers=0):
    """Sequential loader for one condition; order must stay the dataset's own."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


@torch.no_grad()
def evaluate_condition(model, loader, device, criterion):
    """One full pass over the validation split for a single condition.

    Takes a loader rather than a dataset so the caller controls what is being
    iterated - that is what lets the clean reproduction run through data.py's
    own instrumented validation loader instead of a lookalike built here.

    Returns the confusion matrix, the per-sample predicted and true labels in
    dataset order, and the mean cross-entropy. Nothing shuffles, so position `i`
    of the returned vectors is validation position `i` and maps back to a stable
    dataset index.
    """
    assert not model.training, "model must be in eval mode"
    # Measured inside the no_grad scope and returned, so the metadata records
    # the state inference actually ran under. Read in the caller it would only
    # capture ambient process state and would always say True.
    grad_enabled = torch.is_grad_enabled()
    assert not grad_enabled, "gradients must be disabled"
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    predictions = []
    targets = []
    loss_total = 0.0
    seen = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_total += loss.item() * labels.size(0)
        seen += labels.size(0)
        predicted = logits.argmax(dim=1).cpu()
        update_confusion(matrix, predicted, labels.cpu())
        predictions.append(predicted)
        targets.append(labels.cpu())
    return {
        "confusion": matrix,
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
        "loss": loss_total / seen,
        "seen": seen,
        "grad_enabled": grad_enabled,
    }


def check_target_sequence(targets, reference_targets, description):
    """Confirm an evaluated pass saw the expected labels in the expected order.

    Applied to the clean reference run as well as to every sweep condition: the
    reference is what all 31 conditions are compared against, so leaving its own
    label sequence unchecked would make the one anchor the least verified pass.
    """
    if not torch.equal(targets, reference_targets):
        raise RuntimeError(f"{description} saw a different label sequence; "
                           "sample order or split changed")


def independent_metrics(predictions, targets):
    """Accuracy and macro F1 straight from the label vectors.

    Deliberately not routed through `update_confusion` and
    `metrics_from_confusion`: comparing those two against each other would be
    comparing a value with itself, since one is now defined in terms of the
    other. This counts matches and per-class true/predicted/actual totals
    directly, so a bug in the confusion accumulation or in the F1 formula shows
    up as a disagreement rather than cancelling out.
    """
    accuracy = (predictions == targets).to(torch.float64).mean().item()
    f1_total = 0.0
    for class_id in range(NUM_CLASSES):
        predicted_mask = predictions == class_id
        actual_mask = targets == class_id
        denominator = int(predicted_mask.sum()) + int(actual_mask.sum())
        if denominator:
            f1_total += 2 * int((predicted_mask & actual_mask).sum()) / denominator
    return accuracy, f1_total / NUM_CLASSES


def check_order_independence(model, observations, device, criterion, labels,
                             validation_indices, stride=30):
    """Confirm no state leaks from one evaluated condition into the next.

    What varies between the two passes is the sequence in which the conditions
    are evaluated, not the order of samples within a pass. So this rules out
    cross-condition leakage — a shared global RNG advanced by the previous
    condition, a cached parameter, a buffer updated in place — by showing that
    a condition's predictions are unchanged when it is evaluated at a different
    position in that sequence. It is a property check, not a second benchmark,
    so it uses a sample of the split rather than repeating the sweep.

    The complementary property, that a corrupted image depends on its dataset
    index alone and not on sample order, batch layout, or worker count, is
    tested in `verify_corruptions.py` against the corruption suite itself and is
    deliberately not duplicated here.

    The subset is taken with a stride rather than as a leading block: the split
    is ordered by class, so the first few hundred positions are two or three
    classes, and a stride spreads the check over the whole label range and over
    many different occlusion anchors. A stride of 30 reaches every class, since
    the smallest class holds one track of exactly 30 consecutive frames.
    """
    positions = list(range(0, len(observations), stride))
    subset = Subset(observations, positions)
    forward = {}
    for corruption, severity in (("gaussian_noise", 1), ("gaussian_noise", 2),
                                 ("occlusion", 2), ("occlusion", 4)):
        loader = condition_loader(build_condition_dataset(subset, corruption, severity))
        forward[(corruption, severity)] = evaluate_condition(
            model, loader, device, criterion)["predictions"]

    reverse = {}
    for corruption, severity in (("occlusion", 4), ("occlusion", 2),
                                 ("gaussian_noise", 2), ("gaussian_noise", 1)):
        loader = condition_loader(build_condition_dataset(subset, corruption, severity))
        reverse[(corruption, severity)] = evaluate_condition(
            model, loader, device, criterion)["predictions"]

    for key, values in forward.items():
        assert torch.equal(values, reverse[key]), key
    # Without this the comparison could pass trivially by every condition
    # producing the same predictions. Both a stochastic condition reseeded per
    # severity and one holding a fixed anchor have to actually differ.
    for corruption in ("gaussian_noise", "occlusion"):
        low, high = (1, 2) if corruption == "gaussian_noise" else (2, 4)
        assert not torch.equal(forward[(corruption, low)], forward[(corruption, high)]), (
            f"{corruption} severities {low} and {high} gave identical predictions")
    covered = {int(labels[int(validation_indices[position])]) for position in positions}
    assert len(covered) == NUM_CLASSES, sorted(set(range(NUM_CLASSES)) - covered)
    print(f"  {len(positions)} samples (stride {stride}, all {len(covered)} "
          f"classes), 4 conditions: identical predictions when a condition is "
          "evaluated at a different position in the condition sequence, so no "
          "state carries over between conditions")
    print("  severities within each corruption differ from each other, so the "
          "comparison is not vacuous")


def verify_checkpoint():
    """Confirm the artifact on disk is still the accepted baseline."""
    digest = file_md5(CHECKPOINT_PATH)
    if digest != EXPECTED_CHECKPOINT_MD5:
        raise RuntimeError(
            f"checkpoint md5 is {digest}, expected {EXPECTED_CHECKPOINT_MD5}; "
            "the accepted baseline artifact has changed"
        )
    return digest


def write_outputs(results, conditions, metadata, validation_indices, labels,
                  tracks_per_class, results_dir):
    """Write every machine-readable artifact for later failure analysis."""
    results_dir.mkdir(parents=True, exist_ok=True)

    clean = results[(CLEAN_CONDITION, 0)]
    summary_rows = []
    for corruption, severity in conditions:
        record = results[(corruption, severity)]
        summary_rows.append({
            "corruption": corruption,
            "severity": severity,
            "parameter": "" if corruption == CLEAN_CONDITION
                         else CORRUPTIONS[corruption].parameter,
            "parameter_value": "" if corruption == CLEAN_CONDITION
                               else corruption_parameter(corruption, severity),
            "loss": record["loss"],
            "accuracy": record["accuracy"],
            "macro_f1": record["macro_f1"],
            "accuracy_drop": clean["accuracy"] - record["accuracy"],
            "macro_f1_drop": clean["macro_f1"] - record["macro_f1"],
            # Share of the clean score that is lost. Reported alongside the
            # absolute drop, never instead of it: a 50% relative fall means
            # something very different at 0.99 clean accuracy than at 0.20.
            "accuracy_relative_drop": (clean["accuracy"] - record["accuracy"])
                                      / clean["accuracy"],
            "macro_f1_relative_drop": (clean["macro_f1"] - record["macro_f1"])
                                      / clean["macro_f1"],
            "samples": record["seen"],
        })

    summary_path = results_dir / "summary.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (results_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "conditions": summary_rows}, indent=2) + "\n")

    class_path = results_dir / "per_class_metrics.csv"
    with open(class_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        # `tracks` travels with every row because support on its own overstates
        # the evidence: 30 images can be 30 frames of a single sign.
        writer.writerow(["corruption", "severity", "class", "support", "tracks",
                         "recall", "f1"])
        for corruption, severity in conditions:
            per_class = results[(corruption, severity)]["per_class"]
            for class_id in range(NUM_CLASSES):
                writer.writerow([
                    corruption, severity, class_id,
                    int(per_class["support"][class_id]),
                    tracks_per_class[class_id],
                    float(per_class["recall"][class_id]),
                    float(per_class["f1"][class_id]),
                ])

    confusion_path = results_dir / "confusion_matrices.npz"
    np.savez_compressed(confusion_path, **{
        f"{corruption}_s{severity}": results[(corruption, severity)]["confusion"].numpy()
        for corruption, severity in conditions
    })

    # Per-sample rows: index, truth, prediction only. Enough to recover any
    # confusion matrix or per-class score and to find individual failures,
    # without storing logits or activations.
    predictions_path = results_dir / "predictions.csv"
    with open(predictions_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corruption", "severity", "validation_position",
                         "dataset_index", "label", "prediction"])
        for corruption, severity in conditions:
            predicted = results[(corruption, severity)]["predictions"].tolist()
            for position, value in enumerate(predicted):
                index = int(validation_indices[position])
                writer.writerow([corruption, severity, position, index,
                                 int(labels[index]), value])

    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "summary_csv": summary_path,
        "summary_json": results_dir / "summary.json",
        "per_class_csv": class_path,
        "confusion_npz": confusion_path,
        "predictions_csv": predictions_path,
        "metadata_json": results_dir / "metadata.json",
    }


def format_table(summary_rows):
    """Render the condition table the report quotes."""
    header = (f"{'corruption':18s} {'sev':>3s} {'param':>8s} {'accuracy':>9s} "
              f"{'macro F1':>9s} {'acc drop':>9s} {'F1 drop':>9s}")
    lines = [header, "-" * len(header)]
    for row in summary_rows:
        parameter = "" if row["parameter_value"] == "" else f"{row['parameter_value']:g}"
        lines.append(
            f"{row['corruption']:18s} {row['severity']:>3d} {parameter:>8s} "
            f"{row['accuracy']:>9.4f} {row['macro_f1']:>9.4f} "
            f"{row['accuracy_drop']:>9.4f} {row['macro_f1_drop']:>9.4f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()

    results_dir = Path(arguments.output_dir) if arguments.output_dir else RESULTS_DIR
    device = select_device(arguments.device)
    started = time.time()

    print("Frozen corruption specification (src/corruptions.py):")
    print(format_specification())

    print("\nCheckpoint:")
    digest = verify_checkpoint()
    model, checkpoint = load_checkpoint(CHECKPOINT_PATH, device=device)
    if checkpoint["epoch"] != EXPECTED_EPOCH:
        raise RuntimeError(f"checkpoint reports epoch {checkpoint['epoch']}, "
                           f"expected {EXPECTED_EPOCH}")
    print(f"  md5 {digest} matches the accepted baseline")
    print(f"  epoch {checkpoint['epoch']}, recorded validation metrics "
          f"{checkpoint['validation_metrics']}")
    print(f"  architecture {checkpoint['model_config']['architecture']}, "
          f"device {device}, torch {torch.__version__}")

    model.eval()
    assert not model.training
    before = state_fingerprint(model)
    criterion = nn.CrossEntropyLoss()

    observations, validation_indices, labels = build_validation_observations()
    if len(observations) != EXPECTED_VALIDATION_SIZE:
        raise RuntimeError(f"validation split holds {len(observations)} images, "
                           f"expected {EXPECTED_VALIDATION_SIZE}")
    reference_targets = torch.tensor(
        [int(labels[int(index)]) for index in validation_indices], dtype=torch.int64)
    tracks_per_class = validation_tracks_per_class(
        observations.dataset, validation_indices, labels)
    single_track = sorted(c for c, n in tracks_per_class.items() if n == 1)
    print(f"\nValidation split: {len(observations)} images, "
          f"{len(set(int(i) for i in validation_indices))} unique dataset indices, "
          f"{len(reference_targets.unique())} classes present")
    print(f"  {sum(tracks_per_class.values())} tracks; {len(single_track)} classes "
          f"hold a single track: {single_track}")

    # The accepted clean numbers were produced through data.py's own validation
    # loader. Running that exact path here, as well as the wrapper path, is what
    # makes "the clean baseline reproduces" a statement about this evaluation
    # rather than about a different pipeline that happens to agree.
    loaders, _ = create_data_loaders(batch_size=arguments.batch_size, download=False)
    # Rebinding `loaders` to the wrapped dict is deliberate: it leaves no
    # uncounted handle on the official test split anywhere in this function.
    loaders = instrument_loaders(loaders)
    print("\nClean reproduction through data.py's validation loader:")
    reference = evaluate_condition(model, loaders["validation"], device, criterion)
    check_target_sequence(reference["targets"], reference_targets,
                          "the data.py validation loader")
    reference_accuracy, reference_f1 = metrics_from_confusion(reference["confusion"])
    print(f"  loss {reference['loss']:.6f} accuracy {reference_accuracy:.6f} "
          f"macro F1 {reference_f1:.6f}")
    direct_accuracy, direct_f1 = independent_metrics(
        reference["predictions"], reference["targets"])
    for name, through_matrix, direct in (("accuracy", reference_accuracy, direct_accuracy),
                                         ("macro F1", reference_f1, direct_f1)):
        if abs(through_matrix - direct) > METRIC_TOLERANCE:
            raise RuntimeError(
                f"{name} from the confusion matrix is {through_matrix!r} but "
                f"{direct!r} counted directly from the prediction and label "
                "vectors; the metric path is wrong"
            )
    print(f"  recomputed directly from the prediction and label vectors: "
          f"accuracy {direct_accuracy:.6f} macro F1 {direct_f1:.6f}")
    for key, measured in (("loss", reference["loss"]), ("accuracy", reference_accuracy),
                          ("macro_f1", reference_f1)):
        difference = abs(measured - ACCEPTED_CLEAN[key])
        if difference > CLEAN_TOLERANCE:
            raise RuntimeError(
                f"clean {key} is {measured:.6f}, accepted baseline is "
                f"{ACCEPTED_CLEAN[key]:.6f} (difference {difference:.2e}); "
                "stop and diagnose before trusting corrupted results"
            )
    print(f"  matches the accepted epoch-{EXPECTED_EPOCH} values within "
          f"{CLEAN_TOLERANCE:g}")

    print("\nOrder independence:")
    check_order_independence(model, observations, device, criterion,
                             labels=labels, validation_indices=validation_indices)

    conditions = corruption_conditions()
    print(f"\nEvaluating {len(conditions)} conditions "
          f"(1 clean + {len(conditions) - 1} corrupted):")
    results = {}
    for corruption, severity in conditions:
        condition_started = time.time()
        loader = condition_loader(
            build_condition_dataset(observations, corruption, severity),
            batch_size=arguments.batch_size, num_workers=arguments.num_workers,
        )
        record = evaluate_condition(model, loader, device, criterion)
        if record["seen"] != EXPECTED_VALIDATION_SIZE:
            raise RuntimeError(f"{corruption} s{severity} evaluated "
                               f"{record['seen']} samples")
        check_target_sequence(record["targets"], reference_targets,
                              f"{corruption} s{severity}")
        accuracy, macro_f1 = metrics_from_confusion(record["confusion"])
        per_class = per_class_from_confusion(record["confusion"])
        # Independent of the confusion-matrix path, so an error in the
        # accumulation or in the F1 formula cannot hide by appearing on both
        # sides of the comparison.
        direct_accuracy, direct_f1 = independent_metrics(
            record["predictions"], record["targets"])
        for name, through_matrix, direct in (("accuracy", accuracy, direct_accuracy),
                                             ("macro_f1", macro_f1, direct_f1)):
            if abs(through_matrix - direct) > METRIC_TOLERANCE:
                raise RuntimeError(
                    f"{corruption} s{severity} {name}: confusion matrix gives "
                    f"{through_matrix!r}, direct count gives {direct!r}")
        if record["grad_enabled"] is not False:
            raise RuntimeError(f"{corruption} s{severity} ran with gradients enabled")
        record.update({"accuracy": accuracy, "macro_f1": macro_f1,
                       "per_class": per_class})
        results[(corruption, severity)] = record
        print(f"  {corruption:18s} s{severity} | loss {record['loss']:.4f} "
              f"accuracy {accuracy:.4f} macro F1 {macro_f1:.4f} "
              f"| {time.time() - condition_started:.1f}s")

    # The sweep's clean row and the data.py-loader reference are two different
    # code paths over the same images: the wrapper with `severity=None` against
    # the pipeline's own validation loader. Asserting they agree exactly ties
    # the whole corrupted table to the accepted baseline, instead of leaving the
    # clean column as an unchecked 32nd condition.
    clean_row = results[(CLEAN_CONDITION, 0)]
    if not torch.equal(clean_row["predictions"], reference["predictions"]):
        mismatches = int((clean_row["predictions"] != reference["predictions"]).sum())
        raise RuntimeError(
            f"the sweep's clean condition disagrees with the data.py validation "
            f"loader on {mismatches} of {len(reference_targets)} images; the "
            "corruption wrapper is altering uncorrupted inputs"
        )
    for name, swept, referenced in (("loss", clean_row["loss"], reference["loss"]),
                                    ("accuracy", clean_row["accuracy"], reference_accuracy),
                                    ("macro_f1", clean_row["macro_f1"], reference_f1)):
        if abs(swept - referenced) > CLEAN_TOLERANCE:
            raise RuntimeError(
                f"sweep clean {name} {swept:.6f} differs from the reference "
                f"{referenced:.6f} by more than {CLEAN_TOLERANCE:g}")
    print("\nSweep clean row: identical predictions to the data.py reference run, "
          "and all three metrics agree, so every drop is measured against the "
          "accepted baseline")

    after = state_fingerprint(model)
    changed = sorted(name for name in before if before[name] != after[name])
    if changed:
        raise RuntimeError(f"model state changed during evaluation: {changed}")
    gradients = [name for name, parameter in model.named_parameters()
                 if parameter.grad is not None]
    if gradients:
        raise RuntimeError(f"gradients were populated: {gradients}")
    batchnorm_buffers = [name for name in before if "running_" in name
                         or "num_batches_tracked" in name]

    metadata = {
        "checkpoint": {
            "path": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "md5": digest,
            "epoch": checkpoint["epoch"],
            "selection_metric": checkpoint["selection_metric"],
            "recorded_validation_metrics": checkpoint["validation_metrics"],
            "model_config": checkpoint["model_config"],
            "training_config": checkpoint.get("training_config"),
        },
        "accepted_clean_metrics": ACCEPTED_CLEAN,
        "reproduced_clean_metrics": {
            "loss": reference["loss"],
            "accuracy": reference_accuracy,
            "macro_f1": reference_f1,
        },
        "evaluation": {
            "split": "validation",
            "validation_images": len(observations),
            "num_classes": NUM_CLASSES,
            "conditions": len(conditions),
            "batch_size": arguments.batch_size,
            "num_workers": arguments.num_workers,
            "device": str(device),
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mode": "eval (BatchNorm running statistics from clean training, "
                    "no recalibration)",
            # Measured inside the no_grad scope of every condition, not read
            # off ambient process state afterwards.
            "gradients_enabled": any(record["grad_enabled"] for record in results.values()),
            "optimizer": None,
            "seconds": time.time() - started,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "corruption_suite": {
            "namespace": VALIDATION_NAMESPACE,
            "base_seed": SEED,
            "severity_levels": list(SEVERITY_LEVELS),
            "specification": {
                name: {"parameter": spec.parameter, "levels": list(spec.levels),
                       "stochastic": spec.stochastic,
                       "severity_in_seed": spec.severity_in_seed}
                for name, spec in CORRUPTIONS.items()
            },
            "example_seeds": {
                f"{name}_s{severity}_index{int(validation_indices[0])}":
                    sample_seed(VALIDATION_NAMESPACE, name, severity,
                                int(validation_indices[0]))
                for name in ("gaussian_noise", "occlusion")
                for severity in SEVERITY_LEVELS
            },
        },
        "interpretation_notes": {
            "occlusion": OCCLUSION_NOTE,
            "per_class_support": SINGLE_TRACK_WARNING,
        },
        "validation_tracks_per_class": tracks_per_class,
        "integrity": {
            "checkpoint_md5_matches": True,
            "clean_metrics_reproduced": True,
            "sweep_clean_row_matches_reference": True,
            "metrics_cross_checked_without_confusion_matrix": True,
            "model_state_unchanged": True,
            "batchnorm_buffers_checked": len(batchnorm_buffers),
            "state_tensors_checked": len(before),
            "test_loader_iterations": loaders["test"].iterations,
            "test_dataset_reads": loaders["test"].dataset.reads,
            "train_loader_iterations": loaders["train"].iterations,
            "train_dataset_reads": loaders["train"].dataset.reads,
            "validation_loader_iterations": loaders["validation"].iterations,
        },
        "repository": repository_state(),
    }

    paths = write_outputs(results, conditions, metadata, validation_indices,
                          labels, tracks_per_class, results_dir)
    summary_rows = json.loads(paths["summary_json"].read_text())["conditions"]

    print("\n" + format_table(summary_rows))
    print("\nIntegrity:")
    print(f"  all {len(before)} parameter and buffer tensors identical before and "
          f"after evaluation, including {len(batchnorm_buffers)} BatchNorm buffers")
    print("  no gradient was populated; no optimizer was constructed")
    print(f"  official test loader iterations: {loaders['test'].iterations}, "
          f"test images read through the instrumented dataset: "
          f"{loaders['test'].dataset.reads}; "
          f"training loader iterations: {loaders['train'].iterations}")
    print(f"  every condition evaluated exactly {EXPECTED_VALIDATION_SIZE} samples "
          "in the same label order")

    print("\nWritten:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(f"\nTotal {time.time() - started:.1f}s. No training, augmentation, or "
          "BatchNorm recalibration occurred, and the official test split was "
          "never iterated.")


if __name__ == "__main__":
    main()
