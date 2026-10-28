"""Checks on the official-test evaluation path that read no official test image.

    PYTHONPATH=. .venv/bin/python -m src.verify_test_path

Everything the test evaluator and the test comparison do is exercised here on
synthetic images, synthetic labels and synthetic prediction files. The official
test split is never decoded, its loader is never iterated, no checkpoint is
loaded, and no test metric is produced - so this module can be run before the
test evaluation as well as after it, and running it does not spend a look at
the test set.

Two things make that a guarantee rather than an intention. `GTSRB.__getitem__`
is replaced for the duration of the run by a tripwire that raises, so any code
path reaching for a real image fails loudly instead of quietly succeeding; and
`evaluate_test.build_test_observations` is replaced the same way. Both are
restored at the end and their call counts are reported.

The one official file this reads is `GT-final_test.csv`, for its structure: row
count, header and label range, all of which the protocol's pre-run checks
already require. No label distribution is printed and nothing here chooses an
analysis setting from it.
"""

import argparse
import ast
import contextlib
import csv
import hashlib
import inspect
import io
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import GTSRB
from torchvision.datasets import gtsrb as gtsrb_module

from . import (compare_models, compare_test, evaluate_checkpoint,
               evaluate_test, run_final_test)
from .compare_test import (
    bootstrap_image_mean,
    class_balanced_rows,
    compare,
    design_effect_illustration,
    image_draws,
    image_rates,
    paired_image_contrast,
)
from .augmentation import augmentation_seed
from .corruptions import (
    CORRUPTIONS,
    CorruptedDataset,
    SEED,
    VALIDATION_NAMESPACE,
    check_stage_layout,
    sample_seed,
    split_preprocessing,
)
from .corruptions_heldout import (
    HELDOUT_CORRUPTIONS,
    HeldoutDataset,
    _lookup as heldout_lookup,
    heldout_sample_seed,
)
from .data import NUM_CLASSES, build_preprocessing, create_data_loaders
from .evaluate_checkpoint import (
    FINGERPRINT_STRIDE,
    PROJECT_ROOT,
    FROZEN_SUITE,
    HELDOUT_SUITE,
    SMOKE_CONDITIONS,
    SMOKE_STRIDE,
    input_fingerprint,
)
from .evaluate_robustness import (
    CLEAN_CONDITION,
    CountingDataset,
    condition_loader,
    evaluate_condition,
    file_md5,
    independent_metrics,
)
from .evaluate_test import (
    EXPECTED_TEST_SIZE,
    GROUND_TRUTH_FIELDS,
    INVENTORY_PATH,
    PROTECTED_ROOTS,
    PROTOCOL_PATH,
    PROTOCOL_READ_COUNTS,
    TEST_NAMESPACE,
    TEST_RESULTS_ROOT,
    TEST_SUITES,
    access_log_entry,
    append_access_log,
    check_inventory_digests,
    check_output_target,
    check_protocol_identity,
    check_validation_binding,
    class_support,
    expected_reads,
    inventory_record,
    parse_test_ground_truth,
    results_dir,
    write_test_outputs,
)
from .train import metrics_from_confusion, per_class_from_confusion
from .verify_corruptions import independent_seed

# Small enough to run in seconds, large enough that 64x64 corruptions have
# something to act on and that a stride lands more than once.
SYNTHETIC_IMAGES = 96

# The released training split, for the file-count and byte-overlap checks.
EXPECTED_TRAIN_FILES = 26640

# The trees the pre-run snapshot covers: every committed result the test stage
# must leave untouched.
SNAPSHOT_TREES = (PROJECT_ROOT / "artifacts" / "robustness",
                  PROJECT_ROOT / "artifacts" / "analysis" / "augmentation",
                  PROJECT_ROOT / "artifacts" / "baseline")

# The arms, the seeds and the error patterns of the synthetic comparison
# fixture. The patterns are stated here as moduli so the expected arm rates and
# contrasts can be derived from them directly, without going through the module
# under test.
FIXTURE_SEEDS = (42, 43, 44)
FIXTURE_STRIDES = {"clean": 5, "balanced": 7, "targeted": 10}
FIXTURE_SEVERITY_STRIDE = 101
FIXTURE_SEED_STRIDE = {42: 0, 43: 211, 44: 307}


class SyntheticObservations(torch.utils.data.Dataset):
    """`(image in [0, 1], label)` pairs that depend only on the index.

    Stands in for the observation half of the official test pipeline: same
    shape, same dtype, same value range, same "the realisation keys on the
    dataset index" property, and no official pixel anywhere near it. Defined at
    module level so it survives pickling into DataLoader workers.
    """

    def __init__(self, count=SYNTHETIC_IMAGES):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(9000 + index)
        image = torch.rand(3, 64, 64, generator=generator)
        return image, index % NUM_CLASSES


class FixedLogits(torch.nn.Module):
    """A deterministic function of the input with the model's output shape.

    Not a trained model and not meant to be one: the confusion-matrix and
    prediction-schema checks need predictions that depend on the pixels and
    nothing else, so that the accounting can be verified against the
    predictions the forward pass actually produced.
    """

    def forward(self, batch):
        flat = batch.flatten(1)
        columns = flat.shape[1] // NUM_CLASSES * NUM_CLASSES
        return flat[:, :columns].reshape(batch.shape[0], NUM_CLASSES, -1).mean(2)


def check_protocol_and_inventory():
    """The frozen documents and the nine approved checkpoints, as read at runtime."""
    identity = check_protocol_identity()
    print(f"  {identity['final_test_protocol']} sha256 "
          f"{identity['final_test_protocol_sha256'][:16]}... matches the frozen "
          "value")
    print(f"  the {len(identity['augmentation_files_verified'])} files the "
          "augmentation manifest covers are unchanged")

    inventory = json.loads(evaluate_test.INVENTORY_PATH.read_text())
    checks = 0
    for record in inventory["models"]:
        path = evaluate_test.PROJECT_ROOT / record["checkpoint_path"]
        looked_up = inventory_record(path)
        assert looked_up == record, record["checkpoint_path"]
        digests = check_inventory_digests(path, record)
        assert digests["md5"] == record["checkpoint_md5"]
        binding = check_validation_binding(record)
        assert binding["checkpoint_md5"] == record["checkpoint_md5"]
        assert len(binding["validation_metadata"]) == len(TEST_SUITES)
        checks += 1
    print(f"  {checks} checkpoints: inventory entry found by path, md5 and "
          "sha256 as recorded, and both committed validation result sets name "
          "the same checkpoint")

    # The rejections. A path outside the inventory, and a record whose digest
    # has been altered, both have to stop a run.
    try:
        inventory_record(evaluate_test.PROJECT_ROOT / "artifacts" / "nowhere.pt")
    except RuntimeError as error:
        assert "not in" in str(error), error
    else:
        raise AssertionError("a checkpoint outside the inventory was accepted")

    first = inventory["models"][0]
    tampered = dict(first)
    tampered["checkpoint_md5"] = "0" * 32
    try:
        check_inventory_digests(
            evaluate_test.PROJECT_ROOT / first["checkpoint_path"], tampered)
    except RuntimeError as error:
        assert "the inventory records" in str(error), error
    else:
        raise AssertionError("a checkpoint whose md5 does not match its "
                             "inventory record was accepted")

    missing = dict(first)
    missing["arm_label"], missing["seed"] = "clean", 99
    try:
        check_validation_binding(missing)
    except RuntimeError as error:
        assert "no committed validation metadata" in str(error), error
    else:
        raise AssertionError("a checkpoint with no validation binding was "
                             "accepted")
    print("  a path outside the inventory, an altered digest and an unbound "
          "arm and seed are all refused")
    return checks + 3


def check_condition_coverage():
    """The test suites are the validation suites, renamespaced and nothing else."""
    assert sorted(TEST_SUITES) == ["frozen", "heldout"]
    validation = {"frozen": FROZEN_SUITE, "heldout": HELDOUT_SUITE}
    checks = 0
    for name, suite in sorted(TEST_SUITES.items()):
        source = validation[name]
        assert suite.name == source.name
        # By reference, not by value: a copied table could be edited without
        # the comparison noticing, so the check is that there is no copy.
        assert suite.table is source.table, name
        assert suite.dataset_class is source.dataset_class, name
        assert suite.parameter is source.parameter, name
        assert suite.specification is source.specification, name
        assert suite.conditions() == source.conditions(), name
        conditions = suite.conditions()
        assert conditions[0] == (CLEAN_CONDITION, 0)
        corrupted = conditions[1:]
        assert len({family for family, _ in corrupted}) * 5 == len(corrupted)
        assert sorted({severity for _, severity in corrupted}) == [1, 2, 3, 4, 5]
        for corruption, severity in corrupted:
            assert suite.parameter_value(corruption, severity) == \
                source.parameter_value(corruption, severity)
        checks += len(conditions)
        print(f"  {name}: 1 clean + {len(corrupted)} corrupted, table, "
              f"parameters and severities shared by reference with the "
              f"validation suite")
    assert len(TEST_SUITES["frozen"].conditions()) == 31
    assert len(TEST_SUITES["heldout"].conditions()) == 26
    families = {name: {family for family, _ in TEST_SUITES[name].conditions()[1:]}
                for name in TEST_SUITES}
    assert not families["frozen"] & families["heldout"]
    print(f"  the {len(families['frozen'])} frozen families and the "
          f"{len(families['heldout'])} held-out families are disjoint, and "
          "both suites share the one clean condition")
    return checks + 3


def check_corruption_determinism(observations):
    """A test realisation depends on the image and the condition, nothing else."""
    conditions = [("gaussian_noise", 4), ("occlusion", 3), ("fog", 2)]
    positions = list(range(16))
    checks = 0
    for corruption, severity in conditions:
        dataset = evaluate_test.TEST_SUITES["frozen"].dataset(
            observations, corruption, severity)
        assert dataset.namespace == TEST_NAMESPACE
        first = [dataset[index][0] for index in positions]

        torch.manual_seed(1234)
        np.random.seed(1234)
        again = [dataset[index][0] for index in reversed(positions)][::-1]
        for one, other in zip(first, again):
            assert torch.equal(one, other), corruption
        checks += len(positions)

        # Through a shuffled two-worker loader, re-ordered by position, so the
        # realisation cannot depend on batch composition or worker identity.
        loader = DataLoader(IndexedDataset(dataset), batch_size=5, shuffle=True,
                            num_workers=2)
        seen = {}
        for images, _, batch_positions in loader:
            for image, position in zip(images, batch_positions.tolist()):
                seen[position] = image
        assert len(seen) == len(dataset)
        for position in positions:
            assert torch.equal(seen[position], first[position]), corruption
        checks += len(positions)

        # And through a subset, so the seed keys on the dataset index rather
        # than on the position within whatever was passed to the loader.
        subset = Subset(dataset, positions[4:8])
        for offset, position in enumerate(positions[4:8]):
            assert torch.equal(subset[offset][0], first[position]), corruption
        checks += 4
    print(f"  {len(conditions)} conditions: identical under reversed access, a "
          "shuffled two-worker loader, and re-subsetting, and unaffected by "
          "the global torch and numpy seeds")
    return checks


class IndexedDataset(torch.utils.data.Dataset):
    """Yields `(image, label, position)` so a shuffled loader can be re-ordered."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, position):
        image, label = self.dataset[position]
        return image, label, position


def check_namespace_separation(observations):
    """The test namespace must not reproduce a validation realisation.

    The namespace enters the seed, so it can only separate the families that
    draw something. The deterministic families are a function of the pixels and
    the severity alone, so their output is identical under both namespaces -
    which is a property of those corruptions, not a leak, and is asserted in
    that direction rather than asserted away.
    """
    both = dict(CORRUPTIONS)
    both.update(HELDOUT_CORRUPTIONS)
    stochastic = sorted(name for name, spec in both.items() if spec.stochastic)
    deterministic = sorted(name for name, spec in both.items()
                           if not spec.stochastic)
    severities = (1, 3, 5)
    indices = range(8)
    checks, matched = 0, 0
    distinct = {}
    for corruption in stochastic + deterministic:
        held_out = corruption in HELDOUT_CORRUPTIONS
        wrapper = HeldoutDataset if held_out else CorruptedDataset
        seed_of = heldout_sample_seed if held_out else sample_seed
        differed = 0
        for severity in severities:
            validation = wrapper(observations, corruption, severity,
                                 namespace=VALIDATION_NAMESPACE)
            test = wrapper(observations, corruption, severity,
                           namespace=TEST_NAMESPACE)
            assert validation.namespace != test.namespace
            for index in indices:
                same = torch.equal(validation[index][0], test[index][0])
                if corruption in stochastic:
                    # The seeds must differ for every image and condition.
                    # That is the mechanism, and it is exact.
                    assert seed_of(VALIDATION_NAMESPACE, corruption, severity,
                                   index) != seed_of(TEST_NAMESPACE, corruption,
                                                     severity, index)
                    differed += not same
                else:
                    assert same, f"{corruption} s{severity} index {index}"
                    matched += 1
                checks += 1
        if corruption in stochastic:
            distinct[corruption] = differed
    attempts = len(severities) * len(indices)
    for corruption, differed in sorted(distinct.items()):
        # Not every realisation lands on different pixels, and it does not
        # have to. A rotation direction is one of two values and a motion-blur
        # angle or an occlusion anchor comes from a small discrete set, so two
        # different seeds often draw the same one - for a binary draw about
        # half the time. What must hold is that the seed differs for every
        # image and condition, which is checked above and is exact, and that
        # the realisations do move.
        assert differed > 0, f"{corruption}: {differed}/{attempts}"
    report = ", ".join(f"{name} {value}/{attempts}"
                       for name, value in sorted(distinct.items()))
    print(f"  every stochastic realisation seed differs between the "
          f"{VALIDATION_NAMESPACE} and {TEST_NAMESPACE} namespaces over "
          f"{attempts} image-and-condition pairs per family; the pixels differ "
          f"in {report} of them, the remainder being two different seeds that "
          "drew the same value from a small discrete set - a rotation "
          "direction is one of two, so about half of those coincide")
    print(f"  the {len(deterministic)} deterministic families are identical in "
          f"both namespaces on all {matched} pairs, because their output is a "
          "function of the pixels and the severity and there is no realisation "
          "to separate")

    # The clean condition carries no realisation, so it is identical in both -
    # which is what makes the clean row comparable across namespaces.
    clean_validation = CorruptedDataset(observations, None,
                                        namespace=VALIDATION_NAMESPACE)
    clean_test = evaluate_test.TEST_SUITES["frozen"].dataset(
        observations, CLEAN_CONDITION, 0)
    assert clean_test.corruption is None
    for index in range(24):
        assert torch.equal(clean_validation[index][0], clean_test[index][0])
        checks += 1
    print("  the clean condition is unrealised and therefore identical in "
          "both namespaces")

    fingerprints = {
        name: input_fingerprint(
            CorruptedDataset(observations, "gaussian_noise", 3, namespace=name),
            list(range(0, SYNTHETIC_IMAGES, 8)))
        for name in (VALIDATION_NAMESPACE, TEST_NAMESPACE)}
    assert len(set(fingerprints.values())) == 2
    print("  the input fingerprints of one condition differ between the two "
          "namespaces")
    return checks + 1


def synthetic_sweep(observations, suite, conditions, device="cpu"):
    """Measure a few conditions with the stand-in model, as the evaluator does."""
    model = FixedLogits().eval()
    criterion = torch.nn.CrossEntropyLoss()
    results, fingerprints = {}, {}
    positions = list(range(0, len(observations), FINGERPRINT_STRIDE))
    for corruption, severity in conditions:
        dataset = suite.dataset(observations, corruption, severity)
        loader = condition_loader(dataset, batch_size=16, num_workers=0)
        measurement = evaluate_condition(model, loader, torch.device(device),
                                         criterion)
        accuracy, macro_f1 = metrics_from_confusion(measurement["confusion"])
        direct_accuracy, direct_f1 = independent_metrics(
            measurement["predictions"], measurement["targets"])
        assert abs(accuracy - direct_accuracy) < 1e-12
        assert abs(macro_f1 - direct_f1) < 1e-12
        assert measurement["grad_enabled"] is False
        measurement.update({"accuracy": accuracy, "macro_f1": macro_f1,
                            "per_class": per_class_from_confusion(
                                measurement["confusion"])})
        results[(corruption, severity)] = measurement
        fingerprints[f"{corruption}_s{severity}"] = input_fingerprint(
            dataset, positions)
    return results, fingerprints


def check_confusion_accounting(observations, results, conditions):
    """Every number in the artifacts has to follow from the predictions."""
    labels = np.array([observations[index][1] for index in range(len(observations))])
    checks = 0
    for corruption, severity in conditions:
        record = results[(corruption, severity)]
        confusion = record["confusion"].numpy()
        predictions = np.asarray(record["predictions"])
        targets = np.asarray(record["targets"])
        assert confusion.shape == (NUM_CLASSES, NUM_CLASSES)
        assert int(confusion.sum()) == len(observations) == record["seen"]
        assert np.array_equal(targets, labels)

        # The matrix rebuilt from the two vectors, cell by cell.
        rebuilt = np.zeros_like(confusion)
        for target, prediction in zip(targets, predictions):
            rebuilt[target, prediction] += 1
        assert np.array_equal(confusion, rebuilt), (corruption, severity)
        assert int(np.trace(confusion)) == int((predictions == targets).sum())
        assert np.array_equal(confusion.sum(axis=1),
                              np.bincount(targets, minlength=NUM_CLASSES))
        assert np.array_equal(confusion.sum(axis=0),
                              np.bincount(predictions, minlength=NUM_CLASSES))

        per_class = record["per_class"]
        assert np.array_equal(np.asarray(per_class["support"]),
                              confusion.sum(axis=1))
        assert np.array_equal(np.asarray(per_class["predicted"]),
                              confusion.sum(axis=0))
        assert np.array_equal(np.asarray(per_class["true_positive"]),
                              np.diag(confusion))
        assert abs(record["accuracy"]
                   - float(np.trace(confusion)) / confusion.sum()) < 1e-12
        checks += 10
    print(f"  {len(conditions)} conditions: the confusion matrix equals the "
          "matrix rebuilt from the prediction and target vectors, its trace is "
          "the number correct, its margins are the label and prediction "
          "counts, and accuracy follows from the trace")
    return checks


def check_prediction_schema(observations, suite, results, conditions, directory):
    """The artifact set, its columns, and the class-support block."""
    labels = np.array([observations[index][1]
                       for index in range(len(observations))])
    indices = np.arange(len(observations))
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    support = {str(class_id): int(counts[class_id])
               for class_id in range(NUM_CLASSES)}
    metadata = {
        "model": {"name": "synthetic", "seed": 0},
        "evaluation": {"split": "official_test", "test_images": len(observations)},
        "corruption_suite": {"namespace": TEST_NAMESPACE},
        "test_class_support": support,
    }
    target = directory / "synthetic_seed0" / suite.name
    written = write_test_outputs(suite, results, conditions, metadata, indices,
                                 labels, support, target)
    assert sorted(path.name for path in written.values()) == [
        "confusion_matrices.npz", "metadata.json", "per_class_metrics.csv",
        "predictions.csv", "summary.csv", "summary.json"]
    assert sorted(path.name for path in target.iterdir()) == sorted(
        path.name for path in written.values()), "an undeclared file was written"

    with open(written["predictions_csv"]) as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["corruption", "severity", "test_position",
                       "dataset_index", "label", "prediction"]
    assert "filename" not in [name.lower() for name in rows[0]]
    body = rows[1:]
    assert len(body) == len(conditions) * len(observations)
    for row in body:
        assert row[2] == row[3], row
        assert int(row[4]) == int(labels[int(row[3])]), row
        assert 0 <= int(row[5]) < NUM_CLASSES, row
    predicted = {(corruption, severity):
                 [int(row[5]) for row in body
                  if row[0] == corruption and int(row[1]) == severity]
                 for corruption, severity in conditions}
    for key, values in predicted.items():
        assert values == list(np.asarray(results[key]["predictions"])), key

    with open(written["summary_csv"]) as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == len(conditions)
    assert [row["corruption"] for row in summary] == [
        corruption for corruption, _ in conditions]
    for row in summary:
        assert abs(float(row["error_rate"]) + float(row["accuracy"]) - 1) < 1e-12
        assert int(row["samples"]) == len(observations)

    with open(written["per_class_csv"]) as handle:
        per_class = list(csv.DictReader(handle))
    assert list(per_class[0]) == ["corruption", "severity", "class", "support",
                                  "precision", "recall", "f1"]
    assert len(per_class) == len(conditions) * NUM_CLASSES
    assert sum(int(row["support"]) for row in per_class) == \
        len(conditions) * len(observations)

    matrices = np.load(written["confusion_npz"])
    assert sorted(matrices.files) == sorted(
        f"{corruption}_s{severity}" for corruption, severity in conditions)
    for corruption, severity in conditions:
        assert np.array_equal(matrices[f"{corruption}_s{severity}"],
                              results[(corruption, severity)]["confusion"].numpy())

    stored = json.loads(written["metadata_json"].read_text())
    assert stored["evaluation"]["split"] == "official_test"
    assert stored["corruption_suite"]["namespace"] == TEST_NAMESPACE
    assert stored["test_class_support"] == support
    print(f"  6 artifacts written for {len(conditions)} synthetic conditions: "
          "prediction columns as frozen and carrying no filename, positions "
          "equal to dataset indices, labels read from the label array, "
          "summary error rates complementary to accuracy, per-class supports "
          "summing to the sweep, and stored matrices equal to the measured ones")

    # The support block has to be the measured one, not whatever the metadata
    # happened to carry.
    wrong = dict(metadata)
    wrong["test_class_support"] = {key: value + 1 for key, value in support.items()}
    try:
        write_test_outputs(suite, results, conditions, wrong, indices, labels,
                           support, directory / "mismatch" / suite.name)
    except RuntimeError as error:
        assert "not the one measured" in str(error), error
    else:
        raise AssertionError("a metadata support block that disagreed with the "
                             "measured labels was written")
    assert not (directory / "mismatch").exists(), \
        "the refused write left a directory that looks like a result set"
    print("  a metadata support block that disagrees with the measured labels "
          "stops the write")
    return 14


def check_class_support():
    """The split-specific support block, on synthetic labels of the right size."""
    labels = np.arange(EXPECTED_TEST_SIZE) % NUM_CLASSES
    support = class_support(labels)
    assert len(support) == NUM_CLASSES
    assert sum(support.values()) == EXPECTED_TEST_SIZE
    assert set(support) == {str(value) for value in range(NUM_CLASSES)}
    for short in (labels[:-1], np.zeros(EXPECTED_TEST_SIZE + 1, dtype=np.int64)):
        try:
            class_support(short)
        except RuntimeError as error:
            assert "sum to" in str(error), error
        else:
            raise AssertionError("a label array of the wrong length was accepted")
    print(f"  class support covers all {NUM_CLASSES} classes and sums to "
          f"{EXPECTED_TEST_SIZE}; a label array of any other length is refused")
    return 5


def check_output_safety(directory):
    """Where results may go, and the three ways a write is refused."""
    assert results_dir("targeted", 44, "heldout") == \
        TEST_RESULTS_ROOT / "targeted_seed44" / "heldout"
    assert results_dir("clean", 42, "frozen", directory) == \
        directory / "clean_seed42" / "frozen"
    checks = 2

    fresh = directory / "fresh" / "frozen"
    assert check_output_target(fresh) == fresh.resolve()
    assert not fresh.exists(), "a check created an output directory"
    fresh.mkdir(parents=True)
    assert check_output_target(fresh) == fresh.resolve(), \
        "an existing but empty directory is a usable target"
    (fresh / "summary.csv").write_text("occupied\n")
    try:
        check_output_target(fresh)
    except RuntimeError as error:
        assert "already holds results" in str(error), error
    else:
        raise AssertionError("an occupied output directory was accepted")
    checks += 3

    # The accepted artifacts. Writing into one, or into a directory that
    # contains one, both have to be refused before a model is built.
    for protected in PROTECTED_ROOTS:
        for candidate in (protected, protected / "frozen", protected.parent):
            try:
                check_output_target(candidate)
            except RuntimeError as error:
                assert ("holds accepted results" in str(error)
                        or "contains the accepted results" in str(error)
                        or "already holds results" in str(error)), error
            else:
                raise AssertionError(f"{candidate} was accepted as an output "
                                     "directory")
            checks += 1
    names = ", ".join(path.name for path in PROTECTED_ROOTS)
    print(f"  output paths are arithmetic on the arm, seed and suite; an "
          f"occupied directory is refused, and so is any path inside or "
          f"containing {names}")

    # The comparison writes once too.
    analysis = directory / "analysis"
    assert compare_test.check_output_target(analysis) == analysis.resolve()
    analysis.mkdir(parents=True)
    (analysis / "paired_contrasts.csv").write_text("occupied\n")
    try:
        compare_test.check_output_target(analysis)
    except RuntimeError as error:
        assert "written once" in str(error), error
    else:
        raise AssertionError("an occupied analysis directory was accepted")
    print("  the comparison refuses an analysis directory that already holds "
          "one of its 14 files")
    checks += 2

    # The comparison's protected trees are narrower than the evaluator's,
    # because its own output belongs inside artifacts/analysis.
    assert compare_test.check_output_target(compare_test.ANALYSIS_DIR) == \
        compare_test.ANALYSIS_DIR
    for candidate in compare_test.PROTECTED_TREES:
        for target in (candidate, candidate / "clean_seed42" / "frozen",
                       candidate.parent):
            try:
                compare_test.check_output_target(target)
            except RuntimeError as error:
                assert "read-only" in str(error), error
            else:
                raise AssertionError(f"{target} was accepted as a comparison "
                                     "output directory")
            checks += 1
    names = ", ".join(evaluate_test._repository_path(path)
                      for path in compare_test.PROTECTED_TREES)
    print(f"  it also refuses {names} and any directory containing one of "
          "them, while accepting its own analysis directory, which lives "
          "inside artifacts/analysis and so cannot be protected wholesale")
    return checks


class FakeCounter:
    """A loader-shaped object carrying the two counts the log reports."""

    def __init__(self, iterations, reads):
        self.iterations = iterations
        self.dataset = type("Counted", (), {"reads": reads})()


def check_failed_line_fields():
    """A failed run's line says which checkpoint spent the look, and on what.

    The failure case is the one where a look at the test split has been spent
    and no result is kept, so the line has to carry the identity and the counts
    that were true when the run stopped rather than the zeros it started with.
    """
    loaders = {"test": FakeCounter(1, EXPECTED_TEST_SIZE),
               "train": FakeCounter(0, 0),
               "validation": FakeCounter(0, 0)}
    measured = evaluate_test.measured_split_counts(loaders)
    assert measured == {
        "test_loader_iterations": 1,
        "test_loader_reads": EXPECTED_TEST_SIZE,
        "train_loader_iterations": 0, "train_loader_reads": 0,
        "validation_loader_iterations": 0, "validation_loader_reads": 0,
    }, measured

    counters = {"arm": "targeted", "seed": 44, "checkpoint_md5": "a" * 32,
                "suite": "heldout", "conditions": 26,
                "test_images_read": 200000, "output_dir": None}
    counters.update(measured)
    entry = access_log_entry("failed", error="RuntimeError: stopped",
                             elapsed_seconds=None, **counters)
    assert entry["status"] == "failed"
    for field in ("arm", "seed", "checkpoint_md5", "suite", "conditions",
                  "test_images_read", "test_loader_reads"):
        assert entry[field] is not None, field
    assert entry["test_images_read"] == 200000
    assert entry["conditions"] == 26
    print("  a failed line carries the arm, seed, checkpoint md5, suite, "
          "condition count, the test images read so far and the live loader "
          "counts, none of them left at the value the run started with")
    print("  the same six loader counts are what check_split_accounting "
          "asserts, so the log and the assertion read one source")
    return 4


def check_worker_rejection():
    """A worker count that would scatter the read accounting is refused."""
    base = ["--checkpoint", "artifacts/baseline/best_checkpoint.pt",
            "--suite", "frozen"]
    arguments = evaluate_test.parse_arguments(base)
    assert arguments.num_workers == 0
    for workers in (1, 2, 8):
        try:
            evaluate_test.parse_arguments(base + ["--num-workers", str(workers)])
        except SystemExit:
            pass
        else:
            raise AssertionError(f"--num-workers {workers} was accepted, and "
                                 "the sweep's read accounting cannot hold")
    print("  the default is 0 workers, and 1, 2 and 8 are refused at argument "
          "parsing, before anything is read: CountingDataset counts a "
          "worker's own copy, so the budget would fail only after the whole "
          "split had been swept")
    return 4


def check_access_log(directory):
    """Append-only, and honest about runs that did not finish."""
    log = directory / "access" / "test_access_log.jsonl"
    completed = access_log_entry(
        "completed", arm="targeted", seed=44, suite="heldout", conditions=26,
        checkpoint_md5="0" * 32, test_images_read=339326,
        test_loader_iterations=1, test_loader_reads=EXPECTED_TEST_SIZE,
        train_loader_iterations=0, train_loader_reads=0,
        validation_loader_iterations=0, validation_loader_reads=0,
        output_dir="artifacts/final_test/targeted_seed44/heldout",
        elapsed_seconds=12.5)
    failed = access_log_entry(
        "failed", arm="clean", seed=42, suite="frozen", conditions=31,
        checkpoint_md5="1" * 32, test_images_read=52204,
        error="RuntimeError: deliberate")

    for entry in (completed, failed):
        for field in ("timestamp", "module", "status", "git_head", "git_dirty",
                      "arm", "seed", "suite", "conditions", "checkpoint_md5",
                      "test_images_read"):
            assert field in entry, field
        assert entry["module"] == "src/evaluate_test.py"
        assert len(entry["git_head"]) == 40
        assert isinstance(entry["git_dirty"], list)
    assert completed["status"] == "completed"
    assert failed["status"] == "failed"
    assert "error" in failed and "error" not in completed
    assert completed["train_loader_reads"] == 0
    assert completed["validation_loader_reads"] == 0

    assert append_access_log(completed, log) == log
    first = log.read_text()
    append_access_log(failed, log)
    second = log.read_text()
    assert second.startswith(first), "the log was rewritten rather than appended"
    lines = [json.loads(line) for line in second.splitlines()]
    assert len(lines) == 2
    assert lines[0] == completed and lines[1] == failed
    assert [line["status"] for line in lines] == ["completed", "failed"]
    assert sum(1 for line in lines if line["status"] == "completed") == 1
    print("  two entries appended in order, the first line byte-identical "
          "after the second was written, and a failed attempt recorded as an "
          "attempt rather than as a completed run")

    # The log is opened for append and fsynced, so an entry survives without
    # the file being read back in first.
    assert os.path.exists(log)
    third = access_log_entry("completed", arm="balanced", seed=43,
                             suite="frozen", conditions=31)
    append_access_log(third, log)
    assert len(log.read_text().splitlines()) == 3
    assert log.read_text().startswith(second)

    # The frozen verification: 18 completed lines, no repeated triple.
    triples = {(line["arm"], line["seed"], line["suite"])
               for line in [json.loads(value)
                            for value in log.read_text().splitlines()]}
    assert len(triples) == 3
    print("  the arm, seed and suite triple of each line is distinct, which is "
          "the uniqueness the 18-run verification checks")
    return 24


def check_read_accounting(observations):
    """Derive the read budget from the iteration, then compare it to the protocol.

    Measured on synthetic images: a condition reads every image once for the
    evaluation pass and once more at every `FINGERPRINT_STRIDE`-th position for
    the fingerprint, and nothing else reads the observations. The official
    figures then follow from that structure and the two suite sizes, and are
    required to equal the ones the frozen protocol states.
    """
    counted = CountingDataset(observations)
    assert counted.reads == 0
    suite = TEST_SUITES["frozen"]
    conditions = suite.conditions()[:4]
    positions = list(range(0, len(counted), FINGERPRINT_STRIDE))
    model = FixedLogits().eval()
    criterion = torch.nn.CrossEntropyLoss()
    per_condition = []
    for corruption, severity in conditions:
        before = counted.reads
        dataset = suite.dataset(counted, corruption, severity)
        loader = condition_loader(dataset, batch_size=16, num_workers=0)
        evaluate_condition(model, loader, torch.device("cpu"), criterion)
        after_pass = counted.reads
        input_fingerprint(dataset, positions)
        per_condition.append((after_pass - before, counted.reads - after_pass))
    for sweep, fingerprint in per_condition:
        assert sweep == len(counted), sweep
        assert fingerprint == len(positions), fingerprint
    total = counted.reads
    assert total == len(conditions) * (len(counted) + len(positions))
    print(f"  measured on {len(counted)} synthetic images: every condition "
          f"reads the dataset once ({len(counted)} reads) plus one re-read per "
          f"fingerprinted position ({len(positions)} reads), and nothing else "
          f"reads it - {len(conditions)} conditions totalled {total}")

    official_positions = list(range(0, EXPECTED_TEST_SIZE, FINGERPRINT_STRIDE))
    assert len(official_positions) == 421
    derived = {}
    for name, suite in sorted(TEST_SUITES.items()):
        budget = expected_reads(suite, suite.conditions(), official_positions)
        assert budget["per_condition"] == EXPECTED_TEST_SIZE + 421 == 13051
        assert budget["total"] == budget["per_condition"] * budget["conditions"]
        assert budget["total"] == PROTOCOL_READ_COUNTS[name], name
        assert budget["protocol_total"] == PROTOCOL_READ_COUNTS[name]
        derived[name] = budget
        print(f"  {name}: {budget['conditions']} x ({EXPECTED_TEST_SIZE} + "
              f"421) = {budget['total']}, which is the figure Part 9 states")

    # And the disagreement branch: a suite whose conditions do not multiply out
    # to the protocol figure has to stop the run, not be rounded into it.
    try:
        expected_reads(TEST_SUITES["frozen"],
                       TEST_SUITES["frozen"].conditions()[:-1],
                       official_positions)
    except RuntimeError as error:
        assert "disagree" in str(error), error
    else:
        raise AssertionError("a read budget that disagrees with the protocol "
                             "was accepted")
    print("  a sweep whose budget does not equal the stated figure stops the "
          "run rather than adopting either number")

    # The smoke path is a stride subset of the same reads, and writes nothing.
    assert SMOKE_CONDITIONS == 3 and SMOKE_STRIDE == 400
    assert len(range(0, EXPECTED_TEST_SIZE, SMOKE_STRIDE)) == 32
    print(f"  the smoke path would read {SMOKE_CONDITIONS} conditions at "
          f"stride {SMOKE_STRIDE} (32 images each) and writes no artifact and "
          "no access-log line; it is not run here, because it decodes official "
          "test images")
    return 8 + len(per_condition) * 2 + len(derived) * 4


def fixture_labels(count=EXPECTED_TEST_SIZE):
    """Deliberately unbalanced synthetic labels.

    Class sizes have to differ, or the class-balanced table would equal the
    primary one and checking it would prove nothing.
    """
    return np.minimum(np.arange(count) // 300, NUM_CLASSES - 1)


def fixture_errors(arm, seed, suite_name, corruption, severity,
                   count=EXPECTED_TEST_SIZE):
    """Which synthetic images one arm gets wrong under one condition.

    An arithmetic pattern, so the arm rates and the paired contrasts can be
    derived from the pattern itself rather than from the module being checked.
    The offset moves with the seed, the suite and the severity, so per-image
    rates are fractions rather than zeros and ones and the equal-weight mean
    over conditions is actually exercised.
    """
    offset = (FIXTURE_SEED_STRIDE[seed] + severity
              + (0 if suite_name == "frozen" else 1)
              + len(corruption))
    return ((np.arange(count) + offset) % FIXTURE_STRIDES[arm]) == 0


def write_fixture(directory, label_index):
    """A full synthetic result set for all nine models and both suites.

    Same shape as what `evaluate_test.py` writes - same files, same columns,
    same metadata fields - with predictions from the pattern above instead of
    from a model. The numbers that come out of the comparison are fixture
    values and are compared against the pattern, not reported as measurements.
    """
    labels = label_index["label"].to_numpy()
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    support = {str(class_id): int(counts[class_id])
               for class_id in range(NUM_CLASSES)}
    models, inventory = [], []
    for arm in compare_test.ARMS:
        for seed in FIXTURE_SEEDS:
            digest = hashlib.md5(f"{arm}:{seed}".encode()).hexdigest()
            epoch = 20 + FIXTURE_SEEDS.index(seed)
            model_dir = directory / f"{arm}_seed{seed}"
            inventory.append({
                "arm": compare_test.ARM_LABELS[arm],
                "arm_label": arm,
                "seed": seed,
                "checkpoint_path": f"fixture/{arm}_seed{seed}/best_model.pt",
                "checkpoint_md5": digest,
                "checkpoint_sha256": "f" * 64,
                "selected_epoch": epoch,
            })
            for suite_name, suite in sorted(TEST_SUITES.items()):
                conditions = suite.conditions()
                target = model_dir / suite_name
                target.mkdir(parents=True, exist_ok=True)
                frames, summary_rows, class_rows = [], [], []
                for corruption, severity in conditions:
                    if corruption == CLEAN_CONDITION:
                        wrong = fixture_errors(arm, seed, suite_name,
                                               "clean", 0)
                    else:
                        wrong = fixture_errors(arm, seed, suite_name,
                                               corruption, severity)
                    prediction = np.where(wrong, (labels + 1) % NUM_CLASSES,
                                          labels)
                    frames.append({"corruption": corruption,
                                   "severity": severity,
                                   "prediction": prediction,
                                   "wrong": wrong})
                    accuracy = 1.0 - float(wrong.mean())
                    summary_rows.append({
                        "model": arm, "seed": seed, "suite": suite_name,
                        "corruption": corruption, "severity": severity,
                        "parameter": suite.parameter_name(corruption),
                        "parameter_value": suite.parameter_value(corruption,
                                                                 severity),
                        "loss": float(wrong.mean()), "accuracy": accuracy,
                        "error_rate": 1.0 - accuracy,
                        "macro_f1": accuracy * 0.99,
                        "samples": EXPECTED_TEST_SIZE})
                    for class_id in range(NUM_CLASSES):
                        selected = labels == class_id
                        class_rows.append({
                            "corruption": corruption, "severity": severity,
                            "class": class_id,
                            "support": int(selected.sum()),
                            "precision": 0.0,
                            "recall": 1.0 - float(wrong[selected].mean()),
                            "f1": 0.0})

                with open(target / "predictions.csv", "w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["corruption", "severity", "test_position",
                                     "dataset_index", "label", "prediction"])
                    positions = np.arange(EXPECTED_TEST_SIZE)
                    for frame in frames:
                        writer.writerows(zip(
                            [frame["corruption"]] * EXPECTED_TEST_SIZE,
                            [frame["severity"]] * EXPECTED_TEST_SIZE,
                            positions.tolist(), positions.tolist(),
                            labels.tolist(), frame["prediction"].tolist()))

                with open(target / "summary.csv", "w", newline="") as handle:
                    writer = csv.DictWriter(handle,
                                            fieldnames=list(summary_rows[0]))
                    writer.writeheader()
                    writer.writerows(summary_rows)
                with open(target / "per_class_metrics.csv", "w",
                          newline="") as handle:
                    writer = csv.DictWriter(handle,
                                            fieldnames=list(class_rows[0]))
                    writer.writeheader()
                    writer.writerows(class_rows)

                arm_record = ("clean baseline" if arm == "clean"
                              else f"augmented ({arm})")
                (target / "metadata.json").write_text(json.dumps({
                    "model": {"name": arm, "seed": seed, "arm": arm_record,
                              "training_records": {"best_epoch": epoch},
                              "provenance": "synthetic fixture",
                              "protocol": {"protocol": "synthetic fixture"}},
                    "checkpoint": {
                        "path": f"fixture/{arm}_seed{seed}/best_model.pt",
                        "md5": digest, "epoch": epoch, "seed": seed},
                    "evaluation": {"split": "official_test",
                                   "test_images": EXPECTED_TEST_SIZE},
                    "corruption_suite": {
                        "namespace": TEST_NAMESPACE,
                        "input_fingerprints": {
                            f"{corruption}_s{severity}":
                                f"{suite_name}:{corruption}:{severity}"
                            for corruption, severity in conditions}},
                    "test_class_support": support,
                }, indent=2) + "\n")
            models.append({"name": arm, "seed": seed, "dir": model_dir,
                           "key": f"{arm}_seed{seed}"})
    inventory_path = directory / "inventory.json"
    inventory_path.write_text(json.dumps({"models": inventory}, indent=2) + "\n")
    return models, inventory_path


def expected_fixture_tables(label_index):
    """The arm rates and paired contrasts the pattern implies, by direct numpy.

    Computed from the error pattern straight into arrays, with no CSV, no
    merge, no groupby and no pivot - so agreement with the comparison's tables
    is evidence that its aggregation does what it says, not that two copies of
    the same code agree.
    """
    labels = label_index["label"].to_numpy()
    arms, contrasts, balanced = {}, {}, {}
    for suite_name, suite in sorted(TEST_SUITES.items()):
        corrupted = [condition for condition in suite.conditions()
                     if condition[0] != CLEAN_CONDITION]
        for seed in FIXTURE_SEEDS:
            rates, class_means = {}, {}
            for arm in compare_test.ARMS:
                stack = np.stack([
                    fixture_errors(arm, seed, suite_name, corruption, severity)
                    for corruption, severity in corrupted]).astype(float)
                rates[arm] = stack.mean(axis=0)
                arms[(suite_name, seed, arm)] = float(stack.mean())
                per_class = np.stack([
                    np.array([row[labels == class_id].mean()
                              for class_id in range(NUM_CLASSES)])
                    for row in stack])
                class_means[arm] = per_class
                balanced[(suite_name, seed, arm)] = float(
                    per_class.mean(axis=1).mean())
            for left, right in compare_test.CONTRASTS:
                difference = rates[left] - rates[right]
                contrasts[(suite_name, seed, left, right)] = {
                    "difference": float(difference.mean()),
                    "worse_under_left": int((difference > 0).sum()),
                    "unchanged": int((difference == 0).sum()),
                    "class_balanced": float(
                        (class_means[left] - class_means[right])
                        .mean(axis=1).mean()),
                }
    return {"arms": arms, "contrasts": contrasts, "class_balanced": balanced}


def check_comparison_math(directory):
    """The whole comparison, end to end, on synthetic predictions.

    Nine synthetic result sets at the real split size, through the real
    command-line path, checked against the error pattern they were built from.
    No official prediction exists and none is produced.
    """
    label_index = pd.DataFrame({
        "dataset_index": np.arange(EXPECTED_TEST_SIZE),
        "label": fixture_labels()})
    models, inventory_path = write_fixture(directory / "results", label_index)
    expected = expected_fixture_tables(label_index)
    output_dir = directory / "analysis"

    started = time.time()
    outcome = compare(models, output_dir, label_index=label_index,
                      inventory_path=inventory_path,
                      validation_dir=compare_models.ANALYSIS_DIR)
    print(f"\n  the fixture comparison ran in {time.time() - started:.1f}s; "
          "every number above is a fixture value, not a measurement")

    written = {path.name for path in output_dir.iterdir()}
    assert written == set(compare_test.OUTPUT_FILES), sorted(
        written.symmetric_difference(compare_test.OUTPUT_FILES))
    checks = 1

    arm_table = pd.read_csv(output_dir / "arm_error_rates.csv")
    assert len(arm_table) == len(TEST_SUITES) * len(FIXTURE_SEEDS) * 3
    for row in arm_table.to_dict("records"):
        key = (row["suite"], row["seed"], row["model"])
        assert abs(row["mean_image_error"] - expected["arms"][key]) < 1e-12, key
        assert row["images"] == EXPECTED_TEST_SIZE
        assert row["draws"] == compare_test.BOOTSTRAP_DRAWS
        assert row["ci_low"] <= row["mean_image_error"] <= row["ci_high"]
        assert "independent image-level sampling" in row["interval_assumption"]
        checks += 1
    print(f"  {len(arm_table)} arm error rates equal their closed-form values "
          "exactly, each over 12630 images and 2000 draws")

    contrast_table = pd.read_csv(output_dir / "paired_contrasts.csv")
    assert len(contrast_table) == len(TEST_SUITES) * len(FIXTURE_SEEDS) * 3
    for row in contrast_table.to_dict("records"):
        key = (row["suite"], row["seed"], row["left"], row["right"])
        reference = expected["contrasts"][key]
        assert abs(row["difference"] - reference["difference"]) < 1e-12, key
        assert row["images_worse_under_left"] == reference["worse_under_left"], key
        assert row["images_unchanged"] == reference["unchanged"], key
        assert abs(row["left_mean_error"]
                   - expected["arms"][(row["suite"], row["seed"],
                                       row["left"])]) < 1e-12
        assert abs(row["difference"] - (row["left_mean_error"]
                                        - row["right_mean_error"])) < 1e-12
        checks += 1
    print(f"  {len(contrast_table)} paired contrasts equal their closed-form "
          "values exactly, including the count of images the left arm got "
          "wrong more often")

    balanced = pd.read_csv(output_dir / "class_balanced_error.csv")
    for row in balanced.to_dict("records"):
        key = (row["suite"], row["seed"], row["model"])
        assert abs(row["class_balanced_error"]
                   - expected["class_balanced"][key]) < 1e-12, key
        assert row["classes"] == NUM_CLASSES
        checks += 1
    assert "ci_low" not in balanced.columns
    assert "ci_high" not in balanced.columns
    balanced_contrasts = pd.read_csv(output_dir / "class_balanced_contrasts.csv")
    for row in balanced_contrasts.to_dict("records"):
        key = (row["suite"], row["seed"], row["left"], row["right"])
        assert abs(row["mean_difference"]
                   - expected["contrasts"][key]["class_balanced"]) < 1e-12, key
        checks += 1
    assert "ci_low" not in balanced_contrasts.columns
    assert not np.allclose(balanced["class_balanced_error"].to_numpy(),
                           arm_table["mean_image_error"].to_numpy()), \
        "the fixture labels are balanced, so this check proves nothing"
    print(f"  {len(balanced)} class-balanced errors and "
          f"{len(balanced_contrasts)} class-balanced contrasts equal their "
          "closed-form values, carry no interval column, and differ from the "
          "image-weighted table")

    sensitivity = pd.read_csv(output_dir / "design_effect_sensitivity.csv")
    assert len(sensitivity) == len(contrast_table)
    factor = float(np.sqrt(1 + (compare_test.DESIGN_EFFECT_CLUSTER - 1)
                           * compare_test.DESIGN_EFFECT_RHO))
    for row in sensitivity.to_dict("records"):
        assert abs(row["scale_factor"] - factor) < 1e-12
        assert row["assumed_cluster_size"] == 30
        assert abs(row["illustrative_low"] - (
            row["point_estimate"]
            - (row["point_estimate"] - row["image_ci_low"]) * factor)) < 1e-12
        assert abs(row["illustrative_high"] - (
            row["point_estimate"]
            + (row["image_ci_high"] - row["point_estimate"]) * factor)) < 1e-12
        assert "not a confidence bound" in row["status"]
        checks += 1
    print(f"  {len(sensitivity)} sensitivity rows scale each endpoint's "
          f"distance from the point estimate by {factor:.4f}, separately above "
          "and below, and every row states that it is neither a bound nor a "
          "decision input")

    # The illustration appears nowhere else. No other output may carry the
    # scale factor or a scaled endpoint. `comparison.json` names the
    # assumptions and the three negative flags in its protocol block, because
    # that block is the record of what the analysis did; it carries none of
    # the illustration's numbers.
    for path in sorted(output_dir.iterdir()):
        if path.name == "design_effect_sensitivity.csv":
            continue
        text = path.read_text()
        for token in ("5.477", "illustrative", "scale_factor"):
            assert token not in text, f"{path.name} mentions {token}"
        if path.name != "comparison.json":
            for token in ("cluster", "design_effect"):
                assert token not in text, f"{path.name} mentions {token}"
        checks += 1
    recorded_illustration = json.loads(
        (output_dir / "comparison.json").read_text())["protocol"][
            "design_effect_illustration"]
    assert set(recorded_illustration) == {
        "assumed_cluster_size", "assumed_within_cluster_correlation",
        "appears_only_in", "is_a_confidence_bound",
        "is_a_corrected_interval", "enters_the_decision_rule"}
    assert recorded_illustration["appears_only_in"] == \
        "design_effect_sensitivity.csv"
    print("  no other output file carries the scale factor or a scaled "
          "endpoint; comparison.json records only the assumptions, where the "
          "illustration lives, and that it is neither a bound, nor a "
          "corrected interval, nor a decision input")

    seeds = pd.read_csv(output_dir / "across_seeds.csv")
    for row in seeds.to_dict("records"):
        group = contrast_table[(contrast_table["suite"] == row["suite"])
                               & (contrast_table["contrast"] == row["contrast"])]
        assert row["seeds"] == len(group) == len(FIXTURE_SEEDS)
        assert abs(row["mean_difference"]
                   - group["difference"].mean()) < 1e-12
        assert row["seeds_favouring_left"] == int((group["difference"] < 0).sum())
        checks += 1
    assert "ci_low" not in seeds.columns and "p_value" not in seeds.columns
    print(f"  {len(seeds)} across-seed rows average the per-seed point "
          "estimates and carry no interval and no p-value")

    reproduction = pd.read_csv(output_dir / "validation_reproduction.csv")
    validation = pd.read_csv(compare_models.ANALYSIS_DIR / "across_seeds.csv")
    assert len(reproduction) == len(validation)
    for row in reproduction.to_dict("records"):
        source = validation[(validation["suite"] == row["suite"])
                            & (validation["contrast"] == row["contrast"])]
        assert len(source) == 1
        assert abs(row["validation_mean_difference"]
                   - float(source["mean_difference"].iloc[0])) < 1e-12
        assert row["directions_agree"] == (
            row["validation_direction"] == row["test_direction"])
        assert "not comparable" in row["unit_of_analysis"]
        checks += 1
    print(f"  {len(reproduction)} reproduction rows carry the committed "
          "validation means unchanged beside the fixture ones, with no "
          "interval spanning the two")

    report = (output_dir / "comparison.txt").read_text()
    for phrase in ("assume independent image-level sampling",
                   "not definitive population-level",
                   "narrower than a dependence-aware interval",
                   "No significance across training seeds is claimed",
                   "references no interval"):
        assert phrase in report, phrase
        checks += 1
    recorded = json.loads((output_dir / "comparison.json").read_text())
    assert recorded["protocol"]["split"] == "official_test"
    assert recorded["protocol"]["namespace"] == TEST_NAMESPACE
    assert recorded["protocol"]["bootstrap_draws"] == 2000
    assert recorded["protocol"]["bootstrap_seed"] == 42
    assert recorded["protocol"]["interval"] == [2.5, 97.5]
    assert recorded["protocol"]["images"] == EXPECTED_TEST_SIZE
    assert recorded["protocol"]["accounts_for_physical_sign_dependence"] is False
    assert recorded["protocol"]["track_structure_available"] is False
    assert recorded["protocol"]["design_effect_illustration"][
        "enters_the_decision_rule"] is False
    assert recorded["protocol"]["clean_accuracy_limit"] == 0.01
    assert recorded["protocol"][
        "clean_accuracy_limit_is_a_pass_criterion"] is False
    print("  the rendered report carries the independence caveat, the "
          "no-significance statement and the no-interval decision rule, and "
          "comparison.json records the protocol constants")
    checks += 11

    # A second run into the same directory has to be refused.
    try:
        compare(models, output_dir, label_index=label_index,
                inventory_path=inventory_path,
                validation_dir=compare_models.ANALYSIS_DIR)
    except RuntimeError as error:
        assert "written once" in str(error), error
    else:
        raise AssertionError("the comparison overwrote its own output")

    # A validation result set passed as a test one has to be refused.
    stray = models[0]["dir"] / "frozen" / "metadata.json"
    original = stray.read_text()
    altered = json.loads(original)
    altered["corruption_suite"]["namespace"] = "validation"
    stray.write_text(json.dumps(altered, indent=2) + "\n")
    try:
        compare_test.check_test_provenance(models, inventory_path)
    except RuntimeError as error:
        assert "corruption namespace" in str(error), error
    else:
        raise AssertionError("a validation-namespace result set was accepted "
                             "as a test one")
    altered["corruption_suite"]["namespace"] = TEST_NAMESPACE
    altered["evaluation"]["split"] = "validation"
    stray.write_text(json.dumps(altered, indent=2) + "\n")
    try:
        compare_test.check_test_provenance(models, inventory_path)
    except RuntimeError as error:
        assert "records the split" in str(error), error
    else:
        raise AssertionError("a validation result set was accepted as a test one")
    stray.write_text(original)

    # And a checkpoint outside the inventory.
    try:
        compare_test.check_test_provenance(models, evaluate_test.INVENTORY_PATH)
    except RuntimeError as error:
        assert "not one of the" in str(error), error
    else:
        raise AssertionError("a result set whose checkpoint is not in the "
                             "inventory was accepted")
    print("  a second run into the same directory, a validation namespace, a "
          "validation split and an unapproved checkpoint are all refused")

    # A short or over-covered prediction file is refused rather than averaged.
    truncated = directory / "truncated"
    (truncated / "frozen").mkdir(parents=True)
    frame = pd.read_csv(models[0]["dir"] / "frozen" / "predictions.csv")
    frame.iloc[:-1].to_csv(truncated / "frozen" / "predictions.csv", index=False)
    try:
        compare_test.load_suite_images(
            {"name": "clean", "seed": 42, "dir": truncated, "key": "x"},
            "frozen", label_index)
    except RuntimeError as error:
        assert "12629 rows" in str(error) and "occlusion" in str(error), error
    else:
        raise AssertionError("a prediction file missing an image was accepted")
    print("  a prediction file missing one image is refused, naming the "
          "condition that is short, rather than averaged over 12629")
    return checks + 6


def check_bootstrap_reproducibility():
    """The draws are the validation module's draws, and they repeat exactly."""
    from .analyze_placement import bootstrap_draws

    reference = bootstrap_draws(500, 600, 42)
    chunked = image_draws(500, draws=600, seed=42)
    assert np.array_equal(chunked, reference), \
        "the chunked draw matrix is not the one analyze_placement produces"
    assert chunked.dtype == np.int32
    assert np.array_equal(chunked, image_draws(500, draws=600, seed=42))
    assert not np.array_equal(chunked, image_draws(500, draws=600, seed=43))
    print("  the chunked int32 draw matrix is element-for-element the one "
          "analyze_placement.bootstrap_draws returns, repeats exactly on a "
          "rerun, and changes with the seed")

    full = image_draws(EXPECTED_TEST_SIZE)
    assert full.shape == (compare_test.BOOTSTRAP_DRAWS, EXPECTED_TEST_SIZE)
    assert np.array_equal(full, bootstrap_draws(
        EXPECTED_TEST_SIZE, compare_test.BOOTSTRAP_DRAWS,
        compare_test.BOOTSTRAP_SEED))
    assert int(full.min()) == 0 and int(full.max()) == EXPECTED_TEST_SIZE - 1
    print(f"  at the real split size the matrix is "
          f"{full.shape[0]} x {full.shape[1]} int32 "
          f"({full.nbytes / 1e6:.0f} MB) and identical to the monolithic draw")

    values = np.random.default_rng(7).random(EXPECTED_TEST_SIZE)
    first = bootstrap_image_mean(values, matrix=full)
    second = bootstrap_image_mean(values)
    assert first == second, "reusing the matrix changed the interval"
    assert abs(first["mean"] - float(values.mean())) < 1e-12
    assert first["images"] == EXPECTED_TEST_SIZE
    assert first["draws"] == 2000

    original = compare_test.CHUNK_DRAWS
    try:
        compare_test.CHUNK_DRAWS = 37
        third = bootstrap_image_mean(values)
    finally:
        compare_test.CHUNK_DRAWS = original
    assert third == first, "the interval depends on the chunk size"
    print("  the interval is identical whether the draws are built once and "
          "reused, rebuilt, or consumed in chunks of a different size")

    # The estimator itself, against a case with a known answer: a constant
    # difference has a zero-width interval at its own value.
    constant = bootstrap_image_mean(np.full(1000, -0.25), draws=200, seed=1)
    assert constant["mean"] == constant["ci_low"] == constant["ci_high"] == -0.25
    paired = np.array([1.0, -1.0] * 500)
    symmetric = bootstrap_image_mean(paired, draws=500, seed=3)
    assert abs(symmetric["mean"]) < 1e-12
    assert symmetric["ci_low"] < 0 < symmetric["ci_high"]
    print("  a constant difference gives a zero-width interval at its own "
          "value, and a balanced one gives an interval straddling zero")

    # Paired differencing, at the only size the module accepts. The left frame
    # is shuffled and the right is not, so a version that paired on row order
    # rather than on the dataset index would give a different vector.
    positions = np.arange(EXPECTED_TEST_SIZE)
    left = np.linspace(0.0, 1.0, EXPECTED_TEST_SIZE)
    right = np.sqrt(left)
    shuffled = np.random.default_rng(11).permutation(positions)
    rates = pd.concat([
        pd.DataFrame({"model": "left", "dataset_index": positions[shuffled],
                      "rate": left[shuffled]}),
        pd.DataFrame({"model": "right", "dataset_index": positions,
                      "rate": right})], ignore_index=True)
    difference = paired_image_contrast(rates, ["model"], "left", "right")
    assert np.allclose(difference, left - right)
    assert np.allclose(paired_image_contrast(rates, ["model"], "right", "left"),
                       right - left)
    print("  paired differences are taken within an image and in dataset-index "
          "order, so a shuffled input gives the same vector, and reversing the "
          "contrast negates it")

    short = rates[rates["dataset_index"] != 7]
    try:
        paired_image_contrast(short, ["model"], "left", "right")
    except RuntimeError as error:
        assert "expected 12630" in str(error), error
    else:
        raise AssertionError("a contrast missing an image was accepted")

    # One side short of the other. Differencing over the intersection would
    # silently produce a 12,629-image contrast, so the two index sets are
    # required to be equal before anything is subtracted.
    lopsided = rates[(rates["model"] != "left")
                     | (rates["dataset_index"] != 7)]
    try:
        paired_image_contrast(lopsided, ["model"], "left", "right")
    except RuntimeError as error:
        assert "do not cover the same images" in str(error), error
        assert "only in right" in str(error), error
    else:
        raise AssertionError("two arms covering different images were paired")
    try:
        image_rates(pd.DataFrame({
            "model": ["left"] * 3, "corruption": ["fog"] * 3,
            "severity": [1, 1, 1], "dataset_index": [0, 1, 2],
            "rate": [0.0, 1.0, 0.0]}), ["model"])
    except RuntimeError as error:
        assert "expected 12630" in str(error), error
    else:
        raise AssertionError("a condition covering 3 images was accepted")
    print("  two arms covering different image sets are refused by name, and "
          "not differenced over the intersection")
    print("  a contrast or a condition that does not cover all 12630 images is "
          "refused rather than averaged over what is present")
    return 22


def check_tradeoff_logic():
    """The clean-accuracy rule, including which arm the cost is measured against."""
    clean = [
        {"arm": "A", "model": "clean", "seed": 42, "clean_accuracy": 0.9900,
         "clean_macro_f1": 0.98, "clean_loss": 0.04},
        {"arm": "B", "model": "balanced", "seed": 42, "clean_accuracy": 0.9850,
         "clean_macro_f1": 0.97, "clean_loss": 0.05},
        {"arm": "C", "model": "targeted", "seed": 42, "clean_accuracy": 0.9700,
         "clean_macro_f1": 0.96, "clean_loss": 0.06},
    ]
    contrasts = [
        {"suite": "frozen", "seed": 42, "contrast": "B-A", "left": "balanced",
         "right": "clean", "difference": -0.08},
        {"suite": "frozen", "seed": 42, "contrast": "C-A", "left": "targeted",
         "right": "clean", "difference": -0.09},
        {"suite": "frozen", "seed": 42, "contrast": "C-B", "left": "targeted",
         "right": "balanced", "difference": -0.01},
        {"suite": "frozen", "seed": 42, "contrast": "B-A", "left": "balanced",
         "right": "clean", "difference": +0.02},
    ]
    rows = compare_test.clean_tradeoff_rows(clean, contrasts)
    assert len(rows) == 4
    costs = {(row["contrast"], row["corrupted_error_change"]):
             row["clean_accuracy_cost"] for row in rows}
    assert abs(costs[("B-A", -0.08)] - 0.0050) < 1e-12
    assert abs(costs[("C-A", -0.09)] - 0.0200) < 1e-12
    # The C-B cost is measured against B, the contrast's own right-hand arm,
    # and not against A: 0.9850 - 0.9700, not 0.9900 - 0.9700.
    assert abs(costs[("C-B", -0.01)] - 0.0150) < 1e-12
    assert abs(costs[("C-B", -0.01)] - (0.9900 - 0.9700)) > 1e-6
    assert rows[0]["exceeds_clean_limit"] is False
    assert rows[1]["exceeds_clean_limit"] is True
    assert rows[2]["exceeds_clean_limit"] is True
    assert rows[0]["verdict"] == "improvement within the clean-accuracy limit"
    assert rows[1]["verdict"].startswith("trade-off")
    assert rows[3]["verdict"] == "no corrupted gain"
    print(f"  the clean cost of a contrast is its own right-hand arm's clean "
          f"accuracy minus its left arm's, the "
          f"{compare_test.CLEAN_TRADEOFF_LIMIT:g} limit labels a trade-off "
          f"rather than failing it, and a contrast with no corrupted gain is "
          f"labelled as such")
    assert compare_test._tradeoff_verdict(0.5, 0.0) == "no corrupted gain"
    assert compare_test._tradeoff_verdict(0.0, -0.1) == \
        "improvement within the clean-accuracy limit"
    return 12


def check_decision_rule():
    """The frozen rule for C against B, on constructed across-seed summaries."""
    def summary(suite, contrast, favouring_left, seeds=3):
        return {"suite": suite, "contrast": contrast, "seeds": seeds,
                "seeds_favouring_left": favouring_left,
                "mean_difference": -0.01 if favouring_left == seeds else 0.001}

    both = [summary("frozen", "C-B", 3), summary("heldout", "C-B", 3)]
    assert compare_test.targeted_verdict(both, 3)["verdict"] == \
        "targeted preferred"

    frozen_only = [summary("frozen", "C-B", 3), summary("heldout", "C-B", 2)]
    outcome = compare_test.targeted_verdict(frozen_only, 3)
    assert outcome["verdict"] == "frozen-suite only, not a general improvement"
    assert "matched the benchmark" in outcome["detail"]

    neither = [summary("frozen", "C-B", 2), summary("heldout", "C-B", 1)]
    assert compare_test.targeted_verdict(neither, 3)["verdict"] == \
        "no advantage for the targeted weights"

    held_only = [summary("frozen", "C-B", 1), summary("heldout", "C-B", 3)]
    assert compare_test.targeted_verdict(held_only, 3)["verdict"] == \
        "frozen-suite only, not a general improvement"

    incomplete = [summary("frozen", "C-B", 3)]
    assert compare_test.targeted_verdict(incomplete, 3)["verdict"] == \
        "incomplete"

    # A win at two of three seeds is not a win, however large the mean.
    strong = [summary("frozen", "C-B", 2), summary("heldout", "C-B", 2)]
    strong[0]["mean_difference"] = -0.5
    assert compare_test.targeted_verdict(strong, 3)["verdict"] == \
        "no advantage for the targeted weights"
    print("  C is preferred only when it is lower at every seed on both "
          "suites; a frozen-only or held-out-only win is labelled as such, a "
          "missing suite is incomplete, and a large mean with two of three "
          "seeds is still no advantage")

    assert compare_test._direction(3, 0, 3) == \
        "consistent: left lower at every seed"
    assert "seed-sensitive" in compare_test._direction(2, 1, 3)
    print("  the across-seed direction label is the imported one and reports "
          "two of three seeds as inconclusive")
    return 9


def check_ground_truth_structure():
    """The official annotation file, read for its structure and nothing else."""
    parsed = parse_test_ground_truth()
    assert len(parsed) == EXPECTED_TEST_SIZE
    names = [name for name, _ in parsed]
    assert names == [f"{index:05d}.ppm" for index in range(EXPECTED_TEST_SIZE)]
    assert len(set(names)) == EXPECTED_TEST_SIZE
    labels = [label for _, label in parsed]
    assert min(labels) >= 0 and max(labels) < NUM_CLASSES
    assert len(GROUND_TRUTH_FIELDS) == 8
    print(f"  GT-final_test.csv parses to {len(parsed)} rows with the "
          f"{len(GROUND_TRUTH_FIELDS)} declared fields, filenames exactly "
          "00000.ppm to 12629.ppm, and labels inside the class range; no "
          "distribution of them is read or printed here")

    # The agreement and filename checks, against a stand-in whose `_samples`
    # has the shape torchvision's does, so no official image is opened.
    class Samples:
        def __init__(self, samples):
            self._samples = samples

    good = Samples([(f"/wherever/{name}", label) for name, label in parsed])
    agreement = evaluate_test.check_label_agreement(good, parsed)
    assert agreement["images"] == EXPECTED_TEST_SIZE
    assert evaluate_test.check_filename_identity(good) == \
        evaluate_test.FILENAME_RULE

    shifted = Samples([(path, (label + 1) % NUM_CLASSES)
                       for path, label in good._samples])
    try:
        evaluate_test.check_label_agreement(shifted, parsed)
    except RuntimeError as error:
        assert "are not established" in str(error), error
    else:
        raise AssertionError("labels disagreeing with the annotation file were "
                             "accepted")

    renamed = Samples([("/wherever/00001.ppm", good._samples[0][1])]
                      + good._samples[1:])
    try:
        evaluate_test.check_filename_identity(renamed)
    except RuntimeError as error:
        assert "not 00000.ppm" in str(error), error
    else:
        raise AssertionError("a dataset whose order does not follow the "
                             "filenames was accepted")
    print("  a shifted label column and a dataset whose order does not follow "
          "the filenames are both refused")
    return 8


class Tripwire:
    """Replaces the official decode path with something that raises.

    Everything here runs on synthetic images, so nothing should reach
    `GTSRB.__getitem__` or `build_test_observations`. Making that fail loudly
    instead of silently succeeding is what turns "this module reads no official
    test image" from a property of the code I wrote into one the run enforces.
    """

    def __init__(self):
        self.calls = 0
        self.original_getitem = GTSRB.__getitem__
        self.original_observations = evaluate_test.build_test_observations

    def __enter__(self):
        def decode(inner_self, index):
            self.calls += 1
            raise AssertionError(
                "verify_test_path decoded an official test image; the checks "
                "run on synthetic data and this path must not be reached")

        def observations(*arguments, **keywords):
            self.calls += 1
            raise AssertionError(
                "verify_test_path built the official test observations; the "
                "checks run on synthetic data and this path must not be "
                "reached")

        GTSRB.__getitem__ = decode
        evaluate_test.build_test_observations = observations
        return self

    def __exit__(self, *details):
        GTSRB.__getitem__ = self.original_getitem
        evaluate_test.build_test_observations = self.original_observations
        return False


def archive_digests_from_torchvision():
    """The three archive MD5s, read out of torchvision's own source.

    Taken from the library that downloaded the data rather than restated here,
    so this check compares the files on disk against the digests the download
    was verified against, not against a number copied into this repository.
    """
    source = inspect.getsource(gtsrb_module)
    found = re.findall(r'md5="([0-9a-f]{32})"', source)
    if len(found) != 3:
        raise RuntimeError(f"torchvision records {len(found)} archive digests, "
                           "expected 3; the dataset module has changed shape")
    return found


def check_archive_integrity():
    """Part 10 check 3: the downloaded archives and the extracted trees.

    Hashing an archive or an image file reads its bytes. Nothing is decoded, no
    label is read, and the counts below are file counts.
    """
    checks = 0
    recorded = archive_digests_from_torchvision()
    stated = re.findall(r"`([0-9a-f]{32})`", PROTOCOL_PATH.read_text())
    if sorted(recorded) != sorted(set(stated) & set(recorded)):
        raise RuntimeError("the frozen protocol does not state the same three "
                           "archive digests torchvision records")
    print(f"  the 3 digests the frozen protocol states are the 3 "
          "torchvision records; neither set is restated in this module")
    checks += 1

    root = evaluate_test.default_data_root() / "gtsrb"
    archives = {
        "GTSRB-Training_fixed.zip": recorded[0],
        "GTSRB_Final_Test_Images.zip": recorded[1],
        "GTSRB_Final_Test_GT.zip": recorded[2],
    }
    present = 0
    for name, expected in archives.items():
        path = root / name
        if not path.exists():
            continue
        found = file_md5(path)
        assert found == expected, f"{name} md5 {found}, expected {expected}"
        present += 1
        checks += 1
    print(f"  {present} of {len(archives)} archives are still on disk and "
          "every one matches its recorded md5")

    train_files = sorted((root / "GTSRB" / "Training").rglob("*.ppm"))
    test_files = sorted((root / "GTSRB" / "Final_Test").rglob("*.ppm"))
    assert len(train_files) == EXPECTED_TRAIN_FILES, len(train_files)
    assert len(test_files) == EXPECTED_TEST_SIZE, len(test_files)
    print(f"  the extracted trees hold {len(train_files)} training and "
          f"{len(test_files)} test .ppm files, the counts the protocol states")
    return checks + 2


def check_byte_overlap():
    """Part 10 check 8: no test file is byte-identical to a training file.

    Reads bytes to hash them and decodes nothing. It establishes the absence of
    duplicate files and nothing more: a second photograph of the same physical
    sign shares no bytes, and the released test metadata carries no track
    identifier, so cross-split sign overlap stays untestable.
    """
    root = evaluate_test.default_data_root() / "gtsrb"
    digests = {}
    for label, pattern in (("train", root / "GTSRB" / "Training"),
                           ("test", root / "GTSRB" / "Final_Test")):
        seen = {}
        for path in sorted(pattern.rglob("*.ppm")):
            seen.setdefault(evaluate_test.file_sha256(path), []).append(path)
        digests[label] = seen
    train, test = digests["train"], digests["test"]
    assert sum(len(paths) for paths in train.values()) == EXPECTED_TRAIN_FILES
    assert sum(len(paths) for paths in test.values()) == EXPECTED_TEST_SIZE

    shared = sorted(set(train) & set(test))
    assert not shared, (f"{len(shared)} digests appear in both splits, first "
                        f"{[train[shared[0]][0].name, test[shared[0]][0].name]}")
    repeated = {label: sorted(key for key, paths in group.items()
                              if len(paths) > 1)
                for label, group in digests.items()}
    print(f"  {EXPECTED_TRAIN_FILES} training and {EXPECTED_TEST_SIZE} test "
          f"files hash to {len(train)} and {len(test)} distinct sha256 "
          "digests, and the intersection of the two sets is empty")
    print(f"  within-split duplicate groups: {len(repeated['train'])} in "
          f"train, {len(repeated['test'])} in test")
    print("  this rules out duplicated files only; a repeated physical sign "
          "shares no bytes and cannot be detected from the released metadata")
    return 4


def check_label_correctness():
    """Part 10 check 4: the labels, against the dataset and against the disk.

    Constructing `GTSRB(split="test")` populates `_samples` from the annotation
    file and decodes nothing; the tripwire on `__getitem__` is live throughout,
    so an accidental read would raise rather than pass.
    """
    parsed = parse_test_ground_truth()
    root = evaluate_test.default_data_root()
    on_disk = sorted(path.name for path in
                     (root / "gtsrb" / "GTSRB" / "Final_Test").rglob("*.ppm"))
    assert on_disk == [name for name, _ in parsed], \
        "the annotation filenames are not exactly the files on disk"
    print(f"  the {len(parsed)} filenames in GT-final_test.csv are exactly the "
          f"{len(on_disk)} .ppm files in Final_Test, in the same order")

    dataset = GTSRB(root=str(root), split="test")
    assert len(dataset) == EXPECTED_TEST_SIZE, len(dataset)
    assert dataset.transform is None and dataset.target_transform is None
    agreement = evaluate_test.check_label_agreement(dataset, parsed)
    assert agreement["images"] == EXPECTED_TEST_SIZE
    assert evaluate_test.check_filename_identity(dataset) == \
        evaluate_test.FILENAME_RULE
    labels = [label for _, label in dataset._samples]
    assert min(labels) >= 0 and max(labels) < NUM_CLASSES
    print("  the real dataset's own _samples agrees with the independent "
          f"parse on all {agreement['images']} (filename, label) pairs, in "
          "order, and every ClassId is inside the class range")
    print("  no image was opened: the dataset was constructed and its index "
          "read, with the tripwire on __getitem__ live")
    return 6


def check_preprocessing_layout():
    """Part 10 check 5, before the run rather than inside it.

    Inspects transform objects. Constructing the loaders does not iterate them,
    and the tripwire would raise if it did.
    """
    stages = list(build_preprocessing().transforms)
    check_stage_layout(stages)
    names = [type(stage).__name__ for stage in stages]
    assert names == ["Resize", "ToTensor", "Normalize"], names
    print(f"  build_preprocessing() is still {' -> '.join(names)}, through "
          "corruptions.check_stage_layout")

    loaders, _ = create_data_loaders(download=False)
    loader_stages = list(loaders["test"].dataset.transform.transforms)
    assert [type(stage).__name__ for stage in loader_stages] == names
    for built, reference in zip(loader_stages, stages):
        assert repr(built) == repr(reference), (repr(built), repr(reference))
    print("  the test dataset create_data_loaders builds carries that exact "
          "pipeline, stage for stage")

    observation, normalize = split_preprocessing()
    observation_stages = list(observation.transforms)
    assert [type(stage).__name__ for stage in observation_stages] == names[:2]
    assert type(normalize).__name__ == "Normalize"
    # `build_preprocessing()` returns a fresh Compose on every call, so the
    # halves are compared by configuration rather than by object identity.
    assert (normalize.mean, normalize.std) == (stages[-1].mean, stages[-1].std)
    assert repr(normalize) == repr(stages[-1])
    assert not any(isinstance(stage, type(normalize))
                   for stage in observation_stages)
    print("  the observation pipeline the test evaluator builds on carries the "
          "first two stages and not the normalisation, whose mean and standard "
          "deviation are the clean pipeline's third stage's")
    return 5


def independent_heldout_seed(namespace, corruption, severity, sample_index,
                             base_seed=SEED):
    """`verify_corruptions.independent_seed`, for the held-out table.

    That function keys its severity-in-seed decision off a dict covering the
    six frozen families only. The payload is the same one, written out here by
    hand the same way, with the decision read from the held-out condition's own
    specification rather than from the seeding code.
    """
    spec, _ = heldout_lookup(corruption, severity)
    token = str(severity) if spec.severity_in_seed else "-"
    payload = f"{base_seed}|{namespace}|{corruption}|{token}|{sample_index}"
    return int.from_bytes(hashlib.blake2b(payload.encode(),
                                          digest_size=8).digest(), "big") >> 1


def check_seed_derivation(indices=(0, 1, 7, 12629)):
    """Part 10 check 7: the realisation seeds, re-derived from the rule.

    The two `independent_*` derivations write the documented payload out by
    hand instead of calling the seeding code, so agreement here compares the
    implementation against the specification rather than against itself.
    """
    checks = 0
    conditions = [(name, severity) for suite in TEST_SUITES.values()
                  for name, severity in suite.conditions()
                  if name != CLEAN_CONDITION]
    assert len(conditions) == 55, len(conditions)
    for corruption, severity in conditions:
        heldout = corruption in HELDOUT_CORRUPTIONS
        derive = heldout_sample_seed if heldout else sample_seed
        independent = (independent_heldout_seed if heldout
                       else independent_seed)
        for index in indices:
            expected = independent(TEST_NAMESPACE, corruption, severity, index)
            assert derive(TEST_NAMESPACE, corruption, severity, index) == \
                expected, (corruption, severity, index)
            checks += 1
    print(f"  {checks} test seeds over {len(conditions)} corrupted conditions "
          f"and {len(indices)} image indices equal the seed re-derived from "
          "the documented payload by hand")

    for corruption, severity in conditions:
        derive = (heldout_sample_seed if corruption in HELDOUT_CORRUPTIONS
                  else sample_seed)
        for index in indices:
            test = derive(TEST_NAMESPACE, corruption, severity, index)
            validation = derive(VALIDATION_NAMESPACE, corruption, severity,
                                index)
            assert test != validation, (corruption, severity, index)
            for epoch in (1, 15, 30):
                assert test != augmentation_seed(epoch, index), \
                    (corruption, severity, index, epoch)
            checks += 4
    print(f"  every one of them differs from the validation seed at the same "
          "condition and index, and from the training-augmentation seed at "
          "that index in epochs 1, 15 and 30")
    return checks


def snapshot_artifacts(trees=None, checkpoints=None):
    """The pre-run halves of Part 10 checks 9 and 11.

    One digest per file under the committed result trees, and a digest and file
    mode per approved checkpoint. Taken before the test evaluation and
    re-verified after it, because the checkpoint files are mode 0644 rather
    than read-only, so it is the snapshot and not the filesystem that
    establishes nothing was modified.
    """
    trees = tuple(trees if trees is not None else SNAPSHOT_TREES)
    if checkpoints is None:
        inventory = json.loads(INVENTORY_PATH.read_text())
        checkpoints = [PROJECT_ROOT / record["checkpoint_path"]
                       for record in inventory["models"]]
    files = {}
    for tree in trees:
        for path in sorted(Path(tree).rglob("*")):
            if path.is_file():
                files[evaluate_test._repository_path(path)] = \
                    evaluate_test.file_sha256(path)
    recorded = {}
    for path in checkpoints:
        recorded[evaluate_test._repository_path(path)] = {
            "sha256": evaluate_test.file_sha256(path),
            "md5": file_md5(path),
            "mode": oct(path.stat().st_mode & 0o777),
        }
    return {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "trees": [evaluate_test._repository_path(Path(tree)) for tree in trees],
        "files": files,
        "checkpoints": recorded,
    }


def verify_artifact_snapshot(snapshot):
    """Re-verify a snapshot. Returns the differences, so the caller reports."""
    changed, missing = [], []
    for name, expected in snapshot["files"].items():
        path = PROJECT_ROOT / name
        if not path.exists():
            missing.append(name)
        elif evaluate_test.file_sha256(path) != expected:
            changed.append(name)
    for name, expected in snapshot["checkpoints"].items():
        path = PROJECT_ROOT / name
        if not path.exists():
            missing.append(name)
            continue
        if evaluate_test.file_sha256(path) != expected["sha256"]:
            changed.append(name)
        elif oct(path.stat().st_mode & 0o777) != expected["mode"]:
            changed.append(f"{name} (mode)")
    return {"files": len(snapshot["files"]),
            "checkpoints": len(snapshot["checkpoints"]),
            "changed": sorted(changed), "missing": sorted(missing)}


def check_snapshot_mechanism(directory, record_to=None):
    """The snapshot round-trips, and a single altered byte is caught."""
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_artifacts()
    assert len(snapshot["checkpoints"]) == 9, len(snapshot["checkpoints"])
    assert snapshot["files"], "the snapshot covers no file"
    modes = sorted({record["mode"]
                    for record in snapshot["checkpoints"].values()})
    print(f"  {len(snapshot['files'])} files under "
          f"{', '.join(snapshot['trees'])} and {len(snapshot['checkpoints'])} "
          f"checkpoints, whose file modes are {modes}")

    result = verify_artifact_snapshot(snapshot)
    assert not result["changed"] and not result["missing"], result
    print("  re-verifying it immediately reports nothing changed and nothing "
          "missing, which is the comparison step 5 of Part 12 runs")

    # A tamper test on copies, so no committed artifact is touched.
    copied = directory / "copy"
    copied.mkdir()
    original = sorted(snapshot["files"])[0]
    (copied / "artifact.csv").write_bytes((PROJECT_ROOT / original).read_bytes())
    checkpoint = sorted(snapshot["checkpoints"])[0]
    (copied / "checkpoint.pt").write_bytes(
        (PROJECT_ROOT / checkpoint).read_bytes())
    local = snapshot_artifacts(trees=[copied], checkpoints=[])
    assert len(local["files"]) == 2, local["files"]
    assert not verify_artifact_snapshot(local)["changed"]
    with open(copied / "artifact.csv", "ab") as handle:
        handle.write(b"\n")
    altered = verify_artifact_snapshot(local)
    assert len(altered["changed"]) == 1, altered
    (copied / "checkpoint.pt").chmod(0o600)
    renamed = snapshot_artifacts(trees=[], checkpoints=[copied / "checkpoint.pt"])
    (copied / "checkpoint.pt").chmod(0o644)
    assert verify_artifact_snapshot(renamed)["changed"] == \
        [evaluate_test._repository_path(copied / "checkpoint.pt") + " (mode)"], \
        verify_artifact_snapshot(renamed)
    print("  on copies of one artifact and one checkpoint, a single appended "
          "byte and a changed file mode are both reported")

    if record_to is not None:
        target = Path(record_to)
        if target.exists():
            raise RuntimeError(f"{target} already holds a snapshot; the "
                               "pre-run snapshot is written once")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(snapshot, indent=2) + "\n")
        print(f"  recorded to {evaluate_test._repository_path(target)}")
    else:
        print("  not recorded to the repository: --record-snapshot PATH does "
              "that, and Part 12 step 1 is where it belongs")
    return 7


# ------------------------------------------------------------------ #
# Execution readiness: what the console says, what a failure leaves   #
# behind, and the eighteen commands.                                  #
# ------------------------------------------------------------------ #

# Tokens that would mean a performance number had reached the console. Checked
# against the source of every print in the evaluator and the wrapper, and
# against the captured output of a complete synthetic run.
METRIC_TOKENS = ("accuracy", "macro_f1", "macro F1", "error_rate", "f1_score",
                 "loss", "_drop", "recall", "precision")

# Names whose indexed assignment is the module's own bookkeeping. Anything
# else assigned by index or by attribute inside the evaluator would be a
# mutation of something it was given, the frozen severity tables included.
MUTABLE_LOCALS = ("counters", "results", "fingerprints", "measurement",
                  "metadata", "entry", "bound")


def print_calls(module_path, function_names=None):
    """Every `print(...)` in a module, as source, optionally by function."""
    tree = ast.parse(Path(module_path).read_text())
    scopes = []
    if function_names is None:
        scopes = [tree]
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in function_names:
                scopes.append(node)
        found = {node.name for node in scopes}
        assert found == set(function_names), f"{found} != {function_names}"
    calls = []
    for scope in scopes:
        for node in ast.walk(scope):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                calls.append((node.lineno, ast.unparse(node)))
    return calls


def check_metric_suppression():
    """No performance number reaches the console while the sequence is running.

    Three independent readings of the same property: the evaluator and the
    wrapper contain no print that could render a metric, the progress formatter
    is not given one to render, and the captured output of a complete run on
    synthetic data contains none. The third is the one that would catch a leak
    the first two missed, and it is in `check_failure_handling`, which is the
    section that runs the evaluator end to end.
    """
    checks = 0
    for module in (evaluate_test.__file__, run_final_test.__file__):
        calls = print_calls(module)
        assert calls, f"no print found in {module}"
        for line, source in calls:
            for token in METRIC_TOKENS:
                assert token not in source, \
                    f"{Path(module).name}:{line} prints {token}: {source[:90]}"
            checks += 1
    print(f"  {checks} print statements across the evaluator and the "
          "eighteen-run wrapper, and not one of them can render an accuracy, "
          "a macro F1, an error rate, a loss or a drop")

    signature = inspect.signature(evaluate_test.condition_progress)
    assert list(signature.parameters) == [
        "position", "total", "corruption", "severity", "images",
        "condition_seconds", "elapsed"], signature
    line = evaluate_test.condition_progress(7, 31, "gaussian_noise", 3, 12630,
                                            118.33, 830.1)
    for token in METRIC_TOKENS:
        assert token not in line, line
    assert "[ 7/31]" in line and "gaussian_noise" in line and "s3" in line
    assert "12630 images" in line and "elapsed" in line and "ETA" in line
    print("  the progress line is given the condition, the image count and "
          "the clock, and has no parameter through which a measurement could "
          f"reach it: {line.strip()}")
    checks += 4

    # Nothing in the evaluator mutates what it was handed, so no run can adjust
    # a severity table, a condition list or a checkpoint on its way through.
    tree = ast.parse(Path(evaluate_test.__file__).read_text())
    mutations = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute):
                mutations.append((node.lineno, ast.unparse(node)))
            elif isinstance(target, ast.Subscript):
                root = target
                while isinstance(root, ast.Subscript):
                    root = root.value
                if not (isinstance(root, ast.Name)
                        and root.id in MUTABLE_LOCALS):
                    mutations.append((node.lineno, ast.unparse(node)))
    assert not mutations, f"the evaluator mutates state: {mutations}"
    source = Path(evaluate_test.__file__).read_text()
    for token in ("torch.save", "shutil", "os.rename", "os.replace",
                  "rmtree", "write_bytes"):
        assert token not in source, token
    assert source.count("unlink()") == 1 and "marker.unlink()" in source
    print("  it assigns by index only into its own counters, results and "
          "metadata, never into an attribute of anything it was given, and it "
          "contains no checkpoint write, copy, rename or delete: a run cannot "
          "select, replace or retouch a checkpoint or a severity table")
    checks += 2

    # No exit from the sweep except the end of the condition list or a raise.
    evaluate_node = next(node for node in ast.walk(tree)
                         if isinstance(node, ast.FunctionDef)
                         and node.name == "evaluate")
    loops = [node for node in ast.walk(evaluate_node)
             if isinstance(node, ast.For)
             and "conditions" in ast.unparse(node.iter)]
    assert len(loops) == 1, f"{len(loops)} condition loops in evaluate"
    exits = [ast.unparse(node) for node in ast.walk(loops[0])
             if isinstance(node, (ast.Break, ast.Continue, ast.Return))]
    assert not exits, f"the condition loop can exit early: {exits}"
    raises = [node for node in ast.walk(loops[0]) if isinstance(node, ast.Raise)]
    assert len(raises) == 3, f"{len(raises)} raises in the condition loop"
    print(f"  the condition loop has no break, continue or return and runs "
          f"every condition to the end; its {len(raises)} exits are raises on "
          "a wrong image count, a wrong label order, enabled gradients and a "
          "metric that disagrees with its own cross-check, none of them a "
          "comparison between conditions or between models")
    checks += 3

    # The wrapper reads no measurement, so it cannot act on one either.
    wrapper = Path(run_final_test.__file__).read_text()
    for token in ("summary.csv", "summary.json", "predictions.csv",
                  "per_class_metrics.csv", "confusion_matrices.npz"):
        assert token not in wrapper, f"the wrapper reads {token}"
    assert "status" in wrapper and "completed" in wrapper
    print("  the wrapper opens no result file of any kind: the only thing it "
          "reads back between runs is the access log's completed lines, which "
          "carry counts and identities and no measurement")
    return checks + 2

class SyntheticTestSplit(torch.utils.data.Dataset):
    """A stand-in for the official test split, with its interface.

    `_samples` in the shape torchvision builds it, contiguous `NNNNN.ppm`
    filenames so the filename-identity assertion has something to assert, and
    pixels from `SyntheticObservations`. `transform` is carried because the
    preprocessing-layout check reads it; the items are already observations, so
    nothing applies it, which is exactly what the real observation dataset does
    once its transform has run.
    """

    def __init__(self, count, transform=None, normalize=None):
        self.count = count
        self.transform = transform
        self.normalize = normalize
        self.images = SyntheticObservations(count)
        self._samples = [(f"synthetic/Final_Test/Images/{index:05d}.ppm",
                          index % NUM_CLASSES) for index in range(count)]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        image, label = self.images[index]
        if self.normalize is not None:
            image = self.normalize(image)
        return image, label


class FailingNumpy:
    """`numpy`, with one function replaced, bound into the evaluator only.

    Used to fail the confusion-matrix write part-way through the artifact set
    without touching the real module, so the writer's own partial-write
    handling is what gets tested rather than a stand-in for it.
    """

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def savez_compressed(self, *arguments, **keywords):
        raise RuntimeError("synthetic failure: the confusion matrices could "
                           "not be written")


STAND_IN_CHECKPOINT = "synthetic/not_a_real_checkpoint.pt"

# The six stages a run can fail at, and the text each injected failure
# carries, so the access-log line and the traceback can be matched to the
# stage that produced them.
FAILURE_STAGES = {
    "dataset": "synthetic failure: the official test split could not be built",
    "checkpoint": "synthetic failure: the checkpoint digest does not match",
    "corruption": "synthetic failure: the corruption could not be realised",
    "inference": "synthetic failure: the forward pass did not run",
    "writing": "synthetic failure: the confusion matrices could not be written",
    "log": "synthetic failure: the access log could not be appended to",
}


@contextlib.contextmanager
def stubbed_evaluator(access_log, size=SYNTHETIC_IMAGES, failure=None):
    """The evaluator with its split, its checkpoint and its log replaced.

    Everything that would read an official image, load a real checkpoint or
    write the repository's access log is swapped for a synthetic equivalent of
    the same shape, and the split size is reduced so the derived read budget is
    reachable in seconds. `failure`, if given, additionally breaks one stage.

    The swap is a module-attribute swap, restored in `finally` and asserted
    restored by the caller, which is why the real constants are read back at
    the end of the section rather than trusted.
    """
    observation, normalize = split_preprocessing()
    dataset = SyntheticTestSplit(size, transform=observation)
    labels = np.array([label for _, label in dataset._samples])
    indices = np.arange(size)
    loader_split = SyntheticTestSplit(size, transform=build_preprocessing(),
                                      normalize=normalize)
    unused = SyntheticTestSplit(4, transform=build_preprocessing(),
                                normalize=normalize)

    def loaders(batch_size=64, download=False, **keywords):
        return {name: DataLoader(split, batch_size=batch_size, shuffle=False)
                for name, split in (("train", unused),
                                    ("validation", unused),
                                    ("test", loader_split))}, {}

    record = {"arm": "A", "arm_label": "clean", "seed": 42,
              "selected_epoch": 21, "checkpoint_path": STAND_IN_CHECKPOINT,
              "checkpoint_md5": "0" * 32, "checkpoint_sha256": "0" * 64}
    identity = {"path": STAND_IN_CHECKPOINT, "seed": 42, "epoch": 21,
                "selection_metric": "synthetic stand-in"}
    model = FixedLogits()

    inference_calls = []
    original_dataset_method = evaluate_test.TestSuite.dataset

    def condition_dataset(self, observations, corruption, severity):
        if failure == "corruption" and corruption != CLEAN_CONDITION:
            raise RuntimeError(FAILURE_STAGES["corruption"])
        return original_dataset_method(self, observations, corruption, severity)

    def inference(*arguments, **keywords):
        inference_calls.append(1)
        if failure == "inference" and len(inference_calls) == 3:
            raise RuntimeError(FAILURE_STAGES["inference"])
        return evaluate_condition(*arguments, **keywords)

    def digests(path, inventory):
        if failure == "checkpoint":
            raise RuntimeError(FAILURE_STAGES["checkpoint"])
        return {"md5": record["checkpoint_md5"],
                "sha256": record["checkpoint_sha256"]}

    def observations_of_the_split(*arguments, **keywords):
        if failure == "dataset":
            raise RuntimeError(FAILURE_STAGES["dataset"])
        return dataset, indices, labels

    def log_append(entry, path=None):
        if failure == "log":
            raise RuntimeError(FAILURE_STAGES["log"])
        return append_access_log(entry, path)

    replacements = {
        "EXPECTED_TEST_SIZE": size,
        "PROTOCOL_READ_COUNTS": {"frozen": None, "heldout": None},
        "ACCESS_LOG": Path(access_log),
        "build_test_observations": observations_of_the_split,
        "parse_test_ground_truth":
            lambda *a, **k: [(Path(path).name, label)
                             for path, label in dataset._samples],
        "create_data_loaders": loaders,
        "inventory_record": lambda path: record,
        "check_inventory_digests": digests,
        "check_identity": lambda path, device: (model, {}, identity),
        "discover_run_metadata": lambda path: {},
        "check_provenance": lambda path, identity, metadata: {
            "verified": ["synthetic stand-in"], "unverifiable": []},
        "check_validation_binding": lambda record: {
            "validation_metadata": {"frozen": "synthetic stand-in"},
            "established_by": "a synthetic stand-in, not a committed run"},
        "evaluate_condition": inference,
        "append_access_log": log_append,
    }
    if failure == "writing":
        replacements["np"] = FailingNumpy(np)
    saved = {name: getattr(evaluate_test, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(evaluate_test, name, value)
        evaluate_test.TestSuite.dataset = condition_dataset
        yield {"dataset": dataset, "labels": labels, "size": size}
    finally:
        for name, value in saved.items():
            setattr(evaluate_test, name, value)
        evaluate_test.TestSuite.dataset = original_dataset_method


def run_stubbed_evaluation(workspace, name, failure=None,
                           size=SYNTHETIC_IMAGES, output_root=None):
    """One `evaluate_test.main` on synthetic data, with its output captured."""
    root = Path(output_root) if output_root else workspace / name
    target = root / "clean_seed42" / "frozen"
    log = workspace / name / "test_access_log.jsonl"
    argv = ["--checkpoint", STAND_IN_CHECKPOINT, "--suite", "frozen",
            "--model-name", "clean", "--output-dir", str(root),
            "--device", "cpu", "--batch-size", "32"]
    console = io.StringIO()
    outcome = {"target": target, "log": log, "entry": None, "error": None}
    with stubbed_evaluator(log, size=size, failure=failure):
        with contextlib.redirect_stdout(console):
            try:
                outcome["entry"] = evaluate_test.main(argv)
            except BaseException as error:
                outcome["error"] = error
    outcome["console"] = console.getvalue()
    outcome["lines"] = [json.loads(line) for line
                        in log.read_text().splitlines() if line.strip()] \
        if log.exists() else []
    return outcome


def check_failure_handling(workspace):
    """A run that completes, a run that is refused, and six that break.

    All seven go through `evaluate_test.main` itself, on a synthetic split of
    96 images with a synthetic checkpoint and a temporary access log, so the
    control flow under test is the one the eighteen runs will take rather than
    a description of it.
    """
    checks = 0
    budget = (SYNTHETIC_IMAGES + len(range(0, SYNTHETIC_IMAGES,
                                           FINGERPRINT_STRIDE))) * 31

    done = run_stubbed_evaluation(workspace, "completed")
    assert done["error"] is None, done["error"]
    assert done["entry"]["status"] == "completed"
    assert len(done["lines"]) == 1 and done["lines"][0]["status"] == "completed"
    assert done["lines"][0]["test_images_read"] == budget, done["lines"][0]
    assert done["lines"][0]["conditions"] == 31
    assert done["lines"][0]["test_loader_reads"] == SYNTHETIC_IMAGES
    assert done["lines"][0]["train_loader_reads"] == 0
    assert done["lines"][0]["validation_loader_reads"] == 0
    written = sorted(path.name for path in done["target"].iterdir())
    assert written == sorted(evaluate_test.RESULT_FILES), written
    assert evaluate_test.INCOMPLETE_MARKER not in written
    print(f"  a complete run on {SYNTHETIC_IMAGES} synthetic images wrote all "
          f"{len(written)} artifacts, left no partial-write marker, and "
          f"appended exactly one completed line recording {budget} reads, the "
          "derived budget for 31 conditions at this split size")
    checks += 10

    for token in METRIC_TOKENS:
        assert token not in done["console"], \
            f"the console of a complete run mentions {token}"
    assert "ETA" in done["console"] and "[31/31]" in done["console"]
    summary = (done["target"] / "summary.csv").read_text()
    for column in ("accuracy", "macro_f1", "error_rate", "accuracy_drop"):
        assert column in summary.splitlines()[0], column
    print("  its console output carries the progress lines, the ETA and the "
          "integrity statements and not one accuracy, macro F1, error rate, "
          "loss or drop, every one of which is in the summary it wrote")
    checks += 3

    refused = run_stubbed_evaluation(workspace, "refused",
                                     output_root=workspace / "completed")
    assert refused["error"] is not None
    assert "already holds results" in str(refused["error"]), refused["error"]
    assert len(refused["lines"]) == 1
    assert refused["lines"][0]["status"] == "failed"
    assert refused["lines"][0]["test_images_read"] == 0
    assert sorted(path.name for path in done["target"].iterdir()) == written
    print("  a second run at the same directory is refused before the split "
          "is touched: the completed set is unchanged, and the attempt is "
          "logged as failed with 0 test images read")
    checks += 5

    nested = run_stubbed_evaluation(workspace, "nested",
                                    output_root=done["target"])
    assert nested["error"] is not None
    assert "already ends in" in str(nested["error"]), nested["error"]
    assert not any(entry["status"] == "completed" for entry in nested["lines"])
    assert sorted(path.name for path in done["target"].iterdir()) == written
    print("  an --output-dir that is itself an arm, seed and suite directory "
          "is refused rather than nested inside: the root is where those "
          "directories are created, and a run that wrote one level deeper "
          "would leave the directory the sequence names apparently empty")
    checks += 4

    expectations = {
        "dataset": {"reads": 0, "target": "absent"},
        "checkpoint": {"reads": 0, "target": "absent"},
        "corruption": {"reads": "some", "target": "absent"},
        "inference": {"reads": "some", "target": "absent"},
        "writing": {"reads": budget, "target": "partial"},
        "log": {"reads": budget, "target": "complete"},
    }
    for stage, expectation in expectations.items():
        outcome = run_stubbed_evaluation(workspace, stage, failure=stage)
        assert outcome["error"] is not None, f"{stage} did not fail"
        assert outcome["entry"] is None
        text = str(outcome["error"]) + str(outcome["error"].__context__ or "")
        assert FAILURE_STAGES[stage] in text, (stage, text)

        if stage == "log":
            assert not outcome["log"].exists()
            assert '"status": "failed"' in outcome["console"]
            assert FAILURE_STAGES[stage] in outcome["console"]
            printed = [line for line in outcome["console"].splitlines()
                       if line.startswith('{"timestamp"')]
            assert len(printed) == 1, printed
            recovered = json.loads(printed[0])
            assert recovered["status"] == "failed"
            assert recovered["test_images_read"] == budget, recovered
            assert recovered["output_dir"]
        else:
            assert len(outcome["lines"]) == 1, (stage, outcome["lines"])
            line = outcome["lines"][0]
            assert line["status"] == "failed", stage
            assert FAILURE_STAGES[stage] in line["error"], stage
            assert line["checkpoint"] == STAND_IN_CHECKPOINT, stage
            assert line["suite"] == "frozen", stage
            if expectation["reads"] == "some":
                assert 0 < line["test_images_read"] < budget, (stage, line)
            else:
                assert line["test_images_read"] == expectation["reads"], \
                    (stage, line)
        assert not any(entry["status"] == "completed"
                       for entry in outcome["lines"]), stage

        if expectation["target"] == "absent":
            assert not outcome["target"].exists(), stage
        else:
            present = sorted(path.name for path in outcome["target"].iterdir())
            if expectation["target"] == "partial":
                assert evaluate_test.INCOMPLETE_MARKER in present, present
                assert set(present) < set(written) | {
                    evaluate_test.INCOMPLETE_MARKER}, present
                try:
                    check_output_target(outcome["target"])
                except RuntimeError as error:
                    assert "already holds results" in str(error), error
                else:
                    raise AssertionError("a partial result set was accepted "
                                         "as an output directory")
            else:
                assert present == written, (stage, present)
                assert evaluate_test.INCOMPLETE_MARKER not in present
        checks += 6

    print("  six injected failures - dataset setup, checkpoint validation, "
          "corruption generation, inference, output writing and the access "
          "log itself - each stopped the run, re-raised rather than swallowed "
          "the error, and recorded one failed line and no completed one")
    print("  the two that failed before any image was read logged 0 reads, "
          "the two that failed mid-sweep logged what they had read by then, "
          "and the two that failed after it logged the full budget")
    print(f"  the one that failed part-way through writing left its directory "
          f"holding the {evaluate_test.INCOMPLETE_MARKER} marker and a subset "
          "of the artifacts, which the output guard then refuses as a target, "
          "so a partial set cannot be completed or overwritten in place")
    print("  the one whose log could not be written printed the line it could "
          "not append, so the reads it records survive the failure, and "
          "re-raised the original error with the log failure chained to it")

    assert evaluate_test.EXPECTED_TEST_SIZE == EXPECTED_TEST_SIZE
    assert evaluate_test.PROTOCOL_READ_COUNTS == PROTOCOL_READ_COUNTS
    assert evaluate_test.ACCESS_LOG == TEST_RESULTS_ROOT / \
        "test_access_log.jsonl"
    assert evaluate_test.evaluate_condition is evaluate_condition
    assert evaluate_test.append_access_log is append_access_log
    assert evaluate_test.np is np
    assert not TEST_RESULTS_ROOT.exists()
    print("  the module's real split size, read budget, access-log path, "
          "inference function, log writer and numpy binding are all back, and "
          f"{evaluate_test._repository_path(TEST_RESULTS_ROOT)} was never "
          "created")
    return checks + 7

def check_execution_plan(workspace):
    """The eighteen commands, field by field, against the inventory itself."""
    inventory = json.loads(INVENTORY_PATH.read_text())["models"]
    expected = []
    for record in inventory:
        for suite in ("frozen", "heldout"):
            expected.append((f"{record['arm']}{record['seed']}", suite,
                             record["checkpoint_path"], record["arm_label"],
                             int(record["seed"]), record["checkpoint_md5"]))
    assert len(expected) == 18 and len({row[:2] for row in expected}) == 18

    runs = run_final_test.plan_runs("cpu")
    assert [(run["model"], run["suite"], run["checkpoint"], run["arm_label"],
             run["seed"], run["checkpoint_md5"]) for run in runs] == expected
    checks = 2

    for run, row in zip(runs, expected):
        _, suite, checkpoint, arm, seed, digest = row
        command = run["command"]
        assert command[1:3] == ["-m", "src.evaluate_test"], command
        flags = dict(zip(command[3::2], command[4::2]))
        assert flags == {
            "--checkpoint": checkpoint,
            "--suite": suite,
            "--model-name": arm,
            "--device": "cpu",
            "--num-workers": "0",
        }, flags
        assert run["output_dir"] == results_dir(arm, seed, suite)
        assert TEST_RESULTS_ROOT in Path(run["output_dir"]).parents
        assert file_md5(Path(checkpoint)) == digest, checkpoint
        arguments = evaluate_test.parse_arguments(command[3:])
        assert arguments.checkpoint == checkpoint
        assert arguments.suite == suite
        assert arguments.model_name == arm
        assert arguments.num_workers == 0 and arguments.device == "cpu"
        assert not arguments.smoke_test
        assert results_dir(arguments.model_name, seed, arguments.suite,
                           arguments.output_dir) == Path(run["output_dir"])
        checks += 8
    print(f"  {len(runs)} commands in the inventory's own row order, frozen "
          "suite before held-out: every checkpoint path, arm label, seed, "
          "suite, device and worker count equals the inventory's, every "
          "checkpoint md5 was recomputed and matched, and the evaluator's own "
          "parser turns each command into the directory its row names")

    summary = run_final_test.check_plan(runs, verify_digests=False)
    assert summary == {"runs": 18, "models": 9,
                       "suites": ["frozen", "heldout"], "device": "cpu",
                       "digests_verified": False}, summary
    checks += 1

    occupied = workspace / "occupied"
    planned = run_final_test.plan_runs("cpu", occupied, workspace / "logs")
    taken = Path(planned[7]["output_dir"])
    taken.mkdir(parents=True)
    (taken / "summary.csv").write_text("occupied\n")
    damage = [
        ("an output directory that already holds a result", planned,
         "already holds results"),
    ]
    for arm_label in ("targeted",):
        wrong_arm = run_final_test.plan_runs("cpu")
        wrong_arm[0]["arm_label"] = arm_label
        damage.append(("an arm label that would misfile a result", wrong_arm,
                       "would file"))
    wrong_checkpoint = run_final_test.plan_runs("cpu")
    wrong_checkpoint[2]["checkpoint"] = "artifacts/clean_seed44/best_checkpoint.pt"
    damage.append(("a checkpoint that is not the model's own",
                   wrong_checkpoint, "would read"))
    repeated = run_final_test.plan_runs("cpu")
    repeated[3] = dict(repeated[2])
    damage.append(("the same model and suite planned twice", repeated,
                   "planned twice"))
    workers = run_final_test.plan_runs("cpu")
    workers[0]["num_workers"] = 2
    damage.append(("a worker count above zero", workers, "read accounting"))
    devices = run_final_test.plan_runs("cpu")
    devices[9] = run_final_test.plan_runs("mps")[9]
    damage.append(("two devices in one sequence", devices, "devices"))
    mismatched = run_final_test.plan_runs("cpu")
    mismatched[0]["command"] = mismatched[0]["command"] + [
        "--output-dir", str(mismatched[0]["output_dir"])]
    damage.append(("a command that would write somewhere other than the "
                   "directory its row names", mismatched, "would write"))
    short = run_final_test.plan_runs("cpu")[:17]
    damage.append(("a sequence of the wrong length", short, "expected 18"))
    missing = run_final_test.plan_runs("cpu")
    missing[5]["model"] = "A45"
    damage.append(("a model outside the inventory", missing,
                   "not in the checkpoint inventory"))

    for label, plan, message in damage:
        try:
            run_final_test.check_plan(plan, verify_digests=False)
        except RuntimeError as error:
            assert message in str(error), (label, error)
        else:
            raise AssertionError(f"the plan check accepted {label}")
        checks += 1
    print(f"  the plan check refuses all {len(damage)} of: "
          + "; ".join(label for label, _, _ in damage))

    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        code = run_final_test.main(["--device", "cpu", "--skip-digests"])
    assert code == 0
    text = console.getvalue()
    assert "Plan only" in text and "18 runs" in text
    assert text.count("--checkpoint") == 18
    for token in METRIC_TOKENS:
        assert token not in text, f"the plan mentions {token}"
    assert not run_final_test.LOG_DIR.exists()
    assert not TEST_RESULTS_ROOT.exists()
    print("  the default invocation prints the plan, validates it, creates "
          "neither the results root nor the log directory, and carries no "
          "measurement; --execute is what runs anything")
    checks += 7

    # A failure stops the sequence where it happened.
    plan = run_final_test.plan_runs("cpu", workspace / "sequence",
                                    workspace / "sequence_logs")
    log = workspace / "sequence_access_log.jsonl"
    attempted = []

    def stop_at_five(run):
        attempted.append(run["position"])
        if run["position"] == 5:
            return {"returncode": 1, "seconds": 0.1}
        append_access_log(access_log_entry(
            "completed", arm=run["arm_label"], seed=run["seed"],
            suite=run["suite"]), log)
        return {"returncode": 0, "seconds": 0.1}

    silent = []

    def exit_zero_without_logging(run):
        silent.append(run["position"])
        return {"returncode": 0, "seconds": 0.1}

    saved = run_final_test.run_one
    try:
        run_final_test.run_one = stop_at_five
        with contextlib.redirect_stdout(io.StringIO()) as console:
            outcome = run_final_test.execute(plan, log)
        assert attempted == [1, 2, 3, 4, 5], attempted
        assert outcome == {"completed": 4, "failed": 5, "returncode": 1,
                           "seconds": outcome["seconds"]}, outcome
        assert "stops here with 4 of 18 runs completed" in console.getvalue()
        assert "Nothing proceeds to the next checkpoint" in console.getvalue()

        run_final_test.run_one = exit_zero_without_logging
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                run_final_test.execute(plan, workspace / "empty_log.jsonl")
            except RuntimeError as error:
                assert "gained [] rather than exactly one" in str(error), error
            else:
                raise AssertionError("a run that recorded no read was treated "
                                     "as a completed evaluation")
        assert silent == [1], silent
    finally:
        run_final_test.run_one = saved
    print("  a run that exits non-zero stops the sequence at that position "
          "with the earlier runs kept and nothing attempted after it, and a "
          "run that exits zero without appending a completed line stops it "
          "too rather than being counted")
    return checks + 7


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-snapshot", default=None, metavar="PATH",
                        help="write the pre-run artifact and checkpoint digest "
                             "snapshot that Part 12 step 1 records and step 5 "
                             "re-verifies; refuses an existing file")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    started = time.time()
    observations = SyntheticObservations()
    _, normalize = split_preprocessing()
    sample, label = observations[0]
    assert sample.shape == (3, 64, 64) and sample.dtype == torch.float32
    assert 0.0 <= float(sample.min()) and float(sample.max()) <= 1.0
    assert isinstance(label, int)
    assert normalize(sample).shape == sample.shape

    total = 0
    with Tripwire() as tripwire, tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)

        sections = [
            ("Frozen documents and approved checkpoints",
             lambda: check_protocol_and_inventory()),
            ("Condition coverage", check_condition_coverage),
            ("Corruption determinism",
             lambda: check_corruption_determinism(observations)),
            ("Namespace separation",
             lambda: check_namespace_separation(observations)),
            ("Official annotation structure", check_ground_truth_structure),
            ("Official labels, against the dataset and the disk",
             check_label_correctness),
            ("Dataset integrity", check_archive_integrity),
            ("Byte-level overlap with the training split", check_byte_overlap),
            ("Preprocessing layout", check_preprocessing_layout),
            ("Realisation seeds, re-derived", check_seed_derivation),
            ("Pre-run artifact and checkpoint snapshot",
             lambda: check_snapshot_mechanism(
                 workspace / "snapshot", arguments.record_snapshot)),
            ("Class support", check_class_support),
            ("Output safety", lambda: check_output_safety(workspace / "outputs")),
            ("Worker count", check_worker_rejection),
            ("A failed run's access-log line", check_failed_line_fields),
            ("Access log", lambda: check_access_log(workspace / "log")),
            ("Read accounting", lambda: check_read_accounting(observations)),
            ("Bootstrap reproducibility", check_bootstrap_reproducibility),
            ("Clean-accuracy trade-off", check_tradeoff_logic),
            ("Decision rule", check_decision_rule),
            ("What the console says while a sweep runs",
             check_metric_suppression),
            ("Failure handling, through the evaluator itself",
             lambda: check_failure_handling(workspace / "failures")),
            ("The eighteen-run execution plan",
             lambda: check_execution_plan(workspace / "plan")),
        ]
        for title, check in sections:
            print(f"\n{title}:")
            total += check()

        print("\nConfusion accounting and the artifact set "
              "(synthetic images, stand-in model):")
        suite = TEST_SUITES["frozen"]
        conditions = [(CLEAN_CONDITION, 0), ("gaussian_noise", 3),
                      ("occlusion", 5)]
        results, fingerprints = synthetic_sweep(observations, suite, conditions)
        assert len(set(fingerprints.values())) == len(conditions)
        total += check_confusion_accounting(observations, results, conditions)
        total += check_prediction_schema(observations, suite, results,
                                         conditions, workspace / "artifacts")
        print(f"  the {len(conditions)} synthetic conditions have distinct "
              "input fingerprints")
        total += 1

        print("\nComparison, end to end on synthetic predictions for all nine "
              "models:")
        total += check_comparison_math(workspace / "comparison")

        assert tripwire.calls == 0

    assert not TEST_RESULTS_ROOT.exists(), \
        f"{TEST_RESULTS_ROOT} exists; this module must not create it"
    assert not compare_test.ANALYSIS_DIR.exists(), \
        f"{compare_test.ANALYSIS_DIR} exists; this module must not create it"

    print(f"\nAll {total} checks passed in {time.time() - started:.1f}s. No "
          "official test image was decoded, no official test loader was "
          "iterated, no checkpoint was loaded, no test metric was computed, "
          f"and neither {evaluate_test._repository_path(TEST_RESULTS_ROOT)} nor "
          f"{evaluate_test._repository_path(compare_test.ANALYSIS_DIR)} was "
          "created.")


if __name__ == "__main__":
    main()
