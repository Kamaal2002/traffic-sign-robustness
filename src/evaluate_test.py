"""Evaluate one accepted checkpoint on the official GTSRB test split.

    .venv/bin/python -m src.evaluate_test \
        --checkpoint artifacts/baseline/best_checkpoint.pt --suite frozen
    .venv/bin/python -m src.evaluate_test \
        --checkpoint artifacts/augmented_targeted_seed44/best_checkpoint.pt \
        --suite heldout

The final stage of the study, run under `docs/final_test_protocol.md` after
that document was frozen. Nothing about the measurement changes here: the
conditions, the severity tables, the seeding convention, the normalisation, the
metric paths, the fingerprints and the model-integrity checks are the objects
`evaluate_checkpoint.py` already defines, imported rather than restated, so a
test condition differs from its validation counterpart in exactly two ways -
the split it runs on and the namespace its realisations are seeded from.

Three things this module adds, all of them consequences of the split being the
one the study has held back:

* `final_test` as the corruption namespace, so no test realisation can collide
  with a validation or training-augmentation realisation at the same index;
* enforcement rather than recording of the split accounting. The existing
  evaluator writes the train and test loader counters into its metadata; here
  the run asserts them - train 0 reads, validation 0 reads, test exactly one
  iteration of 12,630 - and asserts the exact number of images the sweep
  itself read, so a pass that touched the wrong split fails instead of
  reporting a number;
* an append-only access log, because how many times the test set has been read
  is part of what the final result means.

The split has no track structure (Part 3 of the protocol), so there are no
physical-sign identifiers to carry into the outputs and the validation
tracks-per-class block is replaced by a class-support block. The fingerprint
stride of 30 is kept for comparability, but it is a 1-in-30 systematic sample
of positions here rather than one frame per sign.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.datasets import GTSRB

from .corruptions import (
    EXPECTED_STAGES,
    SEVERITY_LEVELS,
    CorruptedDataset,
    check_stage_layout,
    split_preprocessing,
)
from .data import NUM_CLASSES, SEED, build_preprocessing, create_data_loaders
from .evaluate_checkpoint import (
    FINGERPRINT_STRIDE,
    FROZEN_SUITE,
    HELDOUT_SUITE,
    SMOKE_CONDITIONS,
    SMOKE_STRIDE,
    Suite,
    _repository_path,
    check_identity,
    check_provenance,
    discover_run_metadata,
    input_fingerprint,
    model_record,
    recorded_policy_name,
    resolve_model_name,
    run_smoke_test,
)
from .evaluate_robustness import (
    BATCH_SIZE,
    CLEAN_CONDITION,
    CLEAN_TOLERANCE,
    METRIC_TOLERANCE,
    PROJECT_ROOT,
    CountingDataset,
    check_target_sequence,
    condition_loader,
    evaluate_condition,
    file_md5,
    independent_metrics,
    instrument_loaders,
    repository_state,
    state_fingerprint,
)
from .train import metrics_from_confusion, per_class_from_confusion, select_device

# The namespace every test realisation is seeded from. Distinct from
# `corruptions.VALIDATION_NAMESPACE` and from the training augmenter's `train`,
# so the same image index under the same family and severity gets a different
# realisation on each split: a model cannot have been shown its test corruption
# during training or development, whatever the index.
TEST_NAMESPACE = "final_test"

# The official test release, from the benchmark's own count.
EXPECTED_TEST_SIZE = 12630

TEST_RESULTS_ROOT = PROJECT_ROOT / "artifacts" / "final_test"
ACCESS_LOG = TEST_RESULTS_ROOT / "test_access_log.jsonl"
INVENTORY_PATH = PROJECT_ROOT / "artifacts" / "robustness" / "checkpoint_inventory.json"
VALIDATION_ROOT = PROJECT_ROOT / "artifacts" / "robustness"

# Trees this module must never be able to write into: the accepted baseline,
# the committed validation evaluations, and the committed validation analysis.
# Checked as paths before the model is built, so no result set can be produced
# on top of one.
PROTECTED_ROOTS = (PROJECT_ROOT / "artifacts" / "baseline",
                   PROJECT_ROOT / "artifacts" / "robustness",
                   PROJECT_ROOT / "artifacts" / "analysis")

# The frozen protocol this run executes, pinned by digest. A run whose protocol
# document has changed is not running the protocol that was frozen, and the
# whole point of freezing it before any test number existed is that the
# procedure cannot be adjusted after seeing one.
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "final_test_protocol.md"
PROTOCOL_SHA256 = "41df8eef10ae5d736d203245bfce1156c0ef57c51ee41e0574ba6a5e8eff07ef"

# The augmentation protocol the nine training runs were frozen under, verified
# through its own manifest so the training recipes are shown unchanged as well.
AUGMENTATION_MANIFEST = PROJECT_ROOT / "docs" / "augmentation_manifest.json"

# Read counts Part 9 of the frozen protocol states for a complete sweep. They
# are not used as the expected value - `expected_reads` below derives that from
# the suite and the stride - but the derivation is required to agree with them,
# so a drift between the implementation and the document stops the run instead
# of being absorbed by either side.
PROTOCOL_READ_COUNTS = {"frozen": 404581, "heldout": 339326}

# The official ground-truth file, read once directly and once through
# torchvision, and required to agree.
GROUND_TRUTH = "GT-final_test.csv"
GROUND_TRUTH_FIELDS = ("Filename", "Width", "Height", "Roi.X1", "Roi.Y1",
                       "Roi.X2", "Roi.Y2", "ClassId")

# What `dataset_index` means in `predictions.csv`, asserted per run and
# recorded in the metadata. It is why there is no filename column.
FILENAME_RULE = ("dataset_index i is the official test image "
                 "f'{i:05d}.ppm', asserted for all 12630 entries")

# Written first and removed last by `write_test_outputs`, so a results
# directory that still holds it is a partial write from a run that stopped
# part-way through writing. `check_output_target` refuses any non-empty
# directory, this one included, which is what forces a partial set to be
# examined and moved by hand instead of being completed or overwritten by a
# later run.
INCOMPLETE_MARKER = "INCOMPLETE"

# Everything one complete result set holds. Required to exist before the marker
# is removed, so "the marker is gone" means "all six files are there".
RESULT_FILES = ("summary.csv", "summary.json", "per_class_metrics.csv",
                "confusion_matrices.npz", "predictions.csv", "metadata.json")

# What the console says while a sweep runs, and what it deliberately does not.
# The 18 runs are executed one after another and the comparison is read only
# once all of them have completed and been verified; a per-condition accuracy
# on the terminal would make it possible to notice a result mid-sequence and
# act on it - stop early, rerun a seed, pick a checkpoint - which is exactly
# the degree of freedom the frozen protocol closes. So the progress lines carry
# the condition, the image count and the clock, the formatter below is given no
# metric to print, and every performance number of the run reaches the artifact
# set without passing through stdout. The only place a metric value appears on
# the console is the message of an exception that aborts the run because the
# metric path itself disagreed with its cross-check, and that run is not a
# result.
PROGRESS_NOTE = ("per-condition metrics are written to the result set, not "
                 "printed: no performance number reaches the console while "
                 "the sequence of runs is still in progress")


class TestSuite(Suite):
    """A validation suite's conditions, realised on the official test split.

    Overrides `dataset` and nothing else. The condition table, the parameter
    lookup, the specification renderer, the description, the notes and the
    condition enumeration are the validation suite's own attributes, so a
    severity value cannot drift between the two splits: there is only one copy
    of it.
    """

    def dataset(self, observations, corruption, severity):
        """Wrap the test observations for one condition, clean included.

        Both branches pass `TEST_NAMESPACE`, which is the whole difference
        between this and `Suite.dataset`. The clean condition still goes
        through `CorruptedDataset` with no corruption so that it is the same
        code path as every corrupted condition and the drops are measured
        against a clean row produced by the same wrapper.
        """
        if corruption == CLEAN_CONDITION:
            return CorruptedDataset(observations, None, namespace=TEST_NAMESPACE)
        return self.dataset_class(observations, corruption, severity,
                                  namespace=TEST_NAMESPACE, base_seed=SEED)


def build_test_suite(suite):
    """Build the test counterpart of one validation suite, sharing its tables."""
    return TestSuite(name=suite.name, table=suite.table,
                     dataset_class=suite.dataset_class,
                     parameter=suite.parameter,
                     specification=suite.specification,
                     description=suite.description, notes=suite.notes)


TEST_SUITES = {suite.name: build_test_suite(suite)
               for suite in (FROZEN_SUITE, HELDOUT_SUITE)}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_data_root():
    return PROJECT_ROOT / "data"


def build_test_observations(data_root=None, download=False):
    """The official test split as `[0, 1]` observations, plus indices and labels.

    The counterpart of `corruptions.build_validation_observations`: the same
    `split_preprocessing()` pipeline with the normalisation stage held back, so
    a corruption is inserted at the same point as on validation and the clean
    condition is bit-identical to `data.py`'s own test pipeline.

    Labels come from the dataset's own `_samples`, which is the route
    `data.read_training_index` already uses for the training split, so they are
    exactly the integers the loader will yield rather than a second opinion
    about them. Reading `_samples` decodes no image.
    """
    if data_root is None:
        data_root = default_data_root()
    observation, _ = split_preprocessing()
    dataset = GTSRB(root=data_root, split="test", transform=observation,
                    download=download)
    if len(dataset) != EXPECTED_TEST_SIZE:
        raise RuntimeError(f"the official test split holds {len(dataset)} "
                           f"images, expected {EXPECTED_TEST_SIZE}")
    labels = np.array([int(label) for _, label in dataset._samples])
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise RuntimeError(f"test labels span [{labels.min()}, {labels.max()}], "
                           f"outside the {NUM_CLASSES} declared classes")
    return dataset, np.arange(EXPECTED_TEST_SIZE), labels


def parse_test_ground_truth(data_root=None):
    """Parse `GT-final_test.csv` directly, without torchvision.

    A second, independent reading of the file the dataset class reads, so the
    labels every test number rests on are established by two code paths that
    agree rather than by one that cannot be checked.
    """
    if data_root is None:
        data_root = default_data_root()
    path = Path(data_root) / "gtsrb" / GROUND_TRUTH
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    if not rows:
        raise RuntimeError(f"{path} is empty")
    header = tuple(field.strip() for field in rows[0])
    if header != GROUND_TRUTH_FIELDS:
        raise RuntimeError(f"{path} has header {header}, expected "
                           f"{GROUND_TRUTH_FIELDS}")
    parsed = []
    for row in rows[1:]:
        if not row:
            continue
        if len(row) != len(GROUND_TRUTH_FIELDS):
            raise RuntimeError(f"{path} row {len(parsed) + 2} has {len(row)} "
                               f"fields, expected {len(GROUND_TRUTH_FIELDS)}")
        record = dict(zip(header, (field.strip() for field in row)))
        label = int(record["ClassId"])
        if not 0 <= label < NUM_CLASSES:
            raise RuntimeError(f"{path} gives {record['Filename']} class "
                               f"{label}, outside [0, {NUM_CLASSES})")
        parsed.append((record["Filename"], label))
    if len(parsed) != EXPECTED_TEST_SIZE:
        raise RuntimeError(f"{path} holds {len(parsed)} rows, expected "
                           f"{EXPECTED_TEST_SIZE}")
    return parsed


def check_label_agreement(dataset, parsed):
    """Require the two readings of the ground truth to agree on every image."""
    samples = dataset._samples
    if len(samples) != len(parsed):
        raise RuntimeError(f"the dataset holds {len(samples)} samples and the "
                           f"ground-truth parse {len(parsed)} rows")
    for position, ((path, label), (filename, expected)) in enumerate(
            zip(samples, parsed)):
        name = Path(path).name
        if name != filename or int(label) != expected:
            raise RuntimeError(
                f"position {position}: the dataset reports ({name}, "
                f"{int(label)}) and GT-final_test.csv ({filename}, "
                f"{expected}); the labels the test numbers rest on are not "
                "established")
    return {"images": len(parsed), "source": GROUND_TRUTH,
            "agreed_on": "filename and class, in order"}


def check_filename_identity(dataset):
    """Require `dataset_index` to be the official filename, for every image.

    `GT-final_test.csv` holds its rows in ascending filename order over the
    contiguous names `00000.ppm` to `12629.ppm`, and torchvision builds
    `_samples` in that row order. So the dataset index is the filename, and
    `predictions.csv` carries no filename column. That is a claim about the
    files on disk, so it is asserted per run rather than assumed.
    """
    for position, (path, _) in enumerate(dataset._samples):
        expected = f"{position:05d}.ppm"
        if Path(path).name != expected:
            raise RuntimeError(
                f"dataset position {position} is {Path(path).name}, not "
                f"{expected}; dataset_index is not the filename and "
                "predictions.csv would not be traceable to an image")
    return FILENAME_RULE


def check_preprocessing(observation_transform, loader_transform):
    """Require the two pipelines to be the frozen one, split at the right stage.

    The clean test row is the suite's own, produced through `CorruptedDataset`
    with no corruption, and it is cross-checked against `data.py`'s loader. That
    cross-check is only meaningful if the observation pipeline is the loader's
    pipeline minus its last stage, so the layout is asserted rather than
    inferred from the two agreeing.
    """
    stages = list(build_preprocessing().transforms)
    check_stage_layout(stages)
    full = list(loader_transform.transforms)
    if [type(stage) for stage in full] != list(EXPECTED_STAGES):
        raise RuntimeError(f"the test loader's pipeline is "
                           f"{[type(stage).__name__ for stage in full]}, not "
                           f"{[stage.__name__ for stage in EXPECTED_STAGES]}")
    observation = list(observation_transform.transforms)
    if [type(stage) for stage in observation] != list(EXPECTED_STAGES[:-1]):
        raise RuntimeError(
            f"the observation pipeline is "
            f"{[type(stage).__name__ for stage in observation]}, expected the "
            f"first {len(EXPECTED_STAGES) - 1} frozen stages and not the "
            f"{EXPECTED_STAGES[-1].__name__}")
    return {
        "loader_pipeline": [type(stage).__name__ for stage in full],
        "observation_pipeline": [type(stage).__name__ for stage in observation],
        "corruption_inserted_before": EXPECTED_STAGES[-1].__name__,
    }


def check_protocol_identity():
    """Require the frozen protocol and the frozen training recipes to be intact."""
    if not PROTOCOL_PATH.exists():
        raise RuntimeError(f"no protocol at {PROTOCOL_PATH}")
    digest = file_sha256(PROTOCOL_PATH)
    if digest != PROTOCOL_SHA256:
        raise RuntimeError(
            f"{_repository_path(PROTOCOL_PATH)} hashes to {digest}, not the "
            f"frozen {PROTOCOL_SHA256}; this run would not be executing the "
            "protocol that was frozen before any test number existed")
    manifest = json.loads(AUGMENTATION_MANIFEST.read_text())
    differs = []
    for name, expected in sorted(manifest["files"].items()):
        actual = file_sha256(PROJECT_ROOT / name)
        if actual != expected:
            differs.append(f"{name}: {actual} against {expected}")
    if differs:
        raise RuntimeError("the augmentation protocol the nine runs were "
                           "trained under has changed:\n  " + "\n  ".join(differs))
    return {
        "final_test_protocol": _repository_path(PROTOCOL_PATH),
        "final_test_protocol_sha256": digest,
        "augmentation_manifest": _repository_path(AUGMENTATION_MANIFEST),
        "augmentation_manifest_sha256": file_sha256(AUGMENTATION_MANIFEST),
        "augmentation_files_verified": sorted(manifest["files"]),
    }


def inventory_record(checkpoint_path):
    """The inventory entry for one checkpoint, keyed on its repository path."""
    inventory = json.loads(INVENTORY_PATH.read_text())
    wanted = _repository_path(checkpoint_path)
    for record in inventory["models"]:
        if record["checkpoint_path"] == wanted:
            return record
    raise RuntimeError(
        f"{wanted} is not in {_repository_path(INVENTORY_PATH)}, which lists "
        f"{len(inventory['models'])} approved checkpoints; only those nine may "
        "be evaluated on the official test split")


def check_inventory_digests(checkpoint_path, record):
    """Require the checkpoint file to be the bytes the inventory recorded."""
    digests = {"md5": file_md5(checkpoint_path),
               "sha256": file_sha256(checkpoint_path)}
    for key, measured in sorted(digests.items()):
        expected = record[f"checkpoint_{key}"]
        if measured != expected:
            raise RuntimeError(
                f"{_repository_path(checkpoint_path)} has {key} {measured} but "
                f"the inventory records {expected}; this is not the checkpoint "
                "the validation results were measured from")
    return digests


def check_validation_binding(record):
    """Tie the checkpoint to its arm and seed through the committed validation runs.

    What has to be established is that the file being evaluated is the one the
    arm and seed it is filed under actually produced. That was established on
    these byte-identical files by the committed validation evaluations, so it is
    read back from their metadata rather than re-measured: no validation
    inference runs in the test stage. JSON only.
    """
    arm, seed = record["arm_label"], record["seed"]
    bound = {}
    for suite_name in sorted(TEST_SUITES):
        path = VALIDATION_ROOT / f"{arm}_seed{seed}" / suite_name / "metadata.json"
        if not path.exists():
            raise RuntimeError(
                f"no committed validation metadata at {_repository_path(path)}, "
                f"so the {arm} seed-{seed} checkpoint is not bound to the arm "
                "and seed this result would be filed under")
        metadata = json.loads(path.read_text())
        checkpoint = metadata.get("checkpoint") or {}
        if checkpoint.get("md5") != record["checkpoint_md5"]:
            raise RuntimeError(
                f"{_repository_path(path)} records checkpoint md5 "
                f"{checkpoint.get('md5')} but the inventory records "
                f"{record['checkpoint_md5']}")
        for block, holder, label in (
                ("clean_reproduction", metadata, "a clean_reproduction block"),
                ("recorded_validation_metrics", checkpoint,
                 "a checkpoint.recorded_validation_metrics block")):
            if not holder.get(block):
                raise RuntimeError(f"{_repository_path(path)} carries no "
                                   f"{label}")
        integrity = metadata.get("integrity") or {}
        if integrity.get("clean_metrics_match_checkpoint_record") is not True:
            raise RuntimeError(
                f"{_repository_path(path)} does not record "
                "integrity.clean_metrics_match_checkpoint_record: true, so the "
                "weights were never bound to the metrics this checkpoint "
                "claims")
        bound[suite_name] = _repository_path(path)
    return {
        "arm": arm,
        "seed": seed,
        "checkpoint_md5": record["checkpoint_md5"],
        "validation_metadata": bound,
        "established_by": ("the committed validation evaluations of these "
                           "byte-identical files; no validation inference was "
                           "rerun"),
    }


def class_support(labels):
    """Per-class image counts over the test split.

    The block that replaces validation's tracks-per-class. The test release
    carries no track identifiers, so the only support a class has here is a
    number of images, and recording it as such keeps any later per-class
    reading honest about what it is counting.
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64),
                         minlength=NUM_CLASSES)
    if counts.size != NUM_CLASSES:
        raise RuntimeError(f"test labels run to class id {counts.size - 1}, "
                           f"outside the {NUM_CLASSES} classes the benchmark "
                           "defines")
    if int(counts.sum()) != EXPECTED_TEST_SIZE:
        raise RuntimeError(f"class supports sum to {int(counts.sum())}, "
                           f"expected {EXPECTED_TEST_SIZE}")
    # Part 3 leaves the number of classes actually present in the test labels
    # to a check that counts distinct ids and nothing else. bincount's own size
    # cannot do it: padded to NUM_CLASSES, it reports a missing class as a zero
    # count, and a class with no test images would contribute a 0 to macro F1
    # under the policy train.metrics_from_confusion documents.
    present = len(set(int(label) for label in np.asarray(labels)))
    if present != NUM_CLASSES:
        raise RuntimeError(f"the test labels carry {present} distinct class "
                           f"ids, not {NUM_CLASSES}; a class with no test "
                           "images would be averaged in as a zero")
    return {str(class_id): int(counts[class_id]) for class_id in range(NUM_CLASSES)}


def results_dir(arm, seed, suite_name, output_root=None):
    """Where one model's results for one suite go.

    `output_root` is the root the arm, seed and suite directories are created
    under, not the target itself. A root that already ends in that tail is a
    target path passed by mistake; honouring it would nest the artifacts one
    level deeper than every other run and leave the directory the operator
    named holding a directory instead of a result set, so it is refused.
    """
    root = Path(output_root) if output_root else TEST_RESULTS_ROOT
    tail = (f"{arm}_seed{seed}", suite_name)
    if output_root and tuple(root.parts[-2:]) == tail:
        raise RuntimeError(
            f"--output-dir {root} already ends in {'/'.join(tail)}. It names "
            "the root that the arm, seed and suite directories are created "
            f"under, so this run would write {root.joinpath(*tail)}")
    return root / f"{arm}_seed{seed}" / suite_name


def check_output_target(path):
    """Refuse any output directory that could damage an accepted artifact.

    Three ways a test run could overwrite something that cannot be regenerated,
    all refused here before the model is built: writing inside a protected tree,
    writing a directory that contains one, and writing into a directory that
    already holds results. The last is what makes a rerun deliberate - a failed
    run leaves its partial directory, and it has to be moved or removed by hand
    rather than silently replaced.
    """
    resolved = Path(path).resolve()
    for protected in PROTECTED_ROOTS:
        protected = Path(protected).resolve()
        if resolved == protected or protected in resolved.parents:
            raise RuntimeError(
                f"{resolved} is inside {protected}, which holds accepted "
                "results; the test evaluation writes under "
                f"{TEST_RESULTS_ROOT} and never into a validation tree")
        if resolved in protected.parents:
            raise RuntimeError(
                f"{resolved} contains the accepted results in {protected}; "
                "a test result set gets its own directory")
    if resolved.exists() and any(resolved.iterdir()):
        raise RuntimeError(
            f"{resolved} already holds results; evaluating into it would "
            "overwrite a test result set. Move or remove it deliberately, or "
            "pass a different --output-dir.")
    return resolved


def append_access_log(entry, path=None):
    """Append one line to the access log, and never rewrite an existing one.

    How many times the official test split has been read is part of what the
    final numbers mean, so the record of it is append-only: opened `"a"`,
    flushed, and fsynced, so an interrupted process cannot leave the log
    describing fewer reads than happened.
    """
    path = Path(path) if path else ACCESS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(entry) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def access_log_entry(status, **fields):
    """One access-log line: what was attempted, and whether it completed.

    `status` is the field the frozen schema does not have and the run needs: a
    failed or interrupted evaluation still read test images, so it is recorded,
    but it is recorded as an attempt rather than as a result. Verification
    counts the `completed` lines.
    """
    entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "module": "src/evaluate_test.py", "status": status}
    repository = repository_state()
    entry["git_head"] = repository["head_commit"]
    entry["git_dirty"] = repository["modified_paths"]
    entry.update(fields)
    return entry


def expected_reads(suite, conditions, fingerprint_positions):
    """How many test images one complete sweep reads, derived from the sweep.

    Two reads per image per condition and no more: the evaluation pass over the
    condition's dataset, and - for one position in every `FINGERPRINT_STRIDE` -
    the fingerprint, which re-reads the dataset so that it describes the inputs
    rather than anything the model did with them. Both go through the
    `CountingDataset` wrapping the observations, so both are counted.

    The figure is derived here and then required to equal the one Part 9 of the
    frozen protocol states, so the two cannot drift apart silently.
    """
    per_condition = EXPECTED_TEST_SIZE + len(fingerprint_positions)
    total = per_condition * len(conditions)
    declared = PROTOCOL_READ_COUNTS.get(suite.name)
    if declared is not None and total != declared:
        raise RuntimeError(
            f"the {suite.name} sweep reads {len(conditions)} x "
            f"({EXPECTED_TEST_SIZE} + {len(fingerprint_positions)}) = {total} "
            f"test images, but Part 9 of the frozen protocol states "
            f"{declared}. The implementation and the protocol disagree; stop "
            "and resolve it rather than changing either to match the other.")
    return {
        "per_condition": per_condition,
        "sweep_reads": EXPECTED_TEST_SIZE,
        "fingerprint_reads": len(fingerprint_positions),
        "conditions": len(conditions),
        "total": total,
        "protocol_total": declared,
    }


def measured_split_counts(loaders):
    """The loader counters as they stand, for the access log.

    Read whenever the log might be written, so a run that fails partway
    records the splits it had touched by then rather than a row of zeros.
    """
    return {
        "test_loader_iterations": loaders["test"].iterations,
        "test_loader_reads": loaders["test"].dataset.reads,
        "train_loader_iterations": loaders["train"].iterations,
        "train_loader_reads": loaders["train"].dataset.reads,
        "validation_loader_iterations": loaders["validation"].iterations,
        "validation_loader_reads": loaders["validation"].dataset.reads,
    }


def check_split_accounting(loaders, reads, expected):
    """Require the measured split counters to be exactly what they should be.

    The existing evaluator records these numbers. Here they are assertions,
    which is what turns "the validation split was not used" from a property of
    the control flow into a measured quantity per run. The instrumentation's
    limits are the ones `CountingDataset` documents: reads through the loaders
    `create_data_loaders` returned on this call, in this process, with zero
    workers.
    """
    measured = dict(measured_split_counts(loaders),
                    sweep_test_images_read=reads)
    required = {
        "test_loader_iterations": 1,
        "test_loader_reads": EXPECTED_TEST_SIZE,
        "train_loader_iterations": 0,
        "train_loader_reads": 0,
        "validation_loader_iterations": 0,
        "validation_loader_reads": 0,
        "sweep_test_images_read": expected,
    }
    wrong = {key: (measured[key], required[key]) for key in required
             if measured[key] != required[key]}
    if wrong:
        raise RuntimeError(
            "the split accounting does not hold: "
            + "; ".join(f"{key} is {got} and must be {want}"
                        for key, (got, want) in sorted(wrong.items()))
            + ". A test evaluation that touched another split, or read the "
              "test split the wrong number of times, is not a result.")
    return measured


def format_duration(seconds):
    """`H:MM:SS`, for the progress lines."""
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def condition_progress(position, total, corruption, severity, images,
                       condition_seconds, elapsed):
    """One progress line: where the sweep is, and how long it has left.

    Takes no measurement and is given no way to reach one, which is how
    `PROGRESS_NOTE` is enforced rather than intended. The estimate is the mean
    time per completed condition over the conditions left, which is accurate
    enough for a sweep whose conditions all read the same 12,630 images.
    """
    remaining = (total - position) * elapsed / position
    return (f"  [{position:2d}/{total}] {corruption:20s} s{severity} | "
            f"{images} images | {condition_seconds:5.1f}s | elapsed "
            f"{format_duration(elapsed)} | ETA {format_duration(remaining)}")


def write_test_outputs(suite, results, conditions, metadata, indices, labels,
                       support, results_dir):
    """Write the test artifact set: the validation set, with the split's columns.

    A separate writer for the same reason the module is separate:
    `evaluate_checkpoint.write_outputs` produced the committed validation
    results and stays byte-identical. The differences are the two split-specific
    arguments - class support in place of tracks per class - and the prediction
    position column, which is named for the split it indexes.
    """
    if metadata["test_class_support"] != support:
        raise RuntimeError("the class-support block prepared for the metadata "
                           "is not the one measured from the labels")
    results_dir.mkdir(parents=True, exist_ok=True)
    marker = results_dir / INCOMPLETE_MARKER
    marker.write_text(
        f"{metadata['model']['name']} seed {metadata['model']['seed']}, "
        f"{suite.name} suite: writing started and not yet finished. A "
        "directory holding this file is a partial result set, not a result. "
        "Move it aside and rerun; the evaluator refuses to write into a "
        "directory that is not empty.\n")

    clean = results[(CLEAN_CONDITION, 0)]
    summary_rows = []
    for corruption, severity in conditions:
        record = results[(corruption, severity)]
        summary_rows.append({
            "model": metadata["model"]["name"],
            "seed": metadata["model"]["seed"],
            "suite": suite.name,
            "corruption": corruption,
            "severity": severity,
            "parameter": suite.parameter_name(corruption),
            "parameter_value": suite.parameter_value(corruption, severity),
            "loss": record["loss"],
            "accuracy": record["accuracy"],
            "error_rate": 1.0 - record["accuracy"],
            "macro_f1": record["macro_f1"],
            "accuracy_drop": clean["accuracy"] - record["accuracy"],
            "macro_f1_drop": clean["macro_f1"] - record["macro_f1"],
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
        json.dumps({"metadata": metadata, "conditions": summary_rows},
                   indent=2) + "\n")

    class_path = results_dir / "per_class_metrics.csv"
    with open(class_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corruption", "severity", "class", "support",
                         "precision", "recall", "f1"])
        for corruption, severity in conditions:
            per_class = results[(corruption, severity)]["per_class"]
            predicted = per_class["predicted"]
            true_positive = per_class["true_positive"]
            for class_id in range(NUM_CLASSES):
                count = float(predicted[class_id])
                precision = (float(true_positive[class_id]) / count
                             if count > 0 else 0.0)
                writer.writerow([
                    corruption, severity, class_id,
                    int(per_class["support"][class_id]),
                    precision,
                    float(per_class["recall"][class_id]),
                    float(per_class["f1"][class_id]),
                ])

    confusion_path = results_dir / "confusion_matrices.npz"
    np.savez_compressed(confusion_path, **{
        f"{corruption}_s{severity}":
            results[(corruption, severity)]["confusion"].numpy()
        for corruption, severity in conditions
    })

    predictions_path = results_dir / "predictions.csv"
    with open(predictions_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corruption", "severity", "test_position",
                         "dataset_index", "label", "prediction"])
        for corruption, severity in conditions:
            predicted = results[(corruption, severity)]["predictions"].tolist()
            for position, value in enumerate(predicted):
                index = int(indices[position])
                writer.writerow([corruption, severity, position, index,
                                 int(labels[index]), value])

    (results_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")

    missing = [name for name in RESULT_FILES
               if not (results_dir / name).exists()]
    if missing:
        raise RuntimeError(f"{results_dir} is missing {missing} after the "
                           "write; the partial set keeps its marker")
    marker.unlink()
    return {
        "summary_csv": summary_path,
        "summary_json": results_dir / "summary.json",
        "per_class_csv": class_path,
        "confusion_npz": confusion_path,
        "predictions_csv": predictions_path,
        "metadata_json": results_dir / "metadata.json",
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", required=True, choices=sorted(TEST_SUITES))
    parser.add_argument("--model-name", default=None,
                        help="arm label; read from the run's policy.json if absent")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true",
                        help="a few conditions over a stride subset; reports no "
                             "performance metric and writes nothing")
    arguments = parser.parse_args(argv)
    if arguments.num_workers != 0:
        parser.error(
            f"--num-workers {arguments.num_workers} would put the sweep's "
            "reads in worker processes, where CountingDataset counts a "
            "worker's own copy and this process sees far fewer than the "
            "derived budget. The run would read the whole official test "
            "split and then fail its own accounting, so only 0 is accepted.")
    return arguments


def evaluate(arguments, counters):
    """One checkpoint, one suite, on the official test split.

    `counters` is filled in as the run proceeds so that a failure can be logged
    with the number of test images it actually read rather than with a guess.
    """
    suite = TEST_SUITES[arguments.suite]
    device = select_device(arguments.device)
    started = time.time()

    print(f"Suite {suite.name}: {suite.description}")
    print(f"Namespace {TEST_NAMESPACE}, split official_test")
    print(suite.specification())

    print("\nProtocol:")
    protocol = check_protocol_identity()
    print(f"  {protocol['final_test_protocol']} sha256 "
          f"{protocol['final_test_protocol_sha256'][:16]} (frozen)")
    print(f"  {protocol['augmentation_manifest']}: "
          f"{len(protocol['augmentation_files_verified'])} training-recipe "
          "files unchanged")

    print("\nCheckpoint:")
    record = inventory_record(arguments.checkpoint)
    digests = check_inventory_digests(arguments.checkpoint, record)
    # The pair this run belongs to is known as soon as the inventory names it,
    # and a failure from here on has to be filed against that pair: a failed
    # line that names no pair cannot be matched to a planned run, and the log
    # is append-only, so one would have to be reasoned about by hand forever.
    counters.update({"arm": record["arm_label"], "seed": int(record["seed"]),
                     "checkpoint_md5": digests["md5"], "suite": suite.name})
    model, checkpoint, identity = check_identity(arguments.checkpoint, device)
    run_metadata = discover_run_metadata(arguments.checkpoint)
    model_name = resolve_model_name(run_metadata, arguments.model_name)
    seed = identity["seed"]
    if seed is None:
        raise RuntimeError("the checkpoint records no training seed, so its "
                           "results cannot be filed against a replicate")
    if model_name != record["arm_label"]:
        raise RuntimeError(
            f"this checkpoint would be filed under arm {model_name!r} but the "
            f"inventory records it as {record['arm_label']!r}")
    if seed != record["seed"]:
        raise RuntimeError(
            f"this checkpoint records seed {seed} but the inventory records "
            f"seed {record['seed']}")
    if identity["epoch"] != record["selected_epoch"]:
        raise RuntimeError(
            f"this checkpoint is epoch {identity['epoch']} but the inventory "
            f"records the selected epoch as {record['selected_epoch']}")
    print(f"  {identity['path']}")
    print(f"  md5 {digests['md5']}  sha256 {digests['sha256'][:16]}  "
          f"(inventory entry {record['arm']}{record['seed']})")
    print(f"  arm {model_name}, seed {seed}, epoch {identity['epoch']}, "
          f"selected on {identity['selection_metric']}")
    if run_metadata.get("policy"):
        print(f"  augmentation policy {recorded_policy_name(run_metadata)}, "
              f"from {run_metadata['policy_path']}")
    provenance = check_provenance(arguments.checkpoint, identity, run_metadata)
    print("  provenance: " + "; ".join(provenance["verified"]))
    for limitation in provenance["unverifiable"]:
        print(f"  not verifiable: {limitation}")
    binding = check_validation_binding(record)
    print("  bound to its arm and seed by "
          + ", ".join(sorted(binding["validation_metadata"].values())))
    print(f"  {binding['established_by']}")

    model.eval()
    assert not model.training
    before = state_fingerprint(model)
    criterion = nn.CrossEntropyLoss()

    print("\nOfficial test split:")
    dataset, indices, labels = build_test_observations()
    parsed = parse_test_ground_truth()
    agreement = check_label_agreement(dataset, parsed)
    filename_rule = check_filename_identity(dataset)
    support = class_support(labels)
    print(f"  {len(dataset)} images, labels agreed by two independent readings "
          f"of {agreement['source']}")
    print(f"  {filename_rule}")
    print(f"  class support spans {min(support.values())} to "
          f"{max(support.values())} images over all {NUM_CLASSES} classes, "
          "each of which the labels were confirmed to carry; the split "
          "carries no track identifiers, so there is no per-sign support")

    # Wrapped before any condition is built, so every read the sweep and the
    # fingerprints make goes through one counter.
    observations = CountingDataset(dataset)
    if len(observations) != EXPECTED_TEST_SIZE:
        raise RuntimeError(f"the instrumented test split reports "
                           f"{len(observations)} images")

    if arguments.smoke_test:
        print()
        run_smoke_test(suite, model, observations, device)
        after = state_fingerprint(model)
        changed = sorted(name for name in before if before[name] != after[name])
        if changed:
            raise RuntimeError(f"model state changed during the smoke test: "
                               f"{changed}")
        print(f"\n{observations.reads} test images were read by the smoke "
              "test. Model state unchanged, nothing was written, and no "
              "access-log line is appended: the stride subset is covered by "
              "the line the real run writes.")
        print("This run produced no performance result and must not be quoted "
              "as one.")
        return None

    target = results_dir(model_name, seed, suite.name, arguments.output_dir)
    target = check_output_target(target)
    counters["output_dir"] = _repository_path(target)

    conditions = suite.conditions()
    counters["conditions"] = len(conditions)
    fingerprint_positions = list(range(0, EXPECTED_TEST_SIZE, FINGERPRINT_STRIDE))
    reads = expected_reads(suite, conditions, fingerprint_positions)
    print(f"\nRead budget: {reads['conditions']} conditions x "
          f"({reads['sweep_reads']} + {reads['fingerprint_reads']}) = "
          f"{reads['total']} test image reads, derived from the sweep and "
          f"equal to the {reads['protocol_total']} the frozen protocol states")

    reference_targets = torch.tensor([int(label) for label in labels],
                                     dtype=torch.int64)

    loaders, _ = create_data_loaders(batch_size=arguments.batch_size,
                                     download=False)
    pipelines = check_preprocessing(dataset.transform,
                                    loaders["test"].dataset.transform)
    print(f"  loader pipeline {pipelines['loader_pipeline']}, observation "
          f"pipeline {pipelines['observation_pipeline']}")
    # Rebound to the instrumented dict, so no uncounted handle on any split
    # survives in this function.
    loaders = instrument_loaders(loaders)
    counters.update(measured_split_counts(loaders))

    print("\nClean reproduction through data.py's test loader:")
    reference = evaluate_condition(model, loaders["test"], device, criterion)
    counters.update(measured_split_counts(loaders))
    check_target_sequence(reference["targets"], reference_targets,
                          "the data.py test loader")
    reference_accuracy, reference_f1 = metrics_from_confusion(
        reference["confusion"])
    direct_accuracy, direct_f1 = independent_metrics(reference["predictions"],
                                                     reference["targets"])
    for name, through_matrix, direct in (
            ("accuracy", reference_accuracy, direct_accuracy),
            ("macro F1", reference_f1, direct_f1)):
        if abs(through_matrix - direct) > METRIC_TOLERANCE:
            raise RuntimeError(
                f"the {name} read off the confusion matrix and the {name} "
                "counted directly from the prediction and label vectors "
                f"differ by {abs(through_matrix - direct):.3g}, more than "
                f"{METRIC_TOLERANCE:g}; the metric path is wrong")
    print(f"  {reference['seen']} images in the official label order, "
          "cross-checked against a direct count of the prediction and label "
          "vectors")

    print(f"\nEvaluating {len(conditions)} conditions "
          f"(1 clean + {len(conditions) - 1} corrupted). {PROGRESS_NOTE}:")
    results = {}
    fingerprints = {}
    for corruption, severity in conditions:
        condition_started = time.time()
        condition_dataset = suite.dataset(observations, corruption, severity)
        loader = condition_loader(condition_dataset,
                                  batch_size=arguments.batch_size,
                                  num_workers=arguments.num_workers)
        measurement = evaluate_condition(model, loader, device, criterion)
        if measurement["seen"] != EXPECTED_TEST_SIZE:
            raise RuntimeError(f"{corruption} s{severity} evaluated "
                               f"{measurement['seen']} images")
        check_target_sequence(measurement["targets"], reference_targets,
                              f"{corruption} s{severity}")
        if measurement["grad_enabled"] is not False:
            raise RuntimeError(f"{corruption} s{severity} ran with gradients "
                               "enabled")
        accuracy, macro_f1 = metrics_from_confusion(measurement["confusion"])
        per_class = per_class_from_confusion(measurement["confusion"])
        direct_accuracy, direct_f1 = independent_metrics(
            measurement["predictions"], measurement["targets"])
        for name, through_matrix, direct in (
                ("accuracy", accuracy, direct_accuracy),
                ("macro_f1", macro_f1, direct_f1)):
            if abs(through_matrix - direct) > METRIC_TOLERANCE:
                raise RuntimeError(
                    f"{corruption} s{severity}: the confusion matrix {name} "
                    "and the direct count differ by "
                    f"{abs(through_matrix - direct):.3g}, more than "
                    f"{METRIC_TOLERANCE:g}")
        fingerprints[f"{corruption}_s{severity}"] = input_fingerprint(
            condition_dataset, fingerprint_positions)
        measurement.update({"accuracy": accuracy, "macro_f1": macro_f1,
                            "per_class": per_class})
        results[(corruption, severity)] = measurement
        counters["test_images_read"] = observations.reads
        counters.update(measured_split_counts(loaders))
        print(condition_progress(len(results), len(conditions), corruption,
                                 severity, measurement["seen"],
                                 time.time() - condition_started,
                                 time.time() - started))

    if len(fingerprints) != len(conditions):
        raise RuntimeError(f"{len(fingerprints)} fingerprints for "
                           f"{len(conditions)} conditions")
    if len(set(fingerprints.values())) != len(fingerprints):
        repeated = sorted(key for key in fingerprints
                          if list(fingerprints.values()).count(
                              fingerprints[key]) > 1)
        raise RuntimeError(f"conditions {repeated[:4]} were shown identical "
                           "inputs, so the suite is not realising distinct "
                           "conditions")

    clean_row = results[(CLEAN_CONDITION, 0)]
    if not torch.equal(clean_row["predictions"], reference["predictions"]):
        mismatches = int((clean_row["predictions"]
                          != reference["predictions"]).sum())
        raise RuntimeError(
            f"the sweep's clean condition disagrees with the data.py test "
            f"loader on {mismatches} of {EXPECTED_TEST_SIZE} images; the "
            "corruption wrapper is altering uncorrupted inputs")
    for name, swept, referenced in (
            ("loss", clean_row["loss"], reference["loss"]),
            ("accuracy", clean_row["accuracy"], reference_accuracy),
            ("macro_f1", clean_row["macro_f1"], reference_f1)):
        if abs(swept - referenced) > CLEAN_TOLERANCE:
            raise RuntimeError(
                f"the sweep's clean {name} and the data.py reference run's "
                f"differ by {abs(swept - referenced):.3g}, more than "
                f"{CLEAN_TOLERANCE:g}")
    print("\nSweep clean row: identical predictions to the data.py test "
          "reference run, so every drop is measured against this model's own "
          "clean test score")

    accounting = check_split_accounting(loaders, observations.reads,
                                        reads["total"])
    counters["test_images_read"] = observations.reads
    counters.update({key: accounting[key] for key in (
        "test_loader_iterations", "test_loader_reads",
        "train_loader_iterations", "train_loader_reads",
        "validation_loader_iterations", "validation_loader_reads")})
    print("\nSplit accounting, asserted rather than recorded:")
    print(f"  sweep read {accounting['sweep_test_images_read']} test images, "
          f"exactly the derived budget")
    print(f"  data.py test loader: {accounting['test_loader_iterations']} "
          f"iteration, {accounting['test_loader_reads']} reads")
    print("  train loader: 0 iterations, 0 reads; validation loader: 0 "
          "iterations, 0 reads")

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
        "model": model_record(model_name, seed, run_metadata, provenance),
        "checkpoint": identity,
        "checkpoint_digests": digests,
        "inventory_record": record,
        "validation_binding": binding,
        "protocol_identity": protocol,
        "evaluation": {
            "suite": suite.name,
            "suite_description": suite.description,
            "split": "official_test",
            "test_images": len(dataset),
            "num_classes": NUM_CLASSES,
            "conditions": len(conditions),
            "batch_size": arguments.batch_size,
            "num_workers": arguments.num_workers,
            "device": str(device),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mode": "eval (BatchNorm running statistics from training, no "
                    "recalibration)",
            "gradients_enabled": any(measurement["grad_enabled"]
                                     for measurement in results.values()),
            "optimizer": None,
            "preprocessing": pipelines,
            "seconds": time.time() - started,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "ground_truth": dict(agreement, filename_rule=filename_rule),
        "corruption_suite": {
            "module": "src/corruptions.py" if suite.name == "frozen"
                      else "src/corruptions_heldout.py",
            "namespace": TEST_NAMESPACE,
            "base_seed": SEED,
            "severity_levels": list(SEVERITY_LEVELS),
            "specification": {
                name: {"parameter": spec.parameter, "levels": list(spec.levels),
                       "stochastic": spec.stochastic,
                       "severity_in_seed": spec.severity_in_seed}
                for name, spec in suite.table.items()
            },
            "input_fingerprints": fingerprints,
            "fingerprint_stride": FINGERPRINT_STRIDE,
            "fingerprint_frames": len(fingerprint_positions),
            "fingerprint_note": (
                "One fingerprint per condition, each over "
                f"{len(fingerprint_positions)} positions sampled every "
                f"{FINGERPRINT_STRIDE}th image. On validation that stride was "
                "one frame per physical sign; the test release carries no "
                "track identifiers, so here it is simply a 1-in-"
                f"{FINGERPRINT_STRIDE} systematic sample of positions."),
        },
        "interpretation_notes": suite.notes,
        "test_class_support": support,
        "read_accounting": reads,
        "integrity": {
            "checkpoint_md5": digests["md5"],
            "checkpoint_sha256": digests["sha256"],
            "matches_checkpoint_inventory": True,
            "validation_binding_verified": True,
            "frozen_protocol_sha256_verified": True,
            "ground_truth_agreed_by_two_readings": True,
            "dataset_index_is_filename": True,
            "sweep_clean_row_matches_reference": True,
            "metrics_cross_checked_without_confusion_matrix": True,
            "model_state_unchanged": True,
            "batchnorm_buffers_checked": len(batchnorm_buffers),
            "state_tensors_checked": len(before),
            "conditions_with_distinct_inputs": len(fingerprints),
            **accounting,
        },
        "repository": repository_state(),
    }

    paths = write_test_outputs(suite, results, conditions, metadata, indices,
                               labels, support, target)
    summary_rows = json.loads(paths["summary_json"].read_text())["conditions"]
    if len(summary_rows) != len(conditions):
        raise RuntimeError(f"{paths['summary_json']} holds "
                           f"{len(summary_rows)} conditions and the sweep ran "
                           f"{len(conditions)}")

    print("\nIntegrity:")
    print(f"  all {len(before)} parameter and buffer tensors identical before "
          f"and after evaluation, including {len(batchnorm_buffers)} BatchNorm "
          "buffers")
    print("  no gradient was populated; no optimizer was constructed")
    print(f"  every condition evaluated exactly {EXPECTED_TEST_SIZE} images in "
          "the same label order")
    print(f"  {len(fingerprints)} input fingerprints over "
          f"{len(fingerprint_positions)} positions each, all distinct")

    print("\nWritten:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    entry = access_log_entry(
        "completed", arm=model_name, seed=seed,
        checkpoint_md5=digests["md5"], suite=suite.name,
        conditions=len(conditions),
        test_images_read=observations.reads,
        test_loader_iterations=accounting["test_loader_iterations"],
        test_loader_reads=accounting["test_loader_reads"],
        train_loader_iterations=accounting["train_loader_iterations"],
        train_loader_reads=accounting["train_loader_reads"],
        validation_loader_iterations=accounting["validation_loader_iterations"],
        validation_loader_reads=accounting["validation_loader_reads"],
        output_dir=_repository_path(target),
        elapsed_seconds=time.time() - started)
    log_path = append_access_log(entry)
    print(f"\nAccess log: appended one completed line to "
          f"{_repository_path(log_path)}")
    print(f"Total {time.time() - started:.1f}s on the official test split. No "
          "training, augmentation, or BatchNorm recalibration occurred, and "
          "neither the training nor the validation split was read.")
    return entry


def main(argv=None):
    """Run one evaluation, and record the attempt whether or not it completes.

    A run that fails after reading test images has still read them, so the
    access log is written on the way out either way. The line a failed run
    writes carries `status: "failed"`, the images it actually read, and the
    reason it stopped, so it can never be counted as one of the eighteen
    completed evaluations.
    """
    arguments = parse_arguments(argv)
    counters = {"checkpoint": arguments.checkpoint,
                "arm": None, "seed": None, "checkpoint_md5": None,
                "suite": arguments.suite, "conditions": None,
                "test_images_read": 0, "test_loader_iterations": 0,
                "test_loader_reads": 0, "train_loader_iterations": 0,
                "train_loader_reads": 0, "validation_loader_iterations": 0,
                "validation_loader_reads": 0, "output_dir": None}
    try:
        return evaluate(arguments, counters)
    except BaseException as error:
        if arguments.smoke_test:
            raise
        entry = access_log_entry(
            "failed", error=f"{type(error).__name__}: {error}",
            elapsed_seconds=None, **counters)
        try:
            append_access_log(entry)
        except Exception as log_error:
            # The log is the record of how often the split has been read, so
            # losing a line matters more than the tidiness of the traceback.
            # The line is printed where it cannot be written, and the original
            # failure is re-raised either way.
            print(f"\nThe evaluation did not complete, and the access log "
                  f"could not be written "
                  f"({type(log_error).__name__}: {log_error}). The line that "
                  f"belongs in {_repository_path(ACCESS_LOG)} follows, so the "
                  "reads it records can be appended by hand:")
            print(json.dumps(entry))
        else:
            print(f"\nThe evaluation did not complete. One failed line was "
                  f"appended to {_repository_path(ACCESS_LOG)} recording "
                  f"{counters['test_images_read']} test images read. It is "
                  "not a result and is not counted as a completed "
                  "evaluation.")
        raise


if __name__ == "__main__":
    main()
