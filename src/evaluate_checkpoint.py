"""Evaluate any accepted checkpoint under either frozen corruption suite.

    .venv/bin/python -m src.evaluate_checkpoint \
        --checkpoint artifacts/baseline/best_checkpoint.pt --suite frozen
    .venv/bin/python -m src.evaluate_checkpoint \
        --checkpoint artifacts/augmented_targeted_seed42/best_checkpoint.pt \
        --suite heldout

`evaluate_robustness.py` measured one pinned checkpoint under one suite and is
left exactly as it was, because the accepted baseline robustness table has to
stay reproducible by the code that wrote it. The augmentation comparison needs
the same measurement for nine checkpoints under two suites, so this module
takes the checkpoint and the suite as arguments and imports the machinery -
the condition loader, the evaluation pass, the metric cross-check, the state
fingerprint, the loader instrumentation - from that file rather than restating
it.

What is preserved from the original, because the results are only worth as much
as these guarantees:

* the model is loaded in eval mode, never trained, and its parameters and
  BatchNorm buffers are hashed before and after the sweep;
* metrics are computed twice by independent paths, once through the confusion
  matrix and once by counting matches directly;
* every condition must evaluate all 5,340 validation images in the same label
  order, verified against the labels the split itself reports;
* the clean condition is cross-checked against `data.py`'s own validation
  loader, and against the metrics the checkpoint recorded for itself;
* the official test loader is instrumented and its iteration and read counts
  are written into the metadata.

What is added for the comparison: a per-condition fingerprint of the input
tensors. Two models can only be compared on a corrupted suite if they were
shown the same corrupted pixels, and that claim is checkable after the fact by
comparing the fingerprints in their metadata rather than by trusting that the
seeding is index-based.
"""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

from .corruptions import (
    CORRUPTIONS,
    SEVERITY_LEVELS,
    VALIDATION_NAMESPACE,
    CorruptedDataset,
    build_validation_observations,
    corruption_parameter,
    format_specification,
)
from .augmentation import PROTOCOL_ARTIFACT
from .corruptions_heldout import (
    HELDOUT_CORRUPTIONS,
    HeldoutDataset,
    format_heldout_specification,
    heldout_parameter,
)
from .data import NUM_CLASSES, SEED, create_data_loaders
from .evaluate_robustness import (
    ACCEPTED_CLEAN,
    BATCH_SIZE,
    CLEAN_CONDITION,
    CLEAN_TOLERANCE,
    EXPECTED_CHECKPOINT_MD5,
    EXPECTED_EPOCH,
    EXPECTED_VALIDATION_SIZE,
    METRIC_TOLERANCE,
    OCCLUSION_NOTE,
    PROJECT_ROOT,
    SINGLE_TRACK_WARNING,
    check_target_sequence,
    condition_loader,
    evaluate_condition,
    file_md5,
    format_table,
    independent_metrics,
    instrument_loaders,
    repository_state,
    state_fingerprint,
    validation_tracks_per_class,
)
from .train import (
    load_checkpoint,
    metrics_from_confusion,
    per_class_from_confusion,
    select_device,
)

RESULTS_ROOT = PROJECT_ROOT / "artifacts" / "robustness"


def _repository_path(path):
    """Render a path relative to the project root when it lies inside it."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# Positions sampled for the input fingerprint. A stride of 30 lands one frame
# per track, so the fingerprint covers all 43 classes and 178 distinct signs
# rather than the first two classes.
FINGERPRINT_STRIDE = 30

# Smoke mode evaluates this many conditions over a small stride subset. It is a
# plumbing check: no metric from it is reported or recorded.
SMOKE_CONDITIONS = 3
SMOKE_STRIDE = 400


class Suite:
    """One frozen set of conditions, with the code that realises them.

    Both suites share the clean condition, the split, the normalisation, the
    seeding convention, and the namespace. They differ in the corruption table
    and in the dataset wrapper that reads it, so those are the only two things
    this holds beyond names and notes.
    """

    def __init__(self, name, table, dataset_class, parameter, specification,
                 description, notes):
        self.name = name
        self.table = table
        self.dataset_class = dataset_class
        self.parameter = parameter
        self.specification = specification
        self.description = description
        self.notes = notes

    def conditions(self):
        """Clean, then every family at every severity."""
        conditions = [(CLEAN_CONDITION, 0)]
        for family in self.table:
            conditions.extend((family, severity) for severity in SEVERITY_LEVELS)
        return conditions

    def dataset(self, observations, corruption, severity):
        """Wrap the observation split for one condition, clean included.

        The clean condition always goes through the frozen `CorruptedDataset`
        with no corruption, in both suites, so every suite's clean row is
        byte-identical to the baseline's clean row and the two suites' drops are
        measured from the same reference.
        """
        if corruption == CLEAN_CONDITION:
            return CorruptedDataset(observations, None,
                                    namespace=VALIDATION_NAMESPACE)
        return self.dataset_class(observations, corruption, severity,
                                  namespace=VALIDATION_NAMESPACE, base_seed=SEED)

    def parameter_value(self, corruption, severity):
        if corruption == CLEAN_CONDITION:
            return ""
        return self.parameter(corruption, severity)

    def parameter_name(self, corruption):
        if corruption == CLEAN_CONDITION:
            return ""
        return self.table[corruption].parameter


FROZEN_SUITE = Suite(
    name="frozen",
    table=CORRUPTIONS,
    dataset_class=CorruptedDataset,
    parameter=corruption_parameter,
    specification=format_specification,
    description="the six training-family conditions from src/corruptions.py",
    notes={"occlusion": OCCLUSION_NOTE, "per_class_support": SINGLE_TRACK_WARNING},
)

HELDOUT_SUITE = Suite(
    name="heldout",
    table=HELDOUT_CORRUPTIONS,
    dataset_class=HeldoutDataset,
    parameter=heldout_parameter,
    specification=format_heldout_specification,
    description="the five generalisation conditions from src/corruptions_heldout.py",
    notes={
        "scope": (
            "No model in this study is trained on any held-out family, so these "
            "numbers measure transfer rather than performance on the training "
            "distribution. They are co-primary with the frozen suite, not a "
            "secondary check: a policy that helps on the frozen suite and not "
            "here has learned the benchmark, not robustness."
        ),
        "per_class_support": SINGLE_TRACK_WARNING,
    },
)

SUITES = {suite.name: suite for suite in (FROZEN_SUITE, HELDOUT_SUITE)}


def input_fingerprint(dataset, positions):
    """Hash the input tensors one condition actually feeds the model.

    Computed from the dataset rather than from inside the evaluation loop, so it
    is independent of the model, the device, and the batch layout. Two runs with
    the same fingerprint for a condition were shown the same pixels for the
    sampled positions, which is the precondition for comparing their scores.
    """
    digest = hashlib.sha256()
    for position in positions:
        image, label = dataset[position]
        payload = image.detach().to("cpu").contiguous()
        digest.update(payload.numpy().tobytes())
        digest.update(int(label).to_bytes(2, "big"))
    return digest.hexdigest()


def discover_run_metadata(checkpoint_path):
    """Read the policy and history a training run left beside its checkpoint.

    The arm and seed a checkpoint belongs to are properties of the run that
    produced it, so they are read from that run's own records where possible
    instead of retyped on the command line, where they could disagree with the
    artifact they label.
    """
    directory = Path(checkpoint_path).parent
    record = {}
    policy_path = directory / "policy.json"
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())
        record["policy_path"] = _repository_path(policy_path)
        record["policy"] = policy.get("policy")
        record["policy_config"] = policy.get("config")
        record["policy_realised"] = policy.get("realised")
        record["augmentation_base_seed"] = policy.get("augmentation_base_seed")
        record["leakage_checks"] = policy.get("leakage_checks")
    protocol_path = directory / PROTOCOL_ARTIFACT
    if protocol_path.exists():
        record["protocol_path"] = _repository_path(protocol_path)
        record["protocol"] = json.loads(protocol_path.read_text())
    history_path = directory / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        record["history_path"] = _repository_path(history_path)
        record["best_epoch"] = history.get("best_epoch")
        record["selection_metric"] = history.get("selection_metric")
        record["history_config"] = history.get("config")
        record["history_checkpoint"] = history.get("checkpoint")
        record["history_best_metrics"] = {
            "loss": history.get("best_validation_loss"),
            "accuracy": history.get("best_validation_accuracy"),
            "macro_f1": history.get("best_validation_macro_f1"),
        }
    return record


def recorded_policy_name(run_metadata):
    """The arm name a training run recorded, or None if it recorded no policy.

    `AugmentationPolicy.describe()` files the arm identifier under `policy`.
    Every read of that name in this module goes through this function, so the
    key is written down once: a second spelling of it elsewhere would be a
    `KeyError` on the augmented arms and would leave the clean baseline, which
    has no policy record at all, as the only path that still worked.
    """
    policy = run_metadata.get("policy")
    if policy is None:
        return None
    return policy.get("policy") or None


def resolve_model_name(run_metadata, override=None):
    """Name the arm this checkpoint belongs to, for the output path and tables.

    `clean` is returned only when the run left no policy record at all, which is
    what the baseline recipe does. A run that recorded a policy but no name
    raises instead: filing an augmented arm's results under the clean baseline's
    name would put them in the wrong directory and the wrong column of every
    comparison table.
    """
    if override:
        return override
    if "policy_path" not in run_metadata:
        return "clean"
    name = recorded_policy_name(run_metadata)
    if not name:
        raise RuntimeError(
            f"{run_metadata['policy_path']} records no policy name, so this "
            "checkpoint cannot be filed against an arm; pass --model-name")
    return name


def model_record(model_name, seed, run_metadata, provenance=None):
    """Assemble the `model` block of a result set's metadata.

    This block is what `compare_models.py` later reads to decide which arm and
    which replicate a results directory belongs to, so it is built here, in one
    place, by the same code path that writes every real evaluation, rather than
    inline in the sweep where only a full run would reach it.
    """
    policy = run_metadata.get("policy")
    arm = "clean baseline"
    if policy is not None:
        name = recorded_policy_name(run_metadata)
        if not name:
            raise RuntimeError(
                "the run recorded an augmentation policy without a name, so "
                "its results cannot be filed against an arm")
        arm = f"augmented ({name})"
    return {
        "name": model_name,
        "seed": seed,
        "arm": arm,
        "augmentation_policy": policy,
        "training_records": {key: run_metadata[key] for key in
                             ("policy_path", "history_path", "best_epoch",
                              "selection_metric") if key in run_metadata},
        "training_leakage_checks": run_metadata.get("leakage_checks"),
        "protocol": run_metadata.get("protocol"),
        "provenance": provenance,
    }


def check_identity(checkpoint_path, device):
    """Load the checkpoint and record exactly which artifact was measured.

    The accepted baseline is pinned: if the md5 matches it, the epoch and the
    recorded metrics must match too, so a silently re-trained replacement fails
    here rather than producing a different curve under the baseline's name. Any
    other checkpoint is identified by its own md5 and its own recorded metrics,
    which the clean reproduction is then checked against.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint_path}")
    digest = file_md5(checkpoint_path)
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    is_accepted_baseline = digest == EXPECTED_CHECKPOINT_MD5
    if is_accepted_baseline:
        if checkpoint["epoch"] != EXPECTED_EPOCH:
            raise RuntimeError(f"the accepted baseline md5 reports epoch "
                               f"{checkpoint['epoch']}, expected {EXPECTED_EPOCH}")
        for key, accepted in ACCEPTED_CLEAN.items():
            recorded = checkpoint["validation_metrics"][key]
            if abs(recorded - accepted) > CLEAN_TOLERANCE:
                raise RuntimeError(
                    f"the accepted baseline records {key} {recorded:.6f}, "
                    f"expected {accepted:.6f}")
    identity = {
        "path": _repository_path(checkpoint_path),
        "md5": digest,
        "is_accepted_baseline": is_accepted_baseline,
        "epoch": checkpoint["epoch"],
        "seed": checkpoint.get("seed"),
        "selection_metric": checkpoint["selection_metric"],
        "recorded_validation_metrics": checkpoint["validation_metrics"],
        "model_config": checkpoint["model_config"],
        "training_config": checkpoint.get("training_config"),
        "trained_on_device": checkpoint.get("environment", {}).get("device"),
    }
    return model, checkpoint, identity


def check_clean_against_record(measured, identity, device):
    """Compare the reproduced clean metrics with the checkpoint's own record.

    A checkpoint carries the validation metrics its training run measured at the
    selected epoch. Re-measuring them here on the same device must reproduce
    them, which catches a corrupted artifact, a changed split, or a changed
    preprocessing pipeline for every arm and not only for the pinned baseline.

    Across devices it must not be a hard failure: CPU, MPS, and CUDA use
    different kernels and reduction orders, so the low-order digits legitimately
    differ. The difference is recorded either way.
    """
    recorded = identity["recorded_validation_metrics"]
    same_device = identity["trained_on_device"] == str(device)
    differences = {key: abs(measured[key] - recorded[key]) for key in recorded}
    if same_device:
        for key, difference in differences.items():
            if difference > CLEAN_TOLERANCE:
                raise RuntimeError(
                    f"reproduced clean {key} is {measured[key]:.6f} but the "
                    f"checkpoint recorded {recorded[key]:.6f} (difference "
                    f"{difference:.2e}) on the same device {device}; stop and "
                    "diagnose before trusting corrupted results")
    return {
        "recorded": recorded,
        "reproduced": measured,
        "absolute_differences": differences,
        "same_device_as_training": same_device,
        "enforced_at": CLEAN_TOLERANCE if same_device else None,
    }


def _agreeing(label, left, right, disagreements, missing=None):
    """Compare two copies of one recorded fact, and say which way it failed.

    A field absent on either side is not a disagreement, but it is not a
    verification either, so where the field was expected the caller passes a
    list to record that it could not be checked.
    """
    if left is None or right is None:
        if missing is not None:
            missing.append(f"{label} is not recorded on both sides, so it "
                           "could not be checked")
        return False
    if left != right:
        disagreements.append(f"{label}: {left!r} in the run record, {right!r} "
                             "in the checkpoint")
        return False
    return True


def check_provenance(checkpoint_path, identity, run_metadata):
    """Tie the checkpoint file being evaluated to the run records beside it.

    Every arm's results are filed under a policy name and a seed taken from the
    `policy.json` and `history.json` in the checkpoint's own directory, so those
    records have to be shown to describe this file rather than some other run's.
    Four facts the training run wrote twice are cross-checked: the training
    seed, the selected epoch, the validation metrics at that epoch, and the
    checkpoint filename. The metric comparison is the binding one. Both copies
    were written by one process in one pass, so they agree to the last digit,
    and a checkpoint file copied or moved between two run directories fails
    here instead of being relabelled.

    One thing genuinely cannot be verified and is reported rather than assumed:
    the checkpoint format carries no augmentation-policy field, so the arm label
    rests on the `policy.json` beside it. The seed, the training configuration,
    the selected epoch and the recorded metrics must all agree with that record,
    which rules out a checkpoint from a different recipe, a different seed or a
    different epoch - but two runs that differ only in their augmentation
    policy write identical configurations, so the policy name itself is
    established by the directory's record and not by the checkpoint.

    The accepted clean baseline has no `policy.json`, which is what its recipe
    produces and is treated as the clean arm, not as a missing record.
    """
    checkpoint_path = Path(checkpoint_path)
    disagreements = []
    verified = []
    unverifiable = []

    if "history_path" in run_metadata:
        if _agreeing("best epoch", run_metadata.get("best_epoch"),
                     identity["epoch"], disagreements, unverifiable):
            verified.append(f"selected epoch {identity['epoch']}")
        if _agreeing("selection metric", run_metadata.get("selection_metric"),
                     identity["selection_metric"], disagreements, unverifiable):
            verified.append(f"selected on {identity['selection_metric']}")
        history_config = run_metadata.get("history_config") or {}
        if _agreeing("training seed", history_config.get("seed"),
                     identity["seed"], disagreements, unverifiable):
            verified.append(f"training seed {identity['seed']}")
        recorded = identity["recorded_validation_metrics"]
        best = run_metadata.get("history_best_metrics") or {}
        matched = [key for key in sorted(recorded)
                   if _agreeing(f"validation {key} at the selected epoch",
                                best.get(key), recorded[key], disagreements,
                                unverifiable)]
        if len(matched) == len(recorded):
            verified.append("the validation metrics recorded at that epoch, "
                            "exactly")
        if run_metadata.get("history_checkpoint") is not None:
            if _agreeing("checkpoint filename", run_metadata["history_checkpoint"],
                         checkpoint_path.name, disagreements):
                verified.append(f"the filename the run wrote "
                                f"({checkpoint_path.name})")
        else:
            unverifiable.append("the run's history records no checkpoint filename")
    else:
        unverifiable.append(
            f"no history.json beside {checkpoint_path.name}, so the selected "
            "epoch, the training seed and the recorded metrics have nothing to "
            "be checked against")

    protocol = run_metadata.get("protocol")
    if protocol is not None:
        if _agreeing("training seed in the protocol record",
                     protocol.get("seed"), identity["seed"], disagreements,
                     unverifiable):
            verified.append("the protocol record's training seed")
        frozen = (protocol.get("protocol") or {})
        if frozen.get("matches_manifest"):
            verified.append(f"started under the protocol frozen in "
                            f"{frozen['manifest']}")
        else:
            unverifiable.append(
                "the run was not started under the frozen protocol: "
                + (", ".join(frozen.get("differs_from_manifest") or [])
                   or f"it recorded no digests for {PROTOCOL_ARTIFACT}"))
    else:
        unverifiable.append(
            f"no {PROTOCOL_ARTIFACT} beside {checkpoint_path.name}, so the "
            "protocol version this run was started under is not recorded; the "
            "accepted baseline predates the frozen protocol")

    policy_config = run_metadata.get("policy_config")
    if "policy_path" in run_metadata:
        if not recorded_policy_name(run_metadata):
            disagreements.append(f"{run_metadata['policy_path']} records no "
                                 "policy name")
        if _agreeing("training seed in the policy record",
                     (policy_config or {}).get("seed"), identity["seed"],
                     disagreements, unverifiable):
            verified.append("the policy record's training seed")
        if _agreeing("augmentation base seed",
                     run_metadata.get("augmentation_base_seed"),
                     identity["seed"], disagreements, unverifiable):
            verified.append("the augmentation stream's base seed")
        realised = run_metadata.get("policy_realised") or {}
        if realised.get("best_epoch") is not None:
            if _agreeing("best epoch in the policy record",
                         realised["best_epoch"], identity["epoch"], disagreements):
                verified.append("the policy record's best epoch")
        else:
            unverifiable.append("the policy record has no realised block, so "
                                "the run it describes may not have finished")
        unverifiable.append(
            "the checkpoint carries no policy field, so the arm name comes from "
            f"{run_metadata['policy_path']}; a checkpoint from a run with the "
            "same seed and configuration but a different policy could not be "
            "distinguished by its contents")
    else:
        verified.append("no policy record, which is the clean recipe's output")

    training_config = identity.get("training_config") or {}
    for label, record in (("history", run_metadata.get("history_config")),
                          ("policy", policy_config)):
        for key in sorted(set(record or {}) & set(training_config)):
            _agreeing(f"{label} record's {key}", record[key],
                      training_config[key], disagreements)

    if disagreements:
        raise RuntimeError(
            f"the records beside {checkpoint_path} do not describe it:\n  "
            + "\n  ".join(disagreements)
            + "\nthe arm and seed this result would be filed under cannot be "
              "trusted; do not evaluate until the mismatch is explained")

    return {
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_md5": identity["md5"],
        "training_seed": identity["seed"],
        "best_epoch": identity["epoch"],
        "selection_metric": identity["selection_metric"],
        "policy_name": recorded_policy_name(run_metadata),
        "protocol": (run_metadata.get("protocol") or {}).get("protocol"),
        "protocol_record": run_metadata.get("protocol_path"),
        "policy_record": run_metadata.get("policy_path"),
        "history_record": run_metadata.get("history_path"),
        "verified": verified,
        "unverifiable": unverifiable,
    }


def write_outputs(suite, results, conditions, metadata, validation_indices,
                  labels, tracks_per_class, results_dir):
    """Write the same artifact set as the baseline evaluation, for any suite.

    Deliberately a separate writer rather than a parameter added to
    `evaluate_robustness.write_outputs`: that function resolves parameter names
    through the frozen `CORRUPTIONS` table, and the file it lives in is the one
    that produced the accepted baseline results. Keeping it byte-identical is
    worth more than sharing forty lines of CSV writing.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

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
        json.dumps({"metadata": metadata, "conditions": summary_rows}, indent=2) + "\n")

    class_path = results_dir / "per_class_metrics.csv"
    with open(class_path, "w", newline="") as handle:
        writer = csv.writer(handle)
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


def run_smoke_test(suite, model, observations, device):
    """Check that the suite runs end to end, without producing a number.

    A few conditions over a small stride subset, reporting only that the shapes,
    counts, label order and input fingerprints are what they should be. No
    accuracy or loss is printed or written: on a few hundred frames spanning a
    handful of tracks, any score would be noise, and a number printed here would
    eventually be quoted.
    """
    positions = list(range(0, len(observations), SMOKE_STRIDE))
    subset = Subset(observations, positions)
    conditions = suite.conditions()[:SMOKE_CONDITIONS]
    print(f"Smoke test: {len(conditions)} conditions over {len(positions)} "
          f"frames (stride {SMOKE_STRIDE}). Plumbing only - no accuracy or "
          "loss is computed, reported, or saved.")
    fingerprints = {}
    for corruption, severity in conditions:
        dataset = suite.dataset(subset, corruption, severity)
        image, _ = dataset[0]
        if tuple(image.shape) != (3, 64, 64):
            raise RuntimeError(f"{corruption} s{severity} produced shape "
                               f"{tuple(image.shape)}")
        loader = condition_loader(dataset)
        with torch.no_grad():
            batches = 0
            seen = 0
            for images, labels in loader:
                logits = model(images.to(device))
                if logits.shape != (labels.shape[0], NUM_CLASSES):
                    raise RuntimeError(f"{corruption} s{severity} logits shape "
                                       f"{tuple(logits.shape)}")
                batches += 1
                seen += labels.shape[0]
        if seen != len(positions):
            raise RuntimeError(f"{corruption} s{severity} saw {seen} of "
                               f"{len(positions)} frames")
        key = f"{corruption}_s{severity}"
        fingerprints[key] = input_fingerprint(dataset, range(len(positions)))
        print(f"  {corruption:20s} s{severity} | {batches} batches, {seen} frames, "
              f"input fingerprint {fingerprints[key][:12]}")
    if len(set(fingerprints.values())) != len(fingerprints):
        raise RuntimeError("two smoke conditions produced identical inputs")
    print("  every condition produced distinct inputs and a full forward pass")
    return fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--model-name", default=None,
                        help="arm label; read from the run's policy.json if absent")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true",
                        help="a few conditions over a stride subset; reports no "
                             "performance metric")
    arguments = parser.parse_args()

    suite = SUITES[arguments.suite]
    device = select_device(arguments.device)
    started = time.time()

    print(f"Suite {suite.name}: {suite.description}")
    print(suite.specification())

    print("\nCheckpoint:")
    model, checkpoint, identity = check_identity(arguments.checkpoint, device)
    run_metadata = discover_run_metadata(arguments.checkpoint)
    model_name = resolve_model_name(run_metadata, arguments.model_name)
    seed = identity["seed"]
    if seed is None:
        raise RuntimeError("the checkpoint records no training seed, so its "
                           "results cannot be filed against a replicate")
    print(f"  {identity['path']}")
    print(f"  md5 {identity['md5']}"
          + ("  (the accepted clean baseline)" if identity["is_accepted_baseline"]
             else ""))
    print(f"  arm {model_name}, seed {seed}, epoch {identity['epoch']}, "
          f"selected on {identity['selection_metric']}")
    print(f"  recorded validation metrics {identity['recorded_validation_metrics']}")
    if run_metadata.get("policy"):
        print(f"  augmentation policy {recorded_policy_name(run_metadata)}, "
              f"P(augment) {run_metadata['policy']['augment_probability']:g}, "
              f"from {run_metadata['policy_path']}")
    provenance = check_provenance(arguments.checkpoint, identity, run_metadata)
    print("  provenance: " + "; ".join(provenance["verified"]))
    for limitation in provenance["unverifiable"]:
        print(f"  not verifiable: {limitation}")

    model.eval()
    assert not model.training
    before = state_fingerprint(model)
    criterion = nn.CrossEntropyLoss()

    observations, validation_indices, labels = build_validation_observations()
    if len(observations) != EXPECTED_VALIDATION_SIZE:
        raise RuntimeError(f"validation split holds {len(observations)} images, "
                           f"expected {EXPECTED_VALIDATION_SIZE}")

    if arguments.smoke_test:
        print()
        run_smoke_test(suite, model, observations, device)
        after = state_fingerprint(model)
        changed = sorted(name for name in before if before[name] != after[name])
        if changed:
            raise RuntimeError(f"model state changed during the smoke test: {changed}")
        print("\nModel state unchanged. Nothing was written. This run produced no "
              "performance result and must not be quoted as one.")
        return

    results_dir = (Path(arguments.output_dir) if arguments.output_dir
                   else RESULTS_ROOT / f"{model_name}_seed{seed}" / suite.name)
    if results_dir.exists() and any(results_dir.iterdir()):
        raise RuntimeError(f"{results_dir} already holds results; evaluating into "
                           "it would overwrite an accepted artifact. Pass a "
                           "different --output-dir.")

    reference_targets = torch.tensor(
        [int(labels[int(index)]) for index in validation_indices], dtype=torch.int64)
    tracks_per_class = validation_tracks_per_class(
        observations.dataset, validation_indices, labels)
    print(f"\nValidation split: {len(observations)} images, "
          f"{len(set(int(i) for i in validation_indices))} unique dataset indices, "
          f"{sum(tracks_per_class.values())} tracks, "
          f"{len(reference_targets.unique())} classes")

    loaders, _ = create_data_loaders(batch_size=arguments.batch_size, download=False)
    # Rebound to the instrumented dict, so no uncounted handle on the official
    # test split survives in this function.
    loaders = instrument_loaders(loaders)
    print("\nClean reproduction through data.py's validation loader:")
    reference = evaluate_condition(model, loaders["validation"], device, criterion)
    check_target_sequence(reference["targets"], reference_targets,
                          "the data.py validation loader")
    reference_accuracy, reference_f1 = metrics_from_confusion(reference["confusion"])
    direct_accuracy, direct_f1 = independent_metrics(
        reference["predictions"], reference["targets"])
    for name, through_matrix, direct in (("accuracy", reference_accuracy, direct_accuracy),
                                         ("macro F1", reference_f1, direct_f1)):
        if abs(through_matrix - direct) > METRIC_TOLERANCE:
            raise RuntimeError(
                f"{name} from the confusion matrix is {through_matrix!r} but "
                f"{direct!r} counted directly from the prediction and label "
                "vectors; the metric path is wrong")
    measured_clean = {"loss": reference["loss"], "accuracy": reference_accuracy,
                      "macro_f1": reference_f1}
    print(f"  loss {reference['loss']:.6f} accuracy {reference_accuracy:.6f} "
          f"macro F1 {reference_f1:.6f}")
    print(f"  recomputed without the confusion matrix: accuracy "
          f"{direct_accuracy:.6f} macro F1 {direct_f1:.6f}")
    clean_check = check_clean_against_record(measured_clean, identity, device)
    if clean_check["same_device_as_training"]:
        print(f"  matches the metrics this checkpoint recorded at epoch "
              f"{identity['epoch']} within {CLEAN_TOLERANCE:g}")
    else:
        print(f"  checkpoint was trained on {identity['trained_on_device']} and "
              f"evaluated on {device}; differences from its recorded metrics "
              f"{clean_check['absolute_differences']} are recorded, not enforced")

    conditions = suite.conditions()
    fingerprint_positions = list(range(0, len(observations), FINGERPRINT_STRIDE))
    print(f"\nEvaluating {len(conditions)} conditions "
          f"(1 clean + {len(conditions) - 1} corrupted):")
    results = {}
    fingerprints = {}
    for corruption, severity in conditions:
        condition_started = time.time()
        dataset = suite.dataset(observations, corruption, severity)
        loader = condition_loader(dataset, batch_size=arguments.batch_size,
                                  num_workers=arguments.num_workers)
        record = evaluate_condition(model, loader, device, criterion)
        if record["seen"] != EXPECTED_VALIDATION_SIZE:
            raise RuntimeError(f"{corruption} s{severity} evaluated "
                               f"{record['seen']} samples")
        check_target_sequence(record["targets"], reference_targets,
                              f"{corruption} s{severity}")
        if record["grad_enabled"] is not False:
            raise RuntimeError(f"{corruption} s{severity} ran with gradients enabled")
        accuracy, macro_f1 = metrics_from_confusion(record["confusion"])
        per_class = per_class_from_confusion(record["confusion"])
        direct_accuracy, direct_f1 = independent_metrics(
            record["predictions"], record["targets"])
        for name, through_matrix, direct in (("accuracy", accuracy, direct_accuracy),
                                             ("macro_f1", macro_f1, direct_f1)):
            if abs(through_matrix - direct) > METRIC_TOLERANCE:
                raise RuntimeError(
                    f"{corruption} s{severity} {name}: confusion matrix gives "
                    f"{through_matrix!r}, direct count gives {direct!r}")
        # Recomputed from the dataset, so it describes the inputs rather than
        # anything this model did with them, and is comparable across arms.
        fingerprints[f"{corruption}_s{severity}"] = input_fingerprint(
            dataset, fingerprint_positions)
        record.update({"accuracy": accuracy, "macro_f1": macro_f1,
                       "per_class": per_class})
        results[(corruption, severity)] = record
        print(f"  {corruption:20s} s{severity} | loss {record['loss']:.4f} "
              f"accuracy {accuracy:.4f} macro F1 {macro_f1:.4f} "
              f"| {time.time() - condition_started:.1f}s")

    clean_row = results[(CLEAN_CONDITION, 0)]
    if not torch.equal(clean_row["predictions"], reference["predictions"]):
        mismatches = int((clean_row["predictions"] != reference["predictions"]).sum())
        raise RuntimeError(
            f"the sweep's clean condition disagrees with the data.py validation "
            f"loader on {mismatches} of {len(reference_targets)} images; the "
            "corruption wrapper is altering uncorrupted inputs")
    for name, swept, referenced in (("loss", clean_row["loss"], reference["loss"]),
                                    ("accuracy", clean_row["accuracy"], reference_accuracy),
                                    ("macro_f1", clean_row["macro_f1"], reference_f1)):
        if abs(swept - referenced) > CLEAN_TOLERANCE:
            raise RuntimeError(
                f"sweep clean {name} {swept:.6f} differs from the reference "
                f"{referenced:.6f} by more than {CLEAN_TOLERANCE:g}")
    print("\nSweep clean row: identical predictions to the data.py reference run, "
          "so every drop is measured against this model's own clean score")

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
        "clean_reproduction": clean_check,
        "accepted_baseline_clean_metrics": (ACCEPTED_CLEAN
                                            if identity["is_accepted_baseline"]
                                            else None),
        "evaluation": {
            "suite": suite.name,
            "suite_description": suite.description,
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
            "mode": "eval (BatchNorm running statistics from training, no "
                    "recalibration)",
            "gradients_enabled": any(record["grad_enabled"] for record in results.values()),
            "optimizer": None,
            "seconds": time.time() - started,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "corruption_suite": {
            "module": "src/corruptions.py" if suite.name == "frozen"
                      else "src/corruptions_heldout.py",
            "namespace": VALIDATION_NAMESPACE,
            "base_seed": SEED,
            "severity_levels": list(SEVERITY_LEVELS),
            "specification": {
                name: {"parameter": spec.parameter, "levels": list(spec.levels),
                       "stochastic": spec.stochastic,
                       "severity_in_seed": spec.severity_in_seed}
                for name, spec in suite.table.items()
            },
            # Hashes of the actual input tensors. Equal fingerprints across two
            # arms are what licenses comparing their scores condition by
            # condition; unequal ones mean the comparison is not paired.
            "input_fingerprints": fingerprints,
            "fingerprint_stride": FINGERPRINT_STRIDE,
            "fingerprint_frames": len(fingerprint_positions),
        },
        "interpretation_notes": suite.notes,
        "validation_tracks_per_class": tracks_per_class,
        "integrity": {
            "checkpoint_md5": identity["md5"],
            "is_accepted_baseline": identity["is_accepted_baseline"],
            "clean_metrics_match_checkpoint_record":
                clean_check["same_device_as_training"],
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

    paths = write_outputs(suite, results, conditions, metadata,
                          validation_indices, labels, tracks_per_class, results_dir)
    summary_rows = json.loads(paths["summary_json"].read_text())["conditions"]

    print("\n" + format_table(summary_rows))
    print("\nIntegrity:")
    print(f"  all {len(before)} parameter and buffer tensors identical before and "
          f"after evaluation, including {len(batchnorm_buffers)} BatchNorm buffers")
    print("  no gradient was populated; no optimizer was constructed")
    print(f"  official test loader iterations: {loaders['test'].iterations}, "
          f"test images read through the instrumented dataset: "
          f"{loaders['test'].dataset.reads}")
    print(f"  every condition evaluated exactly {EXPECTED_VALIDATION_SIZE} samples "
          "in the same label order")
    print(f"  {len(fingerprints)} input fingerprints recorded over "
          f"{len(fingerprint_positions)} frames, for cross-arm comparison")

    print("\nWritten:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(f"\nTotal {time.time() - started:.1f}s. No training, augmentation, or "
          "BatchNorm recalibration occurred, and the official test split was "
          "never iterated.")


if __name__ == "__main__":
    main()
