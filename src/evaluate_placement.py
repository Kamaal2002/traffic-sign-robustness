"""Evaluate the frozen baseline under the occlusion-location ablation.

    .venv/bin/python -m src.evaluate_placement --smoke
    .venv/bin/python -m src.evaluate_placement

The frozen corruption suite varies occlusion area and leaves position almost
fixed. This sweep holds area fixed at the sides of severities 1 and 3 and varies
position: centred, flush in a corner, uniform over the valid positions, and
along a ladder of distances from the image centre. Everything else is the
accepted pipeline - the epoch-21 checkpoint, the track-aware validation split,
the mid-grey fill, the same normalisation - so a difference between conditions
is attributable to where the patch sits.

Like the frozen sweep this is a measurement stage: nothing trains, fine-tunes,
augments, or recalibrates BatchNorm, and the official test split is never
iterated. The geometry, the stratum weights and the pass list come from
`occlusion_placement.py`; the inference loop, the integrity checks and the
pinned baseline constants are reused from `evaluate_robustness.py`. This file
adds no experimental parameter of its own, and it writes to its own artifact
directory, leaving the accepted robustness results untouched.

`--smoke` runs the full pass list over a fixed two-track subset to exercise the
mechanics - that every condition runs, places its patch where the protocol says,
sees the same frames in the same order, and leaves the model unchanged. It
reports no accuracy, loss, or any other performance number, and stores none:
the inference path produces them as a side effect, and they are dropped
unexamined so the protocol is frozen before any result from it is seen.
"""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

from .corruptions import VALIDATION_NAMESPACE, build_validation_observations
from .data import IMAGE_SIZE, NUM_CLASSES, SEED, create_data_loaders, read_training_index
from .evaluate_robustness import (
    ACCEPTED_CLEAN,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLEAN_TOLERANCE,
    EXPECTED_EPOCH,
    EXPECTED_VALIDATION_SIZE,
    METRIC_TOLERANCE,
    OCCLUSION_NOTE,
    PROJECT_ROOT,
    SINGLE_TRACK_WARNING,
    build_condition_dataset,
    check_target_sequence,
    condition_loader,
    evaluate_condition,
    independent_metrics,
    instrument_loaders,
    repository_state,
    state_fingerprint,
    verify_checkpoint,
)
from .occlusion_placement import (
    CLEAN_CONDITION,
    IMAGE_CENTRE,
    LADDER_DISTANCES,
    PATCH_SIDES,
    UNIFORM_STRATA,
    PlacedOcclusionDataset,
    evaluation_conditions,
    format_protocol,
    ladder_table,
    stratum_bounds,
    top_left_count,
    uniform_weights,
)
from .train import load_checkpoint, metrics_from_confusion, select_device

RESULTS_DIR = PROJECT_ROOT / "artifacts" / "occlusion_placement"
SMOKE_DIR = RESULTS_DIR / "smoke"

EXPECTED_TRACKS = 178
FRAMES_PER_TRACK = 30

# The pass list follows from the geometry, so it is pinned here as well: a change
# to a policy or to the ladder has to be an explicit edit of both places rather
# than a silent change of cost and of what gets compared. The count includes the
# clean pass, which is one of the conditions: 1 clean and 47 placements.
EXPECTED_PASSES = 48

# Frames audited per condition, pixel by pixel, to confirm the patch really sits
# where the protocol table says. A stride keeps the audit spread across classes
# and across the per-frame uniform draws instead of sitting in one class.
AUDIT_STRIDE = 89

# Tracks in the smoke subset. Taken as the first distinct track keys in
# validation order, which is fixed by the frozen split, so the subset is the
# same on every run without a sampling decision.
SMOKE_TRACKS = 2

# The fill maps to exactly 0 after Normalize(0.5, 0.5), and no unpatched pixel
# can land there: ToTensor produces multiples of 1/255, and 0.5 is not one.
# That makes the zero set of a normalised image exactly its patch.
PATCH_VALUE = 0.0


def validation_track_keys(observations, validation_indices):
    """Track key per validation position, from data.py's own index reader."""
    _, tracks = read_training_index(observations.dataset)
    return [tracks[int(index)] for index in validation_indices]


def smoke_positions(track_keys):
    """Validation positions of the first `SMOKE_TRACKS` distinct tracks."""
    chosen = []
    for key in track_keys:
        if key not in chosen:
            chosen.append(key)
        if len(chosen) == SMOKE_TRACKS:
            break
    selected = [position for position, key in enumerate(track_keys) if key in chosen]
    return chosen, selected


def build_pass_dataset(observations, placement):
    """Dataset for one pass: the frozen clean wrapper, or one placed patch."""
    if placement.policy == CLEAN_CONDITION:
        return build_condition_dataset(observations, CLEAN_CONDITION, 0)
    return PlacedOcclusionDataset(observations, placement)


def realized_placements(placement, dataset_indices):
    """Where this pass puts the patch on each frame, by dataset index."""
    if placement.policy == CLEAN_CONDITION:
        return [None] * len(dataset_indices)
    return [placement.top_left(int(index)) for index in dataset_indices]


def audit_patch_pixels(dataset, placement, placements, positions):
    """Confirm the tensors a pass feeds the model carry the declared patch.

    Reads a strided sample of the pass's own dataset and checks that the zero
    set of each normalised image is exactly the square the protocol table
    assigns to that frame. This is what makes the coordinates written to
    `predictions.csv` a record of what was evaluated rather than a restatement
    of the table.
    """
    side = placement.side
    for position in positions:
        image, _ = dataset[position]
        zeros = image == PATCH_VALUE
        if placement.policy == CLEAN_CONDITION:
            if bool(zeros.any()):
                raise RuntimeError("the clean pass produced a filled region")
            continue
        top, left = placements[position]
        expected = torch.zeros_like(zeros)
        expected[..., top:top + side, left:left + side] = True
        if not torch.equal(zeros, expected):
            raise RuntimeError(
                f"{placement.name}: frame at validation position {position} has "
                f"{int(zeros.sum())} filled pixels, expected the {side}x{side} "
                f"square at ({top}, {left})"
            )
    return len(positions)


def protocol_rows():
    """The frozen pass list, as the rows written to `placements.csv`."""
    rows = []
    for placement in evaluation_conditions():
        rows.append({
            "condition": placement.name,
            "policy": placement.policy,
            "side": "" if placement.side is None else placement.side,
            "top": "" if placement.coordinates is None else placement.coordinates[0],
            "left": "" if placement.coordinates is None else placement.coordinates[1],
            "stratum_row": "" if placement.cell is None else placement.cell[0],
            "stratum_column": "" if placement.cell is None else placement.cell[1],
            "centre_distance": "" if placement.distance is None else placement.distance,
            "weight": placement.weight,
        })
    return rows


def ladder_rows():
    """Which passes supply each rung of the distance ladder."""
    rows = []
    for side, distance, rung, policy in ladder_table():
        rows.append({
            "side": side,
            "centre_distance": distance,
            "placements": " ".join(f"{top}+{left}" for top, left in rung),
            "evaluated_by": policy,
        })
    return rows


def geometry_metadata():
    """The numbers that define the ablation's geometry, for the record."""
    sides = {}
    for side in PATCH_SIDES:
        bounds = stratum_bounds(side)
        weights = uniform_weights(side)
        sides[str(side)] = {
            "valid_top_left_per_axis": top_left_count(side),
            "valid_placements": top_left_count(side) ** 2,
            "strata_per_axis": UNIFORM_STRATA,
            "stratum_bounds": [[low, high] for low, high, _ in bounds],
            "stratum_sizes": [size for _, _, size in bounds],
            "stratum_weights": {f"{row}{column}": weight
                                for (row, column), weight in weights.items()},
            "weight_sum": sum(weights.values()),
            "equal_weights_unbiased": len({size for _, _, size in bounds}) == 1,
            "ladder_distances": list(LADDER_DISTANCES[side]),
        }
    return {
        "image_size": IMAGE_SIZE,
        "image_centre": IMAGE_CENTRE,
        "distance_metric": "Chebyshev distance of the patch centre from the image centre",
        "fill": "mid grey before normalisation, which Normalize(0.5, 0.5) maps to 0",
        "sides": sides,
        "passes": len(evaluation_conditions()),
    }


# Carried into the saved metadata so the location result cannot travel without
# the limit on what "corner" means here.
PLACEMENT_NOTE = (
    "Position is varied within the model's input frame, not relative to the "
    "sign. GTSRB crops are centred on the sign and its annotated box covers a "
    "median 0.571 of the resized 64x64 frame, so a flush corner patch still "
    "overlaps that box on 5,257 to 5,313 of the 5,340 validation frames at "
    "side 14, and on all 5,340 at side 25, covering a median 0.18 to 0.21 of "
    "the patch at side 14 and 0.46 to 0.49 at side 25. Read a corner condition "
    "as a peripheral patch, not as an off-sign patch, and read the "
    "centre-minus-corner contrast as the cost of moving a patch towards the "
    "middle of the frame rather than as the cost of covering the sign itself."
)

PAIRING_NOTE = (
    "Every condition is evaluated on all 5,340 validation frames, so the "
    "contrasts are paired at frame level and differ only in where the patch "
    "sits. Frames are not independent: each physical sign contributes 30 "
    "near-duplicate frames. Placements are averaged within a frame, frames "
    "within a track, and the resulting per-track rates are what the paired "
    "contrasts and the bootstrap over the 178 validation tracks use."
)


def write_outputs(results, conditions, metadata, validation_indices, labels,
                  track_keys, placements, results_dir):
    """Write the per-pass predictions, the pass summary, and the protocol."""
    results_dir.mkdir(parents=True, exist_ok=True)

    clean = results[CLEAN_CONDITION]
    summary_rows = []
    for placement in conditions:
        record = results[placement.name]
        summary_rows.append({
            "condition": placement.name,
            "policy": placement.policy,
            "side": "" if placement.side is None else placement.side,
            "centre_distance": "" if placement.distance is None else placement.distance,
            "weight": placement.weight,
            "loss": record["loss"],
            "accuracy": record["accuracy"],
            "macro_f1": record["macro_f1"],
            "accuracy_drop": clean["accuracy"] - record["accuracy"],
            "macro_f1_drop": clean["macro_f1"] - record["macro_f1"],
            "samples": record["seen"],
        })

    summary_path = results_dir / "summary.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    protocol_path = results_dir / "placements.csv"
    with open(protocol_path, "w", newline="") as handle:
        rows = protocol_rows()
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ladder_path = results_dir / "ladder.csv"
    with open(ladder_path, "w", newline="") as handle:
        rows = ladder_rows()
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Per-frame rows carry the realised placement as well as the prediction, so
    # the analysis can weight, pair and aggregate without re-deriving geometry,
    # and the uniform draws are recoverable from the artifact alone.
    predictions_path = results_dir / "predictions.csv"
    with open(predictions_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "policy", "side", "top", "left", "weight",
                         "validation_position", "dataset_index", "track", "label",
                         "prediction"])
        for placement in conditions:
            predicted = results[placement.name]["predictions"].tolist()
            coordinates = placements[placement.name]
            for position, value in enumerate(predicted):
                index = int(validation_indices[position])
                top, left = ("", "") if coordinates[position] is None else coordinates[position]
                writer.writerow([
                    placement.name, placement.policy,
                    "" if placement.side is None else placement.side,
                    top, left, placement.weight, position, index,
                    track_keys[position], int(labels[index]), value,
                ])

    confusion_path = results_dir / "confusion_matrices.npz"
    np.savez_compressed(confusion_path, **{
        placement.name: results[placement.name]["confusion"].numpy()
        for placement in conditions
    })

    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "summary_csv": summary_path,
        "placements_csv": protocol_path,
        "ladder_csv": ladder_path,
        "predictions_csv": predictions_path,
        "confusion_npz": confusion_path,
        "metadata_json": results_dir / "metadata.json",
    }


def run_smoke(model, observations, validation_indices, labels, track_keys, device,
              criterion, arguments, results_dir):
    """Exercise every pass on a fixed two-track subset, without scoring anything.

    What is checked is mechanical: each pass runs over the expected frames, puts
    its patch where the protocol table says, reports the same label sequence,
    and leaves the model untouched. Running the real inference path also
    produces predictions, a loss and a confusion matrix; they are dropped as
    soon as the pass returns, so nothing here compares a prediction with a
    label and no performance number is reported, stored, or printed.
    """
    tracks, positions = smoke_positions(track_keys)
    if len(positions) != SMOKE_TRACKS * FRAMES_PER_TRACK:
        raise RuntimeError(f"smoke subset holds {len(positions)} frames, expected "
                           f"{SMOKE_TRACKS * FRAMES_PER_TRACK}")
    subset = Subset(observations, positions)
    subset_indices = [int(validation_indices[position]) for position in positions]
    reference_targets = torch.tensor(
        [int(labels[index]) for index in subset_indices], dtype=torch.int64)

    conditions = evaluation_conditions()
    print(f"\nSmoke test on {len(positions)} frames from {SMOKE_TRACKS} tracks "
          f"({', '.join(tracks)}), {len(conditions)} passes:")
    before = state_fingerprint(model)
    audited = 0
    distinct = {}
    started = time.time()
    for placement in conditions:
        dataset = build_pass_dataset(subset, placement)
        coordinates = realized_placements(placement, subset_indices)
        audited += audit_patch_pixels(dataset, placement, coordinates,
                                      range(len(positions)))
        record = evaluate_condition(
            model, condition_loader(dataset, batch_size=arguments.batch_size,
                                    num_workers=arguments.num_workers),
            device, criterion)
        # Dropped before anything else in this function can read them: the smoke
        # test establishes that the protocol runs, not how the model scores
        # under it.
        for key in ("loss", "confusion", "predictions"):
            record.pop(key)
        if record["seen"] != len(positions):
            raise RuntimeError(f"{placement.name} evaluated {record['seen']} frames")
        if record["grad_enabled"] is not False:
            raise RuntimeError(f"{placement.name} ran with gradients enabled")
        check_target_sequence(record["targets"], reference_targets, placement.name)
        distinct[placement.name] = len(set(coordinates))
        del record

    after = state_fingerprint(model)
    changed = sorted(name for name in before if before[name] != after[name])
    if changed:
        raise RuntimeError(f"model state changed during the smoke test: {changed}")
    gradients = [name for name, parameter in model.named_parameters()
                 if parameter.grad is not None]
    if gradients:
        raise RuntimeError(f"gradients were populated: {gradients}")

    per_frame = sorted(name for name, count in distinct.items() if count > 1)
    fixed = sorted(name for name, count in distinct.items() if count == 1)
    print(f"  {len(conditions)} passes x {len(positions)} frames; "
          f"{audited} frames audited pixel by pixel against the protocol table")
    print(f"  {len(fixed)} passes place the patch at one fixed coordinate, "
          f"{len(per_frame)} vary it per frame (the {UNIFORM_STRATA}x"
          f"{UNIFORM_STRATA} uniform cells per side)")
    print("  identical label sequence in every pass; model parameters and "
          "BatchNorm buffers unchanged; no gradient populated")

    results_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "purpose": "mechanical check of the placement protocol; no performance "
                   "metric is reported, stored, or inspected",
        "tracks": tracks,
        "frames": len(positions),
        "passes": len(conditions),
        "frames_audited_pixelwise": audited,
        "fixed_coordinate_passes": fixed,
        "per_frame_coordinate_passes": per_frame,
        "label_sequence_identical_across_passes": True,
        "model_state_unchanged": True,
        "gradients_populated": False,
        "seconds": time.time() - started,
        "geometry": geometry_metadata(),
        "repository": repository_state(),
    }
    path = results_dir / "smoke.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"  written: {path}")


def run_sweep(model, checkpoint, digest, observations, validation_indices, labels,
              track_keys, loaders, reference, reference_metrics, device, criterion,
              arguments, results_dir, started):
    """The full ablation: every pass over all 5,340 validation frames."""
    reference_accuracy, reference_f1 = reference_metrics
    reference_targets = reference["targets"]
    dataset_indices = [int(index) for index in validation_indices]
    conditions = evaluation_conditions()
    if len(conditions) != EXPECTED_PASSES:
        raise RuntimeError(f"the protocol defines {len(conditions)} passes, "
                           f"expected {EXPECTED_PASSES}")

    print(f"\nEvaluating {len(conditions)} passes "
          f"(1 clean + {len(conditions) - 1} placements):")
    before = state_fingerprint(model)
    results = {}
    placements = {}
    audited = 0
    for placement in conditions:
        pass_started = time.time()
        dataset = build_pass_dataset(observations, placement)
        coordinates = realized_placements(placement, dataset_indices)
        audited += audit_patch_pixels(
            dataset, placement, coordinates,
            range(0, len(dataset_indices), AUDIT_STRIDE))
        record = evaluate_condition(
            model, condition_loader(dataset, batch_size=arguments.batch_size,
                                    num_workers=arguments.num_workers),
            device, criterion)
        if record["seen"] != EXPECTED_VALIDATION_SIZE:
            raise RuntimeError(f"{placement.name} evaluated {record['seen']} frames")
        if record["grad_enabled"] is not False:
            raise RuntimeError(f"{placement.name} ran with gradients enabled")
        check_target_sequence(record["targets"], reference_targets, placement.name)
        accuracy, macro_f1 = metrics_from_confusion(record["confusion"])
        direct_accuracy, direct_f1 = independent_metrics(
            record["predictions"], record["targets"])
        for name, through_matrix, direct in (("accuracy", accuracy, direct_accuracy),
                                             ("macro_f1", macro_f1, direct_f1)):
            if abs(through_matrix - direct) > METRIC_TOLERANCE:
                raise RuntimeError(
                    f"{placement.name} {name}: confusion matrix gives "
                    f"{through_matrix!r}, direct count gives {direct!r}")
        record.update({"accuracy": accuracy, "macro_f1": macro_f1})
        results[placement.name] = record
        placements[placement.name] = coordinates
        print(f"  {placement.name:28s} | loss {record['loss']:.4f} "
              f"accuracy {accuracy:.4f} macro F1 {macro_f1:.4f} "
              f"| {time.time() - pass_started:.1f}s")

    # Same check the frozen sweep makes: the ablation's clean pass has to give
    # the accepted baseline's predictions exactly, which is what ties every
    # placement contrast to a known reference point.
    clean = results[CLEAN_CONDITION]
    if not torch.equal(clean["predictions"], reference["predictions"]):
        mismatches = int((clean["predictions"] != reference["predictions"]).sum())
        raise RuntimeError(
            f"the ablation's clean pass disagrees with the data.py validation "
            f"loader on {mismatches} frames")
    for name, swept, referenced in (("loss", clean["loss"], reference["loss"]),
                                    ("accuracy", clean["accuracy"], reference_accuracy),
                                    ("macro_f1", clean["macro_f1"], reference_f1)):
        if abs(swept - referenced) > CLEAN_TOLERANCE:
            raise RuntimeError(f"clean {name} {swept:.6f} differs from the "
                               f"reference {referenced:.6f}")

    after = state_fingerprint(model)
    changed = sorted(name for name in before if before[name] != after[name])
    if changed:
        raise RuntimeError(f"model state changed during evaluation: {changed}")
    gradients = [name for name, parameter in model.named_parameters()
                 if parameter.grad is not None]
    if gradients:
        raise RuntimeError(f"gradients were populated: {gradients}")

    metadata = {
        "experiment": "occlusion location ablation",
        "checkpoint": {
            "path": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "md5": digest,
            "epoch": checkpoint["epoch"],
            "recorded_validation_metrics": checkpoint["validation_metrics"],
            "model_config": checkpoint["model_config"],
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
            "validation_tracks": len(set(track_keys)),
            "num_classes": NUM_CLASSES,
            "passes": len(conditions),
            "forward_passes": len(conditions) * EXPECTED_VALIDATION_SIZE,
            # The reproduction through data.py's own validation loader is a
            # check on the checkpoint and the preprocessing path, not one of
            # the protocol's passes, so it is counted apart from them.
            "reference_passes": 1,
            "model_passes": (len(conditions) + 1) * EXPECTED_VALIDATION_SIZE,
            "batch_size": arguments.batch_size,
            "num_workers": arguments.num_workers,
            "device": str(device),
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mode": "eval (BatchNorm running statistics from clean training, "
                    "no recalibration)",
            "gradients_enabled": any(record["grad_enabled"] for record in results.values()),
            "optimizer": None,
            "seconds": time.time() - started,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "geometry": geometry_metadata(),
        "placement_seeds": {
            "namespace": VALIDATION_NAMESPACE,
            "base_seed": SEED,
            "scheme": "blake2b over base seed | namespace | placement | side | "
                      "cell | dataset index, split into two 32-bit halves and "
                      "reduced into the cell's bounds",
        },
        "interpretation_notes": {
            "placement": PLACEMENT_NOTE,
            "pairing": PAIRING_NOTE,
            "frozen_occlusion": OCCLUSION_NOTE,
            "per_class_support": SINGLE_TRACK_WARNING,
        },
        "integrity": {
            "checkpoint_md5_matches": True,
            "clean_pass_matches_reference": True,
            "metrics_cross_checked_without_confusion_matrix": True,
            "frames_audited_pixelwise": audited,
            "model_state_unchanged": True,
            "state_tensors_checked": len(before),
            "test_loader_iterations": loaders["test"].iterations,
            "test_dataset_reads": loaders["test"].dataset.reads,
            "train_loader_iterations": loaders["train"].iterations,
            "train_dataset_reads": loaders["train"].dataset.reads,
            "validation_loader_iterations": loaders["validation"].iterations,
            "accepted_robustness_artifacts_modified": False,
        },
        "repository": repository_state(),
    }

    paths = write_outputs(results, conditions, metadata, validation_indices, labels,
                          track_keys, placements, results_dir)
    print("\nIntegrity:")
    print(f"  clean pass identical to the data.py reference run")
    print(f"  {audited} frames audited pixel by pixel against the protocol table")
    print(f"  all {len(before)} parameter and buffer tensors unchanged; "
          "no gradient populated; no optimizer constructed")
    print(f"  official test loader iterations: {loaders['test'].iterations}, "
          f"test images read: {loaders['test'].dataset.reads}")
    print("\nWritten:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="run every pass on a fixed two-track subset and "
                             "check the mechanics, computing no metric")
    arguments = parser.parse_args()

    device = select_device(arguments.device)
    started = time.time()

    print("Frozen placement protocol (src/occlusion_placement.py):")
    print(format_protocol())

    print("\nCheckpoint:")
    digest = verify_checkpoint()
    model, checkpoint = load_checkpoint(CHECKPOINT_PATH, device=device)
    if checkpoint["epoch"] != EXPECTED_EPOCH:
        raise RuntimeError(f"checkpoint reports epoch {checkpoint['epoch']}, "
                           f"expected {EXPECTED_EPOCH}")
    print(f"  md5 {digest} matches the accepted baseline, epoch {checkpoint['epoch']}")
    print(f"  device {device}, torch {torch.__version__}")
    model.eval()
    assert not model.training
    criterion = nn.CrossEntropyLoss()

    observations, validation_indices, labels = build_validation_observations()
    if len(observations) != EXPECTED_VALIDATION_SIZE:
        raise RuntimeError(f"validation split holds {len(observations)} images, "
                           f"expected {EXPECTED_VALIDATION_SIZE}")
    track_keys = validation_track_keys(observations, validation_indices)
    if len(set(track_keys)) != EXPECTED_TRACKS:
        raise RuntimeError(f"validation split holds {len(set(track_keys))} tracks, "
                           f"expected {EXPECTED_TRACKS}")
    print(f"\nValidation split: {len(observations)} frames, "
          f"{len(set(track_keys))} tracks")

    if arguments.smoke:
        results_dir = Path(arguments.output_dir) if arguments.output_dir else SMOKE_DIR
        run_smoke(model, observations, validation_indices, labels, track_keys,
                  device, criterion, arguments, results_dir)
        print(f"\nTotal {time.time() - started:.1f}s. Mechanical check only: no "
              "accuracy, loss, or other performance metric was reported, stored, "
              "or inspected.")
        return

    results_dir = Path(arguments.output_dir) if arguments.output_dir else RESULTS_DIR
    loaders, _ = create_data_loaders(batch_size=arguments.batch_size, download=False)
    loaders = instrument_loaders(loaders)
    print("\nClean reproduction through data.py's validation loader:")
    reference = evaluate_condition(model, loaders["validation"], device, criterion)
    reference_targets = torch.tensor(
        [int(labels[int(index)]) for index in validation_indices], dtype=torch.int64)
    check_target_sequence(reference["targets"], reference_targets,
                          "the data.py validation loader")
    reference_accuracy, reference_f1 = metrics_from_confusion(reference["confusion"])
    for key, measured in (("loss", reference["loss"]),
                          ("accuracy", reference_accuracy),
                          ("macro_f1", reference_f1)):
        difference = abs(measured - ACCEPTED_CLEAN[key])
        if difference > CLEAN_TOLERANCE:
            raise RuntimeError(
                f"clean {key} is {measured:.6f}, accepted baseline is "
                f"{ACCEPTED_CLEAN[key]:.6f} (difference {difference:.2e})")
    print(f"  loss {reference['loss']:.6f} accuracy {reference_accuracy:.6f} "
          f"macro F1 {reference_f1:.6f}, matching the accepted baseline within "
          f"{CLEAN_TOLERANCE:g}")

    run_sweep(model, checkpoint, digest, observations, validation_indices, labels,
              track_keys, loaders, reference, (reference_accuracy, reference_f1),
              device, criterion, arguments, results_dir, started)
    print(f"\nTotal {time.time() - started:.1f}s. No training, augmentation, or "
          "BatchNorm recalibration occurred, and the official test split was "
          "never iterated.")


if __name__ == "__main__":
    main()
