"""Structural verification of the occlusion-location protocol.

    .venv/bin/python -m src.verify_placement

These checks are about the placement geometry, the stratum weighting, the
per-frame seeding, the preprocessing path, the pairing across passes, and the
track-level estimators. No checkpoint is loaded, no prediction is computed, and
the official test split is never opened, so nothing here lets model behaviour
feed back into how the conditions are defined. The statistical path is checked
against synthetic outcomes with a known answer for the same reason.

The frozen suite is verified separately in `verify_corruptions.py`; this module
only confirms that the ablation agrees with it where the two meet.
"""

import contextlib
import csv
import io
import json
import subprocess
import sys
import tempfile
from hashlib import blake2b
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import GTSRB

from .analyze_placement import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    POLICIES,
    bootstrap_mean,
    frame_rates,
    ladder_assignment,
    ladder_frame_rates,
    load_predictions,
    paired_contrast,
    placement_noise,
    track_rates,
)
from .analyze_placement import main as analyse
from .corruptions import (
    CORRUPTIONS,
    OCCLUSION_FILL,
    OCCLUSION_LEVELS,
    SEVERITY_LEVELS,
    VALIDATION_NAMESPACE,
    build_validation_observations,
    occlusion_side,
    sample_seed,
    split_preprocessing,
    stable_index,
)
from .data import (
    IMAGE_SIZE,
    NUM_CLASSES,
    SEED,
    build_preprocessing,
    read_training_index,
    split_tracks,
)
from .evaluate_robustness import (
    CLEAN_CONDITION as ACCEPTED_CLEAN_CONDITION,
    PROJECT_ROOT,
    file_md5,
)
from .evaluate_placement import (
    EXPECTED_PASSES,
    EXPECTED_TRACKS,
    FRAMES_PER_TRACK,
    PATCH_VALUE,
    RESULTS_DIR,
    ladder_rows,
    protocol_rows,
    write_outputs,
)
from .occlusion_placement import (
    CLEAN_CONDITION,
    IMAGE_CENTRE,
    LADDER_DISTANCES,
    PATCH_SIDES,
    PLACEMENT_CONDITION,
    UNIFORM_STRATA,
    PlacedOcclusionDataset,
    centre_distance,
    centre_placements,
    corner_placements,
    evaluation_conditions,
    ladder_placements,
    ladder_table,
    place_patch,
    placement_seed,
    stratum_bounds,
    top_left_count,
    uniform_placement,
    uniform_weights,
)

# Recorded once from the implementation and asserted on every run. A change here
# means the placement seeding rule itself changed, so previously drawn uniform
# positions are no longer reproducible and the sweep would have to be rerun.
EXPECTED_PLACEMENT_SEED = 968930985289931410
EXPECTED_PLACEMENT_SEED_ARGS = (14, (1, 2), 17)

# The accepted robustness artifacts, pinned so this ablation fails here if it
# ever writes into the frozen results. Only the deterministic result files are
# pinned: `metadata.json` and `summary.json` embed a timestamp and a duration,
# so their digests would differ on an honest rerun of the same sweep.
ACCEPTED_ARTIFACTS = {
    "artifacts/baseline/best_checkpoint.pt": "7c6ea3f9773d7a644235463d6a02bf0f",
    "artifacts/robustness/baseline/predictions.csv": "64d022b4bf1dd591c2fadc985f08e04e",
    "artifacts/robustness/baseline/summary.csv": "3ea9477b1f5f3bbe5a5ac82ba636cd9e",
    "artifacts/robustness/baseline/per_class_metrics.csv":
        "f5afa74073c3f41bfd5335e7901ae55a",
    "artifacts/robustness/baseline/confusion_matrices.npz":
        "c5393d9dcfab9d1c58f495e45e6bc015",
}

# The frozen modules, pinned by committed blob hash. The working tree is
# compared with HEAD separately, so between them a committed edit and an
# uncommitted one both fail here rather than quietly changing what the accepted
# pipeline means.
FROZEN_SOURCES = {
    "src/corruptions.py": "5bd9b4c496b5e2104d3ca9688b13ad6b5d69b194",
    "src/data.py": "1aeea9393b28eb4fdff088516ac710561ac29ee5",
    "src/model.py": "cfca1ea330235cb870a4286ffb7af03720d88cb1",
    "src/train.py": "7a92b70c900c1acccca2273895dae8c0605630c2",
    "src/evaluate_robustness.py": "b9d524e3ef88ed7cb709147649eb05774d19b835",
}

# Frames used where a check needs many independent per-frame draws.
DRAW_SAMPLE = 5340

# The synthetic fixture labels every frame with one class and indexes the frames
# away from zero, so a check that silently used the position in place of the
# dataset index would change the draws it is comparing against.
SYNTHETIC_LABEL = 1
SYNTHETIC_OFFSET = 1000

# Everything `analyze_placement.main` is expected to leave behind.
EXPECTED_ANALYSIS_FILES = [
    "ladder_contrasts.csv",
    "ladder_rates.csv",
    "ladder_track_rates.csv",
    "paired_contrasts.csv",
    "placement_noise.csv",
    "policy_rates.csv",
    "summary.json",
    "track_rates.csv",
]

# Measured from the GTSRB ground-truth annotations by `check_roi_overlap` and
# pinned so the interpretation note in `evaluate_placement.py` cannot drift from
# what the data says. The sign's annotated box covers this share of the resized
# frame, and a flush corner patch misses it on this many validation frames.
EXPECTED_ROI_MEDIAN_AREA = 0.5714
EXPECTED_CORNER_MISSES = {14: (27, 83), 25: (0, 0)}
EXPECTED_CORNER_OVERLAP = {14: (0.182, 0.214), 25: (0.462, 0.488)}
ROI_ANNOTATIONS = Path("gtsrb/GTSRB/Training")


def independent_placement_seed(side, cell, dataset_index, base_seed=SEED,
                               namespace=VALIDATION_NAMESPACE):
    """Re-derive a placement seed from the documented rule, not the helper.

    Written out here so that editing the payload in `occlusion_placement.py`
    fails this verifier instead of silently redefining every uniform draw.
    """
    payload = (f"{base_seed}|{namespace}|placement|{side}|"
               f"{cell[0]},{cell[1]}|{dataset_index}").encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big") >> 1


def synthetic_outcome(top, left, count):
    """A deterministic 0/1 outcome for the weighting checks.

    The outcome fires when the placement sits in the middle half of the valid
    range, which is the shape the ablation is looking for: an error indicator
    that depends on how central the patch is. A weighting that misstates the
    strata then gives a visibly wrong answer, which a function flat across the
    strata would hide.
    """
    middle = (count - 1) / 2
    return float(max(abs(top - middle), abs(left - middle)) <= (count - 1) / 4)


def synthetic_error(dataset_index, top, left, side):
    """A deterministic error indicator that depends on where the patch sits.

    Central placements always fail and peripheral ones fail on a fixed minority
    of frames, so the policy contrasts are large and signed. A rule whose rate
    came out the same under every policy would let a pairing or sign error pass
    the aggregation checks unnoticed.
    """
    central = centre_distance((top, left), side) <= (IMAGE_SIZE - side) / 4
    return int(central or dataset_index % 11 == 0)


def check_sides():
    """The ablation's patch sides must still be the frozen severities 1 and 3."""
    spec = [occlusion_side(fraction, IMAGE_SIZE, IMAGE_SIZE)
            for fraction in OCCLUSION_LEVELS]
    assert PATCH_SIDES == (spec[0], spec[2]), (PATCH_SIDES, spec)
    print(f"  patch sides {PATCH_SIDES} are the frozen occlusion severities 1 and 3 "
          f"at {IMAGE_SIZE}x{IMAGE_SIZE} (full severity table {spec})")
    print(f"  image centre {IMAGE_CENTRE} for a {IMAGE_SIZE}px frame")


def check_geometry():
    """Valid ranges, centre and corner policies, and attainable distances."""
    for side in PATCH_SIDES:
        count = top_left_count(side)
        assert count == IMAGE_SIZE - side + 1
        grid = [(y, x) for y in range(count) for x in range(count)]
        distances = {centre_distance(placement, side) for placement in grid}

        centres = centre_placements(side)
        centre_values = {centre_distance(placement, side) for placement in centres}
        assert len(centre_values) == 1, centre_values
        assert centre_values.pop() == min(distances)
        expected_centres = 1 if (IMAGE_SIZE - side) % 2 == 0 else 4
        assert len(centres) == expected_centres, centres

        corners = corner_placements(side)
        assert len(set(corners)) == 4
        for top, left in corners:
            assert centre_distance((top, left), side) == max(distances)
            # Each flush patch has to lie inside one 32x32 quadrant, which is
            # what makes "corner" a region rather than a position near a border.
            for value in (top, left):
                assert value + side <= IMAGE_SIZE // 2 or value >= IMAGE_SIZE // 2

        diagonal = {centre_distance((t, t), side) for t in range(count)}
        rungs = LADDER_DISTANCES[side]
        assert set(rungs) <= diagonal, (rungs, sorted(diagonal))
        assert max(rungs) == max(distances)
        assert min(rungs) == min(distances)
        assert list(rungs) == sorted(rungs)
        assert ladder_placements(side, min(rungs)) == centres
        assert ladder_placements(side, max(rungs)) == corners

        print(f"  side {side}: {count}x{count} = {count ** 2} valid placements, "
              f"distances {min(distances):g} to {max(distances):g}, "
              f"{len(centres)} centre and 4 corner placements")
        print(f"    ladder {[f'{value:g}' for value in rungs]} are attainable "
              f"diagonal distances, ends coincide with the centre and corner sets")

    # Positions that do not exist must be refused rather than rounded into
    # existence. Every half-pixel distance from minus the frame width to plus
    # the frame width is tried, so this covers all three refusal modes: a
    # distance no integer coordinate can realise, one that would push the patch
    # out of the frame, and a negative one that would otherwise be mirrored into
    # a valid position at its magnitude.
    for side in PATCH_SIDES:
        count = top_left_count(side)
        attainable = {centre_distance((t, t), side) for t in range(count)}
        accepted = set()
        for step in range(-2 * IMAGE_SIZE, 2 * IMAGE_SIZE + 1):
            distance = step / 2
            try:
                rung = ladder_placements(side, distance)
            except ValueError:
                continue
            accepted.add(distance)
            for placement in rung:
                for value in placement:
                    assert 0 <= value <= IMAGE_SIZE - side, (side, distance, rung)
                assert centre_distance(placement, side) == distance
        assert accepted == attainable, (side, sorted(attainable ^ accepted))
        print(f"  side {side}: of the {4 * IMAGE_SIZE + 1} half-pixel distances "
              f"from -{IMAGE_SIZE} to {IMAGE_SIZE}, exactly the {len(attainable)} "
              f"attainable ones are accepted (max {max(attainable):g}) and the rest "
              "refused, including negative requests that would otherwise mirror")

    for row in protocol_rows():
        if row["side"] == "":
            continue
        if row["top"] == "":
            continue
        assert 0 <= row["top"] <= IMAGE_SIZE - row["side"], row
        assert 0 <= row["left"] <= IMAGE_SIZE - row["side"], row
    print(f"  every fixed placement in the protocol table lies inside the frame")


def check_weighting():
    """The stratified estimator must reproduce the uniform mean exactly."""
    for side in PATCH_SIDES:
        bounds = stratum_bounds(side)
        count = top_left_count(side)
        sizes = [size for _, _, size in bounds]
        assert sum(sizes) == count
        assert [low for low, _, _ in bounds][0] == 0
        assert [high for _, high, _ in bounds][-1] == count - 1
        for (low, high, size), (next_low, _, _) in zip(bounds, bounds[1:]):
            assert high + 1 == next_low, bounds

        weights = uniform_weights(side)
        assert abs(sum(weights.values()) - 1.0) < 1e-12
        for (row, column), weight in weights.items():
            assert abs(weight - sizes[row] * sizes[column] / count ** 2) < 1e-15

        # Exact check, not a simulation: with the cell means known, the weighted
        # sum has to equal the mean over all valid placements, and equal weights
        # have to equal it only when the strata are equally sized.
        grid = np.array([[synthetic_outcome(y, x, count) for x in range(count)]
                         for y in range(count)])
        exact = grid.mean()
        weighted = 0.0
        unweighted = 0.0
        for (row, column), weight in weights.items():
            low_y, high_y, _ = bounds[row]
            low_x, high_x, _ = bounds[column]
            cell_mean = grid[low_y:high_y + 1, low_x:high_x + 1].mean()
            weighted += weight * cell_mean
            unweighted += cell_mean / len(weights)
        assert abs(weighted - exact) < 1e-12, (weighted, exact)
        equal_sized = len(set(sizes)) == 1
        gap = abs(unweighted - exact)
        assert (gap < 1e-12) == equal_sized, (side, gap, equal_sized)
        if not equal_sized:
            assert gap > 1e-3, gap
        print(f"  side {side}: strata {sizes}, weights "
              f"{sorted({round(w, 6) for w in weights.values()})}, sum 1")
        print(f"    stratum weights reproduce the mean over all {count ** 2} "
              f"placements exactly ({weighted:.12f} vs {exact:.12f})")
        print(f"    equal weights give {unweighted:.12f}, "
              + ("identical because the strata are equally sized"
                 if equal_sized else f"off by {gap:.6f} because they are not"))

        # The direction of that error is a claim the documentation makes, so
        # check it rather than restating it: equal weighting moves weight onto
        # the cells the exact weighting holds lightest, which here are the
        # central ones. The worst case over all possible cell means is the
        # total weight displaced.
        displaced = sum(max(1 / len(weights) - weight, 0.0)
                        for weight in weights.values())
        central = weights[(UNIFORM_STRATA // 2, UNIFORM_STRATA // 2)]
        outer = weights[(0, 0)]
        if equal_sized:
            assert abs(central - 1 / len(weights)) < 1e-15
            assert displaced < 1e-15
        else:
            assert central < 1 / len(weights) < outer, (central, outer)
            assert gap <= displaced + 1e-12, (gap, displaced)
            print(f"    equal weighting shifts weight towards central placements: "
                  f"the central cell gets {1 / len(weights):.6f} against its true "
                  f"{central:.6f}, the lowest-coordinate cell gets the same "
                  f"{1 / len(weights):.6f} against its true {outer:.6f}; the largest "
                  f"absolute error any outcome could produce is {displaced:.4f}")

        # Now with the draws the protocol actually makes: averaging the
        # estimator over frames has to converge on the same exact mean, which is
        # what makes the nine placements an estimate of uniform sampling rather
        # than of nine fixed positions.
        estimates = np.zeros(DRAW_SAMPLE)
        equal_estimates = np.zeros(DRAW_SAMPLE)
        for index in range(DRAW_SAMPLE):
            for cell, weight in weights.items():
                outcome = synthetic_outcome(*uniform_placement(side, cell, index),
                                            count)
                estimates[index] += weight * outcome
                equal_estimates[index] += outcome / len(weights)
        error = abs(estimates.mean() - exact)
        tolerance = 4 * estimates.std(ddof=1) / np.sqrt(DRAW_SAMPLE)
        assert error < tolerance, (side, error, tolerance)
        print(f"    over {DRAW_SAMPLE} seeded frames the weighted estimator gives "
              f"{estimates.mean():.6f} (exact {exact:.6f}, |error| {error:.6f} "
              f"within the {tolerance:.6f} sampling tolerance)")
        # The equal-weight estimator converges too, but on the unweighted mean
        # of the cell means rather than on the uniform mean. That is the bias,
        # and it does not shrink with more frames.
        equal_error = abs(equal_estimates.mean() - unweighted)
        assert equal_error < tolerance, (side, equal_error, tolerance)
        print(f"    the equal-weight estimator converges on {unweighted:.6f} "
              f"instead, measured {equal_estimates.mean():.6f}"
              + ("" if not equal_sized else " (the same value for this side)"))


def check_draws():
    """Uniform draws must stay in their stratum and cover it evenly."""
    for side in PATCH_SIDES:
        bounds = stratum_bounds(side)
        diagonal = 0
        for cell in sorted(uniform_weights(side)):
            low_y, high_y, size_y = bounds[cell[0]]
            low_x, high_x, size_x = bounds[cell[1]]
            draws = np.array([uniform_placement(side, cell, index)
                              for index in range(DRAW_SAMPLE)])
            assert draws[:, 0].min() == low_y and draws[:, 0].max() == high_y
            assert draws[:, 1].min() == low_x and draws[:, 1].max() == high_x
            for axis, (low, size) in enumerate(((low_y, size_y), (low_x, size_x))):
                counts = np.bincount(draws[:, axis] - low, minlength=size)
                expected = DRAW_SAMPLE / size
                deviation = np.abs(counts - expected).max() / np.sqrt(expected)
                assert deviation < 6, (side, cell, axis, deviation)
            diagonal += int(((draws[:, 0] - low_y) == (draws[:, 1] - low_x)).sum())
        cells = len(uniform_weights(side))
        share = diagonal / (cells * DRAW_SAMPLE)
        # Both axes read different halves of the digest, so a cell's two
        # coordinates agree about as often as chance allows. A shared phase
        # would put every draw on the cell diagonal and show up here.
        assert abs(share - 1 / bounds[0][2]) < 0.02, share
        print(f"  side {side}: all {cells} cells covered end to end over "
              f"{DRAW_SAMPLE} frames, per-coordinate frequencies within 6 standard "
              f"errors on both axes, coordinates equal on {share:.3f} of draws")


def check_seeds():
    """Placement seeds must be pinned, reproducible, and disjoint from the suite's."""
    side, cell, index = EXPECTED_PLACEMENT_SEED_ARGS
    value = placement_seed(side, cell, index)
    assert value == EXPECTED_PLACEMENT_SEED, (value, EXPECTED_PLACEMENT_SEED)
    assert value == independent_placement_seed(side, cell, index)
    assert 0 <= value < 2 ** 63
    print(f"  placement_seed({side}, {cell}, {index}) = {value}, matching the "
          "pinned value and an independent derivation of the documented payload")

    seeds = {}
    for patch in PATCH_SIDES:
        for key in sorted(uniform_weights(patch)):
            for dataset_index in range(200):
                seeds[(patch, key, dataset_index)] = placement_seed(
                    patch, key, dataset_index)
    assert len(set(seeds.values())) == len(seeds)
    print(f"  {len(seeds)} seeds over both sides, all {UNIFORM_STRATA}x"
          f"{UNIFORM_STRATA} cells and 200 frames are pairwise distinct")

    suite = {sample_seed(VALIDATION_NAMESPACE, name, severity, dataset_index)
             for name in CORRUPTIONS for severity in SEVERITY_LEVELS
             for dataset_index in range(200)}
    assert not suite & set(seeds.values())
    print(f"  disjoint from the {len(suite)} frozen corruption seeds over the same "
          f"frames; the payload carries '{PLACEMENT_CONDITION}' and the namespace "
          f"'{VALIDATION_NAMESPACE}'")

    repeated = [uniform_placement(25, (2, 1), 4242) for _ in range(5)]
    assert len(set(repeated)) == 1, repeated
    print(f"  repeated calls return the same placement {repeated[0]}, so a frame's "
          "position does not depend on call order or batching")


def check_preprocessing(validation):
    """The placed path must be the clean path plus one patch, normalised once."""
    _, normalize = split_preprocessing()
    clean_dataset = GTSRB(root=PROJECT_ROOT / "data", split="train",
                          transform=build_preprocessing(), download=False)
    conditions = {condition.name: condition for condition in evaluation_conditions()}
    checked = 0
    for name in ("centre_s14_y25x25", "corner_s25_y39x39", "uniform_s25_c02",
                 "ladder_s14_d8_y17x33"):
        condition = conditions[name]
        dataset = PlacedOcclusionDataset(validation, condition)
        for position in (0, 137, len(validation) - 1):
            index = stable_index(validation, position)
            observation, _ = validation[position]
            assert float(observation.min()) >= 0.0 and float(observation.max()) <= 1.0
            top, left = condition.top_left(index)
            expected = normalize(place_patch(observation, (top, left), condition.side))
            placed, label = dataset[position]
            assert torch.equal(placed, expected), (name, position)

            reference, reference_label = clean_dataset[index]
            assert label == reference_label
            mask = torch.zeros_like(placed, dtype=torch.bool)
            mask[..., top:top + condition.side, left:left + condition.side] = True
            # Outside the patch the tensor must be the frozen pipeline's own
            # output, bit for bit; inside it must be the fill and nothing else.
            assert torch.equal(placed[~mask], reference[~mask]), (name, position)
            assert torch.equal(placed[mask], torch.full_like(placed[mask], PATCH_VALUE))
            assert not torch.any(placed[~mask] == PATCH_VALUE), (name, position)
            assert int((placed == PATCH_VALUE).sum()) == 3 * condition.side ** 2
            checked += 1
    print(f"  {checked} frames across 4 passes: output == normalize(place_patch("
          "observation)) with the placement derived independently")
    print(f"  outside the patch the tensor is bit-identical to build_preprocessing(); "
          f"inside it is exactly {OCCLUSION_FILL} before normalisation, which the "
          f"frozen Normalize maps to {PATCH_VALUE:g}")
    # The sampled frames are evidence, but the claim is structural: the resize
    # runs on the uint8 image before ToTensor, so every pre-normalisation value
    # sits on the k/255 lattice, and 0.5 = 127.5/255 is not on it. No unpatched
    # pixel can therefore normalise to 0, whatever the frame contains.
    print("  no unpatched pixel reaches that value - it would have to be 127.5/255, "
          "off the k/255 lattice an 8-bit resize produces - so the zero set of a "
          "placed image is exactly its patch and the sweep's pixel audit is sound")

    observation, _ = validation[0]
    for top_left, side in (((-1, 0), 14), ((51, 0), 14), ((0, 40), 25)):
        try:
            place_patch(observation, top_left, side)
        except ValueError:
            continue
        raise AssertionError(f"a {side}px patch at {top_left} was accepted")
    try:
        place_patch(normalize(observation), (0, 0), 14)
    except ValueError:
        pass
    else:
        raise AssertionError("an already-normalised image was accepted")
    print("  out-of-frame placements and already-normalised inputs are refused")


def check_pairing(validation):
    """Every pass must cover the same frames, differing only inside its patch."""
    conditions = evaluation_conditions()
    assert len(conditions) == EXPECTED_PASSES, len(conditions)
    names = [condition.name for condition in conditions]
    assert len(set(names)) == len(names)

    counts = {}
    for condition in conditions:
        counts[(condition.policy, condition.side)] = \
            counts.get((condition.policy, condition.side), 0) + 1
    expected = {(CLEAN_CONDITION, None): 1}
    for side in PATCH_SIDES:
        expected[("centre", side)] = len(centre_placements(side))
        expected[("corner", side)] = 4
        expected[("uniform", side)] = UNIFORM_STRATA ** 2
        expected[("ladder", side)] = sum(
            len(rung) for rung_side, _, rung, policy in ladder_table()
            if policy == "ladder" and rung_side == side)
    assert counts == expected, (counts, expected)
    print(f"  {len(conditions)} passes: " + ", ".join(
        f"{policy}{'' if side is None else f' s{side}'} {count}"
        for (policy, side), count in sorted(counts.items(), key=lambda item: str(item[0]))))

    for side in PATCH_SIDES:
        for policy in POLICIES:
            total = sum(condition.weight for condition in conditions
                        if condition.policy == policy and condition.side == side)
            assert abs(total - 1.0) < 1e-12, (policy, side, total)
    print("  within every policy and side the per-frame weights sum to 1")

    # Read back the weights the protocol table actually stores for each rung,
    # since those are what `placements.csv` and `predictions.csv` record.
    for side, distance, rung, policy in ladder_table():
        assert len(set(rung)) == len(rung)
        stored = [condition.weight for condition in conditions
                  if condition.policy == policy and condition.side == side
                  and condition.coordinates in rung]
        assert len(stored) == len(rung), (side, distance, stored)
        assert abs(sum(stored) - 1.0) < 1e-12, (side, distance, stored)
        for weight in stored:
            assert abs(weight - 1 / len(rung)) < 1e-12, (side, distance, weight)
    reused = [(side, distance, policy) for side, distance, _, policy in ladder_table()
              if policy != "ladder"]
    assert len(reused) == 2 * len(PATCH_SIDES), reused
    print(f"  {len(reused)} ladder rungs are read from the centre and corner passes "
          f"instead of being evaluated twice: {reused}")
    print("  the weight stored for every rung placement is 1 over the rung size, "
          "and each rung's stored weights sum to 1")

    datasets = {condition.name: PlacedOcclusionDataset(validation, condition)
                for condition in conditions if condition.side is not None}
    lengths = {len(dataset) for dataset in datasets.values()}
    assert lengths == {len(validation)}, lengths
    print(f"  every placement pass covers the same {len(validation)} frames")

    first = conditions[1]
    second = next(condition for condition in conditions
                  if condition.policy == "corner" and condition.side == first.side)
    index = stable_index(validation, 11)
    left_image, _ = datasets[first.name][11]
    right_image, _ = datasets[second.name][11]
    differing = torch.zeros_like(left_image, dtype=torch.bool)
    for condition in (first, second):
        top, position = condition.top_left(index)
        differing[..., top:top + condition.side,
                  position:position + condition.side] = True
    assert torch.equal(left_image[~differing], right_image[~differing])
    assert torch.any(left_image != right_image)
    print(f"  {first.name} and {second.name} on the same frame differ only inside "
          "their two patches, so the contrast is paired pixel for pixel")


def synthetic_results(conditions, index_of):
    """Per-pass records shaped like the sweep's, filled from the error rule."""
    results = {}
    placements = {}
    for condition in conditions:
        coordinates = []
        predicted = []
        for _, dataset_index in index_of.values():
            if condition.side is None:
                coordinates.append(None)
                error = int(dataset_index % 50 == 0)
            else:
                top_left = condition.top_left(dataset_index)
                coordinates.append(top_left)
                error = synthetic_error(dataset_index, *top_left, condition.side)
            predicted.append(SYNTHETIC_LABEL - error)
        predictions = torch.tensor(predicted)
        targets = torch.full_like(predictions, SYNTHETIC_LABEL)
        confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        for target, prediction in zip(targets.tolist(), predictions.tolist()):
            confusion[target, prediction] += 1
        correct = float((predictions == targets).to(torch.float64).mean())
        results[condition.name] = {
            "predictions": predictions,
            "targets": targets,
            "confusion": confusion,
            "loss": 1.0 - correct,
            "accuracy": correct,
            "macro_f1": correct,
            "seen": len(index_of),
            "grad_enabled": False,
        }
        placements[condition.name] = coordinates
    return results, placements


def synthetic_predictions(results_dir):
    """Write a full set of sweep artifacts with a known answer.

    Errors are a deterministic function of the frame and of where the patch
    sits, so every quantity the analysis reports can be recomputed directly
    from the rule and compared. No model is involved. The rows are written by
    `evaluate_placement.write_outputs`, the same writer the sweep itself uses,
    so the analysis is read against the schema that will actually be produced
    rather than against a second copy of it kept here.
    """
    index_of = {}
    for track_number in range(EXPECTED_TRACKS):
        for _ in range(FRAMES_PER_TRACK):
            position = len(index_of)
            index_of[position] = (f"track{track_number:03d}",
                                  SYNTHETIC_OFFSET + position)
    conditions = evaluation_conditions()
    results, placements = synthetic_results(conditions, index_of)
    tracks = [track for track, _ in index_of.values()]
    indices = [index for _, index in index_of.values()]
    labels = np.full(max(indices) + 1, SYNTHETIC_LABEL, dtype=np.int64)
    paths = write_outputs(results, conditions, {"synthetic": True}, indices,
                          labels, tracks, placements, results_dir)
    return index_of, paths


def check_outputs():
    """The writer's artifacts, against what the analysis expects of them."""
    conditions = evaluation_conditions()
    with tempfile.TemporaryDirectory() as directory:
        index_of, paths = synthetic_predictions(Path(directory) / "results")
        frames = len(index_of)
        for name, path in paths.items():
            assert path.exists() and path.stat().st_size > 0, name

        with open(paths["predictions_csv"], newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert list(rows[0]) == ["condition", "policy", "side", "top", "left",
                                 "weight", "validation_position", "dataset_index",
                                 "track", "label", "prediction"]
        assert len(rows) == len(conditions) * frames, len(rows)
        assert len({row["condition"] for row in rows}) == len(conditions)
        clean_rows = [row for row in rows if row["policy"] == CLEAN_CONDITION]
        assert len(clean_rows) == frames
        assert all(row["top"] == "" and row["left"] == "" and row["side"] == ""
                   for row in clean_rows)
        assert all(row["top"] != "" for row in rows if row["policy"] != CLEAN_CONDITION)

        # The written coordinates are the provenance record: whoever reads the
        # artifact later reconstructs the uniform draws from them, so they have
        # to be the positions the patches were actually placed at, not a
        # plausible-looking column.
        by_name = {condition.name: condition for condition in conditions}
        occluded = [row for row in rows if row["policy"] != CLEAN_CONDITION]
        for row in occluded[::997]:
            condition = by_name[row["condition"]]
            assert (int(row["top"]), int(row["left"])) == condition.top_left(
                int(row["dataset_index"])), row
            assert int(row["side"]) == condition.side, row
        print(f"  predictions.csv carries {len(rows)} rows, one per frame per pass; "
              f"{len(occluded[::997])} sampled occluded rows record the placement "
              f"the protocol puts on that frame, and clean rows record none")

        with open(paths["summary_csv"], newline="") as handle:
            summary = list(csv.DictReader(handle))
        assert {row["condition"] for row in summary} == {c.name for c in conditions}
        reference = next(row for row in summary if row["policy"] == CLEAN_CONDITION)
        for row in summary:
            assert int(row["samples"]) == frames
            drop = float(reference["accuracy"]) - float(row["accuracy"])
            assert abs(drop - float(row["accuracy_drop"])) < 1e-12, row["condition"]
        print(f"  summary.csv carries {len(summary)} passes and its drops are "
              f"differences against the clean pass in the same file")

        with open(paths["placements_csv"], newline="") as handle:
            protocol = list(csv.DictReader(handle))
        assert len(protocol) == len(conditions)
        assert {row["condition"] for row in protocol} == {c.name for c in conditions}
        with open(paths["ladder_csv"], newline="") as handle:
            ladder = list(csv.DictReader(handle))
        assert len(ladder) == len(ladder_rows()) == len(ladder_table())
        with np.load(paths["confusion_npz"]) as stored:
            assert sorted(stored.files) == sorted(c.name for c in conditions)
            for condition in conditions:
                matrix = stored[condition.name]
                assert matrix.shape == (NUM_CLASSES, NUM_CLASSES)
                assert int(matrix.sum()) == frames, condition.name
        assert json.loads(paths["metadata_json"].read_text())["synthetic"] is True
        print(f"  placements.csv, ladder.csv and the confusion archive name the same "
              f"{len(conditions)} passes, {len(ladder)} ladder rungs, and "
              f"{NUM_CLASSES}x{NUM_CLASSES} matrices summing to {frames} frames")

        # The analysis is run end to end over those artifacts. Its numbers come
        # from the synthetic rule, so they are discarded: only the fact that it
        # consumes the written schema and produces every table is kept.
        output = Path(directory) / "analysis"
        argv = sys.argv
        sys.argv = ["analyze_placement", "--results-dir", str(paths["summary_csv"].parent),
                    "--output-dir", str(output)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                analyse()
        finally:
            sys.argv = argv
        written = sorted(path.name for path in output.iterdir())
        assert written == EXPECTED_ANALYSIS_FILES, written
        print(f"  the analysis entry point reads those files and writes "
              f"{len(written)} tables ({', '.join(written)}); the synthetic "
              f"numbers it produced were discarded")


def check_statistics():
    """Aggregation, pairing and the bootstrap, against a known answer."""
    # As many values as the split has signs, so the percentile interval is not
    # so coarse that two different seeds would land on the same order statistic.
    values = np.linspace(0.0, 1.0, EXPECTED_TRACKS)
    first = bootstrap_mean(values)
    assert first == bootstrap_mean(values)
    assert abs(first["mean"] - values.mean()) < 1e-15
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]
    assert first["draws"] == BOOTSTRAP_DRAWS and first["tracks"] == values.size
    shifted = bootstrap_mean(values, seed=BOOTSTRAP_SEED + 1)
    assert (shifted["ci_low"], shifted["ci_high"]) != (first["ci_low"], first["ci_high"])
    print(f"  bootstrap of {values.size} values is identical on repeat, moves with "
          f"the seed, and uses {BOOTSTRAP_DRAWS} draws at seed {BOOTSTRAP_SEED}")

    with tempfile.TemporaryDirectory() as directory:
        index_of, paths = synthetic_predictions(Path(directory) / "results")
        clean, placement = load_predictions(paths["predictions_csv"])
        frames = frame_rates(placement)
        tracks = track_rates(frames, ["policy", "side"])

        # Recompute one frame's uniform estimate straight from the rule.
        side = 25
        weights = uniform_weights(side)
        dataset_index = index_of[7][1]
        expected = 0.0
        for cell, weight in weights.items():
            top, left = uniform_placement(side, cell, dataset_index)
            expected += weight * synthetic_error(dataset_index, top, left, side)
        got = frames[(frames["policy"] == "uniform") & (frames["side"] == side)
                     & (frames["validation_position"] == 7)]["rate"].item()
        assert abs(got - expected) < 1e-12, (got, expected)
        print(f"  one frame's weighted uniform rate matches a direct computation "
              f"from the stratum weights and the seeded draws ({got:.6f})")

        # And one contrast, by hand over the same synthetic outcomes.
        rates = {}
        for policy in ("centre", "corner"):
            placements = [condition for condition in evaluation_conditions()
                          if condition.policy == policy and condition.side == side]
            per_track = {}
            for position, (track, index) in index_of.items():
                value = sum(condition.weight
                            * synthetic_error(index, *condition.coordinates, side)
                            for condition in placements)
                per_track.setdefault(track, []).append(value)
            rates[policy] = {track: float(np.mean(values))
                             for track, values in per_track.items()}
        differences = np.array([rates["centre"][track] - rates["corner"][track]
                                for track in sorted(rates["centre"])])
        contrast = paired_contrast(tracks[tracks["side"] == side],
                                   ["policy", "side"], "centre", "corner")
        assert abs(contrast["mean"] - differences.mean()) < 1e-12
        # A contrast of zero would make the comparison unable to detect a
        # swapped pair or a dropped placement, so the rule has to separate the
        # two policies by a wide margin.
        assert contrast["mean"] > 0.5, contrast["mean"]
        assert contrast["ci_low"] > 0.5 and contrast["ci_high"] < 1.0
        assert contrast["tracks"] == EXPECTED_TRACKS
        assert abs(contrast["ci_low"] - np.percentile(
            differences[np.random.default_rng(BOOTSTRAP_SEED).integers(
                0, differences.size, size=(BOOTSTRAP_DRAWS, differences.size))
            ].mean(axis=1), 2.5)) < 1e-12
        print(f"  centre-minus-corner at side {side} matches a hand computation over "
              f"{EXPECTED_TRACKS} synthetic signs, mean {contrast['mean']:+.6f}, "
              f"interval [{contrast['ci_low']:+.6f}, {contrast['ci_high']:+.6f}]")

        ladder = track_rates(ladder_frame_rates(placement), ["centre_distance", "side"])
        for patch in PATCH_SIDES:
            present = sorted(ladder[ladder["side"] == patch]["centre_distance"].unique())
            assert present == sorted(LADDER_DISTANCES[patch]), (patch, present)
        assert len(ladder_assignment()) == sum(
            len(rung) for _, _, rung, _ in ladder_table())
        print(f"  the ladder resolves to {sorted(LADDER_DISTANCES[14])} and "
              f"{sorted(LADDER_DISTANCES[25])} rungs, each over all "
              f"{EXPECTED_TRACKS} signs")
        assert len(clean) == EXPECTED_TRACKS * FRAMES_PER_TRACK
        print(f"  the clean pass contributes one row per frame "
              f"({len(clean)}) and is kept out of the policy aggregation")

    # The placement-noise term is an analytic variance; check it against a
    # simulation of the same draws rather than trusting the algebra.
    side = 25
    weights = uniform_weights(side)
    cells = sorted(weights)
    rates = {cell: 0.05 + 0.08 * position for position, cell in enumerate(cells)}
    record = placement_noise(side, rates)
    rng = np.random.default_rng(0)
    samples = np.zeros(200_000)
    for cell in cells:
        samples += weights[cell] * (rng.random(samples.size) < rates[cell])
    simulated = samples.std(ddof=1)
    assert abs(simulated - record["frame_sd_pooled_bound"]) < 0.02 * simulated
    assert record["frame_sd_pooled_bound"] <= record["frame_sd_bound"]
    assert abs(record["sum_squared_weights"]
               - sum(weight ** 2 for weight in weights.values())) < 1e-15
    print(f"  per-frame placement noise with one rate per cell: formula "
          f"{record['frame_sd_pooled_bound']:.6f}, simulation {simulated:.6f}, "
          f"worst case {record['frame_sd_bound']:.6f}")

    # The pooled number is an upper bound rather than an estimate, because
    # `q(1 - q)` is concave. Take frames whose per-frame rates straddle the same
    # cell averages and compute both variances exactly: the real one has to come
    # out smaller, by the spread of the per-frame rates.
    spread = 0.05
    halves = {cell: (max(rate - spread, 0.0), min(rate + spread, 1.0))
              for cell, rate in rates.items()}
    for cell, (low, high) in halves.items():
        assert abs((low + high) / 2 - rates[cell]) < 1e-12, cell
    varying = sum(weights[cell] ** 2 * (low * (1 - low) + high * (1 - high)) / 2
                  for cell, (low, high) in halves.items()) ** 0.5
    assert varying < record["frame_sd_pooled_bound"], varying
    print(f"    with the same cell averages but per-frame rates {spread:g} either "
          f"side of them the exact spread is {varying:.6f}, below the pooled "
          f"{record['frame_sd_pooled_bound']:.6f}, so the reported figure "
          "overstates the placement noise rather than understating it")
    print(f"  per-sign {record['track_sd_bound']:.6f} and grand-mean "
          f"{record['grand_mean_sd_bound']:.6f} worst-case standard deviations, "
          "reported separately from the interval across signs")


def roi_boxes(data_root=None):
    """Annotated sign boxes for the validation frames, in resized coordinates.

    GTSRB ships one bounding box per training frame in `GT-<class>.csv`. The
    box is given in original pixels, so it is rescaled by the same factor the
    preprocessing resize applies - independently per axis, since the resize is
    to a fixed 64x64 with no crop - which puts it in the frame the patch is
    placed in. `Roi.X2` and `Roi.Y2` are taken as exclusive bounds, so the box
    area is understated by about one original pixel per axis; the inclusive
    reading gives a median area of 0.608 rather than 0.571.
    """
    if data_root is None:
        data_root = PROJECT_ROOT / "data"
    dataset = GTSRB(root=data_root, split="train")
    labels, tracks = read_training_index(dataset)
    _, validation_indices = split_tracks(labels, tracks)
    annotations = {}
    root = Path(data_root) / ROI_ANNOTATIONS
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        with open(directory / f"GT-{directory.name}.csv", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                scale_x = IMAGE_SIZE / float(row["Width"])
                scale_y = IMAGE_SIZE / float(row["Height"])
                annotations[f"{directory.name}/{row['Filename']}"] = (
                    float(row["Roi.Y1"]) * scale_y,
                    float(row["Roi.X1"]) * scale_x,
                    float(row["Roi.Y2"]) * scale_y,
                    float(row["Roi.X2"]) * scale_x,
                )
    boxes = []
    for index in validation_indices:
        path = Path(dataset._samples[index][0])
        boxes.append(annotations[f"{path.parent.name}/{path.name}"])
    return np.array(boxes)


def check_roi_overlap():
    """A flush corner patch must still land on the sign, not beside it."""
    boxes = roi_boxes()
    assert len(boxes) == EXPECTED_TRACKS * FRAMES_PER_TRACK, len(boxes)
    top, left, bottom, right = boxes.T
    area = (bottom - top) * (right - left) / IMAGE_SIZE ** 2
    median_area = float(np.median(area))
    assert abs(median_area - EXPECTED_ROI_MEDIAN_AREA) < 5e-5, median_area
    print(f"  the annotated sign box covers a median {median_area:.4f} of the "
          f"resized frame over {len(boxes)} validation frames "
          f"(mean {area.mean():.4f}, range {area.min():.4f}-{area.max():.4f})")

    for side in PATCH_SIDES:
        misses = []
        overlaps = []
        for patch_top, patch_left in corner_placements(side):
            height = np.clip(np.minimum(bottom, patch_top + side) - np.maximum(top, patch_top), 0, None)
            width = np.clip(np.minimum(right, patch_left + side) - np.maximum(left, patch_left), 0, None)
            covered = height * width / side ** 2
            misses.append(int((covered == 0).sum()))
            overlaps.append(float(np.median(covered)))
        assert (min(misses), max(misses)) == EXPECTED_CORNER_MISSES[side], (side, misses)
        expected_low, expected_high = EXPECTED_CORNER_OVERLAP[side]
        assert abs(min(overlaps) - expected_low) < 5e-4, (side, overlaps)
        assert abs(max(overlaps) - expected_high) < 5e-4, (side, overlaps)
        print(f"    side {side}: a flush corner patch misses the box on "
              f"{min(misses)}-{max(misses)} of {len(boxes)} frames depending on the "
              f"corner ({misses}), and covers a median {min(overlaps):.3f}-"
              f"{max(overlaps):.3f} of its own area with sign")


def check_integrity():
    """The frozen suite and the accepted results must be untouched."""
    for relative, expected in sorted(ACCEPTED_ARTIFACTS.items()):
        digest = file_md5(PROJECT_ROOT / relative)
        assert digest == expected, (relative, digest, expected)
    print(f"  {len(ACCEPTED_ARTIFACTS)} accepted artifacts unchanged, including the "
          "epoch-21 checkpoint and the frozen robustness predictions")

    modified = subprocess.run(
        ["git", "status", "--porcelain", *sorted(FROZEN_SOURCES)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    ).stdout.strip()
    assert not modified, modified
    for relative, expected in sorted(FROZEN_SOURCES.items()):
        blob = subprocess.run(["git", "rev-parse", f"HEAD:{relative}"],
                              cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert blob.returncode == 0, (relative, blob.stderr)
        assert blob.stdout.strip() == expected, (relative, blob.stdout.strip())
    print(f"  the {len(FROZEN_SOURCES)} frozen modules match their pinned committed "
          "contents and have no uncommitted modifications")

    # The clean pass is dispatched by label through the frozen
    # `build_condition_dataset`, which only recognises the accepted sweep's name,
    # so a rename here would break the dispatch rather than the comparison.
    assert CLEAN_CONDITION == ACCEPTED_CLEAN_CONDITION, CLEAN_CONDITION
    print(f"  the clean pass keeps the accepted sweep's label "
          f"{CLEAN_CONDITION!r}")

    for path in (RESULTS_DIR.relative_to(PROJECT_ROOT),
                 RESULTS_DIR.relative_to(PROJECT_ROOT) / "predictions.csv"):
        result = subprocess.run(["git", "check-ignore", "-v", str(path)],
                                cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f"NOT ignored by Git: {path}"
    print(f"  ablation outputs land in {RESULTS_DIR.relative_to(PROJECT_ROOT)}, "
          "which Git ignores")
    tracked = subprocess.run(["git", "ls-files", "artifacts", "data"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert not tracked.stdout.strip(), tracked.stdout
    print("  no artifact or dataset path is tracked by Git")


def main():
    print("Patch sides:")
    check_sides()

    print("\nPlacement geometry:")
    check_geometry()

    print("\nUniform estimand and stratum weights:")
    check_weighting()

    print("\nPer-frame uniform draws:")
    check_draws()

    print("\nPlacement seeding:")
    check_seeds()

    validation, _, _ = build_validation_observations()
    print(f"\nPreprocessing alignment ({len(validation)} validation observations):")
    check_preprocessing(validation)

    print("\nPairing across passes:")
    check_pairing(validation)

    print("\nTrack-level estimators:")
    check_statistics()

    print("\nWritten artifacts:")
    check_outputs()

    print("\nAnnotated sign boxes:")
    check_roi_overlap()

    print("\nIntegrity:")
    check_integrity()

    print("\nAll placement checks passed. No checkpoint was loaded, no prediction "
          "was computed, and the official test split was never read.")


if __name__ == "__main__":
    main()
