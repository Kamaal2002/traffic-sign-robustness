"""Structural verification of the augmentation experiment's infrastructure.

    .venv/bin/python -m src.verify_augmentation

Four pieces are checked: the training augmentation policies, the held-out
corruption suite, the checkpoint-parameterised evaluator, and the model
comparison. Nothing here trains a model, and the official test split is never
opened, so no check can be satisfied by model behaviour.

The checks that matter most are the ones that have to fail. A mislabelled
result directory, a checkpoint whose own records describe another run, and a
training run aimed at the accepted baseline's directory are all driven through
the real entry points and required to be refused, because each of them would
otherwise produce a plausible table attributing one model's errors to another.

Most checks run on synthetic fixtures - constructed observations, hand-built
prediction tables with known error rates, temporary output directories - so
that the expected answer is known in advance rather than read off the
implementation. Three groups need the real split: the severity ordering of the
held-out families, which is only meaningful on structured images; the parity
check between zero-probability augmented loading and the accepted clean
recipe; and the leakage checks over the real track-aware split.

No test image is read. The clean-parity check calls `data.create_data_loaders`,
because comparing against the baseline's own loader construction is the point
of that check, and that function builds the official test index along with the
other two; the loader it returns is never iterated.
"""

import io
import json
import math
import subprocess
import tempfile
from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import blake2b, md5
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .augmentation import (
    AUGMENT_PROBABILITY,
    AugmentationPolicy,
    AugmentedDataset,
    BALANCED,
    BALANCED_WEIGHTS,
    EpochLoader,
    OCCLUSION_PLACEMENT,
    POLICIES,
    PROTOCOL_ARTIFACT,
    PROTOCOL_FILES,
    PROTOCOL_MANIFEST,
    SEVERITY_WEIGHTS,
    TARGETED,
    TARGETED_WEIGHTS,
    TRAIN_NAMESPACE,
    augmentation_seed,
    build_augmented_loaders,
    check_observation,
    file_digest,
    protocol_digests,
    protocol_record,
)
from .compare_models import (
    ARM_LABELS,
    CLEAN_TRADEOFF_LIMIT,
    _direction,
    compare,
    targeted_verdict,
)
from .corruptions import (
    CORRUPTIONS,
    OCCLUSION_FILL,
    SEVERITY_LEVELS,
    VALIDATION_NAMESPACE,
    build_validation_observations,
    corruption_seed,
    occlusion_side,
    split_preprocessing,
)
from .corruptions_heldout import (
    HELDOUT_CORRUPTIONS,
    HeldoutDataset,
    NEUTRAL_GREY,
    _motion_kernel,
    apply_heldout,
    heldout_conditions,
    heldout_parameter,
    heldout_sample_seed,
)
from .data import IMAGE_SIZE, NUM_CLASSES, SEED, build_preprocessing, create_data_loaders
from .evaluate_checkpoint import (
    FINGERPRINT_STRIDE,
    SUITES,
    check_identity,
    check_provenance,
    discover_run_metadata,
    input_fingerprint,
    model_record,
    recorded_policy_name,
    resolve_model_name,
    write_outputs,
)
from .evaluate_placement import EXPECTED_TRACKS, FRAMES_PER_TRACK
from .evaluate_robustness import (
    CHECKPOINT_PATH,
    CLEAN_CONDITION,
    EXPECTED_CHECKPOINT_MD5,
    PROJECT_ROOT,
)
from .train import TrainingConfig
from .train_augmented import (
    EXPECTED_TRAIN_IMAGES,
    EXPECTED_VALIDATION_IMAGES,
    PROTECTED_DIRS,
    check_no_leakage,
    check_output_dir,
    claim_output_dir,
    policy_record,
    resolve_policy,
    write_protocol_sidecar,
)
from .train_clean import ACCEPTED_BASELINE, default_output_dir
from .train_clean import run as clean_replicate

# Pinned from the implementation and re-derived from the documented rule below.
# A change here means previously generated augmented epochs are no longer
# reproducible, which for a training-time augmenter means the runs cannot be
# repeated at all.
EXPECTED_TRAIN_SEED_ARGS = (7, 1234)
EXPECTED_TRAIN_SEED = 5129022550016994121

# The declared recipe, written out here rather than imported as a table, so
# editing the policy in augmentation.py fails this verifier instead of silently
# redefining the experiment.
DECLARED_TARGETED = {
    "occlusion": 0.42,
    "gaussian_noise": 0.22,
    "fog": 0.15,
    "brightness_bright": 0.09,
    "brightness_dark": 0.06,
    "gaussian_blur": 0.06,
}
DECLARED_AUGMENT_PROBABILITY = 0.5

# The held-out severity tables, restated for the same reason.
DECLARED_HELDOUT = {
    "motion_blur": ("length", (3, 5, 7, 11, 15), True),
    "jpeg_compression": ("quality", (70, 50, 30, 15, 7), False),
    "contrast_reduction": ("factor", (0.7, 0.55, 0.4, 0.25, 0.15), False),
    "salt_pepper": ("fraction", (0.005, 0.01, 0.03, 0.06, 0.1), True),
    "rotation": ("degrees", (3, 7, 12, 18, 25), True),
}

# Modules whose contents the accepted results depend on. Verified clean against
# HEAD, because "the benchmark was frozen before any model was trained" is the
# claim the whole comparison rests on.
#
# The last two were added when the test path was implemented. `TestSuite`
# subclasses `evaluate_checkpoint.Suite` and `compare_test.py` imports the
# contrast list, the clean-accuracy limit and the decision rule from
# `compare_models.py`, so an edit to either file while writing the test modules
# would move a committed validation number, or the rule that reads it, without
# tripping any other gate.
FROZEN_MODULES = (
    "src/corruptions.py",
    "src/data.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate_robustness.py",
    "src/analyze_placement.py",
    "src/evaluate_checkpoint.py",
    "src/compare_models.py",
)

SAMPLES = 4000
ORDERING_FRAMES = 40


def independent_train_seed(epoch, dataset_index, base_seed=SEED,
                           namespace=TRAIN_NAMESPACE):
    """Re-derive the augmentation seed straight from the documented rule.

    Written from the specification rather than by calling `augmentation_seed`,
    so the checks below compare the implementation against the scheme that was
    agreed instead of against itself.
    """
    payload = f"{base_seed}|{namespace}|{epoch}|{dataset_index}".encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big") >> 1


def observation(seed=0, size=IMAGE_SIZE):
    """A valid `[0, 1]` observation kept away from the clamps."""
    generator = torch.Generator().manual_seed(seed)
    return 0.2 + 0.6 * torch.rand(3, size, size, generator=generator)


def structured_observation(size=IMAGE_SIZE):
    """A smooth, structured image: the regime a real sign crop lives in.

    White noise is useless for checking a geometric or blur severity ladder,
    because one pixel of displacement already decorrelates it completely and
    every severity then looks equally bad. This has gradients and edges at the
    scale a 64x64 sign crop does.
    """
    axis = torch.linspace(0, 1, size)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    radius = ((rows - 0.5) ** 2 + (columns - 0.5) ** 2).sqrt()
    disc = (radius < 0.35).float()
    bar = ((rows > 0.45) & (rows < 0.55)).float()
    image = torch.stack([
        0.15 + 0.7 * disc * (1 - bar),
        0.2 + 0.6 * (columns * disc),
        0.25 + 0.5 * (rows * disc + 0.3 * bar),
    ])
    return image.clamp(0.0, 1.0)


class ListDataset:
    """The smallest dataset the augmenter accepts, for synthetic checks."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index].clone(), self.labels[index]


def check_policies():
    """The two arms must differ only in the family distribution."""
    assert sorted(POLICIES) == ["balanced", "targeted"], sorted(POLICIES)
    assert resolve_policy("balanced") is BALANCED
    assert resolve_policy("targeted") is TARGETED

    for policy in (BALANCED, TARGETED):
        assert policy.augment_probability == DECLARED_AUGMENT_PROBABILITY, policy.name
        families = dict(policy.family_weights)
        assert set(families) == set(CORRUPTIONS), sorted(families)
        assert abs(sum(families.values()) - 1.0) < 1e-12
        severities = dict(policy.severity_weights)
        assert set(severities) == set(SEVERITY_LEVELS)
        assert all(abs(weight - 1 / 5) < 1e-12 for weight in severities.values())
        assert policy.describe()["families_per_image"] == 1
        assert policy.describe()["occlusion_nested"] is False
        assert policy.describe()["occlusion_placement"] == OCCLUSION_PLACEMENT

    assert AUGMENT_PROBABILITY == DECLARED_AUGMENT_PROBABILITY
    assert dict(BALANCED.family_weights) == BALANCED_WEIGHTS
    assert all(abs(weight - 1 / 6) < 1e-12 for weight in BALANCED_WEIGHTS.values())
    assert dict(TARGETED.family_weights) == TARGETED_WEIGHTS
    for family, weight in DECLARED_TARGETED.items():
        assert abs(TARGETED_WEIGHTS[family] - weight) < 1e-12, family
    assert BALANCED.severity_weights == TARGETED.severity_weights
    assert dict(SEVERITY_WEIGHTS) == dict(BALANCED.severity_weights)
    assert BALANCED.augment_probability == TARGETED.augment_probability

    for bad in (
        {"augment_probability": 1.5},
        {"family_weights": (("occlusion", 0.5), ("fog", 0.4))},
        {"family_weights": (("occlusion", 1.0), ("occlusion", 0.0))},
        {"family_weights": (("motion_blur", 1.0),)},
        {"severity_weights": (("1", 1.0),)},
    ):
        fields = {"name": "bad", "augment_probability": 0.5,
                  "family_weights": BALANCED.family_weights,
                  "severity_weights": BALANCED.severity_weights}
        fields.update(bad)
        try:
            AugmentationPolicy(**fields)
        except ValueError:
            continue
        raise AssertionError(f"accepted an invalid policy: {bad}")

    print(f"  two policies, P(augment)={DECLARED_AUGMENT_PROBABILITY} and uniform "
          "severity in both, family weights the only difference")
    print(f"  targeted weights match the declared table: "
          + "  ".join(f"{name}={weight:g}"
                      for name, weight in DECLARED_TARGETED.items()))


def check_seed_derivation():
    """The epoch seed must come from stable identifiers and nothing else."""
    assert augmentation_seed(*EXPECTED_TRAIN_SEED_ARGS) == EXPECTED_TRAIN_SEED
    for epoch in (1, 2, 30):
        for index in (0, 17, 39208):
            assert augmentation_seed(epoch, index) == independent_train_seed(
                epoch, index), (epoch, index)
    assert augmentation_seed(1, 0) < 1 << 63

    # The namespace exists so that a training seed and an evaluation seed for
    # the same index can never coincide.
    collisions = 0
    for index in range(500):
        train = {augmentation_seed(epoch, index) for epoch in range(1, 31)}
        evaluation = {corruption_seed(VALIDATION_NAMESPACE, name, None, index)
                      for name in CORRUPTIONS}
        collisions += len(train & evaluation)
    assert collisions == 0, collisions

    distinct = {augmentation_seed(epoch, index)
                for epoch in range(1, 31) for index in range(200)}
    assert len(distinct) == 30 * 200, len(distinct)

    for bad in (0, -1, 1.0, True, "1"):
        try:
            augmentation_seed(bad, 0)
        except ValueError:
            continue
        raise AssertionError(f"accepted epoch {bad!r}")

    print(f"  seed = blake2b(base|{TRAIN_NAMESPACE}|epoch|index) >> 1, "
          f"re-derived independently; 6,000 (epoch, index) pairs all distinct")
    print("  no training seed collides with any evaluation seed for the same "
          "index over 500 indices and 30 epochs")


def check_plan_distribution():
    """The realised draws must match the declared tables, and vary by epoch."""
    for policy in (BALANCED, TARGETED):
        plans = [policy.plan(augmentation_seed(1 + index % 7, index))
                 for index in range(SAMPLES)]
        augmented = [plan for plan in plans if plan["augmented"]]
        share = len(augmented) / len(plans)
        assert abs(share - policy.augment_probability) < 0.02, (policy.name, share)

        families = pd.Series([plan["family"] for plan in augmented]).value_counts(
            normalize=True)
        for family, weight in policy.family_weights:
            observed = float(families.get(family, 0.0))
            assert abs(observed - weight) < 0.03, (policy.name, family, observed)
        severities = pd.Series([plan["severity"] for plan in augmented]).value_counts(
            normalize=True)
        for severity in SEVERITY_LEVELS:
            assert abs(float(severities[severity]) - 0.2) < 0.03, (policy.name, severity)
        assert all(plan["family"] is None and plan["severity"] is None
                   for plan in plans if not plan["augmented"])
        print(f"  {policy.name}: {share:.3f} augmented over {SAMPLES} draws; "
              "family and severity frequencies within 0.03 of the table")

    # Same image, different epochs: the realisation has to move. A dataset that
    # silently held epoch 1 for thirty epochs would pass every other check.
    # An image left clean in all three epochs is the one legitimate exception,
    # and at P(augment) = 0.5 that is an eighth of them.
    images = 400
    changed = untouched = 0
    for index in range(images):
        plans = [BALANCED.plan(augmentation_seed(epoch, index))
                 for epoch in (1, 2, 3)]
        keys = {(plan["augmented"], plan["family"], plan["severity"],
                 plan["realisation_seed"]) for plan in plans}
        if any(plan["augmented"] for plan in plans):
            assert len(keys) > 1, index
            changed += 1
        else:
            untouched += 1
    assert changed + untouched == images
    expected_untouched = images * (1 - BALANCED.augment_probability) ** 3
    assert abs(untouched - expected_untouched) < 4 * math.sqrt(
        expected_untouched), (untouched, expected_untouched)
    print(f"  every one of the {changed} images augmented in at least one of "
          f"three epochs gets a different realisation in each; the remaining "
          f"{untouched} were left clean in all three (expected "
          f"{expected_untouched:.0f})")

    # The two policies share the draw order, so they augment the same images at
    # the same severities and differ only in which family lands.
    same_decision = same_severity = 0
    different_family = 0
    for index in range(1000):
        seed = augmentation_seed(1, index)
        left, right = BALANCED.plan(seed), TARGETED.plan(seed)
        same_decision += left["augmented"] == right["augmented"]
        if left["augmented"]:
            same_severity += left["severity"] == right["severity"]
            different_family += left["family"] != right["family"]
    assert same_decision == 1000, same_decision
    assert same_severity == sum(
        BALANCED.plan(augmentation_seed(1, index))["augmented"]
        for index in range(1000))
    print(f"  B and C augment exactly the same 1,000 images at exactly the same "
          f"severities; {different_family} of them receive a different family")


def check_occlusion_placement():
    """Training occlusion must be uniform over the whole frame and unnested."""
    for severity in SEVERITY_LEVELS:
        side = occlusion_side(CORRUPTIONS["occlusion"].levels[severity - 1],
                              IMAGE_SIZE, IMAGE_SIZE)
        positions = []
        for index in range(6000):
            plan = TARGETED.plan(augmentation_seed(1 + index % 11, index))
            if plan["augmented"] and plan["family"] == "occlusion" \
                    and plan["severity"] == severity:
                positions.append((plan["top"], plan["left"]))
        assert len(positions) > 100, (severity, len(positions))
        tops = np.array([top for top, _ in positions])
        lefts = np.array([left for _, left in positions])
        limit = IMAGE_SIZE - side
        assert tops.min() >= 0 and tops.max() <= limit
        assert lefts.min() >= 0 and lefts.max() <= limit
        # The frozen benchmark confines the patch centre to [17, 46]^2. Training
        # placement must not: patches have to reach the frame border.
        centres = tops + side / 2
        outside = int(((centres < 17) | (centres > 46)).sum())
        assert outside > 0, (severity, "placement never left the frozen window")
        expected = limit / 2
        assert abs(tops.mean() - expected) < 3.0, (severity, tops.mean(), expected)
        print(f"  s{severity} side {side:2d}px: {len(positions)} placements over "
              f"[0, {limit}] on both axes, mean top {tops.mean():.1f} "
              f"(uniform expects {expected:.1f}), {outside} outside the frozen "
              "central window")

    # One family and one severity per image, so nothing is nested.
    for index in range(2000):
        plan = TARGETED.plan(augmentation_seed(1, index))
        if plan["augmented"]:
            assert isinstance(plan["family"], str)
            assert plan["severity"] in SEVERITY_LEVELS
            assert (plan["side"] is None) == (plan["family"] != "occlusion")

    image = observation(1)
    plan = {"augmented": True, "family": "occlusion", "severity": 3,
            "realisation_seed": 0, "top": 0, "left": 0, "side": 10}
    occluded = TARGETED.apply(image, plan)
    assert torch.allclose(occluded[:, :10, :10],
                          torch.full((3, 10, 10), OCCLUSION_FILL))
    assert torch.equal(occluded[:, 10:, 10:], image[:, 10:, 10:])
    assert torch.equal(image, observation(1)), "apply mutated its input"
    print(f"  patch fill is OCCLUSION_FILL={OCCLUSION_FILL}, applied out of "
          "place, and reaches the (0, 0) corner")


def check_augmented_dataset():
    """The dataset must normalise exactly once, keep labels, and need an epoch."""
    images = [observation(index) for index in range(24)]
    labels = list(range(24))
    source = ListDataset(images, labels)
    dataset = AugmentedDataset(source, TARGETED)

    try:
        dataset[0]
    except RuntimeError:
        pass
    else:
        raise AssertionError("sampled without an epoch set")
    for bad in (0, -3, 1.5, True):
        try:
            dataset.set_epoch(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted epoch {bad!r}")

    _, normalize = split_preprocessing()
    dataset.set_epoch(1)
    clean_positions = []
    for index in range(len(source)):
        plan = dataset.plan_for(index)
        image, label = dataset[index]
        assert label == labels[index], index
        assert tuple(image.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert float(image.min()) >= -1.0 - 1e-6
        assert float(image.max()) <= 1.0 + 1e-6
        if not plan["augmented"]:
            clean_positions.append(index)
            assert torch.equal(image, normalize(images[index])), index
    assert clean_positions, "no unaugmented image in the sample"
    print(f"  {len(source)} items: labels unchanged, outputs inside [-1, 1], and "
          f"the {len(clean_positions)} unaugmented ones are bit-identical to "
          "normalize(observation), so normalisation runs exactly once")

    # Determinism and independence from how the item was reached.
    dataset.set_epoch(4)
    direct = [dataset[index][0] for index in range(len(source))]
    again = [dataset[index][0] for index in reversed(range(len(source)))]
    for index, image in enumerate(direct):
        assert torch.equal(image, again[len(source) - 1 - index]), index

    generator = torch.Generator().manual_seed(123)
    shuffled = DataLoader(dataset, batch_size=5, shuffle=True, generator=generator)
    seen = {}
    for batch_images, batch_labels in shuffled:
        for image, label in zip(batch_images, batch_labels):
            seen[int(label)] = image
    for index in range(len(source)):
        assert torch.equal(seen[index], direct[index]), index
    workers = DataLoader(dataset, batch_size=7, shuffle=False, num_workers=2)
    collected = torch.cat([batch for batch, _ in workers])
    assert torch.equal(collected, torch.stack(direct))
    print("  identical tensors by direct indexing, reversed indexing, a shuffled "
          "loader at batch size 5, and two worker processes at batch size 7")

    # An epoch change must move the realisations, and nothing else may.
    dataset.set_epoch(5)
    later = [dataset[index][0] for index in range(len(source))]
    differing = sum(not torch.equal(a, b) for a, b in zip(direct, later))
    assert differing > 0, "epoch 5 reproduced epoch 4 exactly"
    print(f"  {differing} of {len(source)} items change between epoch 4 and "
          "epoch 5")

    for bad in (images[0] * 2 - 1, images[0].unsqueeze(0),
                (images[0] * 255).to(torch.uint8)):
        try:
            check_observation(bad)
        except (TypeError, ValueError):
            continue
        raise AssertionError("accepted an invalid observation")
    print("  normalised, batched, and integer inputs are all rejected before "
          "augmentation")


def check_epoch_loader():
    """The epoch must advance once per iteration of the training loader."""
    source = ListDataset([observation(index) for index in range(8)], list(range(8)))
    dataset = AugmentedDataset(source, BALANCED)
    loader = EpochLoader(DataLoader(dataset, batch_size=4), dataset)
    assert len(loader) == 2
    assert loader.batch_size == 4, "attribute forwarding broken"
    for epoch in range(1, 6):
        for _ in loader:
            pass
        assert dataset.epoch == epoch, (epoch, dataset.epoch)
        assert loader.epochs == epoch
    # A truncated epoch, as the smoke configuration produces, still advances.
    iterator = iter(loader)
    next(iterator)
    assert dataset.epoch == 6 and loader.epochs == 6
    print("  six iterations advanced the epoch to 6, including a truncated "
          "pass; attribute and len forwarding intact")


def check_clean_parity(batches=6):
    """Zero-probability augmented loading must match the accepted clean recipe.

    The augmented path builds its own datasets and its own shuffle generator, so
    "the only difference from the baseline is the augmentation" needs checking
    rather than asserting. With `P(augment) = 0` the two paths must produce the
    same images in the same order, bit for bit, which also confirms the shuffle
    generator is seeded the way `create_data_loaders` seeds it.
    """
    parity = AugmentationPolicy(
        name="parity", augment_probability=0.0,
        family_weights=BALANCED.family_weights,
        severity_weights=BALANCED.severity_weights,
    )
    loaders, split_info, augmented = build_augmented_loaders(parity, batch_size=64)
    reference, _ = create_data_loaders(batch_size=64, download=False)

    assert len(loaders["train"].dataset) == len(reference["train"].dataset)
    assert len(loaders["validation"].dataset) == len(reference["validation"].dataset)

    augmented_iterator = iter(loaders["train"])
    reference_iterator = iter(reference["train"])
    for position in range(batches):
        left_images, left_labels = next(augmented_iterator)
        right_images, right_labels = next(reference_iterator)
        assert torch.equal(left_labels, right_labels), position
        assert torch.equal(left_images, right_images), position
    assert augmented.epoch == 1

    for position, (left, right) in enumerate(zip(loaders["validation"],
                                                 reference["validation"])):
        assert torch.equal(left[0], right[0]), position
        assert torch.equal(left[1], right[1]), position
        if position >= 2:
            break

    leakage = check_no_leakage(loaders, split_info)
    assert leakage["shared_images"] == 0 and leakage["shared_tracks"] == 0
    assert leakage["validation_is_augmented"] is False
    assert leakage["official_test_split_constructed"] is False
    assert leakage["dataset_splits_constructed"] == ["train"]
    assert leakage["train_images"] == 21300 and leakage["validation_images"] == 5340
    assert str(build_preprocessing()) in str(
        loaders["validation"].dataset.dataset.transform)
    print(f"  {batches} training batches and 3 validation batches are "
          "bit-identical to data.py's own loaders at P(augment)=0")
    print(f"  {leakage['train_tracks']} train and "
          f"{leakage['validation_tracks']} validation tracks, no shared image "
          "or track, validation unaugmented, and only the train split "
          "constructed by the augmented run")


def check_heldout_specification():
    """The held-out tables must match the declared values and be disjoint."""
    assert set(HELDOUT_CORRUPTIONS) == set(DECLARED_HELDOUT), sorted(HELDOUT_CORRUPTIONS)
    overlap = set(HELDOUT_CORRUPTIONS) & set(CORRUPTIONS)
    assert not overlap, overlap
    for name, (parameter, levels, stochastic) in DECLARED_HELDOUT.items():
        spec = HELDOUT_CORRUPTIONS[name]
        assert spec.parameter == parameter, name
        assert tuple(spec.levels) == levels, (name, spec.levels)
        assert spec.stochastic is stochastic, name
        assert spec.severity_in_seed is False, name
        assert len(spec.levels) == len(SEVERITY_LEVELS), name
        for severity in SEVERITY_LEVELS:
            assert heldout_parameter(name, severity) == levels[severity - 1]
        ordered = (all(a < b for a, b in zip(levels, levels[1:]))
                   or all(a > b for a, b in zip(levels, levels[1:])))
        assert ordered, (name, levels)
    assert len(heldout_conditions()) == 25, len(heldout_conditions())

    # No training policy may draw a held-out family, at any weight.
    for policy in POLICIES.values():
        drawn = {name for name, _ in policy.family_weights}
        assert not drawn & set(HELDOUT_CORRUPTIONS), (policy.name, drawn)
    print(f"  5 families x 5 severities = 25 conditions, names disjoint from the "
          f"{len(CORRUPTIONS)} training families")
    print("  no augmentation policy assigns weight to any held-out family")

    for arguments in (("fog", 1), ("motion_blur", 0), ("motion_blur", 6),
                      ("motion_blur", True), ("motion_blur", 2.0)):
        try:
            heldout_parameter(*arguments)
        except ValueError:
            continue
        raise AssertionError(f"accepted {arguments}")
    try:
        apply_heldout(observation(0), "rotation", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("a stochastic family ran without a seed")
    for bad in (observation(0) * 2 - 1, observation(0).unsqueeze(0),
                (observation(0) * 255).to(torch.uint8)):
        try:
            apply_heldout(bad, "rotation", 1, seed=1)
        except (TypeError, ValueError):
            continue
        raise AssertionError("accepted an invalid observation")
    print("  frozen family names, out-of-range severities, missing seeds, and "
          "normalised, batched or integer inputs are all rejected")


def check_heldout_properties():
    """Each held-out family's mechanics must be what its severity claims."""
    image = structured_observation()

    for name in HELDOUT_CORRUPTIONS:
        for severity in SEVERITY_LEVELS:
            seed = heldout_sample_seed(VALIDATION_NAMESPACE, name, severity, 11)
            first = apply_heldout(image, name, severity, seed=seed)
            torch.manual_seed(999)
            second = apply_heldout(image, name, severity, seed=seed)
            assert torch.equal(first, second), (name, severity)
            assert first.shape == image.shape, (name, severity)
            assert first.dtype == image.dtype, (name, severity)
            assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0
            others = [apply_heldout(image, name, severity, seed=seed + step)
                      for step in (1, 2, 3, 4)]
            if HELDOUT_CORRUPTIONS[name].stochastic:
                assert any(not torch.equal(first, other) for other in others), \
                    (name, severity, "seed ignored")
            else:
                assert all(torch.equal(first, other) for other in others), \
                    (name, severity, "not deterministic")
        # The realisation is drawn once per image, so the seed does not move
        # with severity and a per-sample ladder varies only the strength.
        seeds = {heldout_sample_seed(VALIDATION_NAMESPACE, name, severity, 11)
                 for severity in SEVERITY_LEVELS}
        assert len(seeds) == 1, (name, seeds)
    print("  every family: deterministic given its seed, independent of global "
          "RNG, shape and dtype preserved, output inside [0, 1]")
    print("  one realisation seed per image across all five severities, so a "
          "severity curve varies only the strength")

    for length in HELDOUT_CORRUPTIONS["motion_blur"].levels:
        assert length % 2 == 1, length
        for angle in (0.0, 45.0, 90.0, 137.3):
            kernel = _motion_kernel(length, angle)
            assert abs(float(kernel.sum()) - 1.0) < 1e-6, (length, angle)
            assert float(kernel.min()) >= 0.0
        assert int((_motion_kernel(length, 0.0) > 0).sum()) == length
    try:
        _motion_kernel(4, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an even motion-blur length")
    print("  motion kernels are odd-sided, non-negative, and sum to 1 at every "
          "angle")

    for severity, factor in zip(SEVERITY_LEVELS,
                                HELDOUT_CORRUPTIONS["contrast_reduction"].levels):
        reduced = apply_heldout(image, "contrast_reduction", severity)
        ratio = float(reduced.std() / image.std())
        assert abs(ratio - factor) < 1e-4, (severity, ratio, factor)
    print("  contrast reduction scales the pixel standard deviation by exactly "
          "its factor at every severity")

    previous = None
    for severity, fraction in zip(SEVERITY_LEVELS,
                                  HELDOUT_CORRUPTIONS["salt_pepper"].levels):
        seed = heldout_sample_seed(VALIDATION_NAMESPACE, "salt_pepper", severity, 5)
        noisy = apply_heldout(image, "salt_pepper", severity, seed=seed)
        touched = (noisy != image).any(dim=0)
        extremes = noisy[:, touched]
        assert bool(((extremes == 0) | (extremes == 1)).all()), severity
        # Whole pixels, not channels: a touched pixel is black or white in all
        # three channels, which is what distinguishes this from Gaussian noise.
        assert bool(((extremes == 0).all(dim=0) | (extremes == 1).all(dim=0)).all())
        count = int(touched.sum())
        expected = fraction * IMAGE_SIZE * IMAGE_SIZE
        assert abs(count - expected) < 4 * math.sqrt(max(expected, 1)) + 2, \
            (severity, count, expected)
        if previous is not None:
            assert int((previous & ~touched).sum()) == 0, severity
        previous = touched
    print("  salt-and-pepper replaces whole pixels with 0 or 1, at the declared "
          "rate, nested across severities")

    corners = set()
    for index in range(200):
        seed = heldout_sample_seed(VALIDATION_NAMESPACE, "rotation", 5, index)
        rotated = apply_heldout(image, "rotation", 5, seed=seed)
        corners.add(round(float(rotated[0, 0, 0]), 6))
        assert abs(float(rotated[0, 0, 0]) - NEUTRAL_GREY) < 1e-6, index
    assert len(corners) == 1, corners
    signs = set()
    for index in range(200):
        seed = heldout_sample_seed(VALIDATION_NAMESPACE, "rotation", 2, index)
        rotated = apply_heldout(image, "rotation", 2, seed=seed)
        signs.add(float((rotated - image).sum()))
    assert len(signs) >= 2, "rotation never changed direction over 200 draws"
    print(f"  rotation fills the vacated corner with NEUTRAL_GREY="
          f"{NEUTRAL_GREY} and draws both directions over 200 images")

    quality_outputs = {severity: apply_heldout(image, "jpeg_compression", severity)
                       for severity in SEVERITY_LEVELS}
    for severity, output in quality_outputs.items():
        assert torch.equal(output, apply_heldout(image, "jpeg_compression", severity))
        quantised = (output * 255).round()
        assert torch.allclose(output * 255, quantised, atol=1e-4), severity
    assert len({float(output.sum()) for output in quality_outputs.values()}) == 5
    print("  JPEG round trips are bit-stable, land on the 8-bit grid, and differ "
          "at all five qualities")


def check_heldout_ordering(frames):
    """Severity must get harder, measured on real frames and independently.

    Deviation from the clean image is the independent yardstick: it is computed
    here, not by any function in the suite, and it must increase with severity
    for every family. Real frames rather than noise, because a geometric or
    blur ladder saturates immediately on white noise and the check would pass
    or fail for the wrong reason.
    """
    positions = list(range(0, len(frames), max(1, len(frames) // ORDERING_FRAMES)))
    sample = [frames[position][0] for position in positions[:ORDERING_FRAMES]]
    for name in HELDOUT_CORRUPTIONS:
        deviations = []
        for severity in SEVERITY_LEVELS:
            total = 0.0
            for position, image in enumerate(sample):
                seed = heldout_sample_seed(VALIDATION_NAMESPACE, name, severity,
                                           position)
                corrupted = apply_heldout(image, name, severity, seed=seed)
                total += float((corrupted - image).abs().mean())
            deviations.append(total / len(sample))
        assert all(a < b for a, b in zip(deviations, deviations[1:])), \
            (name, deviations)
        print(f"  {name:20s} " + "  ".join(f"{value:.4f}" for value in deviations))
    print(f"  mean absolute deviation increases with severity for all "
          f"{len(HELDOUT_CORRUPTIONS)} families over {len(sample)} real frames")


def check_heldout_dataset(frames, validation_indices):
    """The dataset wrapper must seed by dataset index and normalise once."""
    positions = list(range(0, len(frames), 700))
    subset = Subset(frames, positions)
    _, normalize = split_preprocessing()

    dataset = HeldoutDataset(subset, "rotation", 3)
    direct = [dataset[index][0] for index in range(len(subset))]
    shuffled = DataLoader(dataset, batch_size=3, shuffle=True,
                          generator=torch.Generator().manual_seed(7))
    seen = 0
    for images, _ in shuffled:
        for image in images:
            assert any(torch.equal(image, other) for other in direct)
            seen += 1
    assert seen == len(subset)
    workers = DataLoader(dataset, batch_size=4, num_workers=2)
    assert torch.equal(torch.cat([batch for batch, _ in workers]),
                       torch.stack(direct))

    # Seeded by dataset index, so the same frame gets the same realisation
    # whatever subset it is reached through.
    single = HeldoutDataset(Subset(frames, [positions[2]]), "rotation", 3)
    assert torch.equal(single[0][0], direct[2])

    expected = normalize(apply_heldout(
        frames[positions[1]][0], "rotation", 3,
        seed=heldout_sample_seed(VALIDATION_NAMESPACE, "rotation", 3,
                                 int(validation_indices[positions[1]]))))
    assert torch.equal(direct[1], expected)
    print(f"  {len(subset)} frames: identical under shuffling, two workers, and "
          "a one-element subset, so the realisation keys on the dataset index")
    print("  output equals normalize(apply_heldout(observation)) exactly, so "
          "normalisation runs once")


def check_suites():
    """The evaluator's two suites must cover the declared conditions."""
    assert sorted(SUITES) == ["frozen", "heldout"]
    frozen = SUITES["frozen"].conditions()
    heldout = SUITES["heldout"].conditions()
    assert len(frozen) == 31, len(frozen)
    assert len(heldout) == 26, len(heldout)
    assert frozen[0] == (CLEAN_CONDITION, 0) and heldout[0] == (CLEAN_CONDITION, 0)
    assert len(set(frozen)) == len(frozen) and len(set(heldout)) == len(heldout)
    assert {name for name, _ in frozen[1:]} == set(CORRUPTIONS)
    assert {name for name, _ in heldout[1:]} == set(HELDOUT_CORRUPTIONS)
    for suite in SUITES.values():
        assert suite.parameter_name(CLEAN_CONDITION) == ""
        assert suite.parameter_value(CLEAN_CONDITION, 0) == ""
        for name, severity in suite.conditions()[1:]:
            assert suite.parameter_name(name) == suite.table[name].parameter
            assert suite.parameter_value(name, severity) == \
                suite.table[name].levels[severity - 1]
    print("  frozen: 1 clean + 30 corrupted; held-out: 1 clean + 25 corrupted; "
          "both share the clean condition")


def synthetic_run_records(directory, policy_name, seed, best_epoch, metrics):
    """Write the records a finished training run leaves beside its checkpoint.

    The policy record is produced by `train_augmented.policy_record` and the
    history by the keys `train.py` writes, so the readers in
    `evaluate_checkpoint` are checked against the producers' own schemas rather
    than against a second hand-written copy of them. If the two sides ever
    spell a field differently, that is what has to fail.

    No checkpoint file is written. The returned path names one, which is all
    the metadata readers need from it.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(epochs=best_epoch + 2, seed=seed)
    if policy_name is not None:
        record = policy_record(resolve_policy(policy_name), config, "cpu",
                               {"shared_images": 0, "shared_tracks": 0},
                               EXPECTED_TRAIN_IMAGES)
        record["realised"] = {"train_loader_iterations": config.epochs,
                              "final_augmentation_epoch": config.epochs,
                              "best_epoch": best_epoch, "total_seconds": 1.0}
        (directory / "policy.json").write_text(json.dumps(record, indent=2))
    (directory / "history.json").write_text(json.dumps({
        "config": asdict(config),
        "selection_metric": config.selection_metric,
        "best_epoch": best_epoch,
        "best_validation_accuracy": metrics["accuracy"],
        "best_validation_macro_f1": metrics["macro_f1"],
        "best_validation_loss": metrics["loss"],
        "split_sizes": {"train_images": EXPECTED_TRAIN_IMAGES,
                        "validation_images": EXPECTED_VALIDATION_IMAGES},
        "checkpoint": "best_checkpoint.pt",
    }, indent=2))
    return directory / "best_checkpoint.pt"


def synthetic_identity(path, seed, best_epoch, metrics, digest):
    """The identity block `check_identity` records for one checkpoint.

    Its field names are checked against the real thing in
    `check_provenance_records`, which loads the accepted baseline and compares
    the two key sets, so this fixture cannot drift away from the loader.
    """
    return {
        "path": str(path),
        "md5": digest,
        "is_accepted_baseline": False,
        "epoch": best_epoch,
        "seed": seed,
        "selection_metric": "validation_macro_f1",
        "recorded_validation_metrics": dict(metrics),
        "model_config": {"architecture": "BaselineCNN", "num_classes": NUM_CLASSES,
                         "image_size": IMAGE_SIZE, "dropout": 0.5},
        "training_config": asdict(TrainingConfig(epochs=best_epoch + 2, seed=seed)),
        "trained_on_device": "cpu",
    }


def check_run_metadata():
    """A checkpoint must be filed under the arm that actually produced it."""
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory) / "augmented_targeted_seed43"
        run.mkdir()
        checkpoint = run / "best_checkpoint.pt"
        checkpoint.write_bytes(b"")

        assert resolve_model_name(discover_run_metadata(checkpoint)) == "clean"

        (run / "policy.json").write_text(json.dumps(
            {"policy": TARGETED.describe(),
             "leakage_checks": {"shared_tracks": 0}}))
        metadata = discover_run_metadata(checkpoint)
        assert metadata["policy"]["policy"] == "targeted"
        assert metadata["policy"]["augment_probability"] == \
            DECLARED_AUGMENT_PROBABILITY
        assert metadata["leakage_checks"]["shared_tracks"] == 0
        assert resolve_model_name(metadata) == "targeted"
        assert resolve_model_name(metadata, override="balanced") == "balanced"

        (run / "policy.json").write_text(json.dumps({"leakage_checks": {}}))
        try:
            resolve_model_name(discover_run_metadata(checkpoint))
        except RuntimeError:
            pass
        else:
            raise AssertionError("a nameless policy record was filed as clean")

        (run / "history.json").write_text(json.dumps(
            {"best_epoch": 17, "selection_metric": "validation_macro_f1"}))
        metadata = discover_run_metadata(checkpoint)
        assert metadata["best_epoch"] == 17
        assert metadata["selection_metric"] == "validation_macro_f1"
    print("  a run with no policy record is the clean arm, a run with one is "
          "named by it, and a policy record without a name is refused rather "
          "than filed as clean")


def check_fingerprints(frames):
    """Input fingerprints must identify the pixels, not the model or the run."""
    positions = list(range(0, len(frames), 900))
    subset = Subset(frames, positions)
    digests = {}
    for suite in SUITES.values():
        for name, severity in [(CLEAN_CONDITION, 0)] + [
                (family, severity) for family in suite.table
                for severity in (1, 5)]:
            dataset = suite.dataset(subset, name, severity)
            key = f"{suite.name}:{name}_s{severity}"
            digests[key] = input_fingerprint(dataset, range(len(subset)))
            repeated = input_fingerprint(
                suite.dataset(subset, name, severity), range(len(subset)))
            assert digests[key] == repeated, key
    clean = [value for key, value in digests.items()
             if key.endswith(f"{CLEAN_CONDITION}_s0")]
    assert len(set(clean)) == 1, "the two suites disagree on the clean condition"
    assert len(set(digests.values())) == len(digests) - 1, \
        "two different conditions produced identical inputs"
    assert FINGERPRINT_STRIDE == FRAMES_PER_TRACK
    print(f"  {len(digests)} conditions fingerprinted, each reproducible, all "
          "distinct, and the clean condition identical across both suites")


def check_production_metadata():
    """The metadata an evaluation writes must name the arm that trained it.

    Driven through the functions a real evaluation calls - the policy producer,
    the metadata reader, the model block, the provenance check - so a
    disagreement between how the policy record is written and how it is read
    fails here. A full sweep is not needed to reach them, which is the point:
    the key that named the arm used to be read only after thirty conditions had
    been evaluated.
    """
    described = TARGETED.describe()
    assert described["policy"] == "targeted", described
    assert "name" not in described, \
        "the policy record has both a policy and a name field; readers will split"
    metrics = {"loss": 0.0477, "accuracy": 0.9876, "macro_f1": 0.9744}
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = synthetic_run_records(
            Path(directory) / "augmented_targeted_seed43", "targeted", 43, 17,
            metrics)
        run_metadata = discover_run_metadata(checkpoint)
        identity = synthetic_identity(checkpoint, 43, 17, metrics, "0" * 32)
        provenance = check_provenance(checkpoint, identity, run_metadata)
        assert recorded_policy_name(run_metadata) == "targeted"
        assert resolve_model_name(run_metadata) == "targeted"
        block = model_record(resolve_model_name(run_metadata), 43, run_metadata,
                             provenance)
        assert block["arm"] == "augmented (targeted)", block["arm"]
        assert block["name"] == "targeted" and block["seed"] == 43
        assert block["augmentation_policy"]["policy"] == "targeted"
        assert block["training_records"]["best_epoch"] == 17
        assert block["provenance"]["policy_name"] == "targeted"
        assert block["training_leakage_checks"]["shared_tracks"] == 0

        clean = synthetic_run_records(Path(directory) / "clean_seed43", None,
                                      43, 21, metrics)
        clean_metadata = discover_run_metadata(clean)
        assert recorded_policy_name(clean_metadata) is None
        clean_block = model_record("clean", 43, clean_metadata)
        assert clean_block["arm"] == "clean baseline", clean_block["arm"]
        assert clean_block["augmentation_policy"] is None

        nameless = dict(run_metadata)
        nameless["policy"] = {key: value for key, value in described.items()
                              if key != "policy"}
        for reader in (lambda: resolve_model_name(nameless),
                       lambda: model_record("targeted", 43, nameless)):
            try:
                reader()
            except RuntimeError:
                continue
            raise AssertionError("a policy record without a name was accepted")
    print("  the arm in an evaluation's metadata comes from the policy record "
          "the training run wrote, and a record without a name is refused by "
          "both the name resolver and the metadata builder")


def check_provenance_records():
    """The records beside a checkpoint must be shown to describe that file."""
    if not CHECKPOINT_PATH.exists():
        raise AssertionError(f"no accepted baseline checkpoint at {CHECKPOINT_PATH}")
    _, _, identity = check_identity(CHECKPOINT_PATH, "cpu")
    assert identity["md5"] == EXPECTED_CHECKPOINT_MD5, identity["md5"]
    assert identity["is_accepted_baseline"]
    run_metadata = discover_run_metadata(CHECKPOINT_PATH)
    provenance = check_provenance(CHECKPOINT_PATH, identity, run_metadata)
    assert provenance["policy_name"] is None
    assert provenance["checkpoint_md5"] == EXPECTED_CHECKPOINT_MD5
    assert provenance["best_epoch"] == identity["epoch"]
    assert provenance["training_seed"] == identity["seed"]
    # The accepted baseline predates the protocol sidecar, so exactly one
    # field is unverifiable for it, and it has to be reported rather than
    # quietly dropped.
    assert provenance["unverifiable"] == [
        f"no {PROTOCOL_ARTIFACT} beside {CHECKPOINT_PATH.name}, so the protocol "
        "version this run was started under is not recorded; the accepted "
        "baseline predates the frozen protocol"
    ], provenance["unverifiable"]
    for fragment in ("selected epoch", "training seed", "validation metrics",
                     "filename", "clean recipe"):
        assert any(fragment in entry for entry in provenance["verified"]), fragment
    sample = synthetic_identity(CHECKPOINT_PATH, 42, 21, {}, "0" * 32)
    assert set(sample) == set(identity), \
        f"fixture identity fields {sorted(set(sample) ^ set(identity))}"
    print(f"  the accepted baseline checkpoint ({identity['md5']}) is described "
          f"by its own history: seed {identity['seed']}, epoch "
          f"{identity['epoch']}, selection metric and best validation metrics "
          "all agreeing exactly")

    history = json.loads((ACCEPTED_BASELINE / "history.json").read_text())
    tampered_seed = dict(history["config"], seed=history["config"]["seed"] + 1)
    cases = [
        ("a different best epoch", {"best_epoch": history["best_epoch"] + 1},
         "best epoch"),
        ("a different training seed", {"config": tampered_seed}, "training seed"),
        ("a nudged validation macro F1",
         {"best_validation_macro_f1": history["best_validation_macro_f1"] + 1e-9},
         "validation macro_f1"),
        ("another selection metric", {"selection_metric": "validation_accuracy"},
         "selection metric"),
        ("another checkpoint filename", {"checkpoint": "other.pt"},
         "checkpoint filename"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory) / "baseline_copy"
        run.mkdir()
        named = run / CHECKPOINT_PATH.name
        (run / "history.json").write_text(json.dumps(history))
        untouched = check_provenance(named, identity,
                                     discover_run_metadata(named))
        assert untouched["best_epoch"] == identity["epoch"]

        for label, change, fragment in cases:
            (run / "history.json").write_text(json.dumps({**history, **change}))
            try:
                check_provenance(named, identity, discover_run_metadata(named))
            except RuntimeError as error:
                assert fragment in str(error), (label, str(error))
            else:
                raise AssertionError(f"{label} was accepted as provenance")

        (run / "history.json").write_text(json.dumps(history))
        other = policy_record(resolve_policy("balanced"),
                              TrainingConfig(seed=identity["seed"] + 1), "cpu",
                              {}, EXPECTED_TRAIN_IMAGES)
        (run / "policy.json").write_text(json.dumps(other))
        try:
            check_provenance(named, identity, discover_run_metadata(named))
        except RuntimeError as error:
            assert "policy record" in str(error), error
        else:
            raise AssertionError("a policy record from another seed was accepted")
    print(f"  {len(cases) + 1} altered records - epoch, seed, validation "
          "metric, selection metric, filename, and a policy record from "
          "another seed - are each refused")
    print("  what cannot be verified is reported: the checkpoint format holds "
          "no policy field, so an augmented arm's name rests on the policy.json "
          "beside it")


def check_protocol_identity():
    """The manifest must describe the files as they stand, and runs must say so.

    The manifest is the only thing that makes a recorded digest meaningful, so
    a manifest that has drifted from the implementation it names is a failure
    here rather than a surprise months later.
    """
    manifest_path = PROJECT_ROOT / PROTOCOL_MANIFEST
    assert manifest_path.is_file(), f"no protocol manifest at {manifest_path}"
    manifest = json.loads(manifest_path.read_text())
    observed = protocol_digests()
    assert set(manifest["files"]) == set(PROTOCOL_FILES), sorted(manifest["files"])
    drifted = sorted(name for name in PROTOCOL_FILES
                     if manifest["files"][name] != observed[name])
    assert not drifted, f"{PROTOCOL_MANIFEST} does not describe {drifted}"
    record = protocol_record()
    assert record["matches_manifest"], record["differs_from_manifest"]
    assert record["manifest_sha256"] == file_digest(manifest_path)
    print(f"  {len(PROTOCOL_FILES)} protocol files match their digests in "
          f"{PROTOCOL_MANIFEST}: "
          + ", ".join(f"{Path(name).name} {observed[name][:12]}"
                      for name in PROTOCOL_FILES))

    metrics = {"loss": 0.1, "accuracy": 0.9, "macro_f1": 0.8}
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory) / "run"
        named = synthetic_run_records(run, "targeted", 43, 7, metrics)
        with redirect_stdout(io.StringIO()) as noise:
            write_protocol_sidecar(run, "targeted", 43, "cpu")
        assert PROTOCOL_ARTIFACT in noise.getvalue()
        identity = synthetic_identity(named, 43, 7, metrics, "1" * 32)
        provenance = check_provenance(named, identity,
                                      discover_run_metadata(named))
        assert provenance["protocol"]["matches_manifest"]
        assert provenance["protocol_record"].endswith(PROTOCOL_ARTIFACT)
        assert any("frozen in" in entry for entry in provenance["verified"]), \
            provenance["verified"]

        sidecar = run / PROTOCOL_ARTIFACT
        written = json.loads(sidecar.read_text())
        assert written["repository"]["head_commit"]

        drifted = dict(written)
        drifted["protocol"] = dict(written["protocol"],
                                   differs_from_manifest=["src/augmentation.py"],
                                   matches_manifest=False)
        sidecar.write_text(json.dumps(drifted))
        reported = check_provenance(named, identity,
                                    discover_run_metadata(named))
        assert any("not started under the frozen protocol" in entry
                   for entry in reported["unverifiable"]), reported["unverifiable"]

        sidecar.write_text(json.dumps(dict(written, seed=44)))
        try:
            check_provenance(named, identity, discover_run_metadata(named))
        except RuntimeError as error:
            assert "protocol record" in str(error), error
        else:
            raise AssertionError("a protocol record from another seed was "
                                 "accepted")
    print("  a run records the digests it started under, a drift from the "
          "manifest is reported rather than accepted as a protocol run, and a "
          "sidecar from another seed is refused")
    print("  the sidecar is written beside the checkpoint, so it dates a run "
          "against the protocol; it is not proof that these files produced "
          "that checkpoint")


def check_output_guard():
    """No training run may start on top of an accepted or finished one."""
    protected = [ACCEPTED_BASELINE, ACCEPTED_BASELINE / "nested",
                 PROTECTED_DIRS[1], PROTECTED_DIRS[1] / "baseline",
                 ACCEPTED_BASELINE.parent]
    for target in protected:
        try:
            check_output_dir(target)
        except RuntimeError:
            continue
        raise AssertionError(f"{target} was accepted as a training output "
                             "directory")
    assert ACCEPTED_BASELINE in PROTECTED_DIRS
    # Where a replicate goes is checked as path arithmetic, because seeds 43
    # and 44 have since been trained: `artifacts/clean_seed43` now holds a
    # checkpoint, so the guard refuses it, which is the guard working rather
    # than this check failing. The acceptance half therefore runs on a
    # temporary directory of the same name, and the real directories are
    # required to be refused once they hold output. Nothing here reads,
    # writes, or removes a training artifact.
    trained = []
    for seed in (43, 44):
        target = default_output_dir(seed)
        assert target != ACCEPTED_BASELINE
        assert target.name == f"clean_seed{seed}"
        assert target.parent == ACCEPTED_BASELINE.parent
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / target.name
            assert check_output_dir(fresh) == fresh.resolve()
            assert not fresh.exists(), f"{fresh} was created by a check"
        if not target.exists():
            continue
        trained.append(target.name)
        try:
            check_output_dir(target)
        except RuntimeError as error:
            assert "already holds" in str(error), error
        else:
            raise AssertionError(f"{target} holds a finished run but was "
                                 "accepted as a training output directory")
    with tempfile.TemporaryDirectory() as directory:
        fresh = Path(directory) / "clean_seed43"
        assert claim_output_dir(fresh).is_dir()
        for artifact in ("best_checkpoint.pt", "history.json", "policy.json"):
            written = fresh / artifact
            written.write_bytes(b"")
            try:
                claim_output_dir(fresh)
            except RuntimeError as error:
                assert artifact in str(error), error
            else:
                raise AssertionError(f"a run started on top of {artifact}")
            written.unlink()
    # Nothing here can reach training: `run` settles the seed and claims the
    # directory before it loads any data.
    try:
        clean_replicate(SEED)
    except RuntimeError as error:
        assert "reused as it is" in str(error), error
    else:
        raise AssertionError("a seed-42 replicate defaulted into the accepted "
                             "baseline directory")
    print(f"  the accepted baseline and robustness directories are refused, "
          f"replicates default to {default_output_dir(43).name} and "
          f"{default_output_dir(44).name}, a directory holding any training "
          "artifact is refused, and a seed-42 replicate with no output "
          "directory refuses to start")
    if trained:
        print(f"  the already-trained {', '.join(trained)} are refused as "
              "output directories, so this check stays runnable after the "
              "replicates exist")


def check_output_writer():
    """The suite-parameterised writer must produce the full artifact set."""
    samples = 6
    conditions = [(CLEAN_CONDITION, 0), ("rotation", 1), ("salt_pepper", 5)]
    labels = np.array([index % NUM_CLASSES for index in range(samples)])
    validation_indices = np.arange(samples)
    results = {}
    for position, (name, severity) in enumerate(conditions):
        predictions = torch.tensor(
            [(label + (position > 0)) % NUM_CLASSES for label in labels])
        targets = torch.tensor(labels)
        confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
        for predicted, actual in zip(predictions.tolist(), targets.tolist()):
            confusion[actual, predicted] += 1
        accuracy = float((predictions == targets).float().mean())
        results[(name, severity)] = {
            "loss": 0.1 * (position + 1), "accuracy": accuracy,
            "macro_f1": accuracy, "seen": samples,
            "predictions": predictions, "targets": targets,
            "confusion": confusion,
            "per_class": {"support": np.ones(NUM_CLASSES),
                          "recall": np.full(NUM_CLASSES, accuracy),
                          "f1": np.full(NUM_CLASSES, accuracy)},
        }
    tracks_per_class = {class_id: 1 for class_id in range(NUM_CLASSES)}
    with tempfile.TemporaryDirectory() as directory:
        # The metadata is assembled the way an evaluation assembles it, from a
        # run's own records, rather than hand-written here: the summary columns
        # and the metadata file are the artifacts the comparison reads, so the
        # block they carry has to be the production one.
        measured = {"loss": 0.0477, "accuracy": 0.9876, "macro_f1": 0.9744}
        checkpoint = synthetic_run_records(
            Path(directory) / "augmented_targeted_seed42", "targeted", 42, 11,
            measured)
        run_metadata = discover_run_metadata(checkpoint)
        identity = synthetic_identity(checkpoint, 42, 11, measured, "1" * 32)
        metadata = {
            "model": model_record(resolve_model_name(run_metadata), 42,
                                  run_metadata,
                                  check_provenance(checkpoint, identity,
                                                   run_metadata)),
            "checkpoint": identity,
        }
        assert metadata["model"]["arm"] == "augmented (targeted)"
        paths = write_outputs(SUITES["heldout"], results, conditions, metadata,
                              validation_indices, labels, tracks_per_class,
                              Path(directory) / "results")
        summary = pd.read_csv(paths["summary_csv"])
        assert len(summary) == len(conditions)
        assert list(summary["suite"].unique()) == ["heldout"]
        assert summary.loc[0, "accuracy_drop"] == 0.0
        assert summary.loc[1, "parameter"] == "degrees", summary.loc[1]
        assert summary.loc[1, "parameter_value"] == 3
        assert list(summary["model"].unique()) == ["targeted"]
        assert list(summary["seed"].unique()) == [42]
        predictions = pd.read_csv(paths["predictions_csv"])
        assert len(predictions) == len(conditions) * samples
        assert set(predictions["corruption"]) == {name for name, _ in conditions}
        per_class = pd.read_csv(paths["per_class_csv"])
        assert len(per_class) == len(conditions) * NUM_CLASSES
        matrices = np.load(paths["confusion_npz"])
        assert len(matrices.files) == len(conditions)
        assert json.loads(paths["metadata_json"].read_text()) == metadata
    print(f"  {len(conditions)} synthetic held-out conditions wrote summary, "
          "per-class, confusion, prediction and metadata artifacts with the "
          "expected shapes")


def synthetic_predictions(track_index, conditions, failing):
    """Frame-level predictions whose per-sign error rates are known in advance.

    An arm errs on a prescribed set of whole signs under every corrupted
    condition and on nothing under the clean one, so its mean per-sign error
    rate is exactly `len(failing) / 178` and every paired contrast has a closed
    form. Real track identifiers and all thirty frames per track are used, so
    the invariants the estimators enforce are exercised rather than bypassed.
    """
    base = track_index.reset_index(drop=True)
    base = base.assign(validation_position=np.arange(len(base)))
    wrong = base["track"].isin(failing).to_numpy()
    blocks = []
    for name, severity in conditions:
        block = base.copy()
        block["corruption"] = name
        block["severity"] = severity
        incorrect = wrong & (name != CLEAN_CONDITION)
        block["prediction"] = np.where(
            incorrect, (block["label"] + 1) % NUM_CLASSES, block["label"])
        blocks.append(block)
    frames = pd.concat(blocks, ignore_index=True)
    return frames[["corruption", "severity", "validation_position",
                   "dataset_index", "label", "prediction"]]


def synthetic_results(directory, arms, track_index, seeds, families):
    """Write one evaluator-shaped result directory per arm and seed.

    A subset of each suite's families is written rather than all of them. The
    tables the comparison builds do not depend on how many families are present
    - `check_suites` already pins the condition lists - and a full nine-model,
    fifty-seven-condition fixture is a quarter of a million frames per model for
    no extra coverage.

    Each directory carries the identity block an evaluation writes, built by the
    same `model_record` the evaluator calls, over records written by the policy
    producer. Every arm and seed gets its own checkpoint digest, because the
    comparison is entitled to assume nine distinct checkpoints and has to be
    able to prove it.
    """
    signs = sorted(track_index["track"].unique())
    measured = {"loss": 0.0477, "accuracy": 1.0, "macro_f1": 1.0}
    written = {}
    for seed in seeds:
        for arm, count in arms.items():
            model_dir = Path(directory) / f"{arm}_seed{seed}"
            failing = set(signs[:count])
            best_epoch = 11 + (seed % 10)
            checkpoint = synthetic_run_records(
                model_dir / "run", None if arm == "clean" else arm, seed,
                best_epoch, measured)
            run_metadata = discover_run_metadata(checkpoint)
            identity = synthetic_identity(
                checkpoint, seed, best_epoch, measured,
                md5(f"{arm}_seed{seed}".encode()).hexdigest())
            model_block = model_record(
                arm, seed, run_metadata,
                check_provenance(checkpoint, identity, run_metadata))
            for suite_name, suite in sorted(SUITES.items()):
                conditions = [(CLEAN_CONDITION, 0)] + [
                    (family, severity) for family in families[suite_name]
                    for severity in SEVERITY_LEVELS]
                suite_dir = model_dir / suite_name
                suite_dir.mkdir(parents=True, exist_ok=True)

                frames = synthetic_predictions(track_index, conditions, failing)
                frames.to_csv(suite_dir / "predictions.csv", index=False)

                frames = frames.assign(
                    error=(frames["prediction"] != frames["label"]).astype(float))
                summary = frames.groupby(["corruption", "severity"],
                                         as_index=False)["error"].mean()
                summary["accuracy"] = 1 - summary["error"]
                summary["macro_f1"] = summary["accuracy"]
                summary["loss"] = summary["error"]
                summary.to_csv(suite_dir / "summary.csv", index=False)

                per_class = pd.DataFrame([
                    {"corruption": name, "severity": severity, "class": class_id,
                     "support": 1, "tracks": 1,
                     "recall": 1.0 if name == CLEAN_CONDITION else 0.9,
                     "f1": 1.0}
                    for name, severity in conditions
                    for class_id in range(NUM_CLASSES)])
                per_class.to_csv(suite_dir / "per_class_metrics.csv", index=False)

                (suite_dir / "metadata.json").write_text(json.dumps({
                    "model": model_block,
                    "checkpoint": identity,
                    "corruption_suite": {"input_fingerprints": {
                        f"{name}_s{severity}": f"{suite_name}-{name}-{severity}"
                        for name, severity in conditions}},
                }, indent=2))
            written[(arm, seed)] = model_dir
    return written


def rewrite_metadata(path, change):
    """Apply one change to a result set's metadata and return the original."""
    original = path.read_text()
    payload = json.loads(original)
    change(payload)
    path.write_text(json.dumps(payload, indent=2))
    return original


def check_identity_guards(directory, written, arms, seeds):
    """Mislabelled result directories must stop the comparison.

    The input fingerprints prove the arms saw the same pixels; they say nothing
    about which checkpoint produced a prediction file, and two mislabelled
    directories have identical fingerprints. So each case here is a labelling
    that the fingerprint check cannot see, and each one has to be refused
    before any number is computed. The alternative is a table that reports B's
    error rate in C's column.
    """
    models = {(arm, seed): {"name": arm, "seed": seed, "dir": path,
                            "key": f"{arm}_seed{seed}"}
              for (arm, seed), path in written.items()}
    output = Path(directory) / "analysis_identity"

    def refused(specifications, fragment, label):
        try:
            with redirect_stdout(io.StringIO()):
                compare(specifications, output)
        except RuntimeError as error:
            assert fragment in str(error), (label, str(error))
            return
        raise AssertionError(f"{label} was accepted")

    seed = seeds[0]
    swapped = []
    for (arm, model_seed), model in models.items():
        if model_seed == seed and arm in ("balanced", "targeted"):
            other = "targeted" if arm == "balanced" else "balanced"
            swapped.append(dict(model, dir=models[(other, model_seed)]["dir"]))
        else:
            swapped.append(model)
    refused(swapped, "was supplied as", "B and C result directories swapped")

    wrong_seed = [dict(model) for model in models.values()]
    wrong_seed[0] = dict(wrong_seed[0], seed=max(seeds) + 1,
                         key=f"{wrong_seed[0]['name']}_seed{max(seeds) + 1}")
    refused(wrong_seed, "holds results for seed",
            "a directory from another seed")

    straight = list(models.values())
    duplicate = models[("targeted", seed)]["dir"] / "frozen" / "metadata.json"
    shared = json.loads((models[("balanced", seed)]["dir"] / "frozen"
                         / "metadata.json").read_text())["checkpoint"]["md5"]
    originals = {}
    for suite_name in sorted(SUITES):
        target = models[("targeted", seed)]["dir"] / suite_name / "metadata.json"
        originals[target] = rewrite_metadata(
            target, lambda payload: payload["checkpoint"].update({"md5": shared}))
    refused(straight, "two model runs", "one checkpoint under two arms")
    for target, original in originals.items():
        target.write_text(original)

    original = rewrite_metadata(
        duplicate, lambda payload: payload["checkpoint"].update({"md5": "f" * 32}))
    refused(straight, "did not measure the same checkpoint",
            "two suites measured from different checkpoints")
    duplicate.write_text(original)

    original = rewrite_metadata(
        duplicate, lambda payload: payload["model"].pop("name"))
    refused(straight, "records no model.name", "a result set with no arm name")
    duplicate.write_text(original)

    original = rewrite_metadata(
        duplicate, lambda payload: payload["model"].update({"arm": "clean baseline"}))
    refused(straight, "which is not the", "an augmented run describing itself "
                                          "as the clean baseline")
    duplicate.write_text(original)

    with redirect_stdout(io.StringIO()):
        compare(straight, output)
    recorded = json.loads((output / "comparison.json").read_text())
    assert len(recorded["model_identity"]) == len(models)
    assert {entry["checkpoint_md5"] for entry in recorded["model_identity"].values()} \
        == {md5(key.encode()).hexdigest() for key in
            (f"{arm}_seed{seed}" for arm, seed in models)}
    assert "do not establish" in recorded["input_pairing_note"]
    print(f"  swapped directories, a foreign seed, one checkpoint under two "
          f"arms, two suites from different checkpoints, a missing arm name "
          f"and a mislabelled arm are all refused; the {len(models)} correctly "
          "labelled result sets pass and their checkpoint digests are recorded")


def check_class_balanced(track_index, output_dir, arms, seeds):
    """The secondary class-balanced metric must match its closed form.

    Every failing sign in the fixture is wrong under every corrupted condition
    and right under none, so a class's mean is the fraction of its validation
    signs that fail and the whole metric is the mean of those 43 fractions. The
    primary metric is the fraction of all 178 signs that fail, so the two
    differ here by construction, which is the point of reporting both.
    """
    classes = track_index.groupby("track")["label"].first()
    per_class = pd.DataFrame({"track": classes.index.to_numpy(),
                              "class": classes.to_numpy()})
    signs = sorted(track_index["track"].unique())
    single_sign_classes = int((per_class.groupby("class").size() == 1).sum())
    expected = {}
    for arm, count in arms.items():
        failed = per_class["track"].isin(set(signs[:count]))
        fractions = per_class.assign(failed=failed).groupby("class")["failed"].mean()
        assert len(fractions) == NUM_CLASSES
        expected[arm] = float(fractions.mean())

    table = pd.read_csv(output_dir / "class_balanced_error.csv")
    assert len(table) == len(SUITES) * len(seeds) * len(arms)
    for row in table.to_dict("records"):
        assert abs(row["class_balanced_error"] - expected[row["model"]]) < 1e-12, row
        assert row["classes"] == NUM_CLASSES
        assert row["signs"] == len(signs) == EXPECTED_TRACKS
        assert row["single_sign_classes"] == single_sign_classes
        assert row["ci_low"] <= row["class_balanced_error"] <= row["ci_high"], row

    contrasts = pd.read_csv(output_dir / "class_balanced_contrasts.csv")
    assert len(contrasts) == len(SUITES) * len(seeds) * 3
    for row in contrasts.to_dict("records"):
        difference = expected[row["left"]] - expected[row["right"]]
        assert abs(row["mean_difference"] - difference) < 1e-12, row
        assert row["single_sign_classes"] == single_sign_classes

    primary = pd.read_csv(output_dir / "arm_error_rates.csv")
    distinct = [row for row in primary.to_dict("records")
                if abs(row["mean_sign_error"] - expected[row["model"]]) > 1e-6]
    assert distinct, "the class-balanced metric reproduced the primary one"
    for row in primary.to_dict("records"):
        assert abs(row["mean_sign_error"] - arms[row["model"]] / len(signs)) < 1e-12
    print(f"  class-balanced error equals the mean of {NUM_CLASSES} per-class "
          "fractions for every arm, seed and suite, and its paired contrasts "
          "equal the differences of those means")
    print(f"  {single_sign_classes} classes hold a single validation sign, "
          "recorded in every row; the two co-primary sign-level metrics are "
          "unchanged by its presence")


def check_comparison(track_index):
    """The comparison must recover known contrasts and label them correctly."""
    arms = {"clean": 40, "balanced": 30, "targeted": 20}
    seeds = (42, 43, 44)
    families = {"frozen": ("occlusion", "fog"),
                "heldout": ("rotation", "salt_pepper")}
    signs = int(track_index["track"].nunique())
    with tempfile.TemporaryDirectory() as directory:
        written = synthetic_results(directory, arms, track_index, seeds, families)
        models = [{"name": arm, "seed": seed, "dir": path,
                   "key": f"{arm}_seed{seed}"}
                  for (arm, seed), path in written.items()]
        output_dir = Path(directory) / "analysis"
        # The report is written and discarded rather than printed. Its numbers
        # come from a fixture, and a fabricated comparison table in verifier
        # output is the kind of thing that gets quoted later as a result.
        rendered = io.StringIO()
        with redirect_stdout(rendered):
            result = compare(models, output_dir)
        assert "Targeted-versus-balanced rule" in rendered.getvalue()

        rates = pd.read_csv(output_dir / "arm_error_rates.csv")
        assert len(rates) == len(SUITES) * len(seeds) * len(arms)
        for row in rates.to_dict("records"):
            assert abs(row["mean_sign_error"] - arms[row["model"]] / signs) < 1e-12
            assert row["signs"] == signs == EXPECTED_TRACKS
            assert row["draws"] == 2000

        contrasts = pd.read_csv(output_dir / "paired_contrasts.csv")
        assert len(contrasts) == len(SUITES) * len(seeds) * 3
        for row in contrasts.to_dict("records"):
            expected = (arms[row["left"]] - arms[row["right"]]) / signs
            assert abs(row["difference"] - expected) < 1e-12, (row, expected)
            assert row["signs"] == signs
            # The failing sets are nested, so no sign is worse under the arm
            # with fewer failures and the rest are unchanged.
            assert row["signs_worse_under_left"] == 0, row
            assert row["signs_unchanged"] == signs - (
                arms[row["right"]] - arms[row["left"]]), row
            assert row["excludes_zero"] is np.True_ or row["excludes_zero"]

        across = pd.read_csv(output_dir / "across_seeds.csv")
        assert len(across) == len(SUITES) * 3
        for row in across.to_dict("records"):
            assert row["seeds"] == len(seeds)
            assert row["range"] == 0.0
            assert row["direction"] == "consistent: left lower at every seed"
        assert result["verdict"]["verdict"] == "targeted preferred"

        families_table = pd.read_csv(output_dir / "per_family_contrasts.csv")
        written_families = set(families["frozen"]) | set(families["heldout"])
        assert set(families_table["corruption"]) == written_families
        severities = pd.read_csv(output_dir / "per_severity_contrasts.csv")
        assert set(severities["severity"]) == set(SEVERITY_LEVELS)
        classes = pd.read_csv(output_dir / "per_class_recall.csv")
        assert len(classes) == len(models) * len(SUITES) * NUM_CLASSES
        assert (classes["clean_recall"] == 1.0).all()
        clean = pd.read_csv(output_dir / "clean_performance.csv")
        assert len(clean) == len(models)
        assert (clean["clean_accuracy"] == 1.0).all()
        tradeoff = pd.read_csv(output_dir / "clean_tradeoff.csv")
        assert (~tradeoff["exceeds_clean_limit"]).all()
        print(f"  arms failing on {arms} of {signs} signs over {len(seeds)} "
              "seeds: every arm rate and paired contrast equals its "
              "closed-form value exactly")
        print("  the rendered report is checked for its sections and discarded; "
              "its numbers are fixture values, not measurements")
        print(f"  {len(contrasts)} contrasts, each resampling {signs} whole "
              f"signs {2000} times; per-family, per-severity, per-class, clean "
              "and trade-off tables all written")

        check_class_balanced(track_index, output_dir, arms, seeds)
        check_identity_guards(directory, written, arms, seeds)

        # The estimator must reject artifacts that are not paired.
        tampered = Path(directory) / "targeted_seed42" / "frozen" / "metadata.json"
        payload = json.loads(tampered.read_text())
        key = next(iter(payload["corruption_suite"]["input_fingerprints"]))
        payload["corruption_suite"]["input_fingerprints"][key] = "different"
        tampered.write_text(json.dumps(payload))
        try:
            with redirect_stdout(io.StringIO()):
                compare(models, Path(directory) / "analysis_tampered")
        except RuntimeError as error:
            assert "not paired" in str(error), error
        else:
            raise AssertionError("compared arms evaluated on different inputs")
    print("  a single mismatched input fingerprint stops the comparison instead "
          "of producing an unpaired result")


def check_interpretation_rules():
    """The declared labels must fire on the cases they were written for."""
    assert ARM_LABELS == {"clean": "A", "balanced": "B", "targeted": "C"}
    assert CLEAN_TRADEOFF_LIMIT == 0.01
    assert "consistent" in _direction(3, 0, 3)
    assert "consistent" in _direction(0, 3, 3)
    assert "seed-sensitive" in _direction(2, 1, 3)
    assert "inconclusive" in _direction(2, 1, 3)

    def summaries(frozen_favouring, heldout_favouring, seeds=3):
        return [{"suite": "frozen", "contrast": "C-B", "seeds": seeds,
                 "seeds_favouring_left": frozen_favouring},
                {"suite": "heldout", "contrast": "C-B", "seeds": seeds,
                 "seeds_favouring_left": heldout_favouring}]

    assert targeted_verdict(summaries(3, 3), 3)["verdict"] == "targeted preferred"
    partial = targeted_verdict(summaries(3, 2), 3)
    assert partial["verdict"] == "frozen-suite only, not a general improvement"
    assert "headroom" in partial["detail"]
    assert targeted_verdict(summaries(1, 1), 3)["verdict"] == \
        "no advantage for the targeted weights"
    assert targeted_verdict(summaries(3, 3)[:1], 3)["verdict"] == "incomplete"
    print("  C is only preferred when it wins at every seed on both suites; a "
          "frozen-suite-only win is labelled as such, not as a win")
    print(f"  a clean-accuracy cost above {CLEAN_TRADEOFF_LIMIT:g} is reported as "
          "a trade-off")


def check_frozen_modules():
    """The modules the accepted results depend on must be unchanged at HEAD."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *FROZEN_MODULES],
        capture_output=True, text=True, check=True)
    changed = [line for line in result.stdout.splitlines() if line]
    assert not changed, f"frozen modules modified: {changed}"
    print(f"  {len(FROZEN_MODULES)} modules identical to HEAD: "
          + ", ".join(Path(name).name for name in FROZEN_MODULES))


def main():
    print("Augmentation policies:")
    check_policies()

    print("\nPer-epoch seeding:")
    check_seed_derivation()

    print("\nRealised draw distribution:")
    check_plan_distribution()

    print("\nTraining occlusion placement:")
    check_occlusion_placement()

    print("\nAugmented dataset:")
    check_augmented_dataset()

    print("\nEpoch advance:")
    check_epoch_loader()

    print("\nHeld-out specification:")
    check_heldout_specification()

    print("\nHeld-out mechanics:")
    check_heldout_properties()

    print("\nEvaluation suites:")
    check_suites()

    print("\nRun metadata:")
    check_run_metadata()

    print("\nEvaluation metadata, through the production path:")
    check_production_metadata()

    print("\nCheckpoint provenance:")
    check_provenance_records()

    print("\nProtocol identity:")
    check_protocol_identity()

    print("\nTraining output directories:")
    check_output_guard()

    print("\nOutput writer:")
    check_output_writer()

    frames, validation_indices, _ = build_validation_observations()
    print(f"\nHeld-out severity ordering ({len(frames)} validation frames "
          "available):")
    check_heldout_ordering(frames)

    print("\nHeld-out dataset wrapper:")
    check_heldout_dataset(frames, validation_indices)

    print("\nInput fingerprints:")
    check_fingerprints(frames)

    print("\nClean-recipe parity and split integrity:")
    check_clean_parity()

    from .compare_models import validation_track_index
    track_index = validation_track_index()
    print(f"\nModel comparison ({track_index['track'].nunique()} signs x "
          f"{FRAMES_PER_TRACK} frames):")
    check_comparison(track_index)

    print("\nInterpretation rules:")
    check_interpretation_rules()

    print("\nFrozen modules:")
    check_frozen_modules()

    print("\nAll augmentation checks passed. No model was trained, no checkpoint "
          "was written, and the official test split was never read.")


if __name__ == "__main__":
    main()
