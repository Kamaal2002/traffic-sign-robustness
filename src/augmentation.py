"""Training-time corruption augmentation for the controlled augmentation study.

This module supplies the training distributions for Models B and C. It reads
every corruption definition and every severity parameter from `corruptions.py`,
so an augmented image is degraded by exactly the transformations the frozen
benchmark measures, at exactly the same five parameter values.

Why the evaluation wrapper cannot be reused
-------------------------------------------
`CorruptedDataset` derives its realisation from the dataset index alone, so a
sample receives the same noise field or the same occlusion patch in every epoch.
That is the right property for a benchmark - a condition has to mean one fixed
thing - and the wrong one for training, where it would turn augmentation into a
fixed second copy of the dataset. Here the epoch enters the seed, so the
realisation changes every epoch while remaining reproducible.

Seed derivation
---------------
    base seed | train namespace | epoch | stable dataset index

digested with BLAKE2b exactly as `corruptions.corruption_seed` does, and with a
namespace that no evaluation condition uses. The payload holds no batch
position, no shuffle position and no worker id, so the augmentation a sample
receives is a function of (seed, epoch, image) and nothing else: the same run
replays identically, two arms sharing a seed see the same decisions, and the
loader's shuffle cannot change what any image gets.

Draw order
----------
Four values come off one per-image generator in a fixed order:

1. whether to augment at all,
2. the severity,
3. the family,
4. a realisation seed, passed to the frozen corruption function.

The order is deliberate and is part of the policy contract. Both policies in
this study share the augmentation probability and the severity distribution, so
drawing those first means two arms at the same training seed augment *the same
images at the same severities* and differ only in which family lands on them.
That removes two sources of variation from the B-versus-C contrast, which is
the comparison the study turns on. It is only valid because the severity
distribution does not depend on the family; a policy where it did would need a
different order and a different pairing argument.

Occlusion placement
-------------------
Training occlusion is placed uniformly over the whole frame rather than through
the benchmark's `occlusion_anchor`. The benchmark draws one anchor per image
and nests all five patches around it, which constrains every patch to the
central window that leaves room for the severity-5 patch. Copying that
distribution into training would make the training and evaluation occlusion
distributions identical, and a gain on the benchmark's occlusion conditions
would then be unreadable: it could not be separated from having trained on the
evaluation distribution. Uniform whole-frame placement is a strict superset of
the central window, so robustness measured on the benchmark is transfer from a
broader distribution to a narrower one. The patch side and the mid-grey fill
still come from the frozen definitions, so only the position differs.

The cost is stated rather than hidden: a uniformly placed patch lands on sign
content less often than a centred one, so the share of genuinely hard occlusion
samples is lower than the raw count of occluded images.
"""

import json
from dataclasses import dataclass
from hashlib import blake2b, sha256
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import GTSRB

from .corruptions import (
    CORRUPTIONS,
    OCCLUSION_FILL,
    SEVERITY_LEVELS,
    VALUE_TOLERANCE,
    apply_corruption,
    occlusion_side,
    split_preprocessing,
    stable_index,
)
from .data import (
    IMAGE_SIZE,
    SEED,
    build_preprocessing,
    read_training_index,
    split_tracks,
)

# Namespace for augmentation seeds. Distinct from `VALIDATION_NAMESPACE` and
# from any split name the benchmark uses, so a training realisation can never
# coincide with an evaluated one even at the same dataset index.
TRAIN_NAMESPACE = "train"

# Share of training images that receive one corruption. Fixed before training.
AUGMENT_PROBABILITY = 0.5

# Allows float rounding in a declared weight table, not a missing entry.
WEIGHT_TOLERANCE = 1e-9

# Recorded in metadata so the placement rule travels with every run.
OCCLUSION_PLACEMENT = "uniform_whole_frame"

# Severity is uniform in both arms, so the B-versus-C difference is the family
# distribution alone. A distribution tilted towards low severities would be a
# tuned parameter with nothing measured behind it.
SEVERITY_WEIGHTS = {severity: 1 / len(SEVERITY_LEVELS) for severity in SEVERITY_LEVELS}

# Model B: uniform over the six frozen families. The benchmark allocates five of
# its thirty conditions to each family, so this is the uniform measure over the
# benchmark's own condition set and introduces no further parameter. Note that
# it gives the illumination axis a third of the mass, because underexposure and
# overexposure are two families in the frozen table.
BALANCED_WEIGHTS = {name: 1 / len(CORRUPTIONS) for name in CORRUPTIONS}

# Model C: weights derived from the committed baseline robustness curve, by a
# rule fixed before training. Each family's mean clean-to-corrupted accuracy
# drop over severities 1-5 (occlusion 0.4133, gaussian_noise 0.1897, fog 0.1051,
# brightness_bright 0.0430, brightness_dark 0.0155, gaussian_blur 0.0088) is
# normalised to a probability, a 5% floor is applied per family, the remainder
# is distributed proportionally, and the result is rounded to two decimals with
# the rounding residual taken by fog.
#
# The floor is what keeps C a reweighting of B's support rather than a subset of
# it: an arm that never saw blur could not tell us whether its blur robustness
# held up because blur is intrinsically mild or because general augmentation
# transfers to it, and the comparison would stop being about one variable.
TARGETED_WEIGHTS = {
    "gaussian_blur": 0.06,
    "brightness_dark": 0.06,
    "brightness_bright": 0.09,
    "gaussian_noise": 0.22,
    "fog": 0.15,
    "occlusion": 0.42,
}


def augmentation_seed(epoch, dataset_index, base_seed=SEED,
                      namespace=TRAIN_NAMESPACE):
    """Seed for one image in one epoch, from stable identifiers only.

    Same construction as `corruptions.corruption_seed`: BLAKE2b over a
    pipe-separated payload, truncated to 63 bits so it is always a valid
    `manual_seed` value. `hash()` is unusable here because Python randomises
    string hashing per process.
    """
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError(f"epoch must be an int >= 1; got {epoch!r}")
    payload = f"{base_seed}|{namespace}|{epoch}|{int(dataset_index)}".encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big") >> 1


def _weighted_choice(table, generator):
    """Draw one key from an ordered (key, weight) table with a single uniform.

    The cumulative sum is taken in the table's own order, so the mapping from
    the uniform draw to a key is fixed by the policy rather than by dictionary
    iteration order at call time. One uniform per decision is also what makes
    two policies sharing a draw order comparable: the same number is mapped
    through two different tables.
    """
    draw = torch.rand((1,), generator=generator).item()
    total = 0.0
    for key, weight in table:
        total += weight
        if draw < total:
            return key
    # Reached only when the weights sum to just under 1 in float arithmetic.
    return table[-1][0]


@dataclass(frozen=True)
class AugmentationPolicy:
    """One training augmentation distribution, fixed before training.

    `family_weights` and `severity_weights` are ordered tuples of (key, weight)
    pairs rather than dictionaries, because the order is what the uniform draw
    is mapped through and therefore part of the reproducible definition.
    """

    name: str
    augment_probability: float
    family_weights: tuple
    severity_weights: tuple

    def __post_init__(self):
        if not 0.0 <= self.augment_probability <= 1.0:
            raise ValueError(
                f"augment_probability must lie in [0, 1]; got "
                f"{self.augment_probability!r}")
        for label, table, allowed in (
            ("family", self.family_weights, set(CORRUPTIONS)),
            ("severity", self.severity_weights, set(SEVERITY_LEVELS)),
        ):
            keys = [key for key, _ in table]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{label} weights repeat a key: {keys}")
            unknown = set(keys) - allowed
            if unknown:
                raise ValueError(f"unknown {label} keys {sorted(unknown)}; "
                                 f"expected a subset of {sorted(allowed)}")
            if any(weight <= 0 for _, weight in table):
                raise ValueError(f"{label} weights must be positive: {table}")
            total = sum(weight for _, weight in table)
            if abs(total - 1.0) > WEIGHT_TOLERANCE:
                raise ValueError(
                    f"{label} weights sum to {total!r}, expected 1 within "
                    f"{WEIGHT_TOLERANCE:g}")

    def describe(self):
        """The recipe as plain data, for the run's saved metadata."""
        return {
            "policy": self.name,
            "augment_probability": self.augment_probability,
            "family_weights": {name: weight for name, weight in self.family_weights},
            "severity_weights": {str(severity): weight
                                 for severity, weight in self.severity_weights},
            "families_per_image": 1,
            "occlusion_placement": OCCLUSION_PLACEMENT,
            "occlusion_nested": False,
            "severity_levels": list(SEVERITY_LEVELS),
            "namespace": TRAIN_NAMESPACE,
            "draw_order": ["augment", "severity", "family", "realisation_seed"],
        }

    def plan(self, seed):
        """Decide what happens to one image, without touching the image.

        Separating the decisions from the pixels keeps the draw order in one
        place and lets the verifier enumerate the realised distribution over
        many images and epochs without decoding any image.
        """
        generator = torch.Generator().manual_seed(int(seed))
        if torch.rand((1,), generator=generator).item() >= self.augment_probability:
            return {"augmented": False, "family": None, "severity": None,
                    "realisation_seed": None, "top": None, "left": None,
                    "side": None}
        severity = _weighted_choice(self.severity_weights, generator)
        family = _weighted_choice(self.family_weights, generator)
        # Drawn for every family, including the deterministic ones, so the
        # generator consumes the same number of values whatever lands here and
        # the occlusion placement that follows is not shifted by the choice.
        realisation = int(torch.randint(0, 1 << 62, (1,), generator=generator).item())
        top = left = side = None
        if family == "occlusion":
            side = occlusion_side(
                CORRUPTIONS["occlusion"].levels[severity - 1], IMAGE_SIZE, IMAGE_SIZE)
            top, left = _uniform_patch(IMAGE_SIZE, IMAGE_SIZE, side, generator)
        return {"augmented": True, "family": family, "severity": severity,
                "realisation_seed": realisation, "top": top, "left": left,
                "side": side}

    def apply(self, image, plan):
        """Realise a plan on one `[0, 1]` observation.

        Returns a new tensor; `image` is never modified in place. The label is
        not an argument, so no code path here can alter it.
        """
        if not plan["augmented"]:
            return image
        if plan["family"] == "occlusion":
            return _occlude(image, plan["side"], plan["top"], plan["left"])
        return apply_corruption(image, plan["family"], plan["severity"],
                                seed=plan["realisation_seed"])


BALANCED = AugmentationPolicy(
    name="balanced",
    augment_probability=AUGMENT_PROBABILITY,
    family_weights=tuple((name, BALANCED_WEIGHTS[name]) for name in CORRUPTIONS),
    severity_weights=tuple(SEVERITY_WEIGHTS.items()),
)

TARGETED = AugmentationPolicy(
    name="targeted",
    augment_probability=AUGMENT_PROBABILITY,
    family_weights=tuple((name, TARGETED_WEIGHTS[name]) for name in CORRUPTIONS),
    severity_weights=tuple(SEVERITY_WEIGHTS.items()),
)

POLICIES = {policy.name: policy for policy in (BALANCED, TARGETED)}


def _uniform_patch(height, width, side, generator):
    """Top-left corner drawn uniformly over the positions where the patch fits.

    Both axes are inclusive of the flush positions, so the patch can reach the
    frame border at every severity - the property the benchmark's nested anchor
    gives up. The vertical coordinate is drawn first, matching
    `occlusion_anchor`.
    """
    if side > min(height, width):
        raise ValueError(f"patch side {side} does not fit in {height}x{width}")
    top = torch.randint(0, height - side + 1, (1,), generator=generator).item()
    left = torch.randint(0, width - side + 1, (1,), generator=generator).item()
    return int(top), int(left)


def _occlude(image, side, top, left):
    """Mid-grey square patch at a given position.

    The fill is the benchmark's `OCCLUSION_FILL`, which the frozen
    `Normalize(0.5, 0.5)` maps to exactly 0, so the patch removes information
    without injecting a signal of its own. Patches are not nested across
    severities here: nesting exists in the benchmark so that a per-sample
    severity curve isolates patch area from patch-position luck, and training
    has no such curve to read.
    """
    height, width = image.shape[-2:]
    if not (0 <= top <= height - side and 0 <= left <= width - side):
        raise ValueError(f"patch {side}px at ({top}, {left}) leaves the "
                         f"{height}x{width} frame")
    occluded = image.clone()
    occluded[..., top:top + side, left:left + side] = OCCLUSION_FILL
    return occluded


def check_observation(image):
    """Reject anything that is not a single `[0, 1]` CPU observation.

    The occlusion path does not go through `apply_corruption`, so the guard
    that protects the other families cannot protect it. Checking here means an
    already-normalised tensor fails loudly instead of being augmented as if it
    were an image.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor; got {type(image).__name__}")
    if image.ndim != 3:
        raise ValueError(f"image must have shape (C, H, W); got {tuple(image.shape)}")
    if not torch.is_floating_point(image):
        raise TypeError(f"image must be a float tensor; got dtype {image.dtype}")
    if image.device.type != "cpu":
        raise ValueError("augmentation runs CPU-side before collation; got a "
                         f"{image.device.type} tensor")
    low, high = float(image.min()), float(image.max())
    if low < -VALUE_TOLERANCE or high > 1.0 + VALUE_TOLERANCE:
        raise ValueError(
            f"image values must lie in [0, 1] before augmentation; got "
            f"[{low:.4f}, {high:.4f}]")


class AugmentedDataset(Dataset):
    """Corrupt a share of each epoch's training images, then normalise.

    `dataset` must yield `([0, 1] tensor, label)` pairs - build it with the
    observation half of `split_preprocessing()`, never with the full clean
    pipeline, or normalisation would run twice.

    The epoch has to be set before the first read. A dataset that defaulted to
    epoch 0 would train thirty epochs on one fixed realisation and look exactly
    like a working augmenter, so the unset state raises instead.
    """

    def __init__(self, dataset, policy, base_seed=SEED, namespace=TRAIN_NAMESPACE):
        self.dataset = dataset
        self.policy = policy
        self.base_seed = base_seed
        self.namespace = namespace
        self._epoch = None
        _, self.normalize = split_preprocessing()

    @property
    def epoch(self):
        return self._epoch

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
            raise ValueError(f"epoch must be an int >= 1; got {epoch!r}")
        self._epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def plan_for(self, index, epoch=None):
        """The decision record for one position, without reading the image."""
        epoch = self._epoch if epoch is None else epoch
        if epoch is None:
            raise RuntimeError("set_epoch() must be called before sampling")
        seed = augmentation_seed(epoch, stable_index(self.dataset, index),
                                 self.base_seed, self.namespace)
        plan = self.policy.plan(seed)
        plan["epoch"] = epoch
        plan["dataset_index"] = stable_index(self.dataset, index)
        return plan

    def __getitem__(self, index):
        if self._epoch is None:
            raise RuntimeError("set_epoch() must be called before sampling")
        image, label = self.dataset[index]
        check_observation(image)
        plan = self.policy.plan(
            augmentation_seed(self._epoch, stable_index(self.dataset, index),
                              self.base_seed, self.namespace))
        image = self.policy.apply(image, plan)
        # Every corruption is already clamped - `apply_corruption` does it
        # centrally and the patch fill is a constant in range - so this is an
        # invariant check on the augmented image, not a second clamp that could
        # mask a family leaving the valid domain.
        low, high = float(image.min()), float(image.max())
        if low < -VALUE_TOLERANCE or high > 1.0 + VALUE_TOLERANCE:
            raise RuntimeError(
                f"{plan['family']} s{plan['severity']} left the valid image "
                f"domain: [{low:.4f}, {high:.4f}]")
        return self.normalize(image), label


class EpochLoader:
    """Advance the augmentation epoch each time the training loader is iterated.

    `run_experiment` builds one iterator over the training loader per epoch and
    its docstring already relies on that - a diagnostic pass over the loader
    before epoch 1 would change the first epoch's shuffle, so pre-flight checks
    are required to build their own loaders. Counting iterations here therefore
    counts epochs, and the training loop needs no change at all.

    The count is readable afterwards, so "the augmentation saw thirty distinct
    epochs" is a number the run can be checked against rather than a property
    of control flow. Built on the same forwarding pattern as
    `evaluate_robustness.CountingLoader`: ordinary attribute reads and `len`
    reach the wrapped loader, which is everything the training loop uses.
    """

    def __init__(self, loader, augmented):
        self.loader = loader
        self.augmented = augmented
        self.epochs = 0

    def __getattr__(self, name):
        if name == "loader":
            raise AttributeError(name)
        return getattr(self.loader, name)

    def __iter__(self):
        self.epochs += 1
        self.augmented.set_epoch(self.epochs)
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def build_augmented_loaders(policy, base_seed=SEED, batch_size=64, num_workers=0,
                            data_root=None):
    """Loaders for an augmented training run: augmented train, clean validation.

    The official test split is not constructed. `create_data_loaders` builds all
    three, and a training run has no business holding a handle on the test
    dataset at all, so the two datasets this needs are built directly from the
    same track-aware split and the same preprocessing functions.

    Two properties the comparison depends on:

    * The shuffle generator is seeded with `data.SEED`, exactly as
      `create_data_loaders` does, and not with the training seed. Every arm at
      every seed therefore walks the training set in the same order, which is
      what makes "B and C share a shuffle sequence" true, and true of the clean
      baseline as well.
    * Validation is a separate dataset instance carrying the unmodified clean
      pipeline, so no augmentation can reach it.
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / "data"
    observation, _ = split_preprocessing()
    train_source = GTSRB(root=data_root, split="train", transform=observation)
    validation_source = GTSRB(root=data_root, split="train",
                              transform=build_preprocessing())

    labels, tracks = read_training_index(train_source)
    train_indices, validation_indices = split_tracks(labels, tracks)

    augmented = AugmentedDataset(
        Subset(train_source, train_indices.tolist()), policy, base_seed=base_seed)
    shuffle_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        augmented, batch_size=batch_size, shuffle=True, generator=shuffle_generator,
        num_workers=num_workers, drop_last=False,
    )
    validation_loader = DataLoader(
        Subset(validation_source, validation_indices.tolist()),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False,
    )
    loaders = {"train": EpochLoader(train_loader, augmented),
               "validation": validation_loader}
    split_info = {
        "labels": labels,
        "tracks": tracks,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    return loaders, split_info, augmented


def expected_counts(policy, images):
    """Images per epoch each family is expected to receive, for the metadata.

    Expectations, not a realised tally: the draws are per image and per epoch,
    so the realised counts fluctuate around these.
    """
    augmented = policy.augment_probability * images
    return {
        "images": images,
        "augmented": augmented,
        "clean": images - augmented,
        "per_family": {name: augmented * weight
                       for name, weight in policy.family_weights},
    }


def format_policy(policy):
    """Render a policy as the table the run prints and the report quotes."""
    lines = [f"{policy.name}: P(augment) = {policy.augment_probability:g}, "
             f"one family per augmented image, "
             f"occlusion placement {OCCLUSION_PLACEMENT}"]
    for name, weight in policy.family_weights:
        lines.append(f"  {name:18s} {weight:.4f}")
    severities = "  ".join(f"s{severity}={weight:.2f}"
                           for severity, weight in policy.severity_weights)
    lines.append(f"  severity           {severities}")
    return "\n".join(lines)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The frozen protocol: the document that states the experimental decisions, and
# the two implementations that realise the distributions it describes. The
# training recipe itself is not listed, because it is `train.py` unchanged and
# the commit recorded beside every run already pins it.
PROTOCOL_FILES = (
    "docs/augmentation_protocol.md",
    "src/augmentation.py",
    "src/corruptions_heldout.py",
)

PROTOCOL_MANIFEST = "docs/augmentation_manifest.json"

# The sidecar every training run in this experiment writes before it starts,
# naming the protocol version it was started under. Named here, beside the
# manifest it refers to, so the training entry points and the evaluator agree
# on the filename without one importing the other.
PROTOCOL_ARTIFACT = "protocol.json"


def file_digest(path):
    """SHA-256 of a file's bytes, or None if it is not there."""
    path = Path(path)
    if not path.is_file():
        return None
    return sha256(path.read_bytes()).hexdigest()


def protocol_digests():
    """Digests of the protocol files as they stand on disk right now."""
    return {name: file_digest(PROJECT_ROOT / name) for name in PROTOCOL_FILES}


def protocol_record():
    """Associate a run with the frozen protocol version it was started under.

    The record holds the digest of every protocol file as the run observed it,
    the digest of the manifest, and the list of files whose digest differs from
    the manifest's. A run started against an edited protocol therefore says so
    in its own metadata instead of looking like a run of the frozen one.

    What this does not establish: the record is written beside the checkpoint,
    not inside it, so it is evidence about the state of the repository when the
    run started, not proof that these files produced that checkpoint. Anyone
    holding the directory could write any record into it. The guarantee it does
    give is the useful one in practice - a run whose files had drifted from the
    frozen protocol cannot quietly be reported as a run of the protocol.
    """
    manifest_path = PROJECT_ROOT / PROTOCOL_MANIFEST
    observed = protocol_digests()
    recorded = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        recorded = manifest.get("files", {})
    differences = sorted(name for name in PROTOCOL_FILES
                         if recorded.get(name) != observed[name])
    return {
        "manifest": PROTOCOL_MANIFEST,
        "manifest_sha256": file_digest(manifest_path),
        "files": observed,
        "differs_from_manifest": differences,
        "matches_manifest": bool(recorded) and not differences,
    }
