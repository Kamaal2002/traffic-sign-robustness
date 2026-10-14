"""Image corruptions for the robustness evaluation.

This module is the single source of truth for what each corruption condition
means and how strong each of its five severity levels is. Evaluation and, later,
targeted augmentation both read their parameters from `CORRUPTIONS`, so the two
cannot drift apart and end up comparing different definitions of "severity 3".

Where the corruption is applied
-------------------------------
A corruption is meant to model a degraded *observation* of a sign, not an
arbitrary perturbation of an already-normalised model input. The clean pipeline
in `data.py` is `Resize -> ToTensor -> Normalize`, so corruptions are inserted
between `ToTensor` and `Normalize`: they act on the resized RGB image in
`[0, 1]`, and the frozen `Normalize` then runs exactly once, unchanged.

The halves are taken from `build_preprocessing()` itself rather than rewritten
here, so there is no second preprocessing implementation that could drift from
`data.py`. `split_preprocessing` checks the stage layout and fails loudly if
that pipeline ever changes shape.

Each corruption function implements only its transformation; `apply_corruption`
clamps the result to the valid `[0, 1]` image domain once, centrally. Several
conditions need that clamp as part of what they model (overexposure saturating
highlights, noise clipping at the extremes), and the rest need it because float
arithmetic does not respect an analytic bound exactly: a normalised float32
Gaussian kernel sums to 1 only to about 1e-7, so blurring an all-white region
can land marginally above 1.

Reproducibility
---------------
Every stochastic realisation is derived from `base seed | split | condition |
severity | dataset index`, so it depends on which sample is being corrupted and
never on batch position or iteration order. Severity participates only where a
condition is meant to change realisation with it: noise draws a fresh field per
level, while occlusion draws one anchor per image and grows a nested patch
around it.

Corrupting after the resize is deliberate. GTSRB crops range from roughly 15 to
250 pixels on a side, so a blur or noise applied at native resolution and then
resized would have an effective strength that depends on how large the original
crop was, and severity 3 would mean something different for every image.
Applying it after the resize defines severity in the model's input space, which
is consistent across samples and directly interpretable. The cost is that the
degradation models the observation the model receives rather than a sensor-level
effect, since the image has already been resampled once.
"""

import math
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB
from torchvision.transforms import functional as TF

from .data import SEED, build_preprocessing, read_training_index, split_tracks

SEVERITY_LEVELS = (1, 2, 3, 4, 5)

# Namespace for the split a corrupted sample comes from. It is part of the seed
# payload so that, when the official test split is evaluated at the end of the
# project, a test sample never shares a noise field or occlusion anchor with the
# validation sample that happens to sit at the same dataset index.
VALIDATION_NAMESPACE = "validation"

# Fog veils the image towards white; occlusion fills its patch with mid grey.
FOG_COLOUR = 1.0
OCCLUSION_FILL = 0.5

# Hoisted out of CORRUPTIONS because the occlusion geometry needs the largest
# level before the table itself is built: every severity is centred on one
# anchor, and that anchor has to leave room for the severity-5 patch.
OCCLUSION_LEVELS = (0.05, 0.1, 0.15, 0.2, 0.3)

# Images are float tensors straight out of ToTensor(), so the valid range is
# exact rather than approximate. The tolerance only absorbs float error from
# callers that compute an observation themselves.
VALUE_TOLERANCE = 1e-6


def _blur_kernel_size(sigma):
    """Odd kernel width covering +-3 sigma, where the Gaussian is ~0.

    The kernel is derived from sigma rather than listed per severity so that
    each level has exactly one free parameter and the truncation error stays
    negligible at every level.
    """
    return 2 * math.ceil(3 * sigma) + 1


def occlusion_side(area_fraction, height, width):
    """Side length in pixels of the square patch for a target area fraction.

    Rounding to whole pixels means the realised area differs slightly from the
    requested fraction, so the realised value is what gets reported.
    """
    side = round(math.sqrt(area_fraction * height * width))
    return max(1, min(side, min(height, width)))


def _gaussian_blur(image, sigma, generator):
    """Isotropic Gaussian blur. Deterministic; `generator` is unused.

    A normalised Gaussian kernel is a convex combination of pixel values, so the
    result is analytically inside [0, 1]; the caller's clamp only absorbs the
    float32 error in the kernel weights.
    """
    kernel_size = _blur_kernel_size(sigma)
    return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def _brightness(image, gain, generator):
    """Multiplicative exposure change, `out = gain * image`, then clamped.

    Multiplicative rather than additive because a change in illumination or
    exposure scales scene radiance; an additive offset is a veiling effect,
    which is what `fog` models. With gain > 1 the caller's clamp is what
    saturates highlights, which is the irreversible part of overexposure.

    The gain is fixed per severity and is deliberately not adapted to each
    image's own luminance. GTSRB exposure varies enormously, so the same gain
    pushes an already-dark crop towards black sooner than a bright one, and
    saturates a bright crop sooner than a dark one. That is the intended
    property: a severity level names a fixed photometric change applied to
    every image, not a fixed amount of perceived damage. Normalising the gain
    per image would make the condition depend on the test image's own
    statistics and would no longer measure what a fixed lighting change costs.
    Deterministic; `generator` is unused.
    """
    return image * gain


def _gaussian_noise(image, sigma, generator):
    """Zero-mean i.i.d. Gaussian noise added in the [0, 1] domain.

    The noise drawn here is exactly zero-mean, but the caller's clamp makes the
    realised perturbation slightly biased for pixels near 0 or 1. That is a
    property of any bounded-range noise model and is accepted here.
    """
    noise = torch.randn(
        image.shape, generator=generator, dtype=image.dtype, device=image.device
    )
    return image + sigma * noise


def _fog(image, veil, generator):
    """White veiling, `out = (1 - veil) * image + veil * FOG_COLOUR`.

    This is a uniform contrast reduction towards white, not a physical
    atmospheric model: it has no depth term, so every pixel is veiled equally
    and nothing in it estimates scene distance. Because it is an affine
    contraction, pixel standard deviation is exactly `(1 - veil)` times the
    original and the result is analytically inside [0, 1].
    Deterministic; `generator` is unused.
    """
    return image * (1.0 - veil) + veil * FOG_COLOUR


def occlusion_anchor(shape, generator):
    """Draw the shared patch centre for one image, independent of severity.

    The range is the set of centres for which the severity-5 patch still fits
    inside the image, which is what makes every smaller, concentric patch fit
    too. Returned as `(centre_y, centre_x)`; the vertical coordinate is drawn
    first.
    """
    height, width = shape[-2:]
    largest = occlusion_side(OCCLUSION_LEVELS[-1], height, width)
    # `largest - largest // 2` is ceil(largest / 2), the margin the patch needs
    # below and to the right of its centre.
    margin_after = largest - largest // 2
    centre_y = torch.randint(
        largest // 2, height - margin_after + 1, (1,), generator=generator).item()
    centre_x = torch.randint(
        largest // 2, width - margin_after + 1, (1,), generator=generator).item()
    return centre_y, centre_x


def _occlusion(image, area_fraction, generator):
    """Square mid-grey patch covering the requested fraction of the image.

    Severities are nested: all five patches share one centre, so the region hidden
    at severity N stays hidden at severity N+1 and the only thing severity changes
    is how much of the same region is obscured. That is why `occlusion` is seeded
    without its severity (see `sample_seed`) — the anchor is drawn once per image.
    Without nesting, each level would draw an independent position and a
    per-sample accuracy-versus-severity curve would confound patch area with
    patch-position luck.

    The anchor is a pixel centre drawn uniformly from the positions that leave
    room for the *largest* patch, so every severity fits inside the image without
    a clamp that would break nesting. The draw is stochastic across images and
    reproducible for each one; it is not fixed to the middle of the frame. At
    64x64 with a severity-5 side of 35 that is a 30x30 grid of 900 anchors.
    Nothing about the classifier is consulted: placement depends only on the
    image dimensions and the seed.

    Constraining the anchor that way is a known limitation of this benchmark.
    Because the one anchor has to leave room for the severity-5 patch, the
    smaller patches cannot reach the border either: at 64x64 the severity-1
    patch is confined to the central 43x43 region, and only severity 5 can touch
    the frame edge. The condition is therefore centre-weighted by construction,
    increasingly so at low severity, and since GTSRB crops are centred on the
    sign, low-severity occlusion lands on sign content more often than a
    uniformly placed patch would. This condition is not equivalent to random
    erasing, which draws an independent uniform position at every size; results
    here should not be read as interchangeable with that setting. The trade is
    deliberate: nesting is what lets a severity curve isolate patch area from
    patch-position luck, and it costs border reach to get it.

    The fill is mid grey, which the frozen `Normalize(0.5, 0.5)` maps to exactly
    0, so the patch removes information without injecting a strong signal of its
    own. It does not imitate a textured occluder such as a sticker or foliage.
    """
    height, width = image.shape[-2:]
    side = occlusion_side(area_fraction, height, width)
    centre_y, centre_x = occlusion_anchor(image.shape, generator)

    # Integer centring nests exactly: top = c - side // 2 is non-increasing in
    # side and bottom = c + ceil(side / 2) is non-decreasing, so a smaller patch
    # is always contained in a larger one about the same centre.
    top = centre_y - side // 2
    left = centre_x - side // 2
    occluded = image.clone()
    occluded[..., top:top + side, left:left + side] = OCCLUSION_FILL
    return occluded


@dataclass(frozen=True)
class Corruption:
    """One corruption condition and its five severity parameters.

    `levels` is indexed by `severity - 1`, so severity is an ordered index into
    a fixed table rather than anything recomputed at call time.

    `severity_in_seed` records whether a condition's realisation should change
    with severity. Noise is a fresh field at every level, so it does; occlusion
    draws one anchor per image and grows the patch around it, so it does not.
    Keeping the rule on the spec means `sample_seed` is the only place that
    decides it, rather than each call site.
    """

    parameter: str
    levels: tuple
    function: callable
    stochastic: bool
    summary: str
    severity_in_seed: bool = False


CORRUPTIONS = {
    # Sigma is in output pixels. Severity 5 removes fine glyph detail but leaves
    # sign shape and colour legible at 64x64.
    "gaussian_blur": Corruption(
        parameter="sigma",
        levels=(0.5, 0.75, 1.0, 1.5, 2.0),
        function=_gaussian_blur,
        stochastic=False,
        summary="Gaussian blur, kernel truncated at +-3 sigma",
    ),
    # Underexposure and overexposure are kept as two separate conditions. They
    # are qualitatively different failures: darkening compresses the signal
    # towards 0 and is close to invertible, while brightening saturates highlights
    # against the clamp and destroys information irreversibly. Averaging them into
    # one "brightness" score, or picking a direction at random per image, would
    # mix the two and make the result uninterpretable.
    "brightness_dark": Corruption(
        parameter="gain",
        levels=(0.8, 0.65, 0.5, 0.35, 0.25),
        function=_brightness,
        stochastic=False,
        summary="Multiplicative underexposure",
    ),
    "brightness_bright": Corruption(
        parameter="gain",
        levels=(1.25, 1.55, 1.9, 2.35, 2.9),
        function=_brightness,
        stochastic=False,
        summary="Multiplicative overexposure, saturating highlights",
    ),
    # Standard deviation in [0, 1] units, i.e. severity 5 adds noise with a
    # standard deviation of 15% of the full value range.
    "gaussian_noise": Corruption(
        parameter="sigma",
        levels=(0.02, 0.04, 0.07, 0.11, 0.15),
        function=_gaussian_noise,
        stochastic=True,
        summary="Zero-mean Gaussian pixel noise, clipped to [0, 1]",
        severity_in_seed=True,
    ),
    # Veil strength: severity 5 leaves 25% of the original pixel contrast.
    "fog": Corruption(
        parameter="veil",
        levels=(0.15, 0.3, 0.45, 0.6, 0.75),
        function=_fog,
        stochastic=False,
        summary="Uniform white veil (contrast reduction), no depth term",
    ),
    # Requested area fraction of a single square patch; the realised fraction is
    # set by the rounded integer side length. Severities are concentric and
    # nested, so the seed deliberately excludes severity.
    "occlusion": Corruption(
        parameter="area_fraction",
        levels=OCCLUSION_LEVELS,
        function=_occlusion,
        stochastic=True,
        summary="Nested square mid-grey patch about one per-image anchor",
    ),
}


def corruption_parameter(corruption, severity):
    """Return the parameter value this corruption uses at this severity."""
    return _lookup(corruption, severity)[1]


def _lookup(corruption, severity):
    """Validate a (corruption, severity) pair and return the spec and parameter."""
    if corruption not in CORRUPTIONS:
        raise ValueError(
            f"unknown corruption {corruption!r}; expected one of {sorted(CORRUPTIONS)}"
        )
    # bool is a subclass of int and True == 1, so it is excluded explicitly
    # rather than quietly accepted as severity 1.
    if not isinstance(severity, int) or isinstance(severity, bool):
        raise ValueError(f"severity must be an int in {SEVERITY_LEVELS}; got {severity!r}")
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"severity must be one of {SEVERITY_LEVELS}; got {severity!r}")
    spec = CORRUPTIONS[corruption]
    return spec, spec.levels[severity - 1]


def apply_corruption(image, corruption, severity, seed=None):
    """Return a corrupted copy of one `[0, 1]` RGB observation.

    `image` is a single float tensor of shape (C, H, W) holding values in
    [0, 1] — the output of the resize-and-ToTensor half of the clean pipeline,
    before normalisation. Batched input is rejected: the corruption is defined
    per observation, and a batch sharing one generator would receive one
    occlusion position for every image in it.

    `seed` is required for stochastic corruptions, so a realisation can never
    depend on incidental global RNG state. It is accepted and ignored for
    deterministic ones, which lets callers pass a seed uniformly. The returned
    tensor is always new; `image` is never modified in place.

    CPU tensors only. Corruption belongs in `Dataset.__getitem__`, before
    collation and before anything reaches an accelerator, which is where the
    evaluation applies it; accepting device tensors would add a device argument
    to every random draw for a path the design does not use.
    """
    spec, parameter = _lookup(corruption, severity)

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor; got {type(image).__name__}")
    if image.ndim != 3:
        raise ValueError(f"image must have shape (C, H, W); got {tuple(image.shape)}")
    if not torch.is_floating_point(image):
        raise TypeError(f"image must be a float tensor; got dtype {image.dtype}")
    if image.device.type != "cpu":
        raise ValueError(
            f"corruption runs CPU-side before collation; got a {image.device.type} tensor")
    # Guards against being handed an already-normalised tensor, which would make
    # the corruption a perturbation of model inputs rather than of an image.
    low, high = float(image.min()), float(image.max())
    if low < -VALUE_TOLERANCE or high > 1.0 + VALUE_TOLERANCE:
        raise ValueError(
            f"image values must lie in [0, 1] before corruption; got [{low:.4f}, {high:.4f}]"
        )

    if spec.stochastic and seed is None:
        raise ValueError(f"corruption {corruption!r} is stochastic and requires a seed")
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))

    # One central clamp keeps every condition inside the valid image domain,
    # including any added later, and clamping is itself part of what the
    # brightness and noise conditions model.
    return torch.clamp(spec.function(image, parameter, generator), 0.0, 1.0)


def corruption_seed(namespace, corruption, severity, sample_index, base_seed=SEED):
    """Derive a per-sample seed from stable identifiers only.

    The payload is `base_seed | namespace | corruption | severity | index`, where
    `namespace` names the split the sample came from and `severity` is written as
    `-` when a condition's realisation is deliberately severity-independent. The
    index is the sample's position in the original torchvision dataset order —
    never batch position, iteration order, or worker id. A sample therefore
    receives the same realisation however the loader is shuffled or batched, and
    changing the base seed changes the whole suite.

    BLAKE2b is used instead of `hash()` because Python randomises string hashing
    per process, which would silently break reproducibility across runs. The
    digest is shifted to 63 bits so it is always a valid `manual_seed` value.
    """
    token = "-" if severity is None else str(severity)
    payload = f"{base_seed}|{namespace}|{corruption}|{token}|{sample_index}".encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big") >> 1


def sample_seed(namespace, corruption, severity, sample_index, base_seed=SEED):
    """Seed for one sample under one condition, honouring `severity_in_seed`.

    This is the only place that decides whether severity participates, so a
    nested condition cannot accidentally be re-seeded per level by a call site
    that forgot.
    """
    spec, _ = _lookup(corruption, severity)
    return corruption_seed(
        namespace,
        corruption,
        severity if spec.severity_in_seed else None,
        sample_index,
        base_seed,
    )


EXPECTED_STAGES = (transforms.Resize, transforms.ToTensor, transforms.Normalize)


def check_stage_layout(stages):
    """Fail if the clean pipeline is no longer `Resize -> ToTensor -> Normalize`.

    Splitting the frozen pipeline relies on its final stage being the only
    normalisation step. If `data.py` ever changes shape, the corruption suite
    must be revisited rather than silently splice into the wrong place.
    """
    found = tuple(type(stage) for stage in stages)
    if found != EXPECTED_STAGES:
        raise RuntimeError(
            "clean preprocessing layout changed; corruption insertion point is no "
            f"longer valid. Expected {[c.__name__ for c in EXPECTED_STAGES]}, "
            f"found {[c.__name__ for c in found]}"
        )


def split_preprocessing():
    """Split the clean pipeline into its observation and normalisation halves.

    Returns `(observation, normalize)`, where `observation` resizes and converts
    to a `[0, 1]` tensor and `normalize` is the clean pipeline's own `Normalize`
    stage. Both come from `build_preprocessing()`, so the corrupted path uses
    the identical resize configuration and normalisation constants as the clean
    path, and normalisation happens exactly once.
    """
    stages = list(build_preprocessing().transforms)
    check_stage_layout(stages)
    *observation, normalize = stages
    return transforms.Compose(observation), normalize


def stable_index(dataset, index):
    """Resolve a position in a (possibly nested) Subset to the underlying index.

    Seeds are derived from the index into the original dataset order, so the
    realisation of a sample does not change if the subset is rebuilt or
    re-ordered.
    """
    while isinstance(dataset, Subset):
        index = dataset.indices[index]
        dataset = dataset.dataset
    return int(index)


class CorruptedDataset(Dataset):
    """Wrap an observation dataset, corrupt each sample, then normalise it.

    `dataset` must yield `([0, 1] tensor, label)` pairs — build it with the
    `observation` half of `split_preprocessing()`, not with the full clean
    pipeline, or normalisation would be applied twice.

    `corruption=None` applies no corruption, which makes the clean path through
    this wrapper identical to `build_preprocessing()` and gives evaluation a
    single code path for clean and corrupted conditions.

    `namespace` names the split being corrupted and enters every seed, so the
    final test-split evaluation cannot reuse validation's realisations.
    """

    def __init__(self, dataset, corruption, severity=None,
                 namespace=VALIDATION_NAMESPACE, base_seed=SEED):
        if corruption is None:
            if severity is not None:
                raise ValueError("severity must be None when corruption is None")
        else:
            _lookup(corruption, severity)
        self.dataset = dataset
        self.corruption = corruption
        self.severity = severity
        self.namespace = namespace
        self.base_seed = base_seed
        _, self.normalize = split_preprocessing()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.corruption is not None:
            seed = sample_seed(
                self.namespace,
                self.corruption,
                self.severity,
                stable_index(self.dataset, index),
                self.base_seed,
            )
            image = apply_corruption(image, self.corruption, self.severity, seed=seed)
        return self.normalize(image), label


def build_validation_observations(data_root=None, download=False):
    """Return the validation split as `[0, 1]` observations, plus its indices.

    The split is recomputed with `data.py`'s own `read_training_index` and
    `split_tracks`, so it is the same track-level split and the same seed that
    `create_data_loaders` uses; only the final normalisation stage is held back
    so a corruption can be inserted before it. The official test split is not
    touched.
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / "data"
    observation, _ = split_preprocessing()
    dataset = GTSRB(root=data_root, split="train", transform=observation, download=download)
    labels, tracks = read_training_index(dataset)
    _, validation_indices = split_tracks(labels, tracks)
    return Subset(dataset, validation_indices.tolist()), validation_indices, labels


def format_specification():
    """Render the severity table, so reports quote the spec instead of restating it."""
    lines = []
    for name, spec in CORRUPTIONS.items():
        levels = "  ".join(
            f"s{severity}={spec.levels[severity - 1]:g}" for severity in SEVERITY_LEVELS
        )
        kind = "stochastic" if spec.stochastic else "deterministic"
        lines.append(f"{name:18s} {spec.parameter:14s} {kind:13s} {levels}")
    return "\n".join(lines)
