"""Held-out corruption suite for the generalisation half of the study.

Models B and C are trained on the six families in `corruptions.py`. Measuring
them on those same six families answers a narrow question - did the targeted
distribution help on the conditions it was designed around - and cannot
separate robustness from having trained on the evaluation distribution. This
module supplies the other half: five families that appear in no training
recipe, so a gain here is transfer rather than a distribution match.

The suite is frozen before any augmented model is trained, and its severity
parameters were chosen from the mechanics of each transformation and from the
frozen suite's own conventions. No model's performance entered the choice.

Relationship to `corruptions.py`
--------------------------------
That module stays untouched: it defines the benchmark the committed baseline
results were produced against, and the accepted artifacts have to remain
reproducible by exactly the code that wrote them. Everything here that is
already general is imported from it - the `Corruption` record, the severity
ladder, the seed derivation, the pipeline split, the valid-range tolerance, the
mid-grey fill convention - so the two suites share one notion of what a
condition and a severity are. What is reimplemented is the small amount of
lookup and guard code that has to resolve against this table instead of that
one.

Each family is mechanically distinct from all six training families:

* `motion_blur` is directional, where `gaussian_blur` is isotropic.
* `jpeg_compression` introduces block structure and colour quantisation, which
  no training family produces.
* `contrast_reduction` contracts towards mid grey and preserves mean luminance,
  where `fog` contracts towards white and raises it.
* `salt_pepper` replaces whole pixels with extremes, where `gaussian_noise`
  perturbs every channel of every pixel slightly.
* `rotation` is geometric; the training suite contains no geometric change at
  all.

Severity conventions
--------------------
Five ordered levels per family, indexed by `severity - 1`, exactly as the frozen
table. Where a family is stochastic, the realisation is drawn once per image and
held fixed across severities, so a per-sample severity curve varies only the
strength and not the draw - the same reason the frozen occlusion anchor excludes
severity from its seed. `salt_pepper` achieves this by thresholding one uniform
field, which makes its affected pixels nested across levels.
"""

import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms import functional as TF

from .corruptions import (
    OCCLUSION_FILL,
    SEVERITY_LEVELS,
    VALIDATION_NAMESPACE,
    VALUE_TOLERANCE,
    Corruption,
    corruption_seed,
    stable_index,
    split_preprocessing,
)
from .data import SEED

# Contrast reduction contracts towards this value, and rotation fills the
# corners that rotate into frame with it. Both reuse the frozen occlusion fill:
# `Normalize(0.5, 0.5)` maps it to exactly 0, so "no information here" enters
# the model as no signal rather than as a black frame the network could key on.
NEUTRAL_GREY = OCCLUSION_FILL

# Luminance axis that `contrast_reduction` contracts towards. Separate name from
# the fill because they are the same number for different reasons.
CONTRAST_MIDPOINT = 0.5


def _motion_kernel(length, angle_degrees):
    """Normalised line kernel of a given length at a given angle.

    The segment is rasterised by nearest pixel through the kernel centre, so for
    angles off the axes and diagonals two steps can land on the same cell. The
    kernel is therefore normalised by its own sum rather than by `length`, which
    keeps it an average over the cells it actually covers and keeps the
    transformation a convex combination of pixel values.

    `length` is odd so the centre is a pixel and the blur does not shift the
    image.
    """
    if length % 2 == 0:
        raise ValueError(f"motion blur length must be odd; got {length}")
    kernel = torch.zeros(length, length, dtype=torch.float32)
    centre = (length - 1) / 2
    radians = math.radians(angle_degrees)
    step_y, step_x = -math.sin(radians), math.cos(radians)
    for step in range(length):
        offset = step - centre
        row = round(centre + offset * step_y)
        column = round(centre + offset * step_x)
        kernel[row, column] = 1.0
    return kernel / kernel.sum()


def _motion_blur(image, length, generator):
    """Directional blur along one axis drawn uniformly per image.

    The angle is drawn from [0, 180) because a line kernel is symmetric under a
    half turn, and it is drawn once per image rather than per severity, so the
    severity ladder lengthens one fixed streak instead of re-rolling its
    direction.

    Boundaries are reflected, matching the padding `torchvision`'s Gaussian
    blur uses, so the frame edge is not darkened by padding with zeros. The
    kernel is a normalised non-negative average, so the result is analytically
    inside [0, 1] and the caller's clamp only absorbs float error.
    """
    angle = torch.rand((1,), generator=generator).item() * 180.0
    kernel = _motion_kernel(int(length), angle)
    channels = image.shape[-3]
    weight = kernel.to(image.dtype).expand(channels, 1, *kernel.shape)
    pad = int(length) // 2
    padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(padded, weight, groups=channels)
    return blurred.squeeze(0)


def _jpeg_compression(image, quality, generator):
    """Encode to JPEG at a given quality and decode back. Deterministic.

    The round trip is applied to the resized 64x64 observation, so the block
    grid is 8x8 in the model's input space and severity means the same thing for
    every image. Two lossy steps are involved and both are part of what this
    models: the conversion to 8-bit integers, and the encoder's quantisation and
    default chroma subsampling, which discards colour resolution as well as
    high-frequency luminance detail.

    `generator` is unused; the transformation has no random component.

    One reproducibility caveat, and the only one in this suite: the decoded
    pixels come from whichever libjpeg build `torchvision` is linked against.
    The round trip is bit-stable within an environment, so every model is
    measured on identical images, but it is not guaranteed bit-identical across
    libjpeg versions. The evaluator records the `torchvision` version alongside
    the results for that reason.
    """
    as_bytes = (image * 255.0).round().clamp(0, 255).to(torch.uint8)
    encoded = encode_jpeg(as_bytes, quality=int(quality))
    decoded = decode_jpeg(encoded)
    return decoded.to(image.dtype) / 255.0


def _contrast_reduction(image, factor, generator):
    """Contract pixel values towards mid grey, `out = mid + factor * (x - mid)`.

    Distinct from `fog`, which contracts towards white: this preserves the mean
    of a mid-grey-centred image and reduces contrast only, where fog reduces
    contrast and raises luminance together. Pixel standard deviation is exactly
    `factor` times the original, which makes the severity ordering exact rather
    than empirical. The result is a contraction of [0, 1] about an interior
    point, so it is analytically inside [0, 1]. Deterministic; `generator` is
    unused.
    """
    return CONTRAST_MIDPOINT + factor * (image - CONTRAST_MIDPOINT)


def _salt_pepper(image, fraction, generator):
    """Replace a fraction of whole pixels with black or white, half each.

    One uniform field is drawn per image and shared by all three channels, so
    the result is black and white dots rather than per-channel colour speckle -
    the classic impulse-noise failure, and mechanically different from the
    frozen Gaussian field, which perturbs every channel of every pixel.

    Severity thresholds that one field, so the pixels corrupted at severity N
    stay corrupted at severity N+1 and only their number grows. That makes the
    severity curve of a single image a statement about how many pixels are lost
    rather than about which ones were drawn.
    """
    field = torch.rand(
        (1, *image.shape[-2:]), generator=generator, dtype=image.dtype,
        device=image.device,
    )
    pepper = field < fraction / 2
    salt = (field >= fraction / 2) & (field < fraction)
    corrupted = image.clone()
    corrupted = torch.where(pepper, torch.zeros_like(corrupted), corrupted)
    return torch.where(salt, torch.ones_like(corrupted), corrupted)


def _rotation(image, degrees, generator):
    """Rotate about the image centre by a fixed magnitude and a per-image sign.

    The magnitude is the severity parameter; the direction is drawn once per
    image, so the condition measures sensitivity to rotation rather than
    sensitivity to one direction of rotation, and the sign is held across
    severities so a sample's ladder turns one way.

    Bilinear interpolation, no expansion - the output stays 64x64, which is what
    the model accepts - and the corners that rotate into frame are filled with
    mid grey rather than black, so the fill enters the normalised input as zero.
    Two consequences are part of the condition: interpolation smooths the image
    slightly even at the mildest level, and sign content near the corners is
    rotated out of frame and lost.
    """
    sign = 1.0 if torch.rand((1,), generator=generator).item() < 0.5 else -1.0
    return TF.rotate(
        image, sign * float(degrees),
        interpolation=TF.InterpolationMode.BILINEAR,
        expand=False, fill=[NEUTRAL_GREY] * image.shape[-3],
    )


HELDOUT_CORRUPTIONS = {
    # Kernel length in output pixels, odd so the blur does not shift the image.
    # The spread along the streak is the standard deviation of a uniform run of
    # L cells, sqrt((L^2 - 1) / 12): 0.82, 1.41, 2.00, 3.16, 4.32 pixels. Level
    # 3 therefore matches the frozen `gaussian_blur` severity 5 (sigma 2.0) in
    # one direction while leaving the perpendicular direction sharp, and level 5
    # smears about a third of the width of a median sign box.
    "motion_blur": Corruption(
        parameter="length",
        levels=(3, 5, 7, 11, 15),
        function=_motion_blur,
        stochastic=True,
        summary="Directional blur, angle drawn once per image, reflected edges",
    ),
    # Libjpeg quality factor, so the parameter decreases as severity rises, in
    # the same way the underexposure gains do. 70 is near-invisible at 64x64,
    # 50 is mild, 30 is where block structure becomes plainly visible, and 7 is
    # severe enough to flatten glyph interiors into single blocks.
    "jpeg_compression": Corruption(
        parameter="quality",
        levels=(70, 50, 30, 15, 7),
        function=_jpeg_compression,
        stochastic=False,
        summary="JPEG encode and decode at a fixed quality factor",
    ),
    # Retained share of the original contrast. Pixel standard deviation is
    # exactly this multiple of the clean value, so the ladder is exact. Chosen
    # to span a different range from `fog`, whose veils leave 0.85 down to 0.25
    # of the contrast: this starts lower and ends lower, at 15%.
    "contrast_reduction": Corruption(
        parameter="factor",
        levels=(0.7, 0.55, 0.4, 0.25, 0.15),
        function=_contrast_reduction,
        stochastic=False,
        summary="Linear contraction of contrast towards mid grey",
    ),
    # Fraction of the 4,096 pixels replaced, split evenly between black and
    # white. Level 1 touches about 20 pixels and level 5 about 410, roughly a
    # tenth of the frame. Nested across severities by construction.
    "salt_pepper": Corruption(
        parameter="fraction",
        levels=(0.005, 0.01, 0.03, 0.06, 0.1),
        function=_salt_pepper,
        stochastic=True,
        summary="Impulse noise replacing whole pixels with black or white",
    ),
    # Rotation magnitude in degrees, with the direction drawn per image. GTSRB
    # crops are upright and the training recipe contains no geometric
    # augmentation, so even small angles are genuinely out of distribution. 25
    # degrees is large enough to matter and small enough that a human still
    # reads the sign.
    "rotation": Corruption(
        parameter="degrees",
        levels=(3, 7, 12, 18, 25),
        function=_rotation,
        stochastic=True,
        summary="Rotation about the centre, bilinear, mid-grey corner fill",
    ),
}


def _lookup(corruption, severity):
    """Validate a (corruption, severity) pair against the held-out table."""
    if corruption not in HELDOUT_CORRUPTIONS:
        raise ValueError(f"unknown held-out corruption {corruption!r}; expected "
                         f"one of {sorted(HELDOUT_CORRUPTIONS)}")
    if not isinstance(severity, int) or isinstance(severity, bool):
        raise ValueError(f"severity must be an int in {SEVERITY_LEVELS}; "
                         f"got {severity!r}")
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"severity must be one of {SEVERITY_LEVELS}; "
                         f"got {severity!r}")
    spec = HELDOUT_CORRUPTIONS[corruption]
    return spec, spec.levels[severity - 1]


def heldout_parameter(corruption, severity):
    """Return the parameter value this held-out corruption uses at this severity."""
    return _lookup(corruption, severity)[1]


def apply_heldout(image, corruption, severity, seed=None):
    """Return a corrupted copy of one `[0, 1]` RGB observation.

    Same contract as `corruptions.apply_corruption`, against this table: a
    single CPU float tensor of shape (C, H, W) in [0, 1], a seed required for
    the stochastic families so no realisation can depend on ambient global RNG
    state, and one central clamp back into the valid image domain. Batched input
    is rejected because a batch sharing one generator would receive one blur
    angle, one impulse field or one rotation direction for every image in it.

    The guards are checked here rather than deferred to the family functions so
    that an already-normalised tensor fails loudly instead of being corrupted as
    if it were an image.
    """
    spec, parameter = _lookup(corruption, severity)

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor; got {type(image).__name__}")
    if image.ndim != 3:
        raise ValueError(f"image must have shape (C, H, W); got {tuple(image.shape)}")
    if not torch.is_floating_point(image):
        raise TypeError(f"image must be a float tensor; got dtype {image.dtype}")
    if image.device.type != "cpu":
        raise ValueError("held-out corruption runs CPU-side before collation; "
                         f"got a {image.device.type} tensor")
    low, high = float(image.min()), float(image.max())
    if low < -VALUE_TOLERANCE or high > 1.0 + VALUE_TOLERANCE:
        raise ValueError(
            f"image values must lie in [0, 1] before corruption; got "
            f"[{low:.4f}, {high:.4f}]")

    if spec.stochastic and seed is None:
        raise ValueError(f"held-out corruption {corruption!r} is stochastic and "
                         "requires a seed")
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))

    corrupted = torch.clamp(spec.function(image, parameter, generator), 0.0, 1.0)
    if corrupted.shape != image.shape:
        raise RuntimeError(f"{corruption} s{severity} changed the observation "
                           f"shape from {tuple(image.shape)} to "
                           f"{tuple(corrupted.shape)}")
    return corrupted


def heldout_sample_seed(namespace, corruption, severity, sample_index,
                        base_seed=SEED):
    """Seed for one sample under one held-out condition.

    Delegates to `corruptions.corruption_seed`, which takes the condition name
    as data and so needs no knowledge of which table it came from. The held-out
    family names are disjoint from the frozen six, so the two suites cannot
    collide at the same namespace and index.

    No held-out family currently puts severity in its seed: each one draws its
    realisation once per image and lets severity change the strength only. The
    `severity_in_seed` flag is still honoured so a family added later cannot be
    reseeded per level by a call site that forgot.
    """
    spec, _ = _lookup(corruption, severity)
    return corruption_seed(
        namespace, corruption, severity if spec.severity_in_seed else None,
        sample_index, base_seed,
    )


def heldout_conditions():
    """The 25 held-out conditions: every family at every severity."""
    return [(name, severity) for name in HELDOUT_CORRUPTIONS
            for severity in SEVERITY_LEVELS]


class HeldoutDataset(Dataset):
    """Wrap an observation dataset, apply one held-out condition, normalise.

    The counterpart of `corruptions.CorruptedDataset` for this table. `dataset`
    must yield `([0, 1] tensor, label)` pairs from the observation half of
    `split_preprocessing()`, and the normalisation stage comes from the frozen
    clean pipeline, so an evaluated held-out image differs from a clean one by
    the corruption alone.
    """

    def __init__(self, dataset, corruption, severity,
                 namespace=VALIDATION_NAMESPACE, base_seed=SEED):
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
        seed = heldout_sample_seed(
            self.namespace, self.corruption, self.severity,
            stable_index(self.dataset, index), self.base_seed,
        )
        image = apply_heldout(image, self.corruption, self.severity, seed=seed)
        return self.normalize(image), label


def format_heldout_specification():
    """Render the held-out severity table, so reports quote the spec."""
    lines = []
    for name, spec in HELDOUT_CORRUPTIONS.items():
        levels = "  ".join(f"s{severity}={spec.levels[severity - 1]:g}"
                           for severity in SEVERITY_LEVELS)
        kind = "stochastic" if spec.stochastic else "deterministic"
        lines.append(f"{name:20s} {spec.parameter:10s} {kind:13s} {levels}")
    return "\n".join(lines)
