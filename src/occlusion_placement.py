"""Placement geometry for the occlusion-location ablation.

The accepted occlusion condition in `corruptions.py` draws one anchor per image
from the positions that leave room for the severity-5 patch, which confines the
smaller patches to the middle of the frame: at 64x64 the severity-1 patch cannot
leave the central 43x43 box. The failure analysis found that frames whose anchor
fell nearer the centre failed more often, but that benchmark was not built to
vary position, so the association cannot be read as an effect of placement.

This module defines the placements for an ablation that does vary position while
holding everything else fixed: the same epoch-21 checkpoint, the same 5,340
validation frames, the patch sides of the frozen severities 1 and 3, and the
same mid-grey fill applied before the same normalisation. The frozen suite is
imported, never edited, and the ablation writes its own artifacts.

Coordinates
-----------
A placement is the top-left pixel `(y, x)` of a `side x side` patch on a 64x64
image, so the patch covers `[y, y + side) x [x, x + side)` and lies inside the
frame exactly when `0 <= y, x <= 64 - side`. The patch centre is at
`y + (side - 1) / 2`, and distance from the image centre is measured with the
Chebyshev metric, which matches a square patch: it is the coordinate that
decides how far the patch reaches towards a border.

Policies
--------
centre   placements whose patch centre is as close to the image centre as the
         parity of `64 - side` allows: one position for an even span, four for
         an odd one, so no diagonal is silently preferred.
corner   the four placements flush with the image corners. Both sides fit
         within a 32x32 quadrant, so each patch lies entirely in one quadrant.
uniform  a stratified estimate of the mean outcome over all valid top-left
         coordinates, with stratum weights (see `uniform_weights`).
ladder   four diagonal placements at each of a fixed set of Chebyshev
         distances. The rungs that coincide with the centre or corner
         placements reuse those passes rather than repeating them.

What the ablation measures is the effect of moving a patch within the model's
input frame. It is not an off-sign condition: the GTSRB crop keeps the sign's
annotated box over a median 57% of the resized image, so a flush corner patch
still overlaps that box on at least 98.4% of validation frames at side 14 and
on every frame at side 25. Bounding-box overlap is recorded as a covariate; it
is not a measure of how much sign content is hidden, and no segmentation mask
exists to make one. These figures come from the ground-truth annotations and
are recomputed in `verify_placement.py`.
"""

from dataclasses import dataclass
from hashlib import blake2b

from torch.utils.data import Dataset

from .corruptions import (
    OCCLUSION_FILL,
    VALIDATION_NAMESPACE,
    VALUE_TOLERANCE,
    split_preprocessing,
    stable_index,
)
from .data import IMAGE_SIZE, SEED

# The patch sides of the frozen occlusion severities 1 and 3 at 64x64. They are
# listed rather than recomputed so the ablation cannot silently follow a change
# to the severity table; `verify_placement.py` checks they still agree with it.
PATCH_SIDES = (14, 25)

# Strata per axis for the uniform policy, giving nine placements per frame.
UNIFORM_STRATA = 3

# Seed payload token for this experiment. It differs from every corruption name,
# so an ablation placement can never collide with a frozen benchmark seed.
PLACEMENT_CONDITION = "placement"

# Chebyshev distances of the ladder rungs, in pixels of the model's input. The
# attainable distances follow from the geometry: a patch centre sits at
# `top_left + (side - 1) / 2`, so for side 14 the offsets from the image centre
# are integers up to 25, and for side 25 they are half-integers up to 19.5.
# Both ladders start at the smallest attainable distance (the centre policy) and
# end at the largest (the corner policy), with two intermediate rungs spaced as
# evenly as the attainable values allow.
LADDER_DISTANCES = {
    14: (0.0, 8.0, 17.0, 25.0),
    25: (0.5, 6.5, 13.5, 19.5),
}

IMAGE_CENTRE = (IMAGE_SIZE - 1) / 2


def top_left_count(side):
    """Number of valid top-left coordinates per axis for this patch side."""
    return IMAGE_SIZE - side + 1


def patch_centre(top_left, side):
    """Centre coordinate of the pixels a patch covers, on one axis."""
    return top_left + (side - 1) / 2


def centre_distance(placement, side):
    """Chebyshev distance of a placement's patch centre from the image centre."""
    return max(abs(patch_centre(value, side) - IMAGE_CENTRE) for value in placement)


def check_side(side):
    """Reject a patch that cannot be placed, or that no corner can hold."""
    if side < 1 or side > IMAGE_SIZE:
        raise ValueError(f"patch side {side} does not fit in a {IMAGE_SIZE}px image")
    if side > IMAGE_SIZE // 2:
        raise ValueError(
            f"patch side {side} exceeds one quadrant, so the corner policy is undefined"
        )
    return side


def centre_placements(side):
    """Placements whose patch centre is closest to the image centre.

    The span `64 - side` is even for an even side and odd for an odd one. An
    even span has one exactly centred placement; an odd span has no integer
    coordinate at the image centre, and the two nearest sit half a pixel either
    side of it. Taking all four combinations of those two coordinates keeps the
    policy symmetric instead of committing to one diagonal.
    """
    check_side(side)
    span = IMAGE_SIZE - side
    coordinates = (span // 2,) if span % 2 == 0 else (span // 2, span // 2 + 1)
    return tuple((y, x) for y in coordinates for x in coordinates)


def corner_placements(side):
    """The four placements flush with the image corners."""
    check_side(side)
    far = IMAGE_SIZE - side
    return ((0, 0), (0, far), (far, 0), (far, far))


def stratum_bounds(side, strata=UNIFORM_STRATA):
    """Contiguous strata over one axis of valid top-left coordinates.

    Returns `(low, high, size)` per stratum, inclusive bounds. The valid range
    holds `65 - side` coordinates, which is 51 for side 14 and 40 for side 25,
    so three strata are equal only for side 14. The leftover coordinates are
    given to the first strata and the resulting sizes are carried into
    `uniform_weights`, which is where the inequality has to be accounted for.
    """
    check_side(side)
    count = top_left_count(side)
    if strata < 1 or strata > count:
        raise ValueError(f"{strata} strata do not fit {count} coordinates")
    base, extra = divmod(count, strata)
    bounds = []
    low = 0
    for index in range(strata):
        size = base + (1 if index < extra else 0)
        bounds.append((low, low + size - 1, size))
        low += size
    return tuple(bounds)


def uniform_weights(side, strata=UNIFORM_STRATA):
    """Weight of each stratified cell in the uniform-policy estimate.

    The estimand is the error rate of a frame under a patch whose top-left
    coordinate is drawn uniformly from the `N x N` valid positions,
    `N = 65 - side`:

        e = (1 / N^2) * sum over all valid (y, x) of f(y, x)

    The axis is partitioned into contiguous strata of sizes `n_1..n_S`, so the
    cells `C_jk` hold `n_j * n_k` positions and partition the grid. Splitting the
    sum by cell gives

        e = sum over j, k of (n_j * n_k / N^2) * mean of f over C_jk

    One placement is drawn per cell, uniformly within that cell, so its outcome
    is unbiased for the cell mean and the weighted sum of the nine outcomes is
    unbiased for `e`. The weights are therefore the cell's share of the valid
    positions, `n_j * n_k / N^2`, and they sum to 1 because the cells partition
    the grid.

    Equal weights of 1/9 would estimate the unweighted mean of the cell means,
    which equals `e` only when every stratum holds the same number of
    coordinates. That holds for side 14 (51 = 3 x 17) and fails for side 25
    (40 = 14 + 13 + 13). There equal weights take weight off the widest stratum
    and give it to the two narrow ones, which shifts the estimate towards
    central placements rather than away from them: the central cell would carry
    1/9 of the weight instead of its true share of 0.105625, and the cell of
    lowest coordinates 1/9 instead of 0.1225. The largest absolute error that
    can cause for a 0/1 outcome is the sum of the positive weight differences,
    0.0219. `verify_placement.py` derives that bound, checks the weighting
    against the exact grid mean, and measures the error for an outcome that
    fires on central placements.
    """
    bounds = stratum_bounds(side, strata)
    count = top_left_count(side)
    return {
        (j, k): (bounds[j][2] * bounds[k][2]) / (count * count)
        for j in range(strata)
        for k in range(strata)
    }


def placement_seed(side, cell, dataset_index, base_seed=SEED,
                   namespace=VALIDATION_NAMESPACE):
    """Per-frame seed for one stratified cell.

    The payload is `base seed | namespace | placement | side | cell | index`,
    built the same way as the frozen corruption seeds: BLAKE2b over stable
    identifiers only, so a frame's placement is the same however the loader is
    batched and is recoverable offline from its dataset index. The namespace
    keeps a later test-split evaluation from reusing validation placements, and
    the `placement` token keeps these seeds disjoint from the frozen suite's.
    """
    payload = (
        f"{base_seed}|{namespace}|{PLACEMENT_CONDITION}|{side}|"
        f"{cell[0]},{cell[1]}|{dataset_index}"
    ).encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big") >> 1


def uniform_placement(side, cell, dataset_index, strata=UNIFORM_STRATA,
                      base_seed=SEED, namespace=VALIDATION_NAMESPACE):
    """Draw this frame's placement inside one stratified cell.

    The two axes take independent halves of the digest, so the position is
    uniform over the cell rather than along its diagonal. Reducing a 31- or
    32-bit value modulo a stratum size of at most 17 leaves a relative bias
    below 1e-8, which is far smaller than the Monte-Carlo noise the stratified
    design is there to control.
    """
    bounds = stratum_bounds(side, strata)
    seed = placement_seed(side, cell, dataset_index, base_seed, namespace)
    low_y, _, size_y = bounds[cell[0]]
    low_x, _, size_x = bounds[cell[1]]
    return (low_y + (seed >> 32) % size_y, low_x + (seed & 0xFFFFFFFF) % size_x)


def ladder_placements(side, distance):
    """The diagonal placements at one Chebyshev distance from the image centre.

    A placement at distance `d` has both coordinates offset by `d` from the
    centred coordinate `(64 - side) / 2`, so the rung holds the four diagonal
    positions, or one position when `d` is the smallest attainable distance and
    the two offsets coincide. Distances that are negative, that do not land on
    an integer coordinate, or that fall outside the frame are rejected rather
    than rounded or mirrored: the ladder is only allowed to contain positions
    that exist, at the distance that was asked for.

    The ladder moves along the diagonal because its end points are the centre
    and corner policies. Axis-aligned placements at the same Chebyshev distance
    cover a different region of the image and are a different question.
    """
    check_side(side)
    if distance < 0:
        raise ValueError(f"distance {distance} is negative")
    base = (IMAGE_SIZE - side) / 2
    low, high = base - distance, base + distance
    if low != int(low) or high != int(high):
        raise ValueError(
            f"distance {distance} gives non-integer coordinates for side {side}"
        )
    low, high = int(low), int(high)
    if low < 0 or high > IMAGE_SIZE - side:
        raise ValueError(
            f"distance {distance} places a {side}px patch outside the frame"
        )
    coordinates = (low,) if low == high else (low, high)
    return tuple((y, x) for y in coordinates for x in coordinates)


@dataclass(frozen=True)
class Placement:
    """One evaluation pass: a policy, a patch side, and where the patch goes.

    `coordinates` is the fixed top-left for every frame, or None for a uniform
    cell, whose coordinate is a function of the frame's dataset index.

    `weight` is the placement's share of its policy's per-frame average. It
    lives here rather than in the analysis so the aggregation rule is fixed
    together with the geometry: equal shares within the centre, corner and
    ladder-rung sets, and the stratum weight within the uniform set.
    """

    policy: str
    side: int | None
    name: str
    coordinates: tuple | None = None
    cell: tuple | None = None
    distance: float | None = None
    weight: float = 1.0

    def top_left(self, dataset_index):
        """Where this placement puts the patch for one frame."""
        if self.coordinates is not None:
            return self.coordinates
        return uniform_placement(self.side, self.cell, dataset_index)


CLEAN_CONDITION = "clean"


def policy_placements(side):
    """The centre, corner, uniform and ladder placements for one patch side."""
    placements = []

    centres = centre_placements(side)
    for coordinates in centres:
        placements.append(Placement(
            policy="centre", side=side,
            name=f"centre_s{side}_y{coordinates[0]}x{coordinates[1]}",
            coordinates=coordinates,
            distance=centre_distance(coordinates, side),
            weight=1 / len(centres),
        ))

    corners = corner_placements(side)
    for coordinates in corners:
        placements.append(Placement(
            policy="corner", side=side,
            name=f"corner_s{side}_y{coordinates[0]}x{coordinates[1]}",
            coordinates=coordinates,
            distance=centre_distance(coordinates, side),
            weight=1 / len(corners),
        ))

    weights = uniform_weights(side)
    for (row, column), weight in weights.items():
        placements.append(Placement(
            policy="uniform", side=side,
            name=f"uniform_s{side}_c{row}{column}",
            cell=(row, column),
            weight=weight,
        ))

    # A rung whose placements are already evaluated by the centre or corner
    # policy is read from those passes instead of being run again, so the ladder
    # costs only its intermediate rungs. Membership is decided by comparing the
    # coordinate sets, not by assuming which rungs coincide.
    reused = {centres, corners}
    for distance in LADDER_DISTANCES[side]:
        rung = ladder_placements(side, distance)
        if rung in reused:
            continue
        for coordinates in rung:
            placements.append(Placement(
                policy="ladder", side=side,
                name=f"ladder_s{side}_d{distance:g}_y{coordinates[0]}x{coordinates[1]}",
                coordinates=coordinates,
                distance=distance,
                weight=1 / len(rung),
            ))
    return placements


def evaluation_conditions():
    """Every pass of the ablation, clean reproduction first.

    The clean pass is not a placement condition: it goes through the frozen
    corruption wrapper with no corruption, and its predictions must match the
    accepted sweep's clean rows exactly. That is the check that this experiment
    is running the accepted pipeline.
    """
    conditions = [Placement(policy=CLEAN_CONDITION, side=None, name=CLEAN_CONDITION)]
    for side in PATCH_SIDES:
        conditions.extend(policy_placements(side))
    return conditions


def ladder_table():
    """Rungs of the distance ladder and the passes each one reads.

    Returns `(side, distance, placements, policy)` per rung, where `policy` says
    which passes supply it: the centre and corner policies at the two ends, and
    the ladder's own passes in between.
    """
    rows = []
    for side in PATCH_SIDES:
        centres = centre_placements(side)
        corners = corner_placements(side)
        for distance in LADDER_DISTANCES[side]:
            rung = ladder_placements(side, distance)
            if rung == centres:
                policy = "centre"
            elif rung == corners:
                policy = "corner"
            else:
                policy = "ladder"
            rows.append((side, distance, rung, policy))
    return rows


def place_patch(image, top_left, side):
    """Return a copy of a `[0, 1]` observation with one square patch filled.

    The fill is the frozen suite's mid grey, which the frozen `Normalize(0.5,
    0.5)` maps to exactly 0, so the patch removes information without injecting
    a strong signal. Containment is required rather than clamped: a clamped
    patch would silently change area near a border and break the comparison the
    ablation is built on.
    """
    if image.ndim != 3:
        raise ValueError(f"image must have shape (C, H, W); got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    if (height, width) != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected a {IMAGE_SIZE}x{IMAGE_SIZE} image; got {height}x{width}")
    low, high = float(image.min()), float(image.max())
    if low < -VALUE_TOLERANCE or high > 1.0 + VALUE_TOLERANCE:
        raise ValueError(
            f"image values must lie in [0, 1] before the patch; got [{low:.4f}, {high:.4f}]"
        )
    top, left = top_left
    if top < 0 or left < 0 or top + side > height or left + side > width:
        raise ValueError(
            f"a {side}px patch at {top_left} does not fit inside the image"
        )
    placed = image.clone()
    placed[..., top:top + side, left:left + side] = OCCLUSION_FILL
    return placed


class PlacedOcclusionDataset(Dataset):
    """Apply one placement to every frame, then the frozen normalisation.

    `observations` must yield `([0, 1] tensor, label)` pairs, i.e. the dataset
    built from the observation half of `split_preprocessing()`. Normalisation is
    taken from that same split, so the corrupted path differs from the clean one
    only by the patch.
    """

    def __init__(self, observations, placement):
        if placement.side is None:
            raise ValueError("clean conditions go through CorruptedDataset, not this wrapper")
        self.dataset = observations
        self.placement = placement
        _, self.normalize = split_preprocessing()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        top_left = self.placement.top_left(stable_index(self.dataset, index))
        return self.normalize(place_patch(image, top_left, self.placement.side)), label


def format_protocol():
    """Render the frozen placement table, so reports quote it instead of restating it."""
    lines = []
    for side in PATCH_SIDES:
        count = top_left_count(side)
        sizes = [size for _, _, size in stratum_bounds(side)]
        lines.append(
            f"side {side}: {count}x{count} valid top-left coordinates, "
            f"strata {sizes}, centre {list(centre_placements(side))}"
        )
        lines.append(
            f"  corners {list(corner_placements(side))}; "
            f"ladder {[f'{d:g}' for d in LADDER_DISTANCES[side]]}"
        )
    passes = len(evaluation_conditions())
    lines.append(f"{passes} evaluation passes including the clean reproduction")
    return "\n".join(lines)
