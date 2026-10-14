"""Structural verification of the corruption suite.

These checks are about the corruption transformations themselves: ranges,
shapes, reproducibility, and alignment with the frozen clean pipeline. No model
is built, no checkpoint is loaded, and the official test split is never opened,
so nothing here can let model behaviour influence how severities are defined.

    .venv/bin/python -m src.verify_corruptions
"""

import math
import subprocess
from hashlib import blake2b
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import GTSRB

from .corruptions import (
    CORRUPTIONS,
    OCCLUSION_FILL,
    SEVERITY_LEVELS,
    VALIDATION_NAMESPACE,
    CorruptedDataset,
    _blur_kernel_size,
    apply_corruption,
    build_validation_observations,
    check_stage_layout,
    corruption_parameter,
    corruption_seed,
    format_specification,
    occlusion_side,
    sample_seed,
    split_preprocessing,
    stable_index,
)
from .data import IMAGE_SIZE, SEED, build_preprocessing, create_data_loaders

# Recorded once from the implementation and asserted on every run. BLAKE2b is
# stable across processes and platforms, so a change here means the seeding rule
# itself changed and previously generated corrupted samples are not reproducible.
# It was last re-pinned when the split namespace entered the payload and
# occlusion stopped keying on severity; the earlier value, 7185269765872832186,
# belonged to the `base|corruption|severity|index` payload and no longer applies.
EXPECTED_SEED = 7645950319444661500
EXPECTED_SEED_ARGS = (VALIDATION_NAMESPACE, "gaussian_noise", 3, 17, SEED)

# The intended seeding rule, written out here rather than read off the spec, so
# that flipping `severity_in_seed` in corruptions.py fails this verifier instead
# of silently redefining what the conditions mean.
SEVERITY_IN_SEED = {
    "gaussian_blur": False,
    "brightness_dark": False,
    "brightness_bright": False,
    "gaussian_noise": True,
    "fog": False,
    "occlusion": False,
}

ARTIFACT_DIR = "artifacts/corruptions"


def independent_seed(namespace, corruption, severity, sample_index, base_seed=SEED):
    """Re-derive a seed straight from the documented rule.

    Written out from the specification rather than by calling `sample_seed`, so
    the pipeline checks below compare the implementation against the documented
    scheme instead of against itself.
    """
    token = str(severity) if SEVERITY_IN_SEED[corruption] else "-"
    payload = f"{base_seed}|{namespace}|{corruption}|{token}|{sample_index}".encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big") >> 1


def sample_observation(seed=0, size=IMAGE_SIZE):
    """A random but valid `[0, 1]` RGB observation, kept away from the clamps.

    Values sit in [0.2, 0.8] so that clamping does not dominate the brightness
    and noise checks; checks that specifically need the clamp build their own
    extreme inputs.
    """
    generator = torch.Generator().manual_seed(seed)
    return 0.2 + 0.6 * torch.rand(3, size, size, generator=generator)


def check_specification():
    """The severity tables must be ordered and the derived quantities sensible."""
    print(format_specification())

    assert set(CORRUPTIONS) == {
        "gaussian_blur", "brightness_dark", "brightness_bright",
        "gaussian_noise", "fog", "occlusion",
    }, sorted(CORRUPTIONS)

    for name, spec in CORRUPTIONS.items():
        assert len(spec.levels) == len(SEVERITY_LEVELS), (name, spec.levels)
        # Severity is an ordered index, so the parameter must move in one
        # direction. Darkening is the one condition where "stronger" means a
        # smaller number.
        increasing = name != "brightness_dark"
        for lower, higher in zip(spec.levels, spec.levels[1:]):
            assert (higher > lower) if increasing else (higher < lower), (name, spec.levels)

    kernels = [_blur_kernel_size(sigma) for sigma in CORRUPTIONS["gaussian_blur"].levels]
    assert all(size % 2 == 1 for size in kernels), kernels
    assert all(b >= a for a, b in zip(kernels, kernels[1:])), kernels
    print(f"  blur kernel sizes: {kernels}")

    sides = [
        occlusion_side(fraction, IMAGE_SIZE, IMAGE_SIZE)
        for fraction in CORRUPTIONS["occlusion"].levels
    ]
    realised = [side * side / (IMAGE_SIZE * IMAGE_SIZE) for side in sides]
    assert all(b > a for a, b in zip(sides, sides[1:])), sides
    assert all(b > a for a, b in zip(realised, realised[1:])), realised
    print(f"  occlusion sides at {IMAGE_SIZE}px: {sides} "
          f"-> realised area {[round(value, 6) for value in realised]}")
    largest = sides[-1]
    span = IMAGE_SIZE - (largest - largest // 2) - largest // 2 + 1
    print(f"  occlusion anchors: {span}x{span} = {span * span} centres "
          f"(every severity-5 patch fits without clamping)")
    print("Specification ordered; derived kernel sizes and patch areas monotonic.")


def check_rejected_inputs():
    """Invalid severities, names, domains, and missing seeds must all raise."""
    image = sample_observation()

    for severity in (0, 6, -1, 1.5, "3", None, True):
        try:
            apply_corruption(image, "fog", severity)
        except ValueError:
            continue
        raise AssertionError(f"severity {severity!r} was accepted")
    for severity in SEVERITY_LEVELS:
        apply_corruption(image, "fog", severity)
    print(f"  severities accepted: {list(SEVERITY_LEVELS)}; "
          "rejected: 0, 6, -1, 1.5, '3', None, True")

    for name in ("brightness", "blur", "", "GAUSSIAN_BLUR"):
        try:
            apply_corruption(image, name, 1)
        except ValueError:
            continue
        raise AssertionError(f"corruption {name!r} was accepted")
    print("  unknown corruption names rejected, including a case variant")

    # The guard that matters most: a normalised tensor must not be mistaken for
    # an observation, or the corruption would perturb model inputs instead.
    normalised = build_preprocessing().transforms[-1](image)
    try:
        apply_corruption(normalised, "fog", 3)
        raise AssertionError("a normalised [-1, 1] tensor was accepted")
    except ValueError as error:
        assert "[0, 1]" in str(error), error
    print("  normalised [-1, 1] input rejected")

    for bad in (image[0], image.unsqueeze(0), image.to(torch.uint8)):
        try:
            apply_corruption(bad, "fog", 3)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"input with shape {tuple(bad.shape)} / dtype {bad.dtype} accepted")
    print("  2-D, batched, and integer inputs rejected")

    # Corruption is a CPU-side step that runs before collation; the contract is
    # enforced rather than left to chance.
    if torch.backends.mps.is_available():
        try:
            apply_corruption(image.to("mps"), "fog", 3)
            raise AssertionError("an mps tensor was accepted")
        except ValueError as error:
            assert "CPU-side" in str(error), error
        print("  non-CPU (mps) input rejected: corruption runs before collation")
    else:
        print("  no accelerator available to test the CPU-only contract against")

    for name, spec in CORRUPTIONS.items():
        if not spec.stochastic:
            continue
        try:
            apply_corruption(image, name, 3)
            raise AssertionError(f"{name} ran without a seed")
        except ValueError as error:
            assert "seed" in str(error), error
    print("  stochastic corruptions refuse to run without an explicit seed")


def check_structure():
    """Every condition and severity must execute and return a valid image."""
    image = sample_observation(seed=1)
    extreme = torch.cat([
        torch.zeros(3, IMAGE_SIZE // 2, IMAGE_SIZE),
        torch.ones(3, IMAGE_SIZE - IMAGE_SIZE // 2, IMAGE_SIZE),
    ], dim=1)

    count = 0
    for name in CORRUPTIONS:
        for severity in SEVERITY_LEVELS:
            for source in (image, extreme):
                before = source.clone()
                output = apply_corruption(source, name, severity, seed=11)
                assert output.shape == source.shape, (name, severity, output.shape)
                assert output.dtype == source.dtype, (name, severity, output.dtype)
                assert torch.isfinite(output).all(), (name, severity)
                assert float(output.min()) >= 0.0, (name, severity, float(output.min()))
                assert float(output.max()) <= 1.0, (name, severity, float(output.max()))
                assert output.data_ptr() != source.data_ptr(), (name, severity)
                assert torch.equal(source, before), f"{name} s{severity} mutated its input"
                count += 1
    print(f"  {count} corruption/severity/input combinations executed")
    print("  shape, dtype, finiteness, [0, 1] range, and input immutability hold")
    print("  saturated inputs (all-0 and all-1 halves) stay in range")

    doubled = apply_corruption(image.to(torch.float64), "gaussian_noise", 5, seed=3)
    assert doubled.dtype == torch.float64, doubled.dtype
    print("  float64 input returns float64 (noise is drawn in the input dtype)")


def check_reproducibility():
    """Same image, corruption, severity, and seed must give the same result."""
    image = sample_observation(seed=2)

    for name, spec in CORRUPTIONS.items():
        for severity in SEVERITY_LEVELS:
            first = apply_corruption(image, name, severity, seed=7)
            second = apply_corruption(image, name, severity, seed=7)
            assert torch.equal(first, second), (name, severity)

            other = apply_corruption(image, name, severity, seed=8)
            if spec.stochastic:
                assert not torch.equal(first, other), (
                    f"{name} s{severity} ignored the seed")
            else:
                assert torch.equal(first, other), (
                    f"{name} s{severity} is meant to be deterministic")
    print("  identical (image, corruption, severity, seed) reproduces exactly")
    print("  stochastic conditions change with the seed; deterministic ones do not")

    # A different global RNG state must not leak into a seeded corruption.
    torch.manual_seed(1234)
    reference = apply_corruption(image, "occlusion", 4, seed=99)
    torch.manual_seed(4321)
    torch.rand(100)
    assert torch.equal(reference, apply_corruption(image, "occlusion", 4, seed=99))
    print("  results do not depend on global RNG state")


def check_seed_derivation():
    """Per-sample seeds must be stable, varied, and keyed on the right fields."""
    assert corruption_seed(*EXPECTED_SEED_ARGS) == EXPECTED_SEED, corruption_seed(
        *EXPECTED_SEED_ARGS)
    print(f"  corruption_seed{EXPECTED_SEED_ARGS} == {EXPECTED_SEED}")

    for name in CORRUPTIONS:
        for severity in SEVERITY_LEVELS:
            assert sample_seed(VALIDATION_NAMESPACE, name, severity, 31) == independent_seed(
                VALIDATION_NAMESPACE, name, severity, 31), (name, severity)
    print("  sample_seed matches an independent re-derivation of the documented rule")

    # Severity must change the realisation exactly where the design says it does.
    for name, severity_matters in SEVERITY_IN_SEED.items():
        seeds = {sample_seed(VALIDATION_NAMESPACE, name, s, 31) for s in SEVERITY_LEVELS}
        assert len(seeds) == (len(SEVERITY_LEVELS) if severity_matters else 1), (name, seeds)
    print("  gaussian_noise re-seeds per severity; occlusion keeps one anchor per image")

    seeds = {
        sample_seed(VALIDATION_NAMESPACE, name, severity, index)
        for name in CORRUPTIONS
        for severity in SEVERITY_LEVELS
        for index in range(200)
    }
    severity_specific = sum(SEVERITY_IN_SEED.values())
    expected_total = 200 * (
        severity_specific * len(SEVERITY_LEVELS) + (len(SEVERITY_IN_SEED) - severity_specific))
    assert len(seeds) == expected_total, (len(seeds), expected_total)
    assert all(0 <= value < 2 ** 63 for value in seeds)
    print(f"  {len(seeds)} distinct seeds over 200 indices "
          f"({severity_specific} x {len(SEVERITY_LEVELS)} severity-specific + "
          f"{len(SEVERITY_IN_SEED) - severity_specific} severity-independent conditions)")

    assert sample_seed(VALIDATION_NAMESPACE, "fog", 1, 0) != sample_seed(
        VALIDATION_NAMESPACE, "fog", 1, 0, base_seed=1)
    assert sample_seed(VALIDATION_NAMESPACE, "fog", 1, 0) != sample_seed(
        "test", "fog", 1, 0)
    print("  changing the base seed, or the split namespace, changes the realisation")


def check_monotonic_effects():
    """Only effects that must be monotonic by construction are asserted.

    Blur is deliberately absent: a larger sigma does not guarantee a larger
    pixel-wise change for an arbitrary image, so an image-level monotonic
    assertion there would be an artificial constraint rather than a property of
    the transformation. Its ordering is checked at the parameter level instead.
    """
    image = sample_observation(seed=4)

    # Fog is an affine contraction towards white, so pixel std is exactly
    # (1 - veil) times the clean std.
    clean_std = float(image.std())
    for severity in SEVERITY_LEVELS:
        veil = corruption_parameter("fog", severity)
        observed = float(apply_corruption(image, "fog", severity).std())
        assert math.isclose(observed, (1 - veil) * clean_std, rel_tol=1e-5), (severity, observed)
    print("  fog: pixel std equals (1 - veil) x clean std at every severity")

    # The sample image avoids the clamps, so darkening scales the mean exactly
    # and strictly decreases it.
    dark_means = [float(apply_corruption(image, "brightness_dark", s).mean())
                  for s in SEVERITY_LEVELS]
    assert all(b < a for a, b in zip(dark_means, dark_means[1:])), dark_means
    bright_means = [float(apply_corruption(image, "brightness_bright", s).mean())
                    for s in SEVERITY_LEVELS]
    assert all(b >= a for a, b in zip(bright_means, bright_means[1:])), bright_means
    print(f"  brightness_dark mean strictly decreasing: "
          f"{[round(v, 4) for v in dark_means]}")
    print(f"  brightness_bright mean non-decreasing (clamped): "
          f"{[round(v, 4) for v in bright_means]}")

    noise_deviation = [
        float((apply_corruption(image, "gaussian_noise", s, seed=5) - image).abs().mean())
        for s in SEVERITY_LEVELS
    ]
    assert all(b > a for a, b in zip(noise_deviation, noise_deviation[1:])), noise_deviation
    print(f"  gaussian_noise mean |deviation| increasing: "
          f"{[round(v, 4) for v in noise_deviation]}")

    # Count pixels actually covered rather than trusting the spec: the sample
    # image never takes the fill value, so changed pixels are exactly the patch.
    covered = []
    for severity in SEVERITY_LEVELS:
        output = apply_corruption(image, "occlusion", severity, seed=6)
        changed = (output != image).any(dim=0)
        side = occlusion_side(
            corruption_parameter("occlusion", severity), IMAGE_SIZE, IMAGE_SIZE)
        assert int(changed.sum()) == side * side, (severity, int(changed.sum()), side)
        assert torch.equal(output[:, changed], torch.full((3, side * side), OCCLUSION_FILL))
        covered.append(int(changed.sum()))
    assert all(b > a for a, b in zip(covered, covered[1:])), covered
    print(f"  occlusion covered pixels strictly increasing: {covered} "
          f"of {IMAGE_SIZE * IMAGE_SIZE}")


def occluded_mask(image, severity, seed):
    """Boolean map of the pixels one occlusion severity covers."""
    return (apply_corruption(image, "occlusion", severity, seed=seed) != image).any(dim=0)


def check_occlusion_nesting():
    """Severities must be concentric: every covered pixel stays covered."""
    anchors = set()
    for sample in range(40):
        image = sample_observation(seed=100 + sample)
        seed = sample_seed(VALIDATION_NAMESPACE, "occlusion", 1, sample)

        masks = [occluded_mask(image, severity, seed) for severity in SEVERITY_LEVELS]
        for severity, (inner, outer) in enumerate(zip(masks, masks[1:]), start=1):
            # Containment, not merely a larger count.
            assert not (inner & ~outer).any(), (
                f"sample {sample}: severity {severity} covers pixels that "
                f"severity {severity + 1} leaves visible")

        # Recover the centre from each mask. The patch spans [c - side // 2,
        # + side), so adding side // 2 back to the first covered row returns c
        # exactly for both even and odd sides.
        centres = set()
        for severity, mask in zip(SEVERITY_LEVELS, masks):
            side = occlusion_side(
                corruption_parameter("occlusion", severity), IMAGE_SIZE, IMAGE_SIZE)
            rows = mask.any(dim=1).nonzero()
            columns = mask.any(dim=0).nonzero()
            assert int(rows[-1]) - int(rows[0]) + 1 == side, (sample, severity)
            assert int(columns[-1]) - int(columns[0]) + 1 == side, (sample, severity)
            centres.add((int(rows[0]) + side // 2, int(columns[0]) + side // 2))
        assert len(centres) == 1, (sample, centres)
        anchors |= centres

    print("  40 samples: s1 within s2 within s3 within s4 within s5, every pair")
    print("  all five severities recover one shared centre per sample")
    print(f"  {len(anchors)} distinct anchors across the 40 samples "
          "(placement varies between images)")
    assert len(anchors) > 30, anchors


def check_pipeline_alignment(validation):
    """The corrupted path must reuse the clean pipeline, normalising exactly once."""
    _, normalize = split_preprocessing()
    check_stage_layout(list(build_preprocessing().transforms))
    try:
        check_stage_layout(list(build_preprocessing().transforms)[:2])
        raise AssertionError("a changed pipeline layout was accepted")
    except RuntimeError:
        pass
    print("  stage-layout guard accepts the current pipeline and rejects a changed one")

    raw, _ = validation[0]
    assert float(raw.min()) >= 0.0 and float(raw.max()) <= 1.0, (
        float(raw.min()), float(raw.max()))
    print(f"  observation half yields [0, 1] values: "
          f"[{float(raw.min()):.4f}, {float(raw.max()):.4f}]")

    clean_dataset = GTSRB(
        root=Path(__file__).resolve().parents[1] / "data",
        split="train", transform=build_preprocessing(), download=False,
    )
    clean_wrapper = CorruptedDataset(validation, corruption=None)
    for position in (0, 1, 500, 2500, len(validation) - 1):
        index = stable_index(validation, position)
        through_wrapper, label = clean_wrapper[position]
        reference, reference_label = clean_dataset[index]
        assert torch.equal(through_wrapper, reference), position
        assert label == reference_label, position
    print("  clean path through the wrapper is bit-identical to build_preprocessing()")

    # The corrupted path must be exactly normalize(corrupt(observation)) - one
    # normalisation, applied after the corruption, using the frozen constants.
    # Both stochastic conditions are included, with the seed re-derived from the
    # documented rule rather than by calling the same helper the wrapper uses, so
    # a wrong seed or a swapped corrupt/normalize order cannot pass.
    for corruption, severity in (("fog", 3), ("gaussian_noise", 4), ("occlusion", 3)):
        for position in (0, 137, len(validation) - 1):
            observation_image, _ = validation[position]
            index = stable_index(validation, position)
            seed = independent_seed(VALIDATION_NAMESPACE, corruption, severity, index)
            expected = normalize(
                apply_corruption(observation_image, corruption, severity, seed=seed))
            corrupted, _ = CorruptedDataset(validation, corruption, severity)[position]
            assert torch.equal(corrupted, expected), (corruption, severity, position)
            assert float(corrupted.min()) >= -1.0 and float(corrupted.max()) <= 1.0
        print(f"  {corruption} s{severity}: wrapper output == "
              "normalize(corrupt(observation, independently derived seed)) at 3 positions")


def check_split_agreement(validation_indices):
    """The corruption helper must reuse the exact split the loaders use."""
    loaders, split_info = create_data_loaders(download=False)
    assert (split_info["validation_indices"] == validation_indices).all()
    assert len(loaders["validation"].dataset) == len(validation_indices)
    print(f"  validation indices identical to create_data_loaders: "
          f"{len(validation_indices)} images")
    print("  the official test loader was built by data.py but never iterated here")


class IndexedCorrupted(torch.utils.data.Dataset):
    """Yields `(image, label, position)` so a shuffled loader can be re-ordered.

    Defined at module level so it survives pickling into DataLoader workers.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, position):
        image, label = self.dataset[position]
        return image, label, position


def check_batch_order_independence(validation):
    """Corrupted output must depend on the sample, not on how it was loaded."""
    positions = list(range(96))
    subset = Subset(validation, positions)

    for name, severity in (("gaussian_noise", 4), ("occlusion", 3)):
        dataset = CorruptedDataset(subset, name, severity)

        sequential = [dataset[position][0] for position in positions]

        # Access out of order, with unrelated global RNG consumption in between.
        order = torch.randperm(len(positions), generator=torch.Generator().manual_seed(0))
        shuffled = [positions[i] for i in order.tolist()]
        out_of_order = {}
        for position in shuffled:
            torch.rand(3)
            out_of_order[position] = dataset[position][0]
        for position in positions:
            assert torch.equal(sequential[position], out_of_order[position]), (name, position)

        # A real shuffled, batched, multi-worker loader must agree as well.
        loader = DataLoader(
            IndexedCorrupted(dataset), batch_size=7, shuffle=True,
            num_workers=2, generator=torch.Generator().manual_seed(123),
        )
        seen = 0
        for images, _, batch_positions in loader:
            for image, position in zip(images, batch_positions.tolist()):
                assert torch.equal(image, sequential[position]), (name, position)
                seen += 1
        assert seen == len(positions), seen
        print(f"  {name} s{severity}: {seen} samples identical across sequential, "
              "out-of-order, and shuffled 2-worker access")

    # Seeds follow the underlying dataset index, so re-wrapping a different
    # subset of the same dataset must not change a sample's realisation.
    reversed_subset = Subset(validation, positions[::-1])
    direct = CorruptedDataset(validation, "occlusion", 3)
    through_reversed = CorruptedDataset(reversed_subset, "occlusion", 3)
    for offset, position in enumerate(positions[::-1]):
        assert torch.equal(through_reversed[offset][0], direct[position][0]), position
    print("  realisations survive re-subsetting: seeds key on the dataset index")


def check_artifacts_ignored():
    """Corruption grids must land in a Git-ignored directory."""
    root = Path(__file__).resolve().parents[1]
    for path in (ARTIFACT_DIR, f"{ARTIFACT_DIR}/gaussian_blur.png"):
        result = subprocess.run(
            ["git", "check-ignore", "-v", path],
            cwd=root, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"NOT ignored by Git: {path}"
        print(f"  ignored by {result.stdout.split(chr(9))[0]}: {path}")
    tracked = subprocess.run(
        ["git", "ls-files", "artifacts"], cwd=root, capture_output=True, text=True,
    )
    assert not tracked.stdout.strip(), tracked.stdout
    print("  no artifact path is tracked by Git")


def main():
    print("Corruption specification:")
    check_specification()

    print("\nInput validation:")
    check_rejected_inputs()

    print("\nStructural properties:")
    check_structure()

    print("\nReproducibility:")
    check_reproducibility()

    print("\nPer-sample seeding:")
    check_seed_derivation()

    print("\nMonotonic effects:")
    check_monotonic_effects()

    print("\nOcclusion nesting:")
    check_occlusion_nesting()

    validation, validation_indices, _ = build_validation_observations()
    print(f"\nPipeline alignment ({len(validation)} validation observations):")
    check_pipeline_alignment(validation)
    check_split_agreement(validation_indices)

    print("\nBatch-order independence:")
    check_batch_order_independence(validation)

    print("\nArtifact locations:")
    check_artifacts_ignored()

    print("\nAll corruption checks passed. No model was built, no predictions were "
          "computed, and the official test split was never read.")


if __name__ == "__main__":
    main()
