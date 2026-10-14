"""Render clean-vs-severity grids for visual inspection of the corruption suite.

One PNG per condition, each row a validation sign and each column clean through
severity 5. The point is to confirm by eye that every condition looks like what
its name claims, and to judge where each severity scale actually sits.

Severity 5 is intended as severe degradation near the useful edge of the
benchmark: strong enough to be a real test, while aiming to leave enough visual
information for meaningful recognition in most examples. It is not a guarantee
that every sign stays identifiable under every condition — a dark sign under
heavy noise, or a small glyph under a large occluding patch, can lose the
content that distinguishes its class. Where that happens the measured accuracy
reflects an irreducible ceiling rather than model weakness, which is a reason to
report severities separately instead of averaging over them.

    .venv/bin/python -m src.visualize_corruptions

Sample choice uses dataset and image properties only — class id for shape and
colour variety, mean luminance for exposure variety. No model, checkpoint,
prediction, or saliency is involved, and only validation images are read, so the
severity scale cannot be tuned to classifier behaviour. Stochastic conditions
use the same per-sample seeds the evaluation will use, so the grids show the
exact realisations that will be measured later.

Output goes to `artifacts/corruptions/`, which is Git-ignored.
"""

from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot)

from .corruptions import (  # noqa: E402
    CORRUPTIONS,
    SEVERITY_LEVELS,
    VALIDATION_NAMESPACE,
    apply_corruption,
    build_validation_observations,
    corruption_parameter,
    sample_seed,
    stable_index,
)

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "corruptions"

# Six classes spanning the shape and colour families GTSRB contains: red
# circle, yellow diamond, inverted triangle, red octagon, red triangle, blue
# circle. Chosen from the official class definitions, not from any measurement
# of model behaviour.
PANEL_CLASSES = (
    (1, "speed limit 30 (red circle)"),
    (12, "priority road (yellow diamond)"),
    (13, "yield (inverted triangle)"),
    (14, "stop (red octagon)"),
    (25, "road work (red triangle)"),
    (34, "turn left ahead (blue circle)"),
)

# Luminance quantile used to pick within each class, cycling dim/typical/bright
# so the panel spans apparent brightness as well as shape.
LUMINANCE_QUANTILES = (0.05, 0.5, 0.95)

# ITU-R BT.601 luma weights, applied to the [0, 1] observation.
LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)


def mean_luminance(image):
    return float((image * LUMA_WEIGHTS).sum(dim=0).mean())


def select_samples(validation, labels):
    """Pick one validation image per panel class at a fixed luminance quantile.

    Returns `(position, caption)` pairs. Selection is fully determined by the
    class id, the luminance ordering within that class, and the quantile list,
    so it is reproducible and involves nothing classifier-specific.
    """
    selected = []
    for panel, (class_id, description) in enumerate(PANEL_CLASSES):
        quantile = LUMINANCE_QUANTILES[panel % len(LUMINANCE_QUANTILES)]
        positions = [
            position for position in range(len(validation))
            if labels[stable_index(validation, position)] == class_id
        ]
        ranked = sorted(positions, key=lambda position: mean_luminance(validation[position][0]))
        position = ranked[min(int(quantile * len(ranked)), len(ranked) - 1)]
        luminance = mean_luminance(validation[position][0])
        selected.append((
            position,
            f"class {class_id}\n{description}\nluminance {luminance:.3f} "
            f"(q{quantile:g} of {len(ranked)})",
        ))
    return selected


def render(validation, samples, corruption):
    """Write one PNG: rows are samples, columns are clean then severity 1-5."""
    spec = CORRUPTIONS[corruption]
    columns = 1 + len(SEVERITY_LEVELS)
    figure, axes = plt.subplots(
        len(samples), columns, figsize=(1.6 * columns + 2.2, 1.6 * len(samples)),
    )

    for row, (position, caption) in enumerate(samples):
        image, _ = validation[position]
        index = stable_index(validation, position)
        panels = [("clean", image)]
        for severity in SEVERITY_LEVELS:
            seed = sample_seed(VALIDATION_NAMESPACE, corruption, severity, index)
            panels.append((
                f"s{severity}: {spec.parameter}={corruption_parameter(corruption, severity):g}",
                apply_corruption(image, corruption, severity, seed=seed),
            ))
        for column, (title, panel) in enumerate(panels):
            axis = axes[row][column]
            axis.imshow(panel.permute(1, 2, 0).numpy())
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=8)
        axes[row][0].set_ylabel(caption, fontsize=6, rotation=0, ha="right", va="center")

    seeding = "per-sample seed" if spec.stochastic else "deterministic"
    figure.suptitle(f"{corruption} — {spec.summary} ({seeding})", fontsize=11)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path = ARTIFACT_DIR / f"{corruption}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    validation, _, labels = build_validation_observations()
    print(f"Validation observations: {len(validation)}")

    samples = select_samples(validation, labels)
    print("Selected samples (validation split only, chosen without any model):")
    for position, caption in samples:
        flat = caption.replace("\n", " | ")
        print(f"  position {position:5d} (dataset index {stable_index(validation, position)}): {flat}")

    print("\nGrids written:")
    for corruption in CORRUPTIONS:
        print(f"  {render(validation, samples, corruption)}")
    print("\nNo model was loaded and no official test image was read.")


if __name__ == "__main__":
    main()
