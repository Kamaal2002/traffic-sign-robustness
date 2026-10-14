"""Plot the baseline robustness sweep for human inspection.

    .venv/bin/python -m src.plot_robustness

Reads the saved results of `src.evaluate_robustness` and writes figures next to
them. It recomputes nothing: every number plotted here comes from the CSVs that
the evaluation run produced, so a figure cannot disagree with the table.

Severity 0 is the clean condition, shared by every curve, so each line starts
from the same point and the vertical fall is the degradation.

Absolute-drop curves are deliberately not plotted. With one clean reference the
drop is `clean - score`, i.e. the accuracy and macro F1 figures flipped about a
constant; a separate panel would repeat them without adding information. The
per-class heatmap is included instead, because it shows something the curves
cannot: which classes carry the loss.

The occlusion curve needs the caveat `OCCLUSION_NOTE` carries, which is why it
is annotated on the figures: its severities are nested about one anchor per
image and that anchor is centre-constrained, so the curve measures occluding
the middle of the sign rather than occluding x% of an arbitrary region.
"""

import argparse
import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot)

from .corruptions import CORRUPTIONS, SEVERITY_LEVELS  # noqa: E402
from .data import NUM_CLASSES  # noqa: E402
from .evaluate_robustness import (  # noqa: E402
    CLEAN_CONDITION,
    OCCLUSION_NOTE,
    RESULTS_DIR,
)

# One colour and marker per condition, fixed here so every figure in the
# project identifies a corruption the same way.
STYLES = {
    "gaussian_blur": ("#1f77b4", "o"),
    "brightness_dark": ("#2b2b6f", "v"),
    "brightness_bright": ("#ff7f0e", "^"),
    "gaussian_noise": ("#2ca02c", "s"),
    "fog": ("#9467bd", "D"),
    "occlusion": ("#d62728", "P"),
}


# The figures carry the same wording as the saved metadata rather than a
# paraphrase, so the occlusion curve cannot be read without the caveat.
OCCLUSION_CAPTION = "occlusion: " + textwrap.fill(OCCLUSION_NOTE, width=150)


def add_caption(figure, text):
    figure.text(0.01, 0.005, text, fontsize=5.5, color="#444444", va="bottom")


def read_summary(results_dir):
    """Load summary.csv as {corruption: {severity: row}} plus the clean row."""
    conditions = defaultdict(dict)
    clean = None
    with open(results_dir / "summary.csv") as handle:
        for row in csv.DictReader(handle):
            severity = int(row["severity"])
            for key in ("accuracy", "macro_f1", "accuracy_drop", "macro_f1_drop"):
                row[key] = float(row[key])
            if row["corruption"] == CLEAN_CONDITION:
                clean = row
            else:
                conditions[row["corruption"]][severity] = row
    return clean, conditions


def read_per_class(results_dir):
    """Load per_class_metrics.csv as F1 vectors, plus support and track counts.

    Support and tracks are read from the clean rows, which hold the same split
    for every condition. Both are needed for the row labels: support alone
    would suggest a class with 30 images carries 30 independent samples when it
    is 30 frames of one sign.
    """
    scores = {}
    supports = {}
    tracks = {}
    with open(results_dir / "per_class_metrics.csv") as handle:
        for row in csv.DictReader(handle):
            key = (row["corruption"], int(row["severity"]))
            class_id = int(row["class"])
            scores.setdefault(key, np.zeros(NUM_CLASSES))[class_id] = float(row["f1"])
            if row["corruption"] == CLEAN_CONDITION:
                supports[class_id] = int(row["support"])
                tracks[class_id] = int(row["tracks"])
    return scores, supports, tracks


def plot_metric(clean, conditions, metric, label, path):
    """One curve per corruption, all starting from the shared clean point."""
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    severities = [0, *SEVERITY_LEVELS]
    for name in CORRUPTIONS:
        colour, marker = STYLES[name]
        values = [clean[metric]] + [conditions[name][s][metric] for s in SEVERITY_LEVELS]
        axis.plot(severities, values, marker=marker, color=colour, label=name,
                  linewidth=1.8, markersize=5)
    axis.axhline(clean[metric], color="grey", linewidth=0.8, linestyle=":")
    axis.annotate(f"clean {clean[metric]:.4f}", (0.02, clean[metric]),
                  xycoords=("axes fraction", "data"), va="bottom", fontsize=8,
                  color="grey")
    axis.set_xticks(severities)
    axis.set_xlabel("severity (0 = clean)")
    axis.set_ylabel(label)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="lower left")
    axis.set_title(f"Baseline CNN: validation {label} vs corruption severity")
    figure.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    add_caption(figure, OCCLUSION_CAPTION)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_per_class(scores, supports, tracks, path):
    """Per-class F1 for every condition, classes ordered by validation support.

    The curves show how much is lost; this shows where. Ordering rows by support
    puts the classes with a single validation track at the top, so the reader
    can see at a glance whether the damage concentrates there. Row labels carry
    the track count beside the image count, because a one-track row is a single
    physical sign however many frames it holds.
    """
    order = sorted(range(NUM_CLASSES), key=lambda c: (supports[c], c))

    columns = [(CLEAN_CONDITION, 0)] + [
        (name, severity) for name in CORRUPTIONS for severity in SEVERITY_LEVELS
    ]
    grid = np.column_stack([scores[key][order] for key in columns])

    figure, axis = plt.subplots(figsize=(11.5, 8.0))
    image = axis.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels([f"{name}\ns{severity}" if severity else "clean"
                          for name, severity in columns], fontsize=6, rotation=90)
    axis.set_yticks(range(NUM_CLASSES))
    axis.set_yticklabels([f"{c} (n={supports[c]}, {tracks[c]} trk)" for c in order],
                         fontsize=6)
    axis.set_ylabel("class, ordered by validation support (n images, tracks)")
    axis.set_title("Per-class validation F1 under each condition")
    figure.colorbar(image, ax=axis, label="per-class F1", fraction=0.025)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    add_caption(figure, OCCLUSION_CAPTION)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None,
                        help="directory holding a sweep's CSVs; plots are "
                             "written to its plots/ subdirectory")
    arguments = parser.parse_args()

    results_dir = Path(arguments.results_dir) if arguments.results_dir else RESULTS_DIR
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    clean, conditions = read_summary(results_dir)
    scores, supports, tracks = read_per_class(results_dir)

    written = [
        plot_metric(clean, conditions, "accuracy", "accuracy",
                    plot_dir / "accuracy_vs_severity.png"),
        plot_metric(clean, conditions, "macro_f1", "macro F1",
                    plot_dir / "macro_f1_vs_severity.png"),
        plot_per_class(scores, supports, tracks,
                       plot_dir / "per_class_f1_heatmap.png"),
    ]
    print(f"Read {results_dir}")
    print("Plots written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
