"""Figures for the baseline failure analysis.

    .venv/bin/python -m src.plot_failures

Reads what `analyze_failures.py` wrote and draws it. Nothing is recomputed from
the predictions here, so a figure cannot disagree with the CSV it came from, and
there are only as many figures as there are questions:

    failure_onset.png         when clean-correct frames break, frames and signs
    error_concentration.png   how concentrated the error destinations are
    class_fragility.png       which classes degrade, and on how much evidence
    occlusion_anchor.png      failure rate against occlusion patch position
    examples_onset.png        what a first failure at each severity looks like
    examples_transitions.png  what the dominant confusions look like
    examples_unstable.png     frames whose prediction keeps changing

The three example figures re-render validation frames. The images come from the
frozen corruption suite, addressed by the dataset index recorded in
`examples.csv` and seeded exactly as the evaluation seeded them, so a displayed
image is the realisation the accepted prediction was made on. The predicted
classes drawn on the panels are read from `examples.csv`; no model runs here and
the official test split is never constructed.

Colours and the occlusion caveat are imported from `plot_robustness.py` so a
corruption looks the same in every figure in the project. `plots/sources.json`
records the analysis tables each run drew from and the summary of the analysis
run they came from, by repository-relative path and digest, so a figure can be
traced back to the table behind it.
"""

import hashlib
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot)

from .analyze_failures import (  # noqa: E402
    ANALYSIS_DIR,
    ANCHOR_SEVERITIES,
    EXPECTED_TRACKS,
    MATCHED_ERRORS,
    PROJECT_ROOT,
    TRACK_MAJORITY,
)
from .corruptions import (  # noqa: E402
    SEVERITY_LEVELS,
    VALIDATION_NAMESPACE,
    apply_corruption,
    build_validation_observations,
    corruption_parameter,
    sample_seed,
    stable_index,
)
from .data import NUM_CLASSES  # noqa: E402
from .plot_robustness import OCCLUSION_CAPTION, STYLES, add_caption  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

# Severity colours for the stacked onset bars: pale for a failure that needs the
# strongest level, saturated for one that appears immediately.
SEVERITY_COLOURS = plt.get_cmap("YlOrRd")(np.linspace(0.85, 0.3, len(SEVERITY_LEVELS)))
SURVIVOR_COLOUR = "#d9d9d9"

CORRECT_COLOUR = "#2ca02c"
WRONG_COLOUR = "#d62728"

# Marker per support group in the per-class figure. A class with one validation
# track is one physical sign, which is drawn as a hollow marker so it cannot be
# mistaken for class-level evidence.
SUPPORT_STYLES = {
    "1 track": ("^", "none", "#d62728"),
    "2 tracks": ("s", "#ff7f0e", "#ff7f0e"),
    "3+ tracks": ("o", "#1f77b4", "#1f77b4"),
}


def caption_text(figure, text, top=0.94, width=170):
    """Draw a wrapped caption and give the layout room for it.

    Returns the `rect` for `tight_layout`: the reserved fraction follows the
    number of wrapped lines and the figure height, so a long caption on a short
    figure does not land on the x-axis labels.
    """
    wrapped = textwrap.fill(text, width=width)
    add_caption(figure, wrapped)
    reserved = 0.02 + 0.115 * (wrapped.count("\n") + 1) / figure.get_figheight()
    return (0.0, reserved, 1.0, top)


def support_group(tracks):
    return "1 track" if tracks == 1 else "2 tracks" if tracks == 2 else "3+ tracks"


SOURCES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    SOURCES[str(path.relative_to(PROJECT_ROOT))] = digest(path)
    return path


def read(name):
    """Load an analysis table and record its digest for the provenance file."""
    return pd.read_csv(record(ANALYSIS_DIR / name))


def stacked_onset(axis, frame, value_column, group_column, order, unit_label, total_label):
    """Stacked bars of onset severity, one bar per corruption."""
    names = list(order)
    bottom = np.zeros(len(names))
    for position, severity in enumerate(SEVERITY_LEVELS):
        heights = np.array([
            float(frame[(frame["corruption"] == name) & (frame[group_column] == severity)]
                  [value_column].sum())
            for name in names
        ])
        axis.barh(names, heights, left=bottom, color=SEVERITY_COLOURS[position],
                  label=f"{unit_label} first wrong at s{severity}",
                  edgecolor="white", linewidth=0.4)
        bottom += heights
    survivors = np.array([
        float(frame[(frame["corruption"] == name) & (frame[group_column] == 0)]
              [value_column].sum())
        for name in names
    ])
    axis.barh(names, survivors, left=bottom, color=SURVIVOR_COLOUR,
              label=total_label, edgecolor="white", linewidth=0.4)
    axis.invert_yaxis()
    axis.tick_params(labelsize=8)


def plot_failure_onset(onset, totals, track_onset, path):
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    order = onset["corruption"].drop_duplicates().tolist()

    stacked_onset(axes[0], onset, "frames", "onset_severity", order,
                  "frame", "frame never wrong")
    clean_correct = onset[onset["corruption"] == order[0]]["frames"].sum()
    axes[0].set_xlim(0, clean_correct * 1.35)
    axes[0].set_xlabel(f"clean-correct frames ({clean_correct} total)", fontsize=9)
    axes[0].set_title("Frames, by the severity at which each first fails", fontsize=10)
    # The track counts in `onset` are per bucket and a sign can appear in
    # several of them, so the annotation reads the union from the totals table.
    for position, name in enumerate(order):
        row = totals[totals["corruption"] == name].iloc[0]
        axes[0].text(int(row["frames_ever_wrong"]) + 80, position,
                     f"{int(row['frames_ever_wrong'])} frames, from "
                     f"{int(row['tracks_affected'])} of {EXPECTED_TRACKS} signs",
                     fontsize=7, va="center")

    stacked_onset(axes[1], track_onset, "tracks", "majority_onset_severity", order,
                  "majority", "majority never wrong")
    axes[1].set_xlabel(f"validation tracks ({EXPECTED_TRACKS} total)", fontsize=9)
    axes[1].set_title(
        f"Signs, by the severity at which over {TRACK_MAJORITY:g} of the sign's "
        "clean-correct frames fail",
        fontsize=10,
    )
    # Each panel keeps its own legend: the same colour means "this frame first
    # failed here" on the left and "this sign's majority first failed here" on
    # the right, and one shared legend would have to misname one of them.
    axes[0].legend(fontsize=6.5, loc="upper right", framealpha=0.95)
    axes[1].legend(fontsize=6.5, loc="lower right", framealpha=0.95)

    figure.suptitle(
        "Failure onset: the same failures counted as frames and as physical signs",
        fontsize=12,
    )
    rect = caption_text(figure, (
        "A sign can lose its first frame at one severity and another frame at a stronger one, "
        "so the per-bucket track counts in failure_onset.csv overlap; the signs-affected "
        "annotation is the union over severities, taken from failure_onset_totals.csv. "
        + OCCLUSION_CAPTION
    ), top=0.94)
    figure.tight_layout(rect=rect)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_error_concentration(destinations, path):
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharex=True)
    for name, group in destinations.groupby("corruption", sort=False):
        colour, marker = STYLES[name]
        group = group.sort_values("severity")
        for axis, column, matched in (
            (axes[0], "top3_share", "top3_share_matched"),
            (axes[1], "normalised_entropy", "normalised_entropy_matched"),
        ):
            axis.plot(group["severity"], group[column], color=colour, marker=marker,
                      markersize=5, linewidth=1.4, label=name)
            available = group[group[matched].notna()]
            axis.plot(available["severity"], available[matched], color=colour,
                      marker=marker, markersize=8, linestyle="none",
                      markerfacecolor="none", linewidth=1.0)

    axes[0].set_ylabel("share of induced errors in the top 3 destinations", fontsize=9)
    axes[0].set_title("Top-3 destination share", fontsize=10)
    axes[1].set_ylabel(f"entropy of destinations / log({NUM_CLASSES})", fontsize=9)
    axes[1].set_title("Normalised entropy (1 = spread over all classes)", fontsize=10)
    for axis in axes:
        axis.set_xlabel("severity", fontsize=9)
        axis.set_xticks(list(SEVERITY_LEVELS))
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=8)
    axes[1].legend(fontsize=7, loc="lower left", ncol=2, framealpha=0.95)

    figure.suptitle(
        "Where corruption-induced errors go: filled markers use every error, "
        f"hollow markers a matched {MATCHED_ERRORS}-error draw",
        fontsize=11,
    )
    smallest = destinations["errors_induced"].min()
    rect = caption_text(figure, (
        f"A point with no hollow marker comes from fewer than {MATCHED_ERRORS} induced errors "
        "and is noisy; "
        f"the smallest here is {smallest}. Both statistics are computed over frames, so a high "
        "share can also mean a few signs contributed many frames: the number of distinct tracks "
        "behind each point is in error_destinations.csv. " + OCCLUSION_CAPTION
    ), top=0.91)
    figure.tight_layout(rect=rect)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_class_fragility(classes, rho, pvalue, path):
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    classes = classes.copy()
    classes["group"] = classes["validation_tracks"].map(support_group)

    for group in SUPPORT_STYLES:
        rows = classes[classes["group"] == group]
        marker, face, edge = SUPPORT_STYLES[group]
        for axis, x_column in ((axes[0], "clean_recall"), (axes[1], "validation_tracks")):
            axis.scatter(rows[x_column], rows["mean_recall_drop_s5"], marker=marker,
                         facecolors=face, edgecolors=edge, s=46, linewidths=1.2, alpha=0.8,
                         label=group if axis is axes[0] else None)

    # Label the classes a reader would otherwise have to look up: the largest
    # severity-5 degradations and anything already weak on clean input.
    notable = pd.concat([
        classes.nlargest(8, "mean_recall_drop_s5"),
        classes[classes["clean_recall"] < 0.95],
    ]).drop_duplicates(subset="class")
    for _, row in notable.iterrows():
        axes[0].annotate(int(row["class"]),
                         (row["clean_recall"], row["mean_recall_drop_s5"]),
                         textcoords="offset points", xytext=(5, 3), fontsize=7)

    axes[0].set_xlabel("clean validation recall", fontsize=9)
    axes[0].set_ylabel("mean recall drop at severity 5\n(over the six corruptions)", fontsize=9)
    axes[0].set_title("Clean weakness and degradation are different things", fontsize=10)
    axes[0].legend(fontsize=7, loc="upper left", title="validation support",
                   title_fontsize=7, framealpha=0.95)
    axes[1].set_xlabel("validation tracks (physical signs)", fontsize=9)
    axes[1].set_title(
        f"Degradation against validation support: Spearman rho {rho:+.3f} (p {pvalue:.2f})",
        fontsize=10,
    )
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=8)

    figure.suptitle("Per-class degradation against the evidence behind it", fontsize=12)
    rect = caption_text(figure, (
        "Each point is one of the 43 classes. A class with one validation track is 30 frames of "
        "a single physical sign, so its recall is a single-sign observation and is drawn hollow. "
        "Most classes sit at clean recall 1.0 with degradations from 0.03 to 0.50, which is the "
        "point of the left panel. The right panel is descriptive: with 43 classes a rank "
        "correlation this small is uninformative either way, and because the split is "
        "stratified, validation support is proportional to training support, so support and "
        "learnability cannot be separated here."
    ))
    figure.tight_layout(rect=rect)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_occlusion_anchor(anchor, within_track, path):
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    shades = plt.get_cmap("Reds")(np.linspace(0.45, 0.9, len(ANCHOR_SEVERITIES)))
    for position, severity in enumerate(ANCHOR_SEVERITIES):
        rows = anchor[anchor["severity"] == severity].sort_values("offset_low")
        centres = (rows["offset_low"] + rows["offset_high"]) / 2
        side = int(rows.iloc[0]["patch_side_pixels"])
        axes[0].plot(centres, rows["failure_rate"], marker="o", color=shades[position],
                     label=f"s{severity}: {side}x{side} patch "
                           f"({corruption_parameter('occlusion', severity):g} of the frame)")
    # The bins hold the same frames at every severity, so the counts belong on
    # the axis rather than on each curve.
    bins = anchor[anchor["severity"] == ANCHOR_SEVERITIES[0]].sort_values("offset_low")
    axes[0].set_xticks((bins["offset_low"] + bins["offset_high"]) / 2)
    axes[0].set_xticklabels(
        [f"{int(row.offset_low)}-{int(row.offset_high)}\n{int(row.frames)} frames"
         for row in bins.itertuples()]
    )
    axes[0].set_xlabel("anchor distance from the image centre (pixels, Chebyshev)", fontsize=9)
    axes[0].set_ylabel("failure rate among clean-correct frames", fontsize=9)
    axes[0].set_title(
        "At a fixed patch area, failure rate is lower where the anchor fell farther out",
        fontsize=10,
    )
    axes[0].legend(fontsize=7, framealpha=0.95)

    for position, severity in enumerate(ANCHOR_SEVERITIES):
        differences = within_track[within_track["severity"] == severity]["difference"]
        axes[1].hist(differences, bins=np.linspace(-0.8, 0.8, 33), histtype="step",
                     linewidth=1.4, color=shades[position],
                     label=f"s{severity}: mean {differences.mean():+.3f}, "
                           f"positive in {(differences > 0).sum()}/{len(differences)}")
    axes[1].axvline(0.0, color="#666666", linewidth=0.8)
    axes[1].set_xlabel("within one sign: failure rate of nearer-than-median anchors\n"
                       "minus failure rate of the rest", fontsize=9)
    axes[1].set_ylabel("validation tracks", fontsize=9)
    axes[1].set_title("The same comparison inside each physical sign", fontsize=10)
    axes[1].legend(fontsize=7, framealpha=0.95)
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=8)

    figure.suptitle(
        "Occlusion: the accepted failure rates against where the patch landed",
        fontsize=12,
    )
    rect = caption_text(figure, (
        "The anchor is drawn from the sample's dataset index and not from the image, so position "
        "is assigned independently of content, and the 30 frames of one sign receive 30 different "
        "anchors. The right panel uses that to compare each sign against itself. This is an "
        "association measured inside the accepted benchmark, which was not designed to vary "
        "patch position: it motivates a controlled location ablation rather than standing in for "
        "one. " + OCCLUSION_CAPTION
    ))
    figure.tight_layout(rect=rect)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


class FrameSource:
    """Re-render validation frames under the frozen corruption suite.

    Frames are addressed by their index in the original torchvision ordering,
    which is the index `examples.csv` records and the index every corruption
    seed is derived from. Resolving it back through `stable_index` and comparing
    is what guarantees a displayed image is the frame the accepted prediction
    belongs to, rather than whatever sits at that row of the validation subset.
    """

    def __init__(self):
        self.dataset, self.validation_indices, self.labels = build_validation_observations()
        self.images_read = 0

    def clean(self, dataset_index):
        position = int(np.searchsorted(self.validation_indices, dataset_index))
        if position >= len(self.validation_indices):
            raise RuntimeError(f"dataset index {dataset_index} is outside the validation split")
        resolved = stable_index(self.dataset, position)
        if resolved != dataset_index:
            raise RuntimeError(
                f"validation position {position} resolves to dataset index {resolved}, "
                f"not the requested {dataset_index}"
            )
        image, label = self.dataset[position]
        self.images_read += 1
        return image, int(label)

    def corrupted(self, image, corruption, severity, dataset_index):
        return apply_corruption(
            image, corruption, severity,
            seed=sample_seed(VALIDATION_NAMESPACE, corruption, severity, dataset_index),
        )


def draw_panel(axis, image, title, correct):
    axis.imshow(image.permute(1, 2, 0).numpy())
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=7, color=CORRECT_COLOUR if correct else WRONG_COLOUR)
    for spine in axis.spines.values():
        spine.set_color(CORRECT_COLOUR if correct else WRONG_COLOUR)
        spine.set_linewidth(1.2)


def severity_row(axis_row, source, row):
    """Clean frame followed by severities 1-5, titled with the prediction."""
    image, label = source.clean(row["dataset_index"])
    if label != row["label"]:
        raise RuntimeError(
            f"dataset index {row['dataset_index']} has label {label}, "
            f"but examples.csv recorded {row['label']}"
        )
    draw_panel(axis_row[0], image, f"clean: {label}", True)
    for column, severity in enumerate(SEVERITY_LEVELS, start=1):
        prediction = int(row[f"prediction_s{severity}"])
        draw_panel(
            axis_row[column],
            source.corrupted(image, row["corruption"], severity, row["dataset_index"]),
            f"s{severity}: {prediction}",
            prediction == label,
        )


def plot_severity_grid(examples, source, path, title, note):
    columns = 1 + len(SEVERITY_LEVELS)
    figure, axes = plt.subplots(
        len(examples), columns, figsize=(1.45 * columns + 2.6, 1.6 * len(examples)),
        squeeze=False,
    )
    for position, (_, row) in enumerate(examples.iterrows()):
        severity_row(axes[position], source, row)
        axes[position][0].set_ylabel(
            f"{row['corruption']}\nclass {row['label']}  {row['track']}\n{row['rule_short']}",
            fontsize=6, rotation=0, ha="right", va="center",
        )
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=caption_text(figure, note, top=0.97))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_transition_grid(examples, source, path):
    """One row per condition, three example signs per row, clean beside corrupted."""
    conditions = examples[["corruption", "focus_severity", "rule"]].drop_duplicates()
    rows = len(conditions)
    per_row = examples.groupby(["corruption", "focus_severity"]).size().max()
    figure, axes = plt.subplots(rows, 2 * per_row, figsize=(1.5 * 2 * per_row + 2.4, 2.4 * rows),
                                squeeze=False)
    for position, condition in enumerate(conditions.itertuples(index=False)):
        selected = examples[
            (examples["corruption"] == condition.corruption)
            & (examples["focus_severity"] == condition.focus_severity)
        ].sort_values("row")
        severity = int(condition.focus_severity)
        for column, (_, row) in enumerate(selected.iterrows()):
            image, label = source.clean(row["dataset_index"])
            prediction = int(row[f"prediction_s{severity}"])
            draw_panel(axes[position][2 * column], image, f"clean: {label}", True)
            draw_panel(
                axes[position][2 * column + 1],
                source.corrupted(image, row["corruption"], severity, row["dataset_index"]),
                f"s{severity}: {prediction}",
                prediction == label,
            )
            axes[position][2 * column].set_xlabel(row["track"], fontsize=6)
        axes[position][0].set_ylabel(
            f"{condition.corruption} s{severity}\n{condition.rule.split(': ')[-1]}",
            fontsize=7, rotation=0, ha="right", va="center",
        )
    figure.suptitle(
        "Dominant source to target confusions, three different signs each", fontsize=12
    )
    figure.tight_layout(rect=caption_text(figure, (
        "Each pair is the clean frame and the same frame under the condition named on the left. "
        "The severity is fixed per corruption before any pair is looked at: fog and noise at "
        "their strongest level, occlusion at its mildest, where the question is what the "
        "smallest centre-constrained patch already does. Within that condition the pair shown is "
        "the one covering the most distinct validation tracks, on three different signs. "
        + OCCLUSION_CAPTION
    ), top=0.95))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def main():
    onset = read("failure_onset.csv")
    totals = read("failure_onset_totals.csv")
    track_onset = read("track_majority_onset.csv")
    destinations = read("error_destinations.csv")
    classes = read("class_fragility.csv")
    anchor = read("occlusion_anchor_offset.csv")
    within_track = read("occlusion_anchor_within_track.csv")
    examples = read("examples.csv")
    examples["rule_short"] = examples["rule"].str.split(": ").str[-1]

    plots = ANALYSIS_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    rho, pvalue = spearmanr(classes["validation_tracks"], classes["mean_recall_drop_s5"])

    written = [
        plot_failure_onset(onset, totals, track_onset, plots / "failure_onset.png"),
        plot_error_concentration(destinations, plots / "error_concentration.png"),
        plot_class_fragility(classes, rho, pvalue, plots / "class_fragility.png"),
        plot_occlusion_anchor(anchor, within_track, plots / "occlusion_anchor.png"),
    ]

    source = FrameSource()
    written.append(plot_severity_grid(
        examples[examples["figure"] == "onset"], source,
        plots / "examples_onset.png",
        "Clean-correct frames at the severity where each first fails and stays wrong",
        "Panel titles are the predicted class, green where it is correct. One frame per "
        "corruption and onset severity, each row a different physical sign. " + OCCLUSION_CAPTION,
    ))
    written.append(plot_transition_grid(
        examples[examples["figure"] == "transition"], source,
        plots / "examples_transitions.png",
    ))
    written.append(plot_severity_grid(
        examples[examples["figure"] == "unstable"], source,
        plots / "examples_unstable.png",
        "Frames whose predicted class keeps changing as severity grows",
        "Panel titles are the predicted class, green where it is correct. These are the frames "
        "taking the most distinct predictions across the five severities. " + OCCLUSION_CAPTION,
    ))

    provenance = plots / "sources.json"
    summary = ANALYSIS_DIR / "analysis_summary.json"
    with open(provenance, "w") as handle:
        json.dump({
            "analysis_tables_read": dict(sorted(SOURCES.items())),
            "analysis_run": {
                str(summary.relative_to(PROJECT_ROOT)): digest(summary),
            },
            "figures": sorted(path.name for path in written),
            "validation_frames_rendered": source.images_read,
            "model_inference_performed": False,
            "test_images_read": 0,
        }, handle, indent=2)

    print(f"Rendered {source.images_read} validation frames, all addressed by their original "
          "dataset index and corrupted with the frozen per-sample seeds.")
    print("No model inference; the official test split was not constructed.")
    print("\nWritten:")
    for path in (*written, provenance):
        print(f"  {path}")


if __name__ == "__main__":
    main()
