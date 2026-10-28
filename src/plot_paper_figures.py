"""Draw the paper figures from the official test analysis.

    .venv/bin/python -m src.plot_paper_figures

Reads the CSVs that `src.compare_test` wrote under
`artifacts/analysis/final_test/` and writes PDF and PNG versions of four
figures to `figures/`. It recomputes no error rate and no interval: every
plotted value is read from those files, so a figure cannot disagree with the
analysis it illustrates.

Arm levels per family and per severity are not stored directly. The contrast
tables hold `left_mean_error` and `right_mean_error` for B-A, C-A and C-B, so
each arm's level appears in two of the three contrasts; `arm_levels` reads all
of them and refuses to continue if the two copies disagree, which also catches
a mis-parsed contrast label. Figure 3 plots the severity levels this recovers,
after checking them against `arm_error_rates.csv`, whose means they have to
reproduce. Figure 2 plots the stored differences rather than a reconstruction;
the family levels are recovered only as a consistency check on the table it
reads, since with six and five families there is no suite mean to check them
against.

The suite names are the names in the files: `frozen` is the suite of six
families sampled during training, `heldout` the five families that were not.
Both are evaluated on the same 12,630 official test images.

No bootstrap interval is drawn. The intervals in the analysis resample images
within one checkpoint and say nothing about how much a result moves when the
training seed changes, which is the uncertainty these comparisons turn on, so
the per-seed values are plotted individually instead.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / "analysis" / "final_test"
FIGURE_DIR = PROJECT_ROOT / "figures"

INPUTS = {
    "arm_error_rates": ["suite", "seed", "arm", "model", "mean_image_error"],
    "per_family_contrasts": ["suite", "seed", "corruption", "contrast",
                             "left_mean_error", "right_mean_error",
                             "difference"],
    "per_severity_contrasts": ["suite", "seed", "severity", "contrast",
                               "left_mean_error", "right_mean_error",
                               "difference"],
    "clean_performance": ["arm", "model", "seed", "clean_accuracy"],
}

# The figures label their axes "test", so the analysis they read has to be the
# official test one. compare_models writes the same table names for validation.
SPLIT_FILE = "comparison.json"
SPLIT = "official_test"

SUITES = ["frozen", "heldout"]
SUITE_LABELS = {"frozen": "Original suite (six trained families)",
                "heldout": "Held-out suite (five unseen families)"}
ARMS = ["A", "B", "C"]
SEEDS = [42, 43, 44]
SEVERITIES = [1, 2, 3, 4, 5]

# The families each suite is made of. A condition the suite does not hold, the
# clean condition above all, must not reach a corruption-error figure.
FAMILIES = {
    "frozen": ["brightness_bright", "brightness_dark", "fog", "gaussian_blur",
               "gaussian_noise", "occlusion"],
    "heldout": ["contrast_reduction", "jpeg_compression", "motion_blur",
                "rotation", "salt_pepper"],
}

# The contrast labels in the files, read as (left arm, right arm). A label the
# analysis did not write is an error rather than something to skip.
CONTRAST_ARMS = {"B-A": ("B", "A"), "C-A": ("C", "A"), "C-B": ("C", "B")}

# Okabe-Ito colours, which stay distinguishable under the common forms of
# colour vision deficiency, with one marker per arm so the figures are also
# readable in greyscale.
ARM_STYLE = {
    "A": ("#0072B2", "o", "A clean"),
    "B": ("#D55E00", "s", "B balanced"),
    "C": ("#009E73", "^", "C targeted"),
}
SEED_MARKERS = {42: "o", 43: "s", 44: "D"}
GRID = {"color": "#CCCCCC", "linewidth": 0.5, "alpha": 0.8}

RC_PARAMS = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}

PAGE_WIDTH = 7.0  # inches, a two-column page's full text width

SCALE_NOTE = ("Horizontal scales are shared within each column but differ "
              "between contrasts.")

# The clean-accuracy axis is fixed rather than fitted to the nine values, so
# the narrow spread between the arms is not stretched across the panel.
CLEAN_AXIS = (0.95, 1.0)


def read_inputs(analysis_dir):
    """Load the four analysis tables, checking columns and record counts."""
    split_path = analysis_dir / SPLIT_FILE
    if not split_path.exists():
        raise FileNotFoundError(f"no {split_path}, so the split behind this "
                                "directory cannot be established")
    split = json.loads(split_path.read_text()).get("protocol", {}).get("split")
    if split != SPLIT:
        raise RuntimeError(
            f"{split_path} records split {split!r}, but these figures are "
            f"labelled for {SPLIT!r}")

    tables = {}
    for name, columns in INPUTS.items():
        path = analysis_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"no {path}; run src.compare_test first, since these figures "
                "are drawn from its output and nothing here recomputes it")
        frame = pd.read_csv(path)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise RuntimeError(f"{path} has no column {', '.join(missing)}")
        tables[name] = frame

    arms = tables["arm_error_rates"]
    expect_rows(arms, len(SUITES) * len(ARMS) * len(SEEDS), "arm_error_rates")
    expect_values(arms, "suite", SUITES)
    expect_values(arms, "arm", ARMS)
    expect_values(arms, "seed", SEEDS)

    clean = tables["clean_performance"]
    expect_rows(clean, len(ARMS) * len(SEEDS), "clean_performance")
    expect_values(clean, "arm", ARMS)
    expect_values(clean, "seed", SEEDS)

    for name, column, values in (("per_family_contrasts", "contrast",
                                  sorted(CONTRAST_ARMS)),
                                 ("per_severity_contrasts", "severity",
                                  SEVERITIES)):
        expect_values(tables[name], column, values)
        expect_values(tables[name], "suite", SUITES)
        expect_values(tables[name], "seed", SEEDS)

    severities = tables["per_severity_contrasts"]
    expect_rows(severities,
                len(SUITES) * len(SEEDS) * len(SEVERITIES)
                * len(CONTRAST_ARMS), "per_severity_contrasts")

    corruptions = tables["per_family_contrasts"]
    expect_rows(corruptions,
                len(SEEDS) * len(CONTRAST_ARMS)
                * sum(len(names) for names in FAMILIES.values()),
                "per_family_contrasts")
    for suite, names in FAMILIES.items():
        expect_values(corruptions[corruptions["suite"] == suite],
                      "corruption", names)
    return tables


def expect_rows(frame, count, name):
    if len(frame) != count:
        raise RuntimeError(f"{name} holds {len(frame)} rows, expected {count}")


def expect_values(frame, column, values):
    """Require a column to hold exactly the set of values the figures plot."""
    found = sorted(frame[column].unique())
    if found != sorted(values):
        raise RuntimeError(
            f"{column} holds {found}, expected {sorted(values)}")


def families(table, suite):
    """Family names for one suite, ordered by the B-A mean difference."""
    rows = table[(table["suite"] == suite) & (table["contrast"] == "B-A")]
    order = rows.groupby("corruption")["difference"].mean().sort_values()
    return list(order.index)


def arm_levels(table, key):
    """Reconstruct per-arm error from the contrast table's two error columns.

    Returns {(suite, seed, key value): {arm: error}}. Each arm appears in two
    contrasts, so the two readings are compared and a disagreement is raised
    rather than averaged away.
    """
    levels = {}
    for row in table.itertuples():
        left, right = CONTRAST_ARMS[row.contrast]
        cell = levels.setdefault((row.suite, row.seed, getattr(row, key)), {})
        for arm, error in ((left, row.left_mean_error),
                           (right, row.right_mean_error)):
            previous = cell.get(arm)
            if previous is not None and abs(previous - error) > 1e-12:
                raise RuntimeError(
                    f"{row.suite} seed {row.seed} {getattr(row, key)} gives "
                    f"arm {arm} both {previous} and {error}")
            cell[arm] = error
    for cell_key, cell in levels.items():
        if sorted(cell) != ARMS:
            raise RuntimeError(f"{cell_key} resolves arms {sorted(cell)}")
    return levels


def check_severity_levels(levels, arms):
    """The five severity levels must average to the suite mean per arm."""
    for row in arms.itertuples():
        values = [levels[(row.suite, row.seed, severity)][row.arm]
                  for severity in SEVERITIES]
        mean = sum(values) / len(values)
        if abs(mean - row.mean_image_error) > 1e-9:
            raise RuntimeError(
                f"{row.suite} seed {row.seed} arm {row.arm}: severity levels "
                f"average to {mean} but arm_error_rates records "
                f"{row.mean_image_error}")


def headroom(values, factor=1.12):
    """An upper limit a little above the largest plotted value."""
    return max(values) * factor


def contrast_limits(family_table, contrast, margin=0.05):
    """Shared x limits for one contrast's column of figure 2.

    Both suites get the same scale, so a family in one can be read against a
    family in the other. The range is set by the per-seed values, which bound
    the bars drawn from their means; the means are passed in as well only so a
    change to how a bar is derived cannot push it off the axis unnoticed. Zero
    is always inside, since the sign of a difference is what each panel is
    read for."""
    rows = family_table[family_table["contrast"] == contrast]
    means = rows.groupby(["suite", "corruption"])["difference"].mean()
    values = list(rows["difference"]) + list(means) + [0.0]
    low, high = min(values), max(values)
    pad = (high - low) * margin
    return low - pad, high + pad


def arm_mean(arms, suite, arm):
    rows = arms[(arms["suite"] == suite) & (arms["arm"] == arm)]
    return rows["mean_image_error"].mean()


def arm_seed_value(arms, suite, arm, seed):
    rows = arms[(arms["suite"] == suite) & (arms["arm"] == arm)
                & (arms["seed"] == seed)]
    return rows["mean_image_error"].iloc[0]


def seed_legend(axis, title=None):
    """One legend entry per training seed, drawn from the marker map."""
    handles = [plt.Line2D([], [], linestyle="none", marker=marker,
                          markerfacecolor="white", markeredgecolor="#333333",
                          markeredgewidth=0.7, markersize=4,
                          label=f"seed {seed}")
               for seed, marker in SEED_MARKERS.items()]
    axis.legend(handles=handles, title=title, frameon=False,
                handletextpad=0.3, borderpad=0.2, labelspacing=0.3)


def save(figure, stem, written):
    """Write the PDF and PNG pair. The PDF carries no creation timestamp, so
    rerunning the script on unchanged inputs reproduces the file byte for
    byte."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in ((".pdf", {"metadata": {"CreationDate": None}}),
                            (".png", {"dpi": 300})):
        path = FIGURE_DIR / f"{stem}{suffix}"
        figure.savefig(path, **options)
        written.append(path)
    plt.close(figure)


def figure_overall(arms, written):
    """Figure 1: mean corruption error per arm, both suites, seeds marked."""
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.7),
                                sharey=True)
    offsets = {42: -0.22, 43: 0.0, 44: 0.22}
    for axis, suite in zip(axes, SUITES):
        for position, arm in enumerate(ARMS):
            colour, _, _ = ARM_STYLE[arm]
            mean = arm_mean(arms, suite, arm)
            axis.bar(position, mean, width=0.62, color=colour, alpha=0.30,
                     edgecolor=colour, linewidth=0.8, zorder=2)
            axis.hlines(mean, position - 0.31, position + 0.31, color=colour,
                        linewidth=1.4, zorder=4)
            axis.annotate(f"{mean:.4f}", (position, mean), xytext=(0, 13),
                          textcoords="offset points", ha="center",
                          fontsize=7, color="#333333")
            for seed in SEEDS:
                axis.plot(position + offsets[seed],
                          arm_seed_value(arms, suite, arm, seed),
                          marker=SEED_MARKERS[seed], markersize=4,
                          markerfacecolor="white", markeredgecolor=colour,
                          markeredgewidth=0.9, linestyle="none", zorder=5)
        axis.set_xticks(range(len(ARMS)))
        axis.set_xticklabels([ARM_STYLE[arm][2] for arm in ARMS])
        axis.set_xlim(-0.6, len(ARMS) - 0.4)
        axis.set_ylim(0, headroom(arms["mean_image_error"], 1.20))
        axis.set_title(SUITE_LABELS[suite])
        axis.yaxis.grid(True, **GRID)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("mean per-image error\n(conditions equally weighted)")
    seed_legend(axes[1])
    figure.tight_layout(pad=0.6)
    save(figure, "fig1_overall_corruption_error", written)


def figure_per_family(family_table, written):
    """Figure 2: per-family paired contrasts, B-A and C-B, both suites."""
    figure, axes = plt.subplots(
        2, 2, figsize=(PAGE_WIDTH, 4.6), layout="constrained",
        gridspec_kw={"height_ratios": [len(families(family_table, "frozen")),
                                       len(families(family_table, "heldout"))]})
    offsets = {42: 0.2, 43: 0.0, 44: -0.2}
    limits = {contrast: contrast_limits(family_table, contrast)
              for contrast in ("B-A", "C-B")}
    for row, suite in enumerate(SUITES):
        order = families(family_table, suite)
        positions = range(len(order))
        for column, contrast in ((0, "B-A"), (1, "C-B")):
            axis = axes[row][column]
            colour = ARM_STYLE[CONTRAST_ARMS[contrast][0]][0]
            rows = family_table[(family_table["suite"] == suite)
                                & (family_table["contrast"] == contrast)]
            means = rows.groupby("corruption")["difference"].mean()
            axis.barh(list(positions), [means[name] for name in order],
                      height=0.6, color=colour, alpha=0.30,
                      edgecolor=colour, linewidth=0.8, zorder=2)
            for position, name in zip(positions, order):
                per_seed = rows[rows["corruption"] == name]
                for seed, value in zip(per_seed["seed"],
                                       per_seed["difference"]):
                    axis.plot(value, position + offsets[seed],
                              marker=SEED_MARKERS[seed], markersize=3.4,
                              markerfacecolor="white", markeredgecolor=colour,
                              markeredgewidth=0.8, linestyle="none", zorder=5)
            axis.axvline(0.0, color="#333333", linewidth=0.7, zorder=3)
            axis.set_xlim(*limits[contrast])
            axis.set_yticks(list(positions))
            axis.set_yticklabels(order)
            axis.set_ylim(-0.7, len(order) - 0.3)
            axis.invert_yaxis()
            axis.set_title(f"{contrast.replace('-', ' − ')}"
                           f"   {SUITE_LABELS[suite].split(' (')[0].lower()}")
            axis.xaxis.grid(True, **GRID)
            axis.set_axisbelow(True)
            if row == len(SUITES) - 1:
                axis.set_xlabel("difference in mean per-image error\n"
                                "(negative favours the left arm)")
    seed_legend(axes[0][1])
    # The C-B differences are an order of magnitude smaller than B-A, so the
    # two columns keep their own scales; within a column both suites share one,
    # which is the comparison the stacked panels invite. supxlabel rather than
    # figure.text, so the constrained layout reserves room for it instead of
    # letting it land on the x labels.
    figure.supxlabel(SCALE_NOTE, fontsize=7, style="italic", color="#333333")
    save(figure, "fig2_per_family_contrasts", written)


def figure_severity(severity_levels, severity_table, written):
    """Figure 3: error against severity, and the C-B difference per severity."""
    figure, axes = plt.subplots(2, 2, figsize=(PAGE_WIDTH, 4.8),
                                layout="constrained")
    offsets = {42: -0.11, 43: 0.0, 44: 0.11}
    # Both suites share one scale in each row, so the panels can be read
    # against each other.
    level_limit = headroom([error for cell in severity_levels.values()
                            for error in cell.values()], 1.08)
    differences = severity_table[severity_table["contrast"] == "C-B"]
    difference_limit = headroom(differences["difference"].abs(), 1.15)
    for column, suite in enumerate(SUITES):
        axis = axes[0][column]
        for arm in ARMS:
            colour, marker, label = ARM_STYLE[arm]
            means = [sum(severity_levels[(suite, seed, severity)][arm]
                         for seed in SEEDS) / len(SEEDS)
                     for severity in SEVERITIES]
            axis.plot(SEVERITIES, means, color=colour, marker=marker,
                      markersize=4.5, linewidth=1.1, label=label, zorder=4)
            for seed in SEEDS:
                axis.plot([severity + offsets[seed]
                           for severity in SEVERITIES],
                          [severity_levels[(suite, seed, severity)][arm]
                           for severity in SEVERITIES],
                          marker=SEED_MARKERS[seed], markersize=2.6,
                          markerfacecolor="white", markeredgecolor=colour,
                          markeredgewidth=0.6, linestyle="none", zorder=3)
        axis.set_xticks(SEVERITIES)
        axis.set_ylim(0, level_limit)
        axis.set_title(SUITE_LABELS[suite])
        axis.yaxis.grid(True, **GRID)
        axis.set_axisbelow(True)
        if column == 0:
            axis.set_ylabel("mean per-image error")
            axis.legend(frameon=False, handletextpad=0.4, borderpad=0.2,
                        labelspacing=0.3)
        else:
            seed_legend(axis, "open markers, by seed")

        axis = axes[1][column]
        colour = ARM_STYLE["C"][0]
        rows = severity_table[(severity_table["suite"] == suite)
                              & (severity_table["contrast"] == "C-B")]
        means = rows.groupby("severity")["difference"].mean()
        axis.plot(SEVERITIES, [means[severity] for severity in SEVERITIES],
                  color=colour, marker=ARM_STYLE["C"][1], markersize=4.5,
                  linewidth=1.1, zorder=4)
        for seed in SEEDS:
            per_seed = rows[rows["seed"] == seed].set_index("severity")
            axis.plot([severity + offsets[seed] for severity in SEVERITIES],
                      [per_seed.loc[severity, "difference"]
                       for severity in SEVERITIES],
                      marker=SEED_MARKERS[seed], markersize=3.2,
                      markerfacecolor="white", markeredgecolor=colour,
                      markeredgewidth=0.8, linestyle="none", zorder=5)
        axis.axhline(0.0, color="#333333", linewidth=0.7, zorder=3)
        axis.set_xticks(SEVERITIES)
        axis.set_ylim(-difference_limit, difference_limit)
        axis.set_xlabel("severity level")
        axis.set_title(f"C − B, {SUITE_LABELS[suite].split(' (')[0].lower()}")
        axis.yaxis.grid(True, **GRID)
        axis.set_axisbelow(True)
        if column == 0:
            axis.set_ylabel("difference in mean\nper-image error")
    save(figure, "fig3_severity", written)


def figure_clean_tradeoff(arms, clean, written):
    """Figure 4: clean accuracy against corruption error, nine checkpoints."""
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0), sharey=True,
                                layout="constrained")
    accuracy = {(row.arm, row.seed): row.clean_accuracy
                for row in clean.itertuples()}
    outside = [key for key, value in accuracy.items()
               if not CLEAN_AXIS[0] <= value <= CLEAN_AXIS[1]]
    if outside:
        raise RuntimeError(
            f"clean accuracy for {outside} falls outside the fixed axis "
            f"{CLEAN_AXIS}, which would hide a checkpoint")
    for axis, suite in zip(axes, SUITES):
        for arm in ARMS:
            colour, _, label = ARM_STYLE[arm]
            for seed in SEEDS:
                axis.plot(accuracy[(arm, seed)],
                          arm_seed_value(arms, suite, arm, seed),
                          marker=SEED_MARKERS[seed], markersize=5,
                          markerfacecolor=colour, markeredgecolor="#333333",
                          markeredgewidth=0.5, linestyle="none", zorder=4)
        axis.set_xlim(*CLEAN_AXIS)
        axis.set_ylim(0, headroom(arms["mean_image_error"]))
        axis.set_xlabel("clean test accuracy")
        axis.set_title(SUITE_LABELS[suite])
        axis.grid(True, **GRID)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("mean per-image error\n(conditions equally weighted)")
    handles = [plt.Line2D([], [], linestyle="none", marker="o", markersize=5,
                          markerfacecolor=ARM_STYLE[arm][0],
                          markeredgecolor="#333333", markeredgewidth=0.5,
                          label=ARM_STYLE[arm][2]) for arm in ARMS]
    handles += [plt.Line2D([], [], linestyle="none", marker=marker,
                           markersize=4.5, markerfacecolor="#BBBBBB",
                           markeredgecolor="#333333", markeredgewidth=0.6,
                           label=f"seed {seed}")
                for seed, marker in SEED_MARKERS.items()]
    # Outside the panels: with the clean axis fixed at its reported range the
    # nine points sit in one corner, and a legend inside either panel covers
    # the area a reader checks for points the axis might be hiding.
    figure.legend(handles=handles, frameon=False, ncols=len(handles),
                  loc="outside lower center", handletextpad=0.3,
                  borderpad=0.2, columnspacing=1.4)
    save(figure, "fig4_clean_versus_corruption", written)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default=None,
                        help="directory holding the final-test analysis CSVs")
    parser.add_argument("--figure-dir", default=None,
                        help="where to write the figures; defaults to "
                             "figures/ in the project root")
    arguments = parser.parse_args()

    global FIGURE_DIR
    analysis_dir = (Path(arguments.analysis_dir) if arguments.analysis_dir
                    else ANALYSIS_DIR)
    if arguments.figure_dir:
        FIGURE_DIR = Path(arguments.figure_dir)
    tables = read_inputs(analysis_dir)
    family_levels = arm_levels(tables["per_family_contrasts"], "corruption")
    severity_levels = arm_levels(tables["per_severity_contrasts"], "severity")
    check_severity_levels(severity_levels, tables["arm_error_rates"])

    written = []
    with plt.rc_context(RC_PARAMS):
        figure_overall(tables["arm_error_rates"], written)
        figure_per_family(tables["per_family_contrasts"], written)
        figure_severity(severity_levels, tables["per_severity_contrasts"],
                        written)
        figure_clean_tradeoff(tables["arm_error_rates"],
                              tables["clean_performance"], written)

    print(f"Read {analysis_dir}")
    print(f"Reconstructed {len(family_levels)} family and "
          f"{len(severity_levels)} severity cells, both checked arm by arm")
    print("Figures written:")
    for path in written:
        name = (path.relative_to(PROJECT_ROOT)
                if path.is_relative_to(PROJECT_ROOT) else path)
        print(f"  {name} ({path.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
