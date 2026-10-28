"""Paired, image-level comparison of the three arms on the official test split.

    .venv/bin/python -m src.compare_test \
        --model clean:42:artifacts/final_test/clean_seed42 \
        --model balanced:42:artifacts/final_test/balanced_seed42 \
        --model targeted:42:artifacts/final_test/targeted_seed42 \
        --model clean:43:artifacts/final_test/clean_seed43 ...

Each `--model` is `NAME:SEED:DIR` in the same format `compare_models.py` takes,
and `DIR` is the directory `evaluate_test.py` wrote for that checkpoint.

`compare_models.py` produced the committed validation comparison and is left
byte-identical, so this is a separate module. Everything that could change a
reported conclusion is imported from it rather than restated: the arm names and
labels, the three contrasts and their left-minus-right orientation, the
clean-accuracy limit, the bootstrap constants, the across-seed summary, and the
decision rule for C against B. The contrast list and the verdict function here
are the same objects that produced the validation verdict, not copies of them.

What changes, and the only thing that changes
---------------------------------------------
The resampling unit. The validation analysis resampled 178 whole physical
signs, each entering a draw with all 30 of its frames or none, because the
split carries track identifiers. The official test release does not: it is
documented as being in random order with no track structure, so there is no
grouping to resample and the estimator cannot be transferred. The replacement
is a paired bootstrap over the 12,630 test images.

That estimator assumes independent image-level sampling, and the assumption is
likely violated - the release almost certainly contains repeated frames of the
same physical signs, unlabelled. Three consequences, carried on the face of
every table this module writes:

* the interval does not account for physical-sign dependence, and is not
  corrected or adjusted for it;
* it is not a definitive population-level confidence interval - its nominal
  coverage holds under an assumption that is probably false;
* its width is narrower than a dependence-aware width would be.

`design_effect_sensitivity.csv` shows how far the width could move under
k = 30 and rho = 1. That figure is an illustration under a stated assumption,
not a bound and not a second interval, and it appears in that file and nowhere
else. The decision rule references no interval at all: it is the sign of the
per-seed point estimate at each of three seeds on each of two suites, exactly
as on validation.

Training-seed variability and image-level sampling uncertainty are reported
separately and never combined. Three replicates support a direction, a mean and
a range; no interval, p-value or significance claim is produced across seeds.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# The constants and the functions the validation comparison decided with, by
# reference rather than by copy. Four of them - `read_recorded_identity`,
# `_tradeoff_verdict`, `_direction` and `REQUIRED_IDENTITY` - are reached
# through the public functions above them rather than called here; they are
# named so that the dependency is visible and so the checks can assert these
# are the same objects `compare_models.py` used, not re-implementations.
from .analyze_placement import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, INTERVAL
from .compare_models import (
    ARM_LABELS,
    ARMS,
    CLEAN_TRADEOFF_LIMIT,
    CONTRASTS,
    REQUIRED_IDENTITY,
    _direction,
    _recorded,
    _tradeoff_verdict,
    across_seeds,
    check_identical_examples,
    clean_rows,
    clean_tradeoff_rows,
    parse_model,
    read_recorded_identity,
    targeted_verdict,
    verify_model_identities,
    write_table,
)
from .compare_models import ANALYSIS_DIR as VALIDATION_ANALYSIS_DIR
from .corruptions_heldout import HELDOUT_CORRUPTIONS
from .data import NUM_CLASSES
from .evaluate_robustness import CLEAN_CONDITION, PROJECT_ROOT
from .evaluate_test import (
    EXPECTED_TEST_SIZE,
    INVENTORY_PATH,
    TEST_NAMESPACE,
    TEST_SUITES,
    parse_test_ground_truth,
)

ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / "analysis" / "final_test"
# The trees this module reads and must never write into. It cannot reuse the
# evaluator's tuple unchanged: that one protects the whole of
# `artifacts/analysis`, which is where this module's own output belongs, so
# the analysis entry is narrowed to the committed validation comparison.
PROTECTED_TREES = (PROJECT_ROOT / "artifacts" / "baseline",
                   PROJECT_ROOT / "artifacts" / "robustness",
                   VALIDATION_ANALYSIS_DIR)

# The draw matrix for this split is 2,000 x 12,630. Held as int32 - the index
# range needs 14 bits - and consumed in chunks, so the resampled means are
# accumulated over about 101 MB of indices plus a 20 MB chunk buffer rather
# than materialising 2,000 means of 12,630 values at once.
CHUNK_DRAWS = 200
# `targeted_verdict` is imported so the rule is the same object that produced
# the committed validation conclusion, which means its message says "per-sign".
# The official test release carries no track identifiers, so the unit it is
# applied to here is the image. The rule itself is unchanged; only the data
# under it is.
VERDICT_UNIT_NOTE = (
    "The rule above is the imported validation rule, so its wording says "
    "per-sign. On the official test split there are no physical-sign "
    "identifiers and the unit is the image: the quantity compared at each "
    "seed is the mean per-image error over the equally weighted corrupted "
    "conditions.")

# Two sentences of the frozen protocol describe the clean trade-off's
# reference arm differently. Part 7's authoritative paragraph says
# `compare_models.clean_tradeoff_rows` measures each contrast's clean cost
# against "that contrast's own right-hand arm" at the same seed, "not against
# arm A", and spells out that the C-B row is B's clean accuracy minus C's.
# Part 7's own reporting list then says, of the clean trade-off table, "the
# cost against arm A at the same seed". The two agree on B-A and C-A, whose
# right-hand arm is A, and disagree only on C-B.
#
# The implementation follows the right-hand arm. That is what the named
# function does, the function is frozen and produced the committed validation
# table, the paragraph that names it is specific about the C-B row, and Part 1
# question 4 asks the question the same way. Reading it the other way would
# make the test path's trade-off column incomparable with the validation one,
# which is the reason the paragraph gives for the convention. The protocol is
# frozen and is not edited to resolve this; the note travels with the output
# instead.
CLEAN_COST_REFERENCE_NOTE = (
    "Clean cost is measured against the contrast's own right-hand arm at the "
    "same seed, so the C-B cost is B's clean accuracy minus C's. This follows "
    "the frozen protocol's Part 7 paragraph on clean_tradeoff_rows and the "
    "imported function itself. The bullet in the same part's reporting list "
    "that reads \"the cost against arm A at the same seed\" conflicts with "
    "both; it differs only on the C-B row and was not followed. The limit is "
    "a reporting rule either way and the decision rule is unaffected.")

# The sensitivity illustration of Part 6: a cluster size and a within-cluster
# correlation the released test data cannot confirm, stated on the face of the
# only file the figure appears in. rho = 1 is the most extreme value and is
# implausible on its face; the point of the number is to show how far the
# interval could move, not to bound anything.
DESIGN_EFFECT_CLUSTER = 30
DESIGN_EFFECT_RHO = 1.0

INTERVAL_ASSUMPTION = (
    "percentile interval over 2000 paired resamples of the 12630 test images; "
    "assumes independent image-level sampling, which is likely violated by "
    "repeated images of the same physical signs, so it is not a definitive "
    "population-level confidence interval and does not account for "
    "physical-sign dependence")

OUTPUT_FILES = (
    "arm_error_rates.csv", "paired_contrasts.csv", "across_seeds.csv",
    "clean_performance.csv", "clean_tradeoff.csv", "per_family_contrasts.csv",
    "per_severity_contrasts.csv", "per_class_recall.csv",
    "class_balanced_error.csv", "class_balanced_contrasts.csv",
    "design_effect_sensitivity.csv", "validation_reproduction.csv",
    "comparison.txt", "comparison.json",
)


def test_label_index(data_root=None):
    """Every test image's official class, parsed straight from the ground truth.

    The counterpart of `compare_models.validation_track_index`, minus the track
    column, because there is none. Reads one CSV and decodes no image.
    """
    parsed = parse_test_ground_truth(data_root)
    rows = [{"dataset_index": position, "label": label}
            for position, (_, label) in enumerate(parsed)]
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_TEST_SIZE:
        raise RuntimeError(f"the ground truth holds {len(frame)} images, "
                           f"expected {EXPECTED_TEST_SIZE}")
    return frame


def suite_conditions(suite_name):
    """The condition set one suite must cover, from the suite object itself."""
    return {(corruption, int(severity))
            for corruption, severity in TEST_SUITES[suite_name].conditions()}


def load_suite_images(model, suite_name, label_index=None):
    """Per-image errors for one model under one suite.

    The image-level counterpart of `compare_models.load_suite_frames`. Requires
    the full condition set and exactly 12,630 rows in every condition, and
    cross-checks the labels the evaluator recorded against an independent parse
    of `GT-final_test.csv`, so a prediction file written against anything other
    than the official test split cannot be folded into the comparison.

    `label_index` defaults to that parse. It is a parameter so the comparison
    logic can be exercised on synthetic fixtures without the official labels.
    """
    predictions_path = model["dir"] / suite_name / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"no predictions at {predictions_path}")
    if label_index is None:
        label_index = test_label_index()
    images = pd.read_csv(predictions_path)
    expected_columns = ["corruption", "severity", "test_position",
                        "dataset_index", "label", "prediction"]
    if list(images.columns) != expected_columns:
        raise RuntimeError(f"{predictions_path} has columns "
                           f"{list(images.columns)}, expected "
                           f"{expected_columns}")
    images = images.merge(
        label_index.rename(columns={"label": "official_label"}),
        on="dataset_index", how="left", validate="many_to_one")
    if images["official_label"].isna().any():
        missing = int(images["official_label"].isna().sum())
        raise RuntimeError(f"{predictions_path} holds {missing} rows whose "
                           "dataset index is not in the official test split")
    disagreements = int((images["label"] != images["official_label"]).sum())
    if disagreements:
        raise RuntimeError(f"{predictions_path} disagrees with the official "
                           f"ground truth on {disagreements} rows")
    if (images["test_position"] != images["dataset_index"]).any():
        raise RuntimeError(f"{predictions_path} has rows whose test position "
                           "and dataset index differ; on the official test "
                           "split they are the same number")
    present = set(map(tuple, images[["corruption", "severity"]]
                      .drop_duplicates().to_numpy()))
    present = {(str(name), int(severity)) for name, severity in present}
    required = suite_conditions(suite_name)
    if present != required:
        raise RuntimeError(
            f"{predictions_path} covers {len(present)} conditions, expected "
            f"the {len(required)} of the {suite_name} suite; missing "
            f"{sorted(required - present)[:4]}, unexpected "
            f"{sorted(present - required)[:4]}")
    counts = images.groupby(["corruption", "severity"]).size()
    wrong = counts[counts != EXPECTED_TEST_SIZE]
    if len(wrong):
        raise RuntimeError(f"{predictions_path} holds {wrong.iloc[0]} rows for "
                           f"{wrong.index[0]}, expected {EXPECTED_TEST_SIZE}")
    images["rate"] = (images["prediction"] != images["label"]).astype(float)
    images["model"] = model["name"]
    images["seed"] = model["seed"]
    images["suite"] = suite_name
    return images.drop(columns=["official_label"])


def image_rates(images, keys):
    """Per-condition per-image error, then the equal-weight mean over conditions.

    Every condition contributes exactly one row per image, so the mean over
    conditions is an unweighted mean of equally sized groups: a family spanning
    five severities contributes five of them, and the families therefore carry
    equal weight only because each spans five. Both invariants are enforced
    rather than assumed - 12,630 images in every condition, and 12,630 rates
    per group afterwards.
    """
    counts = images.groupby(list(keys) + ["corruption", "severity"]).size()
    wrong = counts[counts != EXPECTED_TEST_SIZE]
    if len(wrong):
        raise RuntimeError(f"{wrong.index[0]} covers {wrong.iloc[0]} images, "
                           f"expected {EXPECTED_TEST_SIZE}")
    rates = images.groupby(list(keys) + ["dataset_index"],
                           as_index=False)["rate"].mean()
    sizes = rates.groupby(list(keys)).size()
    if set(sizes.unique()) != {EXPECTED_TEST_SIZE}:
        raise RuntimeError(f"groups cover {sorted(set(sizes.unique()))} "
                           f"images, expected {EXPECTED_TEST_SIZE}")
    return rates


def paired_image_contrast(images, keys, left, right):
    """The vector of paired per-image differences for one contrast, left - right.

    `images` is the per-image rate frame `image_rates` returns. The difference
    is taken within an image before anything is averaged, so the two sides of a
    contrast always rest on the same images; a positive entry means the
    left-hand arm got that image wrong more often across the conditions.
    """
    selected = images.set_index(list(keys) + ["dataset_index"])["rate"]
    first = selected.xs(left, level=keys[0])
    second = selected.xs(right, level=keys[0])
    left_images = np.sort(first.index.to_numpy())
    right_images = np.sort(second.index.to_numpy())
    if not np.array_equal(left_images, right_images):
        only_left = np.setdiff1d(left_images, right_images)
        only_right = np.setdiff1d(right_images, left_images)
        raise RuntimeError(
            f"{left} and {right} do not cover the same images: "
            f"{len(only_left)} only in {left} (first "
            f"{only_left[:3].tolist()}), {len(only_right)} only in {right} "
            f"(first {only_right[:3].tolist()}). A difference over the "
            "intersection would not be the paired contrast.")
    joined = pd.concat({"left": first, "right": second}, axis=1).dropna()
    if len(joined) != EXPECTED_TEST_SIZE:
        raise RuntimeError(f"{left} against {right} pairs on {len(joined)} "
                           f"images, expected {EXPECTED_TEST_SIZE}")
    joined = joined.sort_index()
    return (joined["left"] - joined["right"]).to_numpy()


def image_draws(count, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """The resampling index matrix for this split, as int32.

    Identical values to `analyze_placement.bootstrap_draws(count, draws, seed)`
    - the same generator, the same seed, the same row-major fill - built a chunk
    of rows at a time and stored as int32 so the 2,000 x 12,630 matrix costs
    about 101 MB rather than 202, and so no int64 copy of it exists even
    briefly. Built once per comparison, so every contrast on both suites is
    resampled on the same draws and two intervals differ because the contrasts
    differ, not because the draws do.
    """
    matrix = np.empty((draws, count), dtype=np.int32)
    generator = np.random.default_rng(seed)
    for start in range(0, draws, CHUNK_DRAWS):
        stop = min(start + CHUNK_DRAWS, draws)
        matrix[start:stop] = generator.integers(0, count,
                                                size=(stop - start, count))
    return matrix


def bootstrap_image_mean(differences, draws=BOOTSTRAP_DRAWS,
                         seed=BOOTSTRAP_SEED, matrix=None):
    """Mean per-image value with a percentile interval over resampled images.

    The interval is the 2.5th and 97.5th percentiles of 2,000 resampled means,
    under the independence assumption this module's docstring states. `matrix`
    lets a suite's draws be built once and reused across its contrasts; passing
    nothing builds them, which is what makes a single interval reproducible on
    its own.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("expected a non-empty vector of per-image values")
    if matrix is None:
        matrix = image_draws(values.size, draws, seed)
    if matrix.shape != (draws, values.size):
        raise ValueError(f"draw matrix is {matrix.shape}, expected "
                         f"{(draws, values.size)}")
    resampled = np.empty(draws, dtype=float)
    for start in range(0, draws, CHUNK_DRAWS):
        stop = min(start + CHUNK_DRAWS, draws)
        resampled[start:stop] = values[matrix[start:stop]].mean(axis=1)
    low, high = np.percentile(resampled, INTERVAL)
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "images": int(values.size),
        "draws": int(draws),
    }


def design_effect_illustration(interval):
    """The sqrt(30) sensitivity width of Part 6, for one primary interval.

    Each endpoint's distance from the point estimate is scaled by
    sqrt(1 + (k - 1) * rho) with k = 30 and rho = 1, separately above and
    below so the asymmetry of the percentile interval survives. It is an
    illustration under two assumptions the released test data cannot confirm:
    no track identifiers exist, so neither the cluster size nor the correlation
    is observable. It is not a confidence bound, not a corrected interval, not
    a second interval, and it enters no decision.
    """
    factor = float(np.sqrt(1 + (DESIGN_EFFECT_CLUSTER - 1) * DESIGN_EFFECT_RHO))
    point = float(interval["mean"])
    return {
        "point_estimate": point,
        "ci_low": float(interval["ci_low"]),
        "ci_high": float(interval["ci_high"]),
        "assumed_cluster_size": DESIGN_EFFECT_CLUSTER,
        "assumed_within_cluster_correlation": DESIGN_EFFECT_RHO,
        "scale_factor": factor,
        "illustrative_low": point - (point - float(interval["ci_low"])) * factor,
        "illustrative_high": point + (float(interval["ci_high"]) - point) * factor,
        "status": ("assumption-dependent sensitivity illustration, not a "
                   "confidence bound, not a corrected interval, and not part "
                   "of the decision rule"),
    }


def arm_rows(rates, seed, suite_name, matrix=None):
    """The mean per-image error rate of each arm, with its own interval."""
    rows = []
    for model, group in rates.groupby("model"):
        record = bootstrap_image_mean(group.sort_values("dataset_index")
                                      ["rate"].to_numpy(), matrix=matrix)
        rows.append({
            "suite": suite_name,
            "seed": seed,
            "arm": ARM_LABELS[model],
            "model": model,
            "mean_image_error": record["mean"],
            "ci_low": record["ci_low"],
            "ci_high": record["ci_high"],
            "images": record["images"],
            "draws": record["draws"],
            "interval_assumption": INTERVAL_ASSUMPTION,
        })
    return rows


def contrast_rows(rates, seed, suite_name, available, matrix=None):
    """Every declared contrast between the arms present at one seed."""
    means = rates.groupby("model")["rate"].mean()
    rows = []
    for left, right in CONTRASTS:
        if left not in available or right not in available:
            continue
        # The paired mean and the difference of the arm means are the same
        # sum rearranged once both arms cover all 12,630 images, so comparing
        # them would prove nothing. What establishes the pairing is
        # `check_identical_examples` on the input fingerprints, the identical
        # image index required in `paired_image_contrast`, and the 12,630-row
        # requirement per condition in `load_suite_images`.
        differences = paired_image_contrast(rates, ["model"], left, right)
        record = bootstrap_image_mean(differences, matrix=matrix)
        rows.append({
            "suite": suite_name,
            "seed": seed,
            "contrast": f"{ARM_LABELS[left]}-{ARM_LABELS[right]}",
            "left": left,
            "right": right,
            "left_mean_error": float(means.loc[left]),
            "right_mean_error": float(means.loc[right]),
            "difference": record["mean"],
            "ci_low": record["ci_low"],
            "ci_high": record["ci_high"],
            "excludes_zero": bool(record["ci_low"] > 0 or record["ci_high"] < 0),
            "images_worse_under_left": int((differences > 0).sum()),
            "images_unchanged": int((differences == 0).sum()),
            "images": record["images"],
            "draws": record["draws"],
            "interval_assumption": INTERVAL_ASSUMPTION,
        })
    return rows


def breakdown_rows(images, seed, suite_name, grouping, matrix=None):
    """Per-family or per-severity contrasts, aggregated the same way.

    Same protocol as the primary metric over a narrower set of conditions, so
    these decompose the headline number rather than forming a separate
    analysis. There are 5, 6 or 11 of them per suite per seed, they carry no
    multiplicity adjustment, and they inherit the independence assumption of
    every interval here. Structure in the result, not a list of findings.
    """
    corrupted = images[images["corruption"] != CLEAN_CONDITION]
    rows = []
    for value, group in corrupted.groupby(grouping):
        rates = image_rates(group, ["model"])
        available = set(rates["model"])
        means = rates.groupby("model")["rate"].mean()
        for left, right in CONTRASTS:
            if left not in available or right not in available:
                continue
            differences = paired_image_contrast(rates, ["model"], left, right)
            record = bootstrap_image_mean(differences, matrix=matrix)
            rows.append({
                "suite": suite_name,
                "seed": seed,
                grouping: value,
                "contrast": f"{ARM_LABELS[left]}-{ARM_LABELS[right]}",
                "left_mean_error": float(means.loc[left]),
                "right_mean_error": float(means.loc[right]),
                "difference": record["mean"],
                "ci_low": record["ci_low"],
                "ci_high": record["ci_high"],
                "excludes_zero": bool(record["ci_low"] > 0
                                      or record["ci_high"] < 0),
                "images_worse_under_left": int((differences > 0).sum()),
                "interval_assumption": INTERVAL_ASSUMPTION,
            })
    return rows


def class_balanced_rows(images, seed, suite_name):
    """Descriptive class-balanced per-image error, per arm, with contrasts.

    The primary outcome weights every test image equally, so the classes with
    more test images contribute more to it. This weights the classes present
    equally instead, in the declared order:

    1. an image is right or wrong;
    2. image errors average within a class, under one condition;
    3. the class means average with equal weight;
    4. conditions average with equal weight, the clean condition excluded.

    Point estimates only. No interval is produced, no per-class inference is
    drawn, and nothing here enters a decision rule - the class weights are a
    second way of looking at the same predictions, not a second experiment, and
    on this split there is no resampling unit that would make an interval on
    them mean anything the primary interval does not already overstate.
    """
    corrupted = images[images["corruption"] != CLEAN_CONDITION]
    present = sorted(int(value) for value in corrupted["label"].unique())

    # The declared chain, as successive groupby means.
    chain = (corrupted
             .groupby(["model", "corruption", "severity", "label"],
                      as_index=False)["rate"].mean()
             .groupby(["model", "corruption", "severity"],
                      as_index=False)["rate"].mean()
             .groupby("model")["rate"].mean())

    # The same quantity from a class-by-condition matrix, to check the chain
    # rather than trust it, and to difference the arms within a class and a
    # condition before anything is averaged so the contrasts are paired.
    matrices, condition_order = {}, None
    for model_name, group in corrupted.groupby("model"):
        pivot = (group.groupby(["corruption", "severity", "label"])["rate"]
                 .mean().unstack("label"))
        if pivot.isna().to_numpy().any():
            raise RuntimeError(f"{model_name} is missing a class under some "
                               f"condition of the {suite_name} suite")
        if condition_order is None:
            condition_order = list(pivot.index)
        elif list(pivot.index) != condition_order:
            raise RuntimeError(f"{model_name} covers different conditions than "
                               f"the other arms under {suite_name}")
        if [int(value) for value in pivot.columns] != present:
            raise RuntimeError(f"{model_name} covers different classes than "
                               f"the other arms under {suite_name}")
        matrices[model_name] = pivot.to_numpy(dtype=float)

    arm_table, contrast_table, points = [], [], {}
    for model_name, matrix in sorted(matrices.items()):
        point = float(matrix.mean(axis=1).mean())
        independent = float(chain.loc[model_name])
        if abs(point - independent) > 1e-12:
            raise RuntimeError(
                f"the class-balanced error for {model_name} is {point!r} by "
                f"matrix average and {independent!r} by grouped means")
        points[model_name] = point
        arm_table.append({
            "suite": suite_name,
            "seed": seed,
            "arm": ARM_LABELS[model_name],
            "model": model_name,
            "class_balanced_error": point,
            "classes": len(present),
            "classes_defined": NUM_CLASSES,
            "images": EXPECTED_TEST_SIZE,
            "status": "descriptive, no interval, enters no decision rule",
        })

    for left, right in CONTRASTS:
        if left not in matrices or right not in matrices:
            continue
        # Differenced class by class and condition by condition before any
        # averaging. As above, the equal-weight chain makes this identical to
        # the difference of the two arm estimates, so it is not re-asserted
        # here as if it were a check.
        delta = matrices[left] - matrices[right]
        point = float(delta.mean(axis=1).mean())
        contrast_table.append({
            "suite": suite_name,
            "seed": seed,
            "contrast": f"{ARM_LABELS[left]}-{ARM_LABELS[right]}",
            "left": left,
            "right": right,
            "mean_difference": point,
            "classes": len(present),
            "status": "descriptive, no interval, enters no decision rule",
        })
    return arm_table, contrast_table


def per_class_rows(models):
    """Mean per-class recall under corruption, per arm and suite.

    Recall is averaged over the corrupted conditions of a suite. The class's
    test support travels with every row, because the official test split is not
    class-balanced and a recall change in a small class rests on few images.
    Descriptive, like the class-balanced table: no interval, no decision.
    """
    rows = []
    for model in models:
        for suite_name in sorted(TEST_SUITES):
            path = model["dir"] / suite_name / "per_class_metrics.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            corrupted = frame[frame["corruption"] != CLEAN_CONDITION]
            clean = frame[frame["corruption"] == CLEAN_CONDITION]
            grouped = corrupted.groupby("class", as_index=False).agg(
                corrupted_recall=("recall", "mean"),
                support=("support", "first"))
            clean_recall = clean.set_index("class")["recall"]
            for record in grouped.to_dict("records"):
                rows.append({
                    "suite": suite_name,
                    "arm": ARM_LABELS[model["name"]],
                    "model": model["name"],
                    "seed": model["seed"],
                    "class": int(record["class"]),
                    "test_support": int(record["support"]),
                    "clean_recall": float(clean_recall.loc[record["class"]]),
                    "mean_corrupted_recall": float(record["corrupted_recall"]),
                })
    return rows


def sensitivity_rows(contrast_table):
    """The sqrt(30) illustration for every primary contrast interval.

    The only place this calculation appears. Each row carries the assumptions
    it rests on and the statement that it is neither a bound nor a decision
    input, so the figure cannot be read out of this file without them.
    """
    rows = []
    for row in contrast_table:
        illustration = design_effect_illustration({
            "mean": row["difference"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
        })
        rows.append({
            "suite": row["suite"],
            "seed": row["seed"],
            "contrast": row["contrast"],
            "point_estimate": illustration["point_estimate"],
            "image_ci_low": illustration["ci_low"],
            "image_ci_high": illustration["ci_high"],
            "assumed_cluster_size": illustration["assumed_cluster_size"],
            "assumed_within_cluster_correlation":
                illustration["assumed_within_cluster_correlation"],
            "scale_factor": illustration["scale_factor"],
            "illustrative_low": illustration["illustrative_low"],
            "illustrative_high": illustration["illustrative_high"],
            "status": illustration["status"],
        })
    return rows


def validation_reproduction_rows(models, test_seed_table, test_contrasts,
                                 validation_dir=None):
    """Whether the official test split reproduces each validation conclusion.

    Reads the committed validation across-seed summary and sets it beside the
    test one, contrast by contrast and suite by suite. The validation directory
    is opened read-only and nothing in it is rewritten.

    Two things this is not. It is not a test of validation against test - the
    two estimators have different resampling units and different sample sizes,
    and no interval here spans both. And it is not an opportunity to revise
    anything: a test result that contradicts the validation conclusion is
    recorded in this table as a contradiction. The C-B direction was
    pre-registered as weak evidence either way, so its non-reproduction is the
    outcome the design anticipated, not a finding.
    """
    validation_dir = Path(validation_dir or VALIDATION_ANALYSIS_DIR)
    source = validation_dir / "across_seeds.csv"
    if not source.exists():
        raise FileNotFoundError(f"no validation summary at {source}")
    validation = pd.read_csv(source)
    test_by_key = {(row["suite"], row["contrast"]): row
                   for row in test_seed_table}
    per_seed = defaultdict(dict)
    for row in test_contrasts:
        per_seed[(row["suite"], row["contrast"])][row["seed"]] = row["difference"]
    rows = []
    for record in validation.to_dict("records"):
        key = (record["suite"], record["contrast"])
        test = test_by_key.get(key)
        if test is None:
            continue
        agree = record["direction"] == test["direction"]
        rows.append({
            "suite": key[0],
            "contrast": key[1],
            "validation_mean_difference": float(record["mean_difference"]),
            "test_mean_difference": test["mean_difference"],
            "validation_seeds_favouring_left":
                int(record["seeds_favouring_left"]),
            "test_seeds_favouring_left": test["seeds_favouring_left"],
            "validation_direction": record["direction"],
            "test_direction": test["direction"],
            "directions_agree": bool(agree),
            "test_seed_differences": " ".join(
                f"{seed}:{value:+.4f}"
                for seed, value in sorted(per_seed[key].items())),
            "unit_of_analysis": ("validation resampled 178 physical signs; "
                                 "the test split has no track structure and "
                                 "resamples 12630 images, so the two intervals "
                                 "are not comparable and none is reported here"),
            "models": len(models),
        })
    return rows


def check_test_provenance(models, inventory_path=None):
    """Confirm every result set is an official-test run of an approved model.

    `verify_model_identities` establishes that a directory holds the arm and
    seed it was supplied as. Three things it cannot know are checked here,
    because this module reads prediction files and a validation result set has
    the same schema as a test one:

    * the evaluator recorded `official_test` as the split and `final_test` as
      the corruption namespace, so a validation directory passed on the command
      line is rejected instead of being compared as though it were a test run;
    * the checkpoint digest appears in the frozen inventory, so only the nine
      approved checkpoints can contribute;
    * both suites of a model carry the same checkpoint digest.
    """
    inventory_path = Path(inventory_path or INVENTORY_PATH)
    inventory = json.loads(inventory_path.read_text())
    approved = {record["checkpoint_md5"]: record
                for record in inventory["models"]}
    records = {}
    for model in models:
        digests = set()
        for suite_name in sorted(TEST_SUITES):
            path = model["dir"] / suite_name / "metadata.json"
            if not path.exists():
                raise RuntimeError(f"no metadata at {path}")
            metadata = json.loads(path.read_text())
            split = _recorded(metadata, ("evaluation", "split"), path)
            namespace = _recorded(metadata, ("corruption_suite", "namespace"),
                                  path)
            if split != "official_test":
                raise RuntimeError(f"{path} records the split as {split!r}; "
                                   "this module compares official test "
                                   "results only")
            if namespace != TEST_NAMESPACE:
                raise RuntimeError(f"{path} records the corruption namespace "
                                   f"as {namespace!r}, expected "
                                   f"{TEST_NAMESPACE!r}")
            digest = str(_recorded(metadata, ("checkpoint", "md5"), path))
            if digest not in approved:
                raise RuntimeError(
                    f"{path} records checkpoint md5 {digest}, which is not one "
                    f"of the {len(approved)} in {inventory_path}")
            digests.add(digest)
        if len(digests) != 1:
            raise RuntimeError(f"{model['key']} reports {len(digests)} "
                               "different checkpoints across its two suites")
        digest = digests.pop()
        record = approved[digest]
        if record["arm_label"] != model["name"] or record["seed"] != model["seed"]:
            raise RuntimeError(
                f"{model['key']} holds the checkpoint the inventory files as "
                f"{record['arm_label']} seed {record['seed']}")
        records[model["key"]] = {
            "checkpoint_md5": digest,
            "checkpoint_sha256": record["checkpoint_sha256"],
            "selected_epoch": record["selected_epoch"],
            "split": "official_test",
            "namespace": TEST_NAMESPACE,
        }
    return records


def check_output_target(output_dir):
    """Refuse to write where a comparison already exists.

    The official test comparison is written once. Overwriting it would destroy
    the only record of what the first run of the decision rule produced, which
    is exactly the record that makes the rule credible, so an existing file
    stops the run rather than being replaced.
    """
    output_dir = Path(output_dir).resolve()
    for protected in PROTECTED_TREES:
        protected = protected.resolve()
        if (output_dir == protected or protected in output_dir.parents
                or output_dir in protected.parents):
            raise RuntimeError(
                f"{output_dir} is inside, equal to, or a parent of "
                f"{protected}, which holds committed results this module "
                "reads. The comparison writes only to its own analysis "
                "directory, and the read-only treatment of the validation "
                "and baseline trees is enforced here rather than assumed.")
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"{output_dir} is not a directory")
    present = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if present:
        raise RuntimeError(
            f"{output_dir} already holds {len(present)} comparison files "
            f"(first {present[:3]}); the official test comparison is written "
            "once, so this run would overwrite the recorded result. Move the "
            "existing directory aside deliberately if it has to be redone.")
    return output_dir


def format_report(arm_table, contrast_table, seed_table, clean_table,
                  tradeoff_table, verdict, balanced_table=(),
                  balanced_contrasts=(), reproduction_table=()):
    """Render the tables the write-up quotes."""
    lines = []
    lines.append("Official test split: mean per-image error rate under "
                 "corruption, by arm")
    header = (f"{'suite':9s} {'seed':>5s} {'arm':>3s} {'model':10s} "
              f"{'error':>8s} {'interval':>18s}")
    lines += [header, "-" * len(header)]
    for row in arm_table:
        interval = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        lines.append(f"{row['suite']:9s} {row['seed']:>5d} {row['arm']:>3s} "
                     f"{row['model']:10s} {row['mean_image_error']:>8.4f} "
                     f"{interval:>18s}")

    lines.append("")
    lines.append("Paired within-image contrasts, per seed "
                 "(negative favours the left arm)")
    header = (f"{'suite':9s} {'seed':>5s} {'contrast':>8s} {'difference':>11s} "
              f"{'interval':>20s} {'images worse':>13s}")
    lines += [header, "-" * len(header)]
    for row in contrast_table:
        interval = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        lines.append(f"{row['suite']:9s} {row['seed']:>5d} {row['contrast']:>8s} "
                     f"{row['difference']:>+11.4f} {interval:>20s} "
                     f"{row['images_worse_under_left']:>13d}")
    lines.append("Intervals resample the 12630 test images. They assume "
                 "independent image-level sampling, which repeated images of "
                 "the same physical signs likely violate; they do not account "
                 "for that dependence, are not definitive population-level "
                 "confidence intervals, and are narrower than a "
                 "dependence-aware interval would be. The official test "
                 "release carries no track identifiers, so the dependence "
                 "cannot be measured from it.")

    lines.append("")
    lines.append("Across training seeds")
    header = (f"{'suite':9s} {'contrast':>8s} {'seeds':>5s} {'mean':>9s} "
              f"{'range':>9s}  direction")
    lines += [header, "-" * len(header)]
    for row in seed_table:
        lines.append(f"{row['suite']:9s} {row['contrast']:>8s} "
                     f"{row['seeds']:>5d} {row['mean_difference']:>+9.4f} "
                     f"{row['range']:>9.4f}  {row['direction']}")
    lines.append("Three runs support a direction and a spread. No significance "
                 "across training seeds is claimed, and seed variability is "
                 "not combined with image-level sampling uncertainty.")

    lines.append("")
    lines.append("Clean test performance and the trade-off")
    header = (f"{'arm':>3s} {'model':10s} {'seed':>5s} {'clean acc':>10s} "
              f"{'clean F1':>9s}")
    lines += [header, "-" * len(header)]
    for row in clean_table:
        lines.append(f"{row['arm']:>3s} {row['model']:10s} {row['seed']:>5d} "
                     f"{row['clean_accuracy']:>10.4f} "
                     f"{row['clean_macro_f1']:>9.4f}")
    for row in tradeoff_table:
        lines.append(f"  {row['suite']:9s} seed {row['seed']} {row['contrast']}: "
                     f"clean cost {row['clean_accuracy_cost']:+.4f}, corrupted "
                     f"error {row['corrupted_error_change']:+.4f} - "
                     f"{row['verdict']}")
    lines.append(f"  {CLEAN_COST_REFERENCE_NOTE}")
    lines.append(f"  The {CLEAN_TRADEOFF_LIMIT:g} limit is a reporting rule, "
                 f"not a pass criterion.")

    if balanced_table:
        lines.append("")
        lines.append("Descriptive: class-balanced per-image error, classes "
                     "weighted equally")
        header = (f"{'suite':9s} {'seed':>5s} {'arm':>3s} {'model':10s} "
                  f"{'error':>8s} {'classes':>8s}")
        lines += [header, "-" * len(header)]
        for row in balanced_table:
            lines.append(f"{row['suite']:9s} {row['seed']:>5d} {row['arm']:>3s} "
                         f"{row['model']:10s} "
                         f"{row['class_balanced_error']:>8.4f} "
                         f"{row['classes']:>8d}")
        for row in balanced_contrasts:
            lines.append(f"  {row['suite']:9s} seed {row['seed']} "
                         f"{row['contrast']}: "
                         f"{row['mean_difference']:+.4f}")
        lines.append("Point estimates only. No interval, no per-class "
                     "inference, and no decision depends on this table.")

    if reproduction_table:
        lines.append("")
        lines.append("Validation conclusions beside the official test result")
        header = (f"{'suite':9s} {'contrast':>8s} {'validation':>11s} "
                  f"{'test':>11s} {'agree':>6s}")
        lines += [header, "-" * len(header)]
        for row in reproduction_table:
            lines.append(f"{row['suite']:9s} {row['contrast']:>8s} "
                         f"{row['validation_mean_difference']:>+11.4f} "
                         f"{row['test_mean_difference']:>+11.4f} "
                         f"{str(row['directions_agree']):>6s}")
        lines.append("The two columns come from different resampling units and "
                     "different sample sizes; the comparison is of directions "
                     "and magnitudes, not a test of one against the other. A "
                     "disagreement is reported as a disagreement.")

    lines.append("")
    lines.append(f"Targeted-versus-balanced rule: {verdict['verdict']}")
    lines.append(f"  {verdict['detail']}")
    lines.append("  The rule is the sign of the per-seed point estimate at "
                 "every seed on both suites. It references no interval.")
    lines.append(f"  {VERDICT_UNIT_NOTE}")
    return "\n".join(lines)


def compare(models, output_dir, label_index=None, inventory_path=None,
            validation_dir=None):
    """Run every declared comparison on the official test results and write it.

    `label_index`, `inventory_path` and `validation_dir` default to the real
    ground truth, the frozen inventory and the committed validation analysis.
    They are parameters so the comparison logic can be exercised end to end on
    synthetic fixtures without official test predictions existing.
    """
    output_dir = check_output_target(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Before any number is computed: every directory has to show that it holds
    # the arm and seed it was supplied as, and that it holds an official test
    # evaluation of an approved checkpoint rather than a validation one.
    identities = verify_model_identities(models)
    provenance = check_test_provenance(models, inventory_path)
    print("Verified model identities, from each result set's own metadata:")
    for key, record in sorted(identities.items()):
        print(f"  {key:16s} {record['recorded_arm']:22s} epoch "
              f"{record['checkpoint_epoch']:>3d}  md5 "
              f"{record['checkpoint_md5']} ({record['checkpoint_path']})")
    print(f"All {len(provenance)} result sets record the official test split "
          f"and the {TEST_NAMESPACE} namespace, and every checkpoint digest is "
          "in the frozen inventory.")
    print()

    if label_index is None:
        label_index = test_label_index()

    by_seed = defaultdict(list)
    for model in models:
        by_seed[model["seed"]].append(model)

    arm_table, contrast_table = [], []
    family_table, severity_table = [], []
    balanced_table, balanced_contrasts = [], []
    pairing = {}

    # One draw matrix for the whole comparison: the resampling unit is the test
    # image and the split is fixed, so every interval in every table is taken
    # on the same 2000 resamples of the same 12630 positions. Two intervals
    # therefore differ because the contrasts differ, not because the draws did.
    draw_matrix = image_draws(EXPECTED_TEST_SIZE)

    for suite_name in sorted(TEST_SUITES):
        # Checked over every arm and seed at once, not within a seed: the
        # corrupted realisations depend on the dataset index alone, so all nine
        # checkpoints must have been shown the same pixels.
        pairing[suite_name] = check_identical_examples(models, suite_name)
        for seed, group in sorted(by_seed.items()):
            images = pd.concat(
                [load_suite_images(model, suite_name, label_index)
                 for model in group],
                ignore_index=True)
            corrupted = images[images["corruption"] != CLEAN_CONDITION]
            rates = image_rates(corrupted, ["model"])
            arm_table += arm_rows(rates, seed, suite_name, draw_matrix)
            contrast_table += contrast_rows(rates, seed, suite_name,
                                            available=set(rates["model"]),
                                            matrix=draw_matrix)
            family_table += breakdown_rows(images, seed, suite_name,
                                           "corruption", draw_matrix)
            severity_table += breakdown_rows(images, seed, suite_name,
                                             "severity", draw_matrix)
            balanced_arms, balanced_pairs = class_balanced_rows(
                images, seed, suite_name)
            balanced_table += balanced_arms
            balanced_contrasts += balanced_pairs
            del images, corrupted, rates

    seed_table = across_seeds(contrast_table)
    clean_table = clean_rows(models)
    tradeoff_table = clean_tradeoff_rows(clean_table, contrast_table)
    verdict = targeted_verdict(seed_table, len(by_seed))
    class_table = per_class_rows(models)
    sensitivity_table = sensitivity_rows(contrast_table)
    reproduction_table = validation_reproduction_rows(
        models, seed_table, contrast_table, validation_dir)

    tables = {
        "arm_error_rates.csv": arm_table,
        "paired_contrasts.csv": contrast_table,
        "across_seeds.csv": seed_table,
        "clean_performance.csv": clean_table,
        "clean_tradeoff.csv": tradeoff_table,
        "per_family_contrasts.csv": family_table,
        "per_severity_contrasts.csv": severity_table,
        "per_class_recall.csv": class_table,
        "class_balanced_error.csv": balanced_table,
        "class_balanced_contrasts.csv": balanced_contrasts,
        "design_effect_sensitivity.csv": sensitivity_table,
        "validation_reproduction.csv": reproduction_table,
    }
    unknown = set(tables) - set(OUTPUT_FILES)
    if unknown:
        raise RuntimeError(f"undeclared output files: {sorted(unknown)}")
    written = []
    for name, rows in tables.items():
        if not rows:
            continue
        write_table(output_dir / name, rows)
        written.append(output_dir / name)

    report = format_report(arm_table, contrast_table, seed_table, clean_table,
                           tradeoff_table, verdict, balanced_table,
                           balanced_contrasts, reproduction_table)
    (output_dir / "comparison.txt").write_text(report + "\n")
    (output_dir / "comparison.json").write_text(json.dumps({
        "models": [{"name": model["name"], "seed": model["seed"],
                    "dir": str(model["dir"])} for model in models],
        "model_identity": identities,
        "identity_fields_enforced": [".".join(keys)
                                     for keys in REQUIRED_IDENTITY],
        "test_provenance": provenance,
        "protocol": {
            "split": "official_test",
            "namespace": TEST_NAMESPACE,
            "primary_metric": "mean per-image error rate, conditions equally "
                              "weighted, clean excluded",
            "aggregation": ["image error under one condition",
                            "equal mean over the suite's corrupted conditions",
                            "paired bootstrap over whole images"],
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interval": list(INTERVAL),
            "images": EXPECTED_TEST_SIZE,
            "resampling_unit": "one official test image",
            "interval_assumption": INTERVAL_ASSUMPTION,
            "accounts_for_physical_sign_dependence": False,
            "track_structure_available": False,
            "clean_accuracy_limit": CLEAN_TRADEOFF_LIMIT,
            "clean_accuracy_limit_is_a_pass_criterion": False,
            "clean_cost_measured_against": "the contrast's own right-hand arm "
                                           "at the same seed",
            "clean_cost_reference_note": CLEAN_COST_REFERENCE_NOTE,
            "verdict_wording": VERDICT_UNIT_NOTE,
            "decision_rule": "C is preferred over B only if C's mean per-image "
                             "error is lower at every training seed on both "
                             "suites; the rule references no interval",
            "suites": sorted(TEST_SUITES),
            "suites_are_co_primary": True,
            "held_out_families": sorted(HELDOUT_CORRUPTIONS),
            "seed_variability": "reported as a direction, a mean and a range "
                                "over three runs; never combined with "
                                "image-level sampling uncertainty and never "
                                "reported as significance",
            "descriptive_metrics": {
                "class_balanced_error": {
                    "aggregation": ["image error under one condition",
                                    "mean within a class",
                                    "equal mean over the classes present",
                                    "equal mean over conditions"],
                    "intervals": False,
                    "per_class_inference": False,
                    "used_for_checkpoint_selection": False,
                    "used_for_the_success_criterion": False,
                },
                "per_class_recall": {"intervals": False,
                                     "used_for_the_success_criterion": False},
            },
            "design_effect_illustration": {
                "assumed_cluster_size": DESIGN_EFFECT_CLUSTER,
                "assumed_within_cluster_correlation": DESIGN_EFFECT_RHO,
                "appears_only_in": "design_effect_sensitivity.csv",
                "is_a_confidence_bound": False,
                "is_a_corrected_interval": False,
                "enters_the_decision_rule": False,
            },
        },
        "input_pairing": pairing,
        "input_pairing_note": "equal fingerprints establish that the arms were "
                              "shown the same inputs; they do not establish "
                              "which checkpoint produced the predictions, "
                              "which is what model_identity records",
        "arm_error_rates": arm_table,
        "paired_contrasts": contrast_table,
        "across_seeds": seed_table,
        "clean_performance": clean_table,
        "clean_tradeoff": tradeoff_table,
        "class_balanced_error": balanced_table,
        "class_balanced_contrasts": balanced_contrasts,
        "validation_reproduction": reproduction_table,
        "targeted_versus_balanced": verdict,
    }, indent=2) + "\n")
    written += [output_dir / "comparison.txt", output_dir / "comparison.json"]

    print(report)
    print("\nWritten:")
    for path in written:
        print(f"  {path}")
    return {"report": report, "verdict": verdict, "paths": written,
            "test_provenance": provenance}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True,
                        metavar="NAME:SEED:DIR",
                        help="one evaluated checkpoint; repeat for each arm "
                             "and seed")
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args(argv)

    models = [parse_model(specification) for specification in arguments.model]
    keys = [model["key"] for model in models]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate arm and seed: {sorted(keys)}")
    expected = {f"{arm}{seed}" for arm in ARMS for seed in (42, 43, 44)}
    supplied = {f"{model['name']}{model['seed']}" for model in models}
    if supplied != expected:
        raise ValueError(
            f"the comparison needs all nine approved models; missing "
            f"{sorted(expected - supplied)}, unexpected "
            f"{sorted(supplied - expected)}")
    output_dir = Path(arguments.output_dir) if arguments.output_dir else ANALYSIS_DIR
    compare(models, output_dir)


if __name__ == "__main__":
    main()
