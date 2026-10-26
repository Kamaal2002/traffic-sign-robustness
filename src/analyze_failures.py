"""Failure analysis of the accepted baseline robustness sweep.

    .venv/bin/python -m src.analyze_failures

The sweep in `evaluate_robustness.py` reports one accuracy and one macro F1 per
condition. That says how much the clean-trained baseline loses but nothing about
how it loses it, so this module re-reads the saved per-sample predictions and
answers four questions offline:

1. how clean-correct frames turn into corruption failures, and at which
   severity each one first breaks;
2. where corrupted predictions go — which classes absorb the errors, and how
   concentrated that is;
3. which classes degrade, and how much of that is claimable given that ten
   validation classes hold a single physical sign;
4. which examples are worth looking at, chosen by rules fixed before any image
   is rendered.

No model runs here. Every prediction comes from `predictions.csv`, and the only
images touched are validation frames re-read to check that a regenerated
occlusion patch sits where the recovered anchor says it does. The official test
split is not constructed, loaded, or read anywhere in this module.

Frames are not independent observations
---------------------------------------
A GTSRB track is 30 consecutive frames of one physical sign, and the split is
track-level: validation holds 178 tracks, 30 frames each. A frame-level count
can therefore be large because many signs are affected or because one sign
contributed 30 near-duplicates, and the two mean very different things. Every
frame-level count below is reported next to the number of distinct validation
tracks behind it, and the track identifiers come from `data.py`'s own reader
rather than from assuming that each block of 30 rows is a track.

What the outputs claim
----------------------
The CSVs and the JSON hold measurements: counts, rates, shares, and the
concentration statistics defined in `destination_table`. Mechanisms are not
inferred from them. Where a result is consistent with more than one explanation
the note in the printed report says so rather than picking one.
"""

import hashlib
import json
import subprocess
from hashlib import blake2b
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torchvision.datasets import GTSRB

from .corruptions import (
    CORRUPTIONS,
    OCCLUSION_FILL,
    OCCLUSION_LEVELS,
    SEVERITY_LEVELS,
    VALIDATION_NAMESPACE,
    apply_corruption,
    build_validation_observations,
    corruption_parameter,
    occlusion_anchor,
    occlusion_side,
    sample_seed,
    stable_index,
)
from .data import IMAGE_SIZE, NUM_CLASSES, read_training_index, split_tracks
from .evaluate_robustness import (
    CLEAN_CONDITION,
    EXPECTED_VALIDATION_SIZE,
    OCCLUSION_NOTE,
    RESULTS_DIR,
)
from .train import metrics_from_confusion, per_class_from_confusion, update_confusion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / "failure_analysis" / "baseline"

FRAMES_PER_TRACK = 30
EXPECTED_TRACKS = 178

# Agreement required when a saved metric is recomputed from the predictions.
# The saved tables are written with full float64 precision, so the recomputed
# values differ only by the order of the summations behind them; 1e-9 is far
# below any difference that would indicate a different experiment and far above
# that reordering noise, which in practice lands near 1e-16.
METRIC_TOLERANCE = 1e-9

# A track "majority-fails" at the first severity where more than this fraction
# of its clean-correct frames are wrong. Half is the only threshold that makes
# the phrase mean what it says; it is named here so the figures and the CSV
# cannot drift apart.
TRACK_MAJORITY = 0.5

# Error-destination entropy is estimated from a sample and is biased low when
# that sample is small, so conditions with 90 errors and conditions with 3,686
# errors cannot be compared directly. Concentration is therefore also reported
# on repeated draws of a common number of errors.
MATCHED_ERRORS = 300
MATCHED_DRAWS = 200
MATCHED_SEED = 42

# Chebyshev distance of the occlusion anchor from the image centre, in pixels.
# The anchor box is the central 30x30 grid of a 64x64 image, so the largest
# attainable offset is 14.5 and the top edge of 16 covers it. That edge follows
# from the image size and the severity-5 area, so the binning below checks that
# every clean-correct frame lands in a bin instead of assuming it.
ANCHOR_BIN_EDGES = (0, 3, 6, 9, 12, 16)

# Severities shown in the occlusion-anchor table: severity 1, where accuracy
# already falls from 0.9876 to 0.8369 while every other corruption is still
# within 0.005 of clean at its own severity 1, plus a middle and the strongest.
ANCHOR_SEVERITIES = (1, 3, 5)

# Positions sampled for the anchor-recovery check. A stride keeps the check
# spread across classes and tracks instead of sitting inside the first few
# tracks, and 60 images is enough to catch a wrong anchor formula.
ANCHOR_CHECK_STRIDE = 89

CORRUPTION_NAMES = tuple(CORRUPTIONS)

# Conditions the example figures are drawn from: the three corruptions with the
# largest accuracy drops at severity 5. The transition figure fixes one severity per
# corruption before any pair is looked at, and the three are not chosen on the
# same grounds: fog and noise are shown at s5, their strongest level, while
# occlusion is shown at s1 because the question there is what the mildest
# centre-constrained patch already does, not what the largest one does. Within
# the chosen condition the pair is then the one covering the most distinct
# tracks. Each row of examples.csv carries the rule it was selected by.
EXAMPLE_CORRUPTIONS = ("gaussian_noise", "fog", "occlusion")
EXAMPLE_TRANSITIONS = (("fog", 5), ("gaussian_noise", 5), ("occlusion", 1))
EXAMPLE_TRANSITION_ROLES = {
    ("fog", 5): "strongest severity",
    ("gaussian_noise", 5): "strongest severity",
    ("occlusion", 1): "mildest severity",
}
EXAMPLE_TRANSITION_ROWS = 3
EXAMPLE_UNSTABLE_ROWS = 2


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_predictions():
    """Load `predictions.csv` and check it holds the conditions it should.

    The checks are about the file being the accepted sweep rather than about the
    analysis: 31 conditions, the severity grid of `corruptions.py`, one row per
    validation frame in each condition, and one fixed `(position, index, label)`
    mapping shared by every condition. If any of that fails, nothing below is
    comparable across conditions and the analysis should not run at all.
    """
    frame = pd.read_csv(RESULTS_DIR / "predictions.csv")
    expected = {(CLEAN_CONDITION, 0)}
    expected |= {(name, severity) for name in CORRUPTIONS for severity in SEVERITY_LEVELS}
    found = set(map(tuple, frame[["corruption", "severity"]].drop_duplicates().to_numpy()))
    if found != expected:
        raise RuntimeError(
            f"predictions.csv conditions do not match the frozen suite: "
            f"missing {sorted(expected - found)}, unexpected {sorted(found - expected)}"
        )

    sizes = frame.groupby(["corruption", "severity"]).size()
    if set(sizes.unique()) != {EXPECTED_VALIDATION_SIZE}:
        raise RuntimeError(
            f"every condition must hold {EXPECTED_VALIDATION_SIZE} predictions; "
            f"found sizes {sorted(sizes.unique())}"
        )

    per_position = frame.groupby("validation_position").agg(
        labels=("label", "nunique"), indices=("dataset_index", "nunique")
    )
    if len(per_position) != EXPECTED_VALIDATION_SIZE:
        raise RuntimeError(f"found {len(per_position)} distinct validation positions")
    if per_position["labels"].max() != 1 or per_position["indices"].max() != 1:
        raise RuntimeError(
            "a validation position carries more than one label or dataset index, "
            "so the conditions are not aligned to the same frames"
        )
    return frame


def condition_predictions(frame):
    """Predictions per condition as arrays ordered by validation position.

    Returns `(reference, predictions, compared)` where `reference` is the shared
    per-position frame table, `predictions[(corruption, severity)]` is an
    `int64` array aligned to `reference`, and `compared` counts the conditions
    whose index and label order were checked against that reference. Building
    the arrays by sorting and asserting the position vector, rather than
    pivoting, keeps the alignment explicit and keeps the predictions integral.
    """
    positions = np.arange(EXPECTED_VALIDATION_SIZE)
    reference = None
    predictions = {}
    compared = 0
    for (name, severity), group in frame.groupby(["corruption", "severity"], sort=True):
        group = group.sort_values("validation_position")
        if not np.array_equal(group["validation_position"].to_numpy(), positions):
            raise RuntimeError(f"{name} s{severity} does not cover positions 0..N-1 exactly once")
        columns = group[["dataset_index", "label"]].reset_index(drop=True)
        if reference is None:
            reference = columns
        elif not reference.equals(columns):
            raise RuntimeError(f"{name} s{severity} has a different index/label order")
        else:
            compared += 1
        predictions[(name, severity)] = group["prediction"].to_numpy(dtype=np.int64)
    return reference, predictions, compared


def track_identity(dataset_indices):
    """Map each validation frame to its physical sign, recomputing the split.

    The split is rebuilt with `data.py`'s own reader and splitter, so the track
    identifiers and the validation membership are the ones the sweep evaluated
    and not a reconstruction from row order. Only the training archive's file
    listing is read; no image is decoded here.
    """
    dataset = GTSRB(root=PROJECT_ROOT / "data", split="train", transform=None, download=False)
    labels, tracks = read_training_index(dataset)
    train_indices, validation_indices = split_tracks(labels, tracks)

    if set(dataset_indices.tolist()) != set(validation_indices.tolist()):
        raise RuntimeError(
            "predictions.csv does not cover the validation split that data.py produces"
        )
    frame_tracks = tracks[dataset_indices]
    counts = pd.Series(frame_tracks).value_counts()
    if len(counts) != EXPECTED_TRACKS or set(counts.unique()) != {FRAMES_PER_TRACK}:
        raise RuntimeError(
            f"expected {EXPECTED_TRACKS} validation tracks of {FRAMES_PER_TRACK} frames; "
            f"found {len(counts)} tracks with sizes {sorted(counts.unique())}"
        )
    if set(tracks[train_indices]) & set(frame_tracks):
        raise RuntimeError("a validation track also appears in the training split")
    return frame_tracks, labels, tracks, train_indices, validation_indices


def tracks_per_class(labels, tracks, indices):
    """Distinct tracks per class over a set of dataset indices.

    Counting signs rather than frames, on either side of the split: the
    validation counts say how much evidence a per-class score rests on, and the
    training counts say how many signs of that class the baseline ever saw.
    """
    per_class = {class_id: set() for class_id in range(NUM_CLASSES)}
    for index in indices:
        per_class[int(labels[index])].add(tracks[index])
    return {class_id: len(names) for class_id, names in per_class.items()}


def confusion_of(predictions, labels):
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    # Copies: the arrays come from pandas and are not writable, and
    # `torch.from_numpy` would otherwise share that buffer.
    update_confusion(matrix, torch.tensor(predictions), torch.tensor(labels))
    return matrix


def check_accepted_tables(predictions, labels):
    """Recompute every saved table from the predictions and compare.

    `summary.csv`, `per_class_metrics.csv`, and `confusion_matrices.npz` are
    derived quantities; `predictions.csv` is the raw record. Rebuilding the
    three from the one proves this analysis reads the same experiment the
    accepted numbers came from, and it also checks the severity-to-parameter
    mapping against the frozen `CORRUPTIONS` table.
    """
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    per_class = pd.read_csv(RESULTS_DIR / "per_class_metrics.csv")
    saved_confusion = np.load(RESULTS_DIR / "confusion_matrices.npz")
    checked = 0

    for (name, severity), prediction in sorted(predictions.items()):
        matrix = confusion_of(prediction, labels)
        key = f"{name}_s{severity}"
        if not np.array_equal(matrix.numpy(), saved_confusion[key]):
            raise RuntimeError(f"recomputed confusion matrix for {key} differs from the saved one")

        accuracy, macro_f1 = metrics_from_confusion(matrix)
        row = summary[(summary["corruption"] == name) & (summary["severity"] == severity)]
        if len(row) != 1:
            raise RuntimeError(f"summary.csv has {len(row)} rows for {key}")
        row = row.iloc[0]
        for metric, value in (("accuracy", accuracy), ("macro_f1", macro_f1)):
            if abs(row[metric] - value) > METRIC_TOLERANCE:
                raise RuntimeError(
                    f"{key} {metric}: recomputed {value!r} against saved {row[metric]!r}"
                )
        if name != CLEAN_CONDITION:
            expected_parameter = corruption_parameter(name, severity)
            if abs(row["parameter_value"] - expected_parameter) > METRIC_TOLERANCE:
                raise RuntimeError(
                    f"{key} parameter {row['parameter_value']} is not the frozen "
                    f"level {expected_parameter}"
                )

        scores = per_class_from_confusion(matrix)
        saved_rows = per_class[
            (per_class["corruption"] == name) & (per_class["severity"] == severity)
        ].sort_values("class")
        if len(saved_rows) != NUM_CLASSES:
            raise RuntimeError(f"per_class_metrics.csv has {len(saved_rows)} rows for {key}")
        for column, vector in (("recall", scores["recall"]), ("f1", scores["f1"])):
            if np.abs(saved_rows[column].to_numpy() - vector.numpy()).max() > METRIC_TOLERANCE:
                raise RuntimeError(f"recomputed per-class {column} for {key} differs")
        if not np.array_equal(saved_rows["support"].to_numpy(), scores["support"].numpy()):
            raise RuntimeError(f"recomputed per-class support for {key} differs")
        checked += 1

    severities = sorted({severity for name, severity in predictions if name != CLEAN_CONDITION})
    if severities != list(SEVERITY_LEVELS):
        raise RuntimeError(f"severity grid is {severities}, not {list(SEVERITY_LEVELS)}")
    for name in CORRUPTIONS:
        levels = [corruption_parameter(name, severity) for severity in SEVERITY_LEVELS]
        ordered = levels == sorted(levels) or levels == sorted(levels, reverse=True)
        if not ordered:
            raise RuntimeError(f"{name} severity levels are not monotone: {levels}")
    return checked, summary, per_class


def build_trajectories(reference, predictions, tracks):
    """One row per (frame, corruption) describing how that frame behaves.

    Columns beyond the identifiers:

    `onset_severity`     first severity at which the prediction is wrong, 0 if
                         the frame is right at all five.
    `persistent`         wrong at every severity from the onset upwards. False
                         means the prediction returns to the true class at some
                         stronger severity, which is a real behaviour and not a
                         bookkeeping artifact.
    `distinct_predictions`  number of distinct classes predicted across the
                         five severities, so 1 means the prediction never moved
                         and larger values mean the frame changes identity as
                         severity grows.
    `prediction_changes` number of adjacent severity steps where the prediction
                         changes, which separates a single switch from churn.

    Frames the clean model already gets wrong are kept with
    `clean_correct=False` so error totals can be reconciled, but every onset
    statistic below is restricted to the clean-correct frames: a corruption
    cannot be said to have broken a frame the baseline never had.
    """
    labels = reference["label"].to_numpy()
    clean_prediction = predictions[(CLEAN_CONDITION, 0)]
    rows = []
    for name in CORRUPTION_NAMES:
        stack = np.stack([predictions[(name, severity)] for severity in SEVERITY_LEVELS], axis=1)
        correct = stack == labels[:, None]
        any_wrong = ~correct.all(axis=1)
        onset = np.zeros(len(labels), dtype=np.int64)
        onset[any_wrong] = (~correct[any_wrong]).argmax(axis=1) + 1
        persistent = np.zeros(len(labels), dtype=bool)
        for position in np.flatnonzero(any_wrong):
            persistent[position] = not correct[position, onset[position] - 1:].any()
        distinct = np.array([len(set(row.tolist())) for row in stack])
        changes = (stack[:, 1:] != stack[:, :-1]).sum(axis=1)
        rows.append(pd.DataFrame({
            "corruption": name,
            "validation_position": np.arange(len(labels)),
            "dataset_index": reference["dataset_index"].to_numpy(),
            "track": tracks,
            "label": labels,
            "clean_prediction": clean_prediction,
            "clean_correct": clean_prediction == labels,
            "onset_severity": onset,
            "persistent": np.where(any_wrong, persistent, False),
            "distinct_predictions": distinct,
            "prediction_changes": changes,
            **{
                f"prediction_s{severity}": stack[:, severity - 1]
                for severity in SEVERITY_LEVELS
            },
        }))
    return pd.concat(rows, ignore_index=True)


def onset_table(trajectories):
    """Frame-level and track-level onset counts per corruption.

    The partition `{never, onset 1..5}` is exclusive and covers every
    clean-correct frame, so the counts reconcile against the prediction rows.
    `tracks` is the number of distinct physical signs behind each frame count,
    which is what distinguishes "many signs lost a frame" from "one sign lost
    thirty". It is a per-bucket count: a sign whose frames first fail at
    different severities appears in several rows, so these track counts must
    not be added up or maximised over to describe a corruption as a whole.
    `onset_totals_table` holds the per-corruption totals.
    """
    rows = []
    clean_correct = trajectories[trajectories["clean_correct"]]
    for name, group in clean_correct.groupby("corruption", sort=False):
        for onset in (0, *SEVERITY_LEVELS):
            selected = group[group["onset_severity"] == onset]
            rows.append({
                "corruption": name,
                "onset_severity": onset,
                "frames": len(selected),
                "tracks": selected["track"].nunique(),
                "persistent_frames": int(selected["persistent"].sum()),
                "frames_with_three_or_more_predictions":
                    int((selected["distinct_predictions"] >= 3).sum()),
            })
    return pd.DataFrame(rows)


def onset_totals_table(trajectories, onsets):
    """Per-corruption totals over the onset buckets.

    `tracks_affected` is the union over severities of the signs with at least
    one clean-correct frame that fails: a sign can lose its first frame at s1
    and another at s4, so the per-bucket track counts overlap and neither their
    sum nor their maximum is the number of signs a corruption touches.
    """
    rows = []
    clean_correct = trajectories[trajectories["clean_correct"]]
    for name, group in clean_correct.groupby("corruption", sort=False):
        failing = group[group["onset_severity"] > 0]
        buckets = onsets[(onsets["corruption"] == name) & (onsets["onset_severity"] > 0)]
        rows.append({
            "corruption": name,
            "clean_correct_frames": len(group),
            "frames_ever_wrong": len(failing),
            "tracks_affected": failing["track"].nunique(),
            "tracks_total": group["track"].nunique(),
            "largest_onset_bucket_tracks": int(buckets["tracks"].max()),
            "persistent_frames": int(failing["persistent"].sum()),
            "frames_with_three_or_more_predictions":
                int((group["distinct_predictions"] >= 3).sum()),
        })
    return pd.DataFrame(rows)


def induced_table(trajectories):
    """Per severity: clean-correct frames that are wrong at that severity.

    This is the cumulative view that matches the accuracy curve, where
    `onset_table` is the first-failure view. Both are reported because they
    answer different things: one says how many frames are broken at severity s,
    the other says when each frame broke.
    """
    rows = []
    clean_correct = trajectories[trajectories["clean_correct"]]
    for name, group in clean_correct.groupby("corruption", sort=False):
        for severity in SEVERITY_LEVELS:
            wrong = group[group[f"prediction_s{severity}"] != group["label"]]
            rows.append({
                "corruption": name,
                "severity": severity,
                "induced_error_frames": len(wrong),
                "induced_error_tracks": wrong["track"].nunique(),
                "share_of_clean_correct": len(wrong) / len(group),
            })
    return pd.DataFrame(rows)


def track_onset_table(trajectories):
    """When each track loses the majority of its clean-correct frames.

    A track is one physical sign. Reporting the severity at which more than
    `TRACK_MAJORITY` of its clean-correct frames are misclassified answers a
    question frame counts cannot: whether a corruption takes down whole signs or
    takes a minority of frames from many signs. Tracks whose majority never
    flips are counted at onset 0.
    """
    rows = []
    clean_correct = trajectories[trajectories["clean_correct"]]
    for name, group in clean_correct.groupby("corruption", sort=False):
        onsets = []
        for _, track_rows in group.groupby("track", sort=False):
            wrong_share = [
                (track_rows[f"prediction_s{severity}"] != track_rows["label"]).mean()
                for severity in SEVERITY_LEVELS
            ]
            onsets.append(next(
                (severity for severity, share in zip(SEVERITY_LEVELS, wrong_share)
                 if share > TRACK_MAJORITY),
                0,
            ))
        onsets = np.array(onsets)
        for onset in (0, *SEVERITY_LEVELS):
            rows.append({
                "corruption": name,
                "majority_onset_severity": onset,
                "tracks": int((onsets == onset).sum()),
            })
    return pd.DataFrame(rows)


def matched_concentration(destinations, generator):
    """Top-3 share and normalised entropy on repeated equal-sized draws.

    Entropy computed on 90 errors and on 3,686 errors are not comparable: the
    estimate is biased towards concentration when few errors are observed. Each
    draw takes `MATCHED_ERRORS` errors without replacement, so every condition
    large enough is described at the same sample size. Conditions with fewer
    errors than that return no value rather than an incomparable one.
    """
    if len(destinations) < MATCHED_ERRORS:
        return None, None
    entropies = []
    shares = []
    for _ in range(MATCHED_DRAWS):
        draw = destinations[generator.choice(len(destinations), MATCHED_ERRORS, replace=False)]
        fractions = np.bincount(draw, minlength=NUM_CLASSES) / MATCHED_ERRORS
        entropies.append(normalised_entropy(fractions))
        shares.append(np.sort(fractions)[::-1][:3].sum())
    return float(np.mean(shares)), float(np.mean(entropies))


def normalised_entropy(fractions):
    """Shannon entropy of a destination distribution, divided by log(43).

    The distribution is over predicted classes among a condition's errors. The
    denominator is the entropy of errors spread evenly over all 43 classes, so 1
    means maximally diffuse and 0 means every error lands on one class. Fixing
    the denominator at `log(NUM_CLASSES)` rather than at the number of
    destinations actually observed is what makes two conditions comparable:
    otherwise a condition that reaches fewer destinations would be rescaled to
    look more diffuse than it is.
    """
    nonzero = fractions[fractions > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / np.log(NUM_CLASSES))


def destination_table(trajectories):
    """Where corruption-induced errors land, and how concentrated that is.

    Restricted to clean-correct frames that the corruption breaks, so the 66
    frames the baseline already misclassifies cannot masquerade as corruption
    behaviour — they matter here because at mild severity they would otherwise
    dominate a small error set.

    Two concentration statistics are reported and no more: the top-3 share is
    directly readable ("x% of errors land on three classes") and the normalised
    entropy accounts for the whole distribution rather than its head. Both are
    frame-level, so each is accompanied by the number of distinct tracks behind
    the errors; a top-3 share of 0.56 drawn from two signs is not evidence of a
    model-level attractor.
    """
    generator = np.random.default_rng(MATCHED_SEED)
    rows = []
    for name in CORRUPTION_NAMES:
        group = trajectories[trajectories["corruption"] == name]
        for severity in SEVERITY_LEVELS:
            column = f"prediction_s{severity}"
            wrong = group[group[column] != group["label"]]
            induced = wrong[wrong["clean_correct"]]
            destinations = induced[column].to_numpy()
            if len(induced):
                fractions = np.bincount(destinations, minlength=NUM_CLASSES) / len(induced)
                order = np.argsort(fractions)[::-1]
                head = {
                    "top_destination": int(order[0]),
                    "top_destination_share": float(fractions[order[0]]),
                    "top_destination_tracks":
                        int(induced[induced[column] == order[0]]["track"].nunique()),
                    "top3_share": float(fractions[order[:3]].sum()),
                    "normalised_entropy": normalised_entropy(fractions),
                }
            else:
                head = dict.fromkeys(
                    ("top_destination", "top_destination_share", "top_destination_tracks",
                     "top3_share", "normalised_entropy")
                )
                fractions = np.zeros(NUM_CLASSES)
            matched_top3, matched_entropy = matched_concentration(destinations, generator)
            rows.append({
                "corruption": name,
                "severity": severity,
                "errors_total": len(wrong),
                "errors_induced": len(induced),
                "errors_clean_already_wrong": len(wrong) - len(induced),
                "induced_error_tracks": induced["track"].nunique(),
                "distinct_destinations": int((fractions > 0).sum()),
                **head,
                "top3_share_matched": matched_top3,
                "normalised_entropy_matched": matched_entropy,
            })
    return pd.DataFrame(rows)


def transition_table(trajectories):
    """Every source class to predicted class pair among induced errors.

    `frames` and `tracks` are both kept for every pair, because a pair made of
    30 frames of one sign and a pair made of 30 frames from nine signs are
    different claims. Pair counts sum to the induced error count per condition,
    which `reconcile` checks.
    """
    rows = []
    for name in CORRUPTION_NAMES:
        group = trajectories[(trajectories["corruption"] == name) & trajectories["clean_correct"]]
        for severity in SEVERITY_LEVELS:
            column = f"prediction_s{severity}"
            induced = group[group[column] != group["label"]]
            pairs = induced.groupby(["label", column]).agg(
                frames=("track", "size"), tracks=("track", "nunique")
            ).reset_index()
            pairs.columns = ["source", "target", "frames", "tracks"]
            pairs.insert(0, "severity", severity)
            pairs.insert(0, "corruption", name)
            rows.append(pairs)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["corruption", "severity", "frames"], ascending=[True, True, False]
    ).reset_index(drop=True)


def class_tables(per_class, validation_tracks, train_tracks_per_class):
    """Per-class clean score, degradation, and the uncertainty attached to it.

    Three things that get conflated are kept apart. Clean weakness is the
    baseline's score before any corruption. Corruption-induced degradation is
    `clean recall - corrupted recall` at a given severity. Support uncertainty
    is the number of validation tracks: a class with one track is 30 frames of
    one physical sign, so its per-class score is a single-sign observation and
    cannot support a claim about the class.

    The training track count is carried alongside because the split is
    stratified: a class with one validation track is also a class with few
    training signs, so clean weakness, degradation, and measurement noise are
    confounded in the same direction and a correlation between support and
    degradation cannot separate them.
    """
    clean = per_class[per_class["corruption"] == CLEAN_CONDITION].set_index("class")
    degradation = per_class[per_class["corruption"] != CLEAN_CONDITION].copy()
    degradation["clean_recall"] = degradation["class"].map(clean["recall"])
    degradation["clean_f1"] = degradation["class"].map(clean["f1"])
    degradation["recall_drop"] = degradation["clean_recall"] - degradation["recall"]
    degradation["f1_drop"] = degradation["clean_f1"] - degradation["f1"]
    degradation["validation_tracks"] = degradation["class"].map(validation_tracks)
    degradation["train_tracks"] = degradation["class"].map(train_tracks_per_class)

    summary = pd.DataFrame({
        "class": clean.index,
        "support": clean["support"].to_numpy(),
        "validation_tracks": [validation_tracks[c] for c in clean.index],
        "train_tracks": [train_tracks_per_class[c] for c in clean.index],
        "clean_recall": clean["recall"].to_numpy(),
        "clean_f1": clean["f1"].to_numpy(),
    })
    for severity in (3, 5):
        at_severity = degradation[degradation["severity"] == severity]
        mean_drop = at_severity.groupby("class")["recall_drop"].mean()
        worst = at_severity.loc[at_severity.groupby("class")["recall_drop"].idxmax()]
        summary[f"mean_recall_drop_s{severity}"] = summary["class"].map(mean_drop)
        summary[f"worst_corruption_s{severity}"] = summary["class"].map(
            worst.set_index("class")["corruption"]
        )
        summary[f"worst_recall_drop_s{severity}"] = summary["class"].map(
            worst.set_index("class")["recall_drop"]
        )
    summary["single_validation_track"] = summary["validation_tracks"] == 1
    # Three independent signs is the smallest support at which a per-class
    # pattern is not one or two signs; it is a reporting threshold, not a test.
    summary["claimable_support"] = summary["validation_tracks"] >= 3
    return summary, degradation


def occlusion_anchor_offsets(dataset_indices):
    """Recover each frame's occlusion anchor and its distance from the centre.

    The anchor is a deterministic function of the frozen per-sample seed, so it
    can be recovered exactly without re-running the corruption or the model.
    Offset is the Chebyshev distance of the anchor from the image centre, which
    matches the square patch: it is the coordinate that decides how far the
    patch reaches towards a border.

    This is the measurement the occlusion caveat needs. Because the anchor is
    drawn from the sample's index and nothing about its content, the offset is
    assigned independently of what the image shows, and the 30 frames of a track
    receive 30 different anchors.
    """
    offsets = np.empty(len(dataset_indices), dtype=np.float64)
    anchors = np.empty((len(dataset_indices), 2), dtype=np.int64)
    centre = (IMAGE_SIZE - 1) / 2
    for row, index in enumerate(dataset_indices):
        generator = torch.Generator().manual_seed(
            sample_seed(VALIDATION_NAMESPACE, "occlusion", 1, int(index))
        )
        anchor_y, anchor_x = occlusion_anchor((3, IMAGE_SIZE, IMAGE_SIZE), generator)
        anchors[row] = (anchor_y, anchor_x)
        offsets[row] = max(abs(anchor_y - centre), abs(anchor_x - centre))
    return anchors, offsets


def occlusion_anchor_table(trajectories, offsets):
    """Failure rate against anchor offset, with a within-track pairing.

    Two views of the same data. The binned rates compare frames across the whole
    split, which mixes signs of different difficulty. The paired column compares
    each track against itself: within one sign's 30 frames, the failure rate of
    the frames whose anchor is nearer the centre than that track's median minus
    the rate of the rest. Since anchor position is assigned from the sample
    index and not from the image, that difference isolates the patch position
    from which signs happen to be hard.
    """
    group = trajectories[
        (trajectories["corruption"] == "occlusion") & trajectories["clean_correct"]
    ].copy()
    group["offset"] = offsets[group["validation_position"].to_numpy()]
    bins = np.digitize(group["offset"].to_numpy(), ANCHOR_BIN_EDGES[1:], right=False)

    rows = []
    pairs = []
    paired = {}
    binned = 0
    for severity in ANCHOR_SEVERITIES:
        wrong = (group[f"prediction_s{severity}"] != group["label"]).to_numpy()
        side = occlusion_side(OCCLUSION_LEVELS[severity - 1], IMAGE_SIZE, IMAGE_SIZE)
        for bin_index in range(len(ANCHOR_BIN_EDGES) - 1):
            selected = bins == bin_index
            if not selected.any():
                continue
            binned += int(selected.sum())
            rows.append({
                "severity": severity,
                "patch_side_pixels": side,
                "offset_low": ANCHOR_BIN_EDGES[bin_index],
                "offset_high": ANCHOR_BIN_EDGES[bin_index + 1],
                "frames": int(selected.sum()),
                "tracks": group.loc[selected, "track"].nunique(),
                "failures": int(wrong[selected].sum()),
                "failure_rate": float(wrong[selected].mean()),
            })

        for name, track_rows in group.assign(wrong=wrong).groupby("track", sort=False):
            median = track_rows["offset"].median()
            near = track_rows[track_rows["offset"] < median]["wrong"]
            far = track_rows[track_rows["offset"] >= median]["wrong"]
            if not len(near) or not len(far):
                continue
            pairs.append({
                "severity": severity,
                "track": name,
                "frames_near": len(near),
                "frames_far": len(far),
                "failure_rate_near": float(near.mean()),
                "failure_rate_far": float(far.mean()),
                "difference": float(near.mean() - far.mean()),
            })
        differences = np.array(
            [row["difference"] for row in pairs if row["severity"] == severity]
        )
        paired[severity] = {
            "tracks": int(len(differences)),
            "mean_difference": float(differences.mean()),
            "median_difference": float(np.median(differences)),
            "tracks_with_higher_failure_near_centre": int((differences > 0).sum()),
        }
    if binned != len(group) * len(ANCHOR_SEVERITIES):
        raise RuntimeError("anchor offset bins do not cover every clean-correct frame")
    return pd.DataFrame(rows), pd.DataFrame(pairs), paired


def verify_anchor_recovery(dataset_indices, anchors):
    """Check the recovered anchor against the frozen corruption on real frames.

    The anchor analysis and the example grids both assume the patch sits where
    `occlusion_anchor` says it does. This re-applies the frozen corruption to a
    strided sample of validation frames and requires, exactly: every pixel
    inside the predicted box equals the fill value, and every pixel outside it
    is untouched. Checking both directions catches a shifted box, a wrong side
    length, and a stale seed, and it does so without assuming the patch differs
    from the image it covers.

    Returns the number of images read, which is also the only image decoding
    this module does, and the number of patch boxes checked on them.
    """
    dataset, validation_indices, _ = build_validation_observations()
    members = set(validation_indices.tolist())
    positions = list(range(0, len(dataset), ANCHOR_CHECK_STRIDE))
    read = 0
    boxes = 0
    for position in positions:
        index = stable_index(dataset, position)
        if index not in members:
            raise RuntimeError(f"position {position} resolves outside the validation split")
        row = int(np.flatnonzero(dataset_indices == index)[0])
        image, _ = dataset[position]
        read += 1
        anchor_y, anchor_x = anchors[row]
        for severity in SEVERITY_LEVELS:
            side = occlusion_side(OCCLUSION_LEVELS[severity - 1], IMAGE_SIZE, IMAGE_SIZE)
            top = anchor_y - side // 2
            left = anchor_x - side // 2
            corrupted = apply_corruption(
                image, "occlusion", severity,
                seed=sample_seed(VALIDATION_NAMESPACE, "occlusion", severity, index),
            )
            patch = corrupted[..., top:top + side, left:left + side]
            if not torch.all(patch == OCCLUSION_FILL):
                raise RuntimeError(
                    f"index {index} s{severity}: predicted patch box is not filled"
                )
            mask = torch.ones_like(corrupted, dtype=torch.bool)
            mask[..., top:top + side, left:left + side] = False
            if not torch.equal(corrupted[mask], image[mask]):
                raise RuntimeError(
                    f"index {index} s{severity}: pixels outside the predicted box changed"
                )
            boxes += 1
    return read, boxes


def reconcile(trajectories, onsets, totals, induced, destinations, transitions, summary):
    """Cross-check the derived tables against each other and against the sweep.

    Each equality here is a place where a silent slice or a mis-join would
    change a headline number, so they are checked rather than assumed:

    * onset counts partition the clean-correct frames;
    * the per-corruption totals agree with the buckets they summarise, and the
      union of affected tracks lies between the largest single bucket and the
      number of validation tracks;
    * the cumulative induced-error count equals the frames whose onset is at or
      below that severity and that are still wrong there;
    * induced errors plus already-wrong frames equal the condition's total
      errors implied by the accepted accuracy;
    * transition-pair frames sum to the induced error count.

    Returns the clean-correct frame count and the number of equalities checked.
    """
    clean_correct = int(trajectories[trajectories["corruption"] == CORRUPTION_NAMES[0]]
                        ["clean_correct"].sum())
    checks = 0
    for name in CORRUPTION_NAMES:
        rows = onsets[onsets["corruption"] == name]
        if rows["frames"].sum() != clean_correct:
            raise RuntimeError(f"{name}: onset counts sum to {rows['frames'].sum()}")
        checks += 1

        total = totals[totals["corruption"] == name].iloc[0]
        failing = rows[rows["onset_severity"] > 0]
        if total["frames_ever_wrong"] != failing["frames"].sum():
            raise RuntimeError(f"{name}: frames ever wrong disagree with the onset buckets")
        if total["clean_correct_frames"] != clean_correct:
            raise RuntimeError(f"{name}: clean-correct frame count differs")
        if not (total["largest_onset_bucket_tracks"] <= total["tracks_affected"]
                <= EXPECTED_TRACKS):
            raise RuntimeError(
                f"{name}: {total['tracks_affected']} affected tracks is outside the range "
                f"[{total['largest_onset_bucket_tracks']}, {EXPECTED_TRACKS}]"
            )
        correct = trajectories[
            (trajectories["corruption"] == name) & trajectories["clean_correct"]
        ]
        union = set()
        for severity in SEVERITY_LEVELS:
            union |= set(correct.loc[correct["onset_severity"] == severity, "track"])
        if len(union) != total["tracks_affected"]:
            raise RuntimeError(
                f"{name}: {total['tracks_affected']} affected tracks against {len(union)} "
                "accumulated over the onset severities"
            )
        checks += 4

        group = trajectories[trajectories["corruption"] == name]
        for severity in SEVERITY_LEVELS:
            wrong = (group[f"prediction_s{severity}"] != group["label"])
            total_errors = int(wrong.sum())
            accepted = summary[
                (summary["corruption"] == name) & (summary["severity"] == severity)
            ].iloc[0]
            implied = round((1 - accepted["accuracy"]) * EXPECTED_VALIDATION_SIZE)
            if total_errors != implied:
                raise RuntimeError(
                    f"{name} s{severity}: {total_errors} errors against {implied} "
                    "implied by the accepted accuracy"
                )
            row = destinations[
                (destinations["corruption"] == name) & (destinations["severity"] == severity)
            ].iloc[0]
            if row["errors_total"] != total_errors:
                raise RuntimeError(f"{name} s{severity}: destination table error count differs")
            if row["errors_induced"] + row["errors_clean_already_wrong"] != total_errors:
                raise RuntimeError(f"{name} s{severity}: induced and clean-wrong do not sum")

            induced_row = induced[
                (induced["corruption"] == name) & (induced["severity"] == severity)
            ].iloc[0]
            if induced_row["induced_error_frames"] != row["errors_induced"]:
                raise RuntimeError(f"{name} s{severity}: induced counts disagree")
            onset_reachable = int(((group["clean_correct"])
                                   & (group["onset_severity"] > 0)
                                   & (group["onset_severity"] <= severity)
                                   & wrong).sum())
            if onset_reachable != row["errors_induced"]:
                raise RuntimeError(
                    f"{name} s{severity}: {onset_reachable} frames with onset <= severity and "
                    f"still wrong against {row['errors_induced']} induced errors"
                )
            pairs = transitions[
                (transitions["corruption"] == name) & (transitions["severity"] == severity)
            ]
            if pairs["frames"].sum() != row["errors_induced"]:
                raise RuntimeError(f"{name} s{severity}: transition frames do not sum to errors")
            checks += 6
    return clean_correct, checks


def selection_key(rule, dataset_index):
    """Deterministic, class-independent ordering of candidate frames.

    Taking the smallest dataset index would be reproducible but biased: the
    torchvision training archive is ordered by class, so "smallest index" means
    "lowest class id", and consecutive indices are usually the same track. A
    digest of the rule and the index keeps the choice fixed for a given rule
    while making it independent of class and of position within a track.
    """
    return blake2b(f"{rule}|{dataset_index}".encode(), digest_size=8).digest()


def select_examples(trajectories, transitions):
    """Pick the frames to render, by rules fixed before any image is looked at.

    Three rules, each tied to a question:

    onset        for each corruption and each onset severity, one clean-correct
                 frame that first fails there and stays wrong. Shows what
                 "fails at severity s" looks like.
    transition   for three strong conditions, the source-to-target pair
                 covering the most distinct tracks, then the most frames; three
                 frames of that pair from different tracks. Shows what a
                 dominant confusion actually looks like.
    unstable     frames that take the most distinct predictions across the five
                 severities. Shows prediction churn rather than a single switch.

    Within one figure and corruption every row comes from a different physical
    sign, so a grid cannot fill with near-duplicates, and candidates are ordered
    by `selection_key` rather than by how dramatic the image looks.
    """
    rows = []

    def take(candidates, limit, rule, used, **fields):
        ordered = candidates.assign(
            order=[selection_key(rule, index) for index in candidates["dataset_index"]]
        ).sort_values("order")
        taken = 0
        for _, row in ordered.iterrows():
            if row["track"] in used:
                continue
            used.add(row["track"])
            rows.append({
                "rule": rule,
                **fields,
                "row": taken,
                "corruption": row["corruption"],
                "validation_position": int(row["validation_position"]),
                "dataset_index": int(row["dataset_index"]),
                "track": row["track"],
                "label": int(row["label"]),
                "onset_severity": int(row["onset_severity"]),
                "distinct_predictions": int(row["distinct_predictions"]),
                **{
                    f"prediction_s{severity}": int(row[f"prediction_s{severity}"])
                    for severity in SEVERITY_LEVELS
                },
            })
            taken += 1
            if taken == limit:
                return
        raise RuntimeError(f"only {taken} of {limit} examples available for rule: {rule}")

    clean_correct = trajectories[trajectories["clean_correct"]]
    for name in EXAMPLE_CORRUPTIONS:
        group = clean_correct[clean_correct["corruption"] == name]
        used = set()
        for severity in SEVERITY_LEVELS:
            take(
                group[(group["onset_severity"] == severity) & group["persistent"]],
                1, f"{name}: first persistent failure at s{severity}", used,
                figure="onset", focus_severity=severity,
            )

    for name, severity in EXAMPLE_TRANSITIONS:
        pairs = transitions[
            (transitions["corruption"] == name) & (transitions["severity"] == severity)
        ].sort_values(["tracks", "frames"], ascending=False)
        source, target = int(pairs.iloc[0]["source"]), int(pairs.iloc[0]["target"])
        group = clean_correct[clean_correct["corruption"] == name]
        take(
            group[(group["label"] == source) & (group[f"prediction_s{severity}"] == target)],
            EXAMPLE_TRANSITION_ROWS,
            f"{name} s{severity}: {source}->{target}, the pair covering the most tracks",
            set(), figure="transition", focus_severity=severity,
        )

    for name in EXAMPLE_CORRUPTIONS:
        group = clean_correct[clean_correct["corruption"] == name]
        ceiling = group["distinct_predictions"].max()
        take(
            group[group["distinct_predictions"] == ceiling],
            EXAMPLE_UNSTABLE_ROWS,
            f"{name}: {ceiling} distinct predictions across the five severities",
            set(), figure="unstable", focus_severity=0,
        )
    return pd.DataFrame(rows)


def repository_state():
    def run(*command):
        return subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout
    status = run("git", "status", "--porcelain")
    return {
        "head_commit": run("git", "rev-parse", "HEAD").strip(),
        "dirty": bool(status.strip()),
        "modified_paths": [line[3:] for line in status.splitlines() if line],
    }


def write_csv(frame, path):
    frame.to_csv(path, index=False)
    return path


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    frame = read_predictions()
    reference, predictions, order_compared = condition_predictions(frame)
    dataset_indices = reference["dataset_index"].to_numpy()
    labels = reference["label"].to_numpy()
    tracks, all_labels, all_tracks, train_indices, validation_indices = track_identity(
        dataset_indices
    )

    checked, summary, per_class = check_accepted_tables(predictions, labels)
    accuracy, macro_f1 = metrics_from_confusion(
        confusion_of(predictions[(CLEAN_CONDITION, 0)], labels)
    )
    print("Inputs")
    print(f"  predictions.csv: {len(frame)} rows, {checked} conditions, "
          f"{EXPECTED_VALIDATION_SIZE} frames each")
    print(f"  recomputed clean accuracy {accuracy:.6f}, macro F1 {macro_f1:.6f} "
          "(matches the accepted baseline)")
    frames_per_track = sorted({int(count) for count in pd.Series(tracks).value_counts()})
    print(f"  validation tracks {len(set(tracks))}, frames per track {frames_per_track}")
    print(f"  summary.csv, per_class_metrics.csv and confusion_matrices.npz all rebuilt "
          f"from these predictions and matched to within {METRIC_TOLERANCE:g}")

    trajectories = build_trajectories(reference, predictions, tracks)
    onsets = onset_table(trajectories)
    totals = onset_totals_table(trajectories, onsets)
    induced = induced_table(trajectories)
    track_onsets = track_onset_table(trajectories)
    destinations = destination_table(trajectories)
    transitions = transition_table(trajectories)

    validation_tracks_per_class = tracks_per_class(all_labels, all_tracks, validation_indices)
    train_tracks_per_class = tracks_per_class(all_labels, all_tracks, train_indices)
    classes, degradation = class_tables(
        per_class, validation_tracks_per_class, train_tracks_per_class
    )

    anchors, offsets = occlusion_anchor_offsets(dataset_indices)
    anchor_rates, anchor_pairs, anchor_paired = occlusion_anchor_table(trajectories, offsets)
    images_read, boxes_checked = verify_anchor_recovery(dataset_indices, anchors)

    clean_correct, reconciliation_checks = reconcile(
        trajectories, onsets, totals, induced, destinations, transitions, summary
    )
    examples = select_examples(trajectories, transitions)

    claimable = classes[classes["claimable_support"]]
    rho, pvalue = spearmanr(
        classes["validation_tracks"], classes["mean_recall_drop_s5"]
    )

    written = {
        "trajectories": write_csv(trajectories, ANALYSIS_DIR / "failure_trajectories.csv"),
        "onset": write_csv(onsets, ANALYSIS_DIR / "failure_onset.csv"),
        "onset_totals": write_csv(totals, ANALYSIS_DIR / "failure_onset_totals.csv"),
        "induced": write_csv(induced, ANALYSIS_DIR / "induced_errors.csv"),
        "track_onset": write_csv(track_onsets, ANALYSIS_DIR / "track_majority_onset.csv"),
        "destinations": write_csv(destinations, ANALYSIS_DIR / "error_destinations.csv"),
        "transitions": write_csv(transitions, ANALYSIS_DIR / "error_transitions.csv"),
        "classes": write_csv(classes, ANALYSIS_DIR / "class_fragility.csv"),
        "degradation": write_csv(degradation, ANALYSIS_DIR / "class_degradation.csv"),
        "anchor": write_csv(anchor_rates, ANALYSIS_DIR / "occlusion_anchor_offset.csv"),
        "anchor_tracks": write_csv(
            anchor_pairs, ANALYSIS_DIR / "occlusion_anchor_within_track.csv"
        ),
        "examples": write_csv(examples, ANALYSIS_DIR / "examples.csv"),
    }

    report = {
        "source": {
            "predictions_csv": str(
                (RESULTS_DIR / "predictions.csv").relative_to(PROJECT_ROOT)
            ),
            "predictions_sha256": file_sha256(RESULTS_DIR / "predictions.csv"),
            "summary_sha256": file_sha256(RESULTS_DIR / "summary.csv"),
            "per_class_sha256": file_sha256(RESULTS_DIR / "per_class_metrics.csv"),
            "confusion_sha256": file_sha256(RESULTS_DIR / "confusion_matrices.npz"),
        },
        "inputs": {
            "conditions": checked,
            "frames": EXPECTED_VALIDATION_SIZE,
            "tracks": EXPECTED_TRACKS,
            "clean_correct_frames": clean_correct,
            "clean_accuracy": accuracy,
            "clean_macro_f1": macro_f1,
        },
        "definitions": {
            "induced_error": "clean-correct frame misclassified under the corruption",
            "onset_severity": "first severity at which a clean-correct frame is wrong",
            "persistent": "wrong at every severity from the onset upwards",
            "track_majority_onset": (
                f"first severity at which more than {TRACK_MAJORITY:g} of a track's "
                "clean-correct frames are wrong"
            ),
            "normalised_entropy": (
                "Shannon entropy of the predicted-class distribution among induced "
                "errors, divided by log(43); 1 is evenly spread over all classes, 0 is "
                "a single destination"
            ),
            "matched_concentration": (
                f"mean over {MATCHED_DRAWS} draws of {MATCHED_ERRORS} induced errors "
                "without replacement, so conditions with different error counts are "
                "comparable"
            ),
            "anchor_offset": (
                "Chebyshev distance in pixels of the occlusion anchor from the image "
                "centre, recovered from the frozen per-sample seed"
            ),
            "tracks_affected": (
                "distinct validation tracks with at least one clean-correct frame that "
                "fails at some severity, taken as a union over the severities rather "
                "than from any single onset bucket"
            ),
        },
        "example_rules": {
            "onset": (
                "for each corruption and each severity, one clean-correct frame whose "
                "first failure is at that severity and which stays wrong above it"
            ),
            "transition": (
                "one severity per corruption fixed in advance - fog and gaussian_noise "
                "at severity 5, their strongest level, and occlusion at severity 1, its "
                "mildest - then the source-to-target pair covering the most distinct "
                "tracks at that condition, shown on three different signs"
            ),
            "unstable": (
                "the frames taking the most distinct predictions across the five "
                "severities, two per corruption"
            ),
            "ordering": (
                "candidates are ordered by a blake2b digest of the rule and the dataset "
                "index, which fixes the choice without favouring low class ids, and no "
                "two rows of one figure and corruption come from the same track"
            ),
            "transition_conditions": {
                f"{name}_s{severity}": EXAMPLE_TRANSITION_ROLES[(name, severity)]
                for name, severity in EXAMPLE_TRANSITIONS
            },
        },
        "occlusion_anchor_paired": anchor_paired,
        "class_support_correlation": {
            "statistic": "spearman",
            "pairs": "validation track count against mean recall drop at severity 5",
            "rho": float(rho),
            "p_value": float(pvalue),
            "note": (
                "descriptive only: the 43 classes are not independent observations of "
                "difficulty, and validation track count is proportional to training "
                "track count, so support and learnability cannot be separated here"
            ),
        },
        "claimable_classes": int(len(claimable)),
        "single_track_classes": sorted(
            classes.loc[classes["single_validation_track"], "class"].tolist()
        ),
        # Counts rather than flags: each value is what a check actually
        # covered, so deleting the check that produces it changes the number
        # instead of leaving a `true` behind.
        "verification": {
            "conditions_matching_frozen_suite": len(predictions),
            "frames_per_condition": len(reference),
            "conditions_checked_against_the_reference_order": order_compared,
            "conditions_rebuilt_from_predictions": checked,
            "metric_tolerance": METRIC_TOLERANCE,
            "validation_tracks_from_split": len(set(tracks)),
            "reconciliation_equalities_checked": reconciliation_checks,
            "occlusion_anchor_boxes_verified": boxes_checked,
            "validation_images_read": images_read,
            "test_dataset_constructed": False,
            "test_images_read": 0,
            "model_inference_performed": False,
        },
        "interpretation_notes": {"occlusion": OCCLUSION_NOTE},
        "repository": repository_state(),
    }
    with open(ANALYSIS_DIR / "analysis_summary.json", "w") as handle:
        json.dump(report, handle, indent=2)

    print("\nFailure onset among the clean-correct frames (frames / tracks behind them)")
    print(f"{'corruption':18s} {'never':>13s} " + " ".join(f"{f's{s}':>13s}" for s in SEVERITY_LEVELS))
    for name in CORRUPTION_NAMES:
        rows = onsets[onsets["corruption"] == name].set_index("onset_severity")
        cells = " ".join(
            f"{rows.loc[s, 'frames']:>8d}/{rows.loc[s, 'tracks']:<4d}"
            for s in (0, *SEVERITY_LEVELS)
        )
        print(f"{name:18s} {cells}")
    print("  per-bucket track counts overlap; signs touched at any severity:")
    for _, row in totals.iterrows():
        print(f"    {row['corruption']:18s} {row['frames_ever_wrong']:>5d} frames across "
              f"{row['tracks_affected']:>3d} of {row['tracks_total']} signs "
              f"(largest single onset bucket {row['largest_onset_bucket_tracks']})")

    print(f"\nTracks whose clean-correct majority is wrong, by first severity "
          f"(of {EXPECTED_TRACKS})")
    print(f"{'corruption':18s} {'never':>7s} " + " ".join(f"{f's{s}':>7s}" for s in SEVERITY_LEVELS))
    for name in CORRUPTION_NAMES:
        rows = track_onsets[track_onsets["corruption"] == name].set_index(
            "majority_onset_severity"
        )
        cells = " ".join(f"{rows.loc[s, 'tracks']:>7d}" for s in (0, *SEVERITY_LEVELS))
        print(f"{name:18s} {cells}")

    print("\nNon-monotone trajectories and prediction churn among clean-correct frames")
    for name in CORRUPTION_NAMES:
        group = trajectories[
            (trajectories["corruption"] == name) & trajectories["clean_correct"]
        ]
        failed = group[group["onset_severity"] > 0]
        recovers = failed[~failed["persistent"]]
        churn = group[group["distinct_predictions"] >= 3]
        print(f"  {name:18s} failed {len(failed):5d} frames, of which {len(recovers):4d} "
              f"({recovers['track'].nunique():3d} tracks) are correct again at a stronger "
              f"severity; {len(churn):5d} frames take 3+ distinct predictions")

    print("\nWhere induced errors go")
    header = ("corruption", "sev", "errors", "trk", "dest", "top1", "top3", "H", "top3*", "H*")
    print("{:18s} {:>3s} {:>7s} {:>4s} {:>5s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s}".format(*header))
    for _, row in destinations.iterrows():
        matched_top3, matched_entropy = (
            "" if pd.isna(row[column]) else f"{row[column]:.3f}"
            for column in ("top3_share_matched", "normalised_entropy_matched")
        )
        print(f"{row['corruption']:18s} {row['severity']:>3d} {row['errors_induced']:>7d} "
              f"{row['induced_error_tracks']:>4d} {row['distinct_destinations']:>5d} "
              f"{row['top_destination_share']:>6.3f} {row['top3_share']:>6.3f} "
              f"{row['normalised_entropy']:>6.3f} {matched_top3:>6s} {matched_entropy:>6s}")
    print("  top1/top3 are shares of induced errors; H is normalised entropy; "
          f"* is the matched {MATCHED_ERRORS}-error draw")

    print("\nStrongest source -> target pairs by distinct tracks")
    for name in CORRUPTION_NAMES:
        for severity in (1, 5):
            pairs = transitions[
                (transitions["corruption"] == name) & (transitions["severity"] == severity)
            ].sort_values(["tracks", "frames"], ascending=False).head(3)
            listed = "; ".join(
                f"{int(p['source'])}->{int(p['target'])} {int(p['frames'])}f/{int(p['tracks'])}t"
                for _, p in pairs.iterrows()
            )
            print(f"  {name:18s} s{severity}  {listed}")

    print("\nClasses with the largest mean recall drop at severity 5")
    ranked = classes.sort_values("mean_recall_drop_s5", ascending=False).head(10)
    print(f"{'class':>5s} {'val trk':>7s} {'trn trk':>7s} {'clean':>7s} {'drop s5':>8s} "
          f"{'worst condition':>20s}")
    for _, row in ranked.iterrows():
        worst = f"{row['worst_corruption_s5']} {row['worst_recall_drop_s5']:.3f}"
        print(f"{row['class']:>5d} {row['validation_tracks']:>7d} {row['train_tracks']:>7d} "
              f"{row['clean_recall']:>7.3f} {row['mean_recall_drop_s5']:>8.3f} {worst:>20s}")
    perfect = classes[(classes["clean_recall"] >= 1.0) & classes["claimable_support"]]
    degrading = perfect.sort_values("mean_recall_drop_s5", ascending=False).head(5)
    print("  perfect clean recall with 3+ validation tracks, largest severity-5 drops: "
          + ", ".join(
              f"class {int(r['class'])} ({r['validation_tracks']} trk, "
              f"drop {r['mean_recall_drop_s5']:.3f})" for _, r in degrading.iterrows()
          ))
    print(f"  single-track classes (one physical sign each): "
          f"{report['single_track_classes']}")
    print(f"  validation track count against mean severity-5 drop: Spearman rho "
          f"{rho:+.3f} (p {pvalue:.3f}), descriptive only")

    print("\nOcclusion anchor offset against failure rate (clean-correct frames)")
    for severity in ANCHOR_SEVERITIES:
        rows = anchor_rates[anchor_rates["severity"] == severity]
        side = int(rows.iloc[0]["patch_side_pixels"])
        cells = " ".join(
            f"{int(r['offset_low'])}-{int(r['offset_high'])}px {r['failure_rate']:.3f} "
            f"({int(r['frames'])}f)"
            for _, r in rows.iterrows()
        )
        print(f"  s{severity} ({side}x{side} patch): {cells}")
        paired = anchor_paired[severity]
        print(f"     within-track, nearer-than-median minus farther: mean "
              f"{paired['mean_difference']:+.3f}, higher near the centre in "
              f"{paired['tracks_with_higher_failure_near_centre']}/{paired['tracks']} tracks")

    print(f"\nSelected {len(examples)} examples by fixed rule "
          f"({examples['figure'].nunique()} figures); see examples.csv")
    print("\nIntegrity")
    print(f"  no model inference; {images_read} validation images decoded, all inside the "
          "validation split")
    print(f"  {boxes_checked} occlusion anchor boxes verified exactly against the frozen "
          f"corruption, on {images_read} frames x {len(SEVERITY_LEVELS)} severities")
    print("  the official test split was not constructed and no test image was read")
    print("\nWritten:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    print(f"  summary: {ANALYSIS_DIR / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
