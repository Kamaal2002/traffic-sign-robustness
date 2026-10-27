"""Paired, sign-level comparison of the three augmentation arms.

    .venv/bin/python -m src.compare_models \
        --model clean:42:artifacts/robustness/clean_seed42 \
        --model balanced:42:artifacts/robustness/balanced_seed42 \
        --model targeted:42:artifacts/robustness/targeted_seed42 \
        --model clean:43:artifacts/robustness/clean_seed43 ...

Each `--model` is `NAME:SEED:DIR`, where `DIR` is the directory
`evaluate_checkpoint.py` wrote for that checkpoint and holds one subdirectory
per suite.

Arms
----
A `clean`      the original recipe, no augmentation
B `balanced`   each of the six training families with probability 1/6
C `targeted`   the weighted family distribution, occlusion-heavy

Both suites are co-primary. The frozen suite says whether a policy helps on the
conditions it was built around; the held-out suite says whether anything
transferred. A policy that wins on one and not the other has not been shown to
improve robustness, and this module labels that outcome rather than picking the
flattering half.

Units of analysis
-----------------
The estimators are imported from `analyze_placement.py` rather than rewritten,
because the aggregation order they encode is the same one this comparison needs
and was fixed before any augmented model existed:

1. a frame is right or wrong;
2. frames average within a track, giving one error rate per physical sign;
3. conditions average with equal weight, giving one number per sign per model;
4. whole tracks are resampled, giving the interval.

Frames are never resampled. Each of the 178 validation tracks is 30
near-duplicate frames of one sign, so a frame bootstrap would overstate the
evidence roughly thirtyfold. Steps 2 and 3 commute here, since every condition
covers every track with exactly 30 frames, but they are applied in the declared
order so the code states the protocol rather than an equivalent shortcut.

What the interval does and does not cover
----------------------------------------
The bootstrap resamples signs. Its interval therefore describes how much a
contrast would move if the 178 validation signs had been a different sample of
signs - nothing else. It says nothing about training-seed variation: two runs
of the same policy differ by initialisation and augmentation draws, and that
variation appears here only as the spread of the three per-seed point
estimates, reported as a range and a direction count. A paired interval that
excludes zero at one seed is not evidence that the policy helps in general, and
no claim of statistical significance across seeds is made or supported by
anything in this file.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_placement import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    INTERVAL,
    bootstrap_mean,
    paired_contrast,
    track_rates,
    write_table,
)
from .corruptions import build_validation_observations
from .corruptions_heldout import HELDOUT_CORRUPTIONS
from .data import NUM_CLASSES, read_training_index
from .evaluate_checkpoint import SUITES
from .evaluate_placement import EXPECTED_TRACKS, FRAMES_PER_TRACK
from .evaluate_robustness import CLEAN_CONDITION, PROJECT_ROOT

ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / "analysis" / "augmentation"

# The three arms, in the order the report discusses them.
ARMS = ("clean", "balanced", "targeted")
ARM_LABELS = {"clean": "A", "balanced": "B", "targeted": "C"}

# The contrasts declared before the comparison was run. Each is (left, right)
# and reported as left minus right, so a negative mean means the left arm makes
# fewer errors on the same signs.
CONTRASTS = (("balanced", "clean"), ("targeted", "clean"),
             ("targeted", "balanced"))

# A robustness gain that costs more than one percentage point of clean accuracy
# is reported as a trade-off rather than as an improvement, whatever the
# corrupted numbers say. Fixed here, not chosen after seeing the results.
CLEAN_TRADEOFF_LIMIT = 0.01


def parse_model(specification):
    """Parse one `NAME:SEED:DIR` argument."""
    parts = specification.split(":")
    if len(parts) != 3:
        raise ValueError(f"expected NAME:SEED:DIR, got {specification!r}")
    name, seed, directory = parts
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; expected one of {list(ARMS)}")
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")
    return {"name": name, "seed": int(seed), "dir": directory,
            "key": f"{name}_seed{seed}"}


def validation_track_index():
    """Map every validation dataset index to its track and label.

    The track identifiers come from `data.py`'s own reader, through
    `corruptions.py`'s observation builder, so this is the same notion of a
    track the split was constructed on rather than a second parse of the
    filenames.
    """
    observations, validation_indices, labels = build_validation_observations()
    _, tracks = read_training_index(observations.dataset)
    rows = [{"dataset_index": int(index), "track": tracks[int(index)],
             "label": int(labels[int(index)])} for index in validation_indices]
    frame = pd.DataFrame(rows)
    if frame["track"].nunique() != EXPECTED_TRACKS:
        raise RuntimeError(f"validation split holds {frame['track'].nunique()} "
                           f"tracks, expected {EXPECTED_TRACKS}")
    return frame


def load_suite_frames(model, suite_name, track_index):
    """Frame-level errors for one model under one suite, with signs attached.

    The labels the evaluator recorded are checked against the labels the split
    itself reports for the same dataset indices, so a prediction file written
    against a different split cannot be silently folded into the comparison.
    """
    predictions_path = model["dir"] / suite_name / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"no predictions at {predictions_path}")
    frames = pd.read_csv(predictions_path)
    frames = frames.merge(
        track_index.rename(columns={"label": "split_label"}),
        on="dataset_index", how="left", validate="many_to_one")
    if frames["track"].isna().any():
        missing = int(frames["track"].isna().sum())
        raise RuntimeError(f"{predictions_path} holds {missing} rows whose "
                           "dataset index is not in the validation split")
    disagreements = int((frames["label"] != frames["split_label"]).sum())
    if disagreements:
        raise RuntimeError(f"{predictions_path} disagrees with the split's "
                           f"labels on {disagreements} rows")
    frames["rate"] = (frames["prediction"] != frames["label"]).astype(float)
    frames["model"] = model["name"]
    frames["seed"] = model["seed"]
    frames["suite"] = suite_name
    return frames


def check_identical_examples(models, suite_name):
    """Confirm every arm was measured on the same inputs under this suite.

    The comparison is paired at sign level, which is only meaningful if the
    corrupted pixels were identical across arms. The evaluator recorded a hash
    of the input tensors per condition for exactly this check, so it is verified
    from the artifacts rather than argued from the seeding scheme.
    """
    if not models:
        raise ValueError("no models to compare")
    reference = None
    for model in models:
        metadata_path = model["dir"] / suite_name / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        fingerprints = metadata["corruption_suite"]["input_fingerprints"]
        if reference is None:
            reference = (model["key"], fingerprints)
            continue
        name, expected = reference
        if fingerprints != expected:
            differing = sorted(key for key in expected
                              if fingerprints.get(key) != expected[key])
            raise RuntimeError(
                f"{model['key']} and {name} were evaluated on different inputs "
                f"under the {suite_name} suite ({len(differing)} conditions "
                f"differ, first {differing[:3]}); the arms are not paired")
    return {"conditions_matched": len(reference[1]), "reference": reference[0]}


# Every identity field a result set must record for its predictions to be
# attributable to a model. A directory missing any of them is rejected rather
# than trusted: the arm and seed on the command line are labels, and the whole
# comparison is a set of claims about which model produced which errors.
REQUIRED_IDENTITY = (("model", "name"), ("model", "seed"), ("model", "arm"),
                     ("model", "training_records", "best_epoch"),
                     ("checkpoint", "path"), ("checkpoint", "md5"),
                     ("checkpoint", "epoch"), ("checkpoint", "seed"))


def _recorded(metadata, keys, source):
    """Read one required identity field, or say which one is missing."""
    value = metadata
    for key in keys:
        if not isinstance(value, dict) or value.get(key) is None:
            raise RuntimeError(
                f"{source} records no {'.'.join(keys)}, so the predictions in "
                "that directory cannot be attributed to a model; re-evaluate "
                "the checkpoint with the current evaluator")
        value = value[key]
    return value


def read_recorded_identity(model, suite_name):
    """The model identity one result set records for itself.

    Read from the metadata the evaluator wrote beside the predictions, which is
    the only place the identity of the model that produced them exists. The
    command line supplies a label and a directory; nothing in a `NAME:SEED:DIR`
    string is evidence about what is in the directory.
    """
    metadata_path = model["dir"] / suite_name / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"no metadata at {metadata_path}, so the model "
                           "behind those predictions is unidentified")
    metadata = json.loads(metadata_path.read_text())

    def read(*keys):
        return _recorded(metadata, keys, metadata_path)

    return {
        "suite": suite_name,
        "source": str(metadata_path),
        "recorded_name": str(read("model", "name")),
        "recorded_seed": int(read("model", "seed")),
        "recorded_arm": str(read("model", "arm")),
        "recorded_best_epoch": int(read("model", "training_records", "best_epoch")),
        "checkpoint_path": str(read("checkpoint", "path")),
        "checkpoint_md5": str(read("checkpoint", "md5")),
        "checkpoint_epoch": int(read("checkpoint", "epoch")),
        "checkpoint_seed": int(read("checkpoint", "seed")),
        "provenance": (metadata["model"].get("provenance")),
        "protocol": (metadata["model"].get("protocol") or {}).get("protocol"),
    }


def verify_model_identities(models):
    """Check every supplied directory holds the arm and seed it was supplied as.

    The input fingerprints checked elsewhere establish that the arms were shown
    the same pixels. They say nothing about which checkpoint produced the
    predictions: two result sets with identical fingerprints are still two
    anonymous prediction files. So each directory is made to identify itself,
    and five things have to hold before any number is computed.

    * the arm name recorded by the evaluator matches the arm requested on the
      command line, so swapping two directories is rejected rather than
      silently relabelling B as C;
    * the training seed recorded for the model and inside the checkpoint both
      match the requested seed, so a replicate cannot be filed under another
      seed's results;
    * the best epoch the training run recorded matches the epoch in the
      checkpoint that was evaluated;
    * the two suites of one model were produced by the same checkpoint file and
      the same md5, so a model cannot be half one run and half another;
    * no two distinct arm-and-seed keys share a checkpoint md5, so one
      checkpoint cannot appear twice under two identities and be contrasted
      with itself.

    The arm string is checked too, because it is derived from the training run's
    own policy record rather than from the `--model-name` label: a result set
    filed as `clean` whose metadata says `augmented (targeted)` is a
    mislabelling even though both fields are internally consistent.
    """
    if not models:
        raise ValueError("no models to compare")
    verified = {}
    by_md5 = {}
    for model in models:
        recorded = [read_recorded_identity(model, suite_name)
                    for suite_name in sorted(SUITES)]
        for entry in recorded:
            if entry["recorded_name"] != model["name"]:
                raise RuntimeError(
                    f"{entry['source']} holds results for arm "
                    f"{entry['recorded_name']!r} but {model['dir']} was "
                    f"supplied as {model['name']!r}; the directories are "
                    "mismatched and relabelling them is not an option")
            if entry["recorded_seed"] != model["seed"]:
                raise RuntimeError(
                    f"{entry['source']} holds results for seed "
                    f"{entry['recorded_seed']} but {model['dir']} was supplied "
                    f"as seed {model['seed']}")
            if entry["checkpoint_seed"] != entry["recorded_seed"]:
                raise RuntimeError(
                    f"{entry['source']} files its results under seed "
                    f"{entry['recorded_seed']} but the checkpoint it evaluated "
                    f"was trained with seed {entry['checkpoint_seed']}")
            expected_arm = ("clean baseline" if model["name"] == "clean"
                            else f"augmented ({model['name']})")
            if entry["recorded_arm"] != expected_arm:
                raise RuntimeError(
                    f"{entry['source']} describes its model as "
                    f"{entry['recorded_arm']!r}, which is not the "
                    f"{expected_arm!r} it was supplied as")
            if entry["recorded_best_epoch"] != entry["checkpoint_epoch"]:
                raise RuntimeError(
                    f"{entry['source']} evaluated epoch "
                    f"{entry['checkpoint_epoch']} but the training run it "
                    f"names selected epoch {entry['recorded_best_epoch']}")
        first = recorded[0]
        for entry in recorded[1:]:
            for field in ("checkpoint_md5", "checkpoint_path",
                          "checkpoint_epoch", "recorded_name", "recorded_seed"):
                if entry[field] != first[field]:
                    raise RuntimeError(
                        f"{model['key']} has {field} {first[field]!r} under the "
                        f"{first['suite']} suite and {entry[field]!r} under the "
                        f"{entry['suite']} suite; the two suites did not "
                        "measure the same checkpoint and cannot be reported as "
                        "one model")
        owner = by_md5.setdefault(first["checkpoint_md5"], model["key"])
        if owner != model["key"]:
            raise RuntimeError(
                f"{model['key']} and {owner} both evaluated the checkpoint with "
                f"md5 {first['checkpoint_md5']}; one checkpoint cannot stand as "
                "two model runs, and contrasting it with itself would report a "
                "difference of zero as a result")
        verified[model["key"]] = {
            "name": model["name"],
            "seed": model["seed"],
            "dir": str(model["dir"]),
            "checkpoint_path": first["checkpoint_path"],
            "checkpoint_md5": first["checkpoint_md5"],
            "checkpoint_epoch": first["checkpoint_epoch"],
            "recorded_arm": first["recorded_arm"],
            "suites_verified": [entry["suite"] for entry in recorded],
            "training_provenance": first["provenance"],
            # The protocol version the training run recorded for itself, or
            # None for a run that recorded none. Carried into the comparison
            # output so a reported result names the protocol it was produced
            # under; it is a record written beside the checkpoint, not a proof
            # of which code produced it.
            "training_protocol": first["protocol"],
        }
    return verified


def condition_track_rates(frames):
    """Per-condition, per-sign error rates, in the declared aggregation order.

    `track_rates` enforces the invariants the protocol depends on: exactly 30
    frames behind every (model, condition, sign) rate, and exactly 178 signs
    behind every condition.
    """
    keys = ["model", "corruption", "severity"]
    return track_rates(frames, keys)


def primary_rates(frames):
    """One mean per-sign error rate per arm: the primary metric.

    Conditions are averaged with equal weight after the within-sign average, so
    a family with five severities contributes as much as any other family and no
    condition's sample size can tilt the metric. The clean condition is excluded
    - it is reported separately, as the cost side of the trade-off, not as a
    thirty-first corruption.
    """
    corrupted = frames[frames["corruption"] != CLEAN_CONDITION]
    per_condition = condition_track_rates(corrupted)
    rates = per_condition.groupby(["model", "track"], as_index=False)["rate"].mean()
    per_model = rates.groupby("model", as_index=False).size()
    if set(per_model["size"]) != {EXPECTED_TRACKS}:
        raise RuntimeError(f"arms cover {sorted(set(per_model['size']))} signs, "
                           f"expected {EXPECTED_TRACKS}")
    return rates


def sign_classes(frames):
    """The class of every physical sign, taken from its own frames' labels."""
    per_sign = frames.groupby("track")["label"].agg(["nunique", "first"])
    inconsistent = per_sign.index[per_sign["nunique"] != 1].tolist()
    if inconsistent:
        raise RuntimeError(f"signs {inconsistent[:5]} carry more than one "
                           "class; the split's tracks are not single signs")
    return per_sign["first"].astype(int)


def _class_layout(classes):
    """Signs ordered by class, with the segment boundaries the averaging uses."""
    signs = pd.DataFrame({"track": classes.index.to_numpy(),
                          "class": classes.to_numpy()})
    signs = signs.sort_values(["class", "track"], ignore_index=True)
    if signs["class"].nunique() != NUM_CLASSES:
        raise RuntimeError(f"the validation signs cover "
                           f"{signs['class'].nunique()} classes, expected "
                           f"{NUM_CLASSES}")
    counts = signs.groupby("class").size().to_numpy()
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return signs, starts, counts


def _class_balanced(values, columns, starts, counts):
    """Average signs within classes, then the classes, then the conditions.

    `values` is one row per condition and one column per sign, already averaged
    within each sign. `columns` selects the signs, in class order, so a
    resampled draw is passed in exactly where the identity selection would be.
    """
    selected = values[:, columns]
    class_means = np.add.reduceat(selected, starts, axis=1) / counts
    return float(class_means.mean(axis=1).mean())


def _within_class_draws(starts, counts, draws=BOOTSTRAP_DRAWS,
                        seed=BOOTSTRAP_SEED):
    """Resampling indices that keep each class's sign count fixed.

    Signs are resampled inside their own class, so all 43 class weights stay
    equal across draws and the interval describes the sampling of signs within
    classes. The same draws are reused on both sides of a contrast, so the
    difference stays paired sign by sign.
    """
    low = np.repeat(starts, counts)
    high = low + np.repeat(counts, counts)
    return np.random.default_rng(seed).integers(low, high,
                                                size=(draws, low.size))


def class_balanced_rows(frames, seed, suite_name):
    """Secondary class-balanced per-sign error, per arm, with paired contrasts.

    The two co-primary outcomes weight every physical sign equally, so the
    classes that contribute more validation signs contribute more to them. This
    secondary metric weights the 43 classes equally instead, in the declared
    order:

    1. a frame is right or wrong;
    2. frames average within a physical sign;
    3. sign error rates average within a class;
    4. the 43 class means average with equal weight;
    5. conditions average with equal weight, the clean condition excluded.

    It answers a different question - whether a policy helps across the sign
    types rather than across the validation signs - and it answers it weakly.
    Some validation classes hold a single physical sign, so their class mean is
    one sign's error rate, and resampling inside such a class returns that same
    sign every time. The interval therefore understates how much this metric
    would move on another sample of signs, and the metric is not independent
    confirmation from 43 well-sampled classes. It is reported alongside the
    primary outcomes, never in place of them, and nothing selects a checkpoint
    or decides the experiment on it.
    """
    corrupted = frames[frames["corruption"] != CLEAN_CONDITION]
    per_condition = condition_track_rates(corrupted)
    classes = sign_classes(corrupted)
    signs, starts, counts = _class_layout(classes)
    columns = signs["track"].to_numpy()
    single_sign_classes = int((counts == 1).sum())

    values, condition_order = {}, None
    for model_name, group in per_condition.groupby("model"):
        pivot = group.pivot(index=["corruption", "severity"], columns="track",
                            values="rate")
        if pivot.isna().to_numpy().any():
            raise RuntimeError(f"{model_name} is missing signs under some "
                               f"condition of the {suite_name} suite")
        if condition_order is None:
            condition_order = list(pivot.index)
        elif list(pivot.index) != condition_order:
            raise RuntimeError(f"{model_name} covers different conditions than "
                               f"the other arms under {suite_name}")
        values[model_name] = pivot[columns].to_numpy(dtype=float)

    # The same five steps written as successive groupby means, to check the
    # segmented average the bootstrap relies on rather than trust it.
    attached = per_condition.merge(classes.rename("class").reset_index(),
                                   on="track", validate="many_to_one")
    checked = (attached
               .groupby(["model", "corruption", "severity", "class"],
                        as_index=False)["rate"].mean()
               .groupby(["model", "corruption", "severity"],
                        as_index=False)["rate"].mean()
               .groupby("model")["rate"].mean())

    identity = np.arange(len(columns))
    draws = _within_class_draws(starts, counts)
    arm_table, contrast_table, points = [], [], {}
    for model_name, matrix in sorted(values.items()):
        point = _class_balanced(matrix, identity, starts, counts)
        independent = float(checked.loc[model_name])
        if abs(point - independent) > 1e-12:
            raise RuntimeError(
                f"the class-balanced error for {model_name} is {point!r} by "
                f"segmented average and {independent!r} by grouped means")
        resampled = np.array([_class_balanced(matrix, draw, starts, counts)
                              for draw in draws])
        low, high = np.percentile(resampled, INTERVAL)
        points[model_name] = point
        arm_table.append({
            "suite": suite_name,
            "seed": seed,
            "arm": ARM_LABELS[model_name],
            "model": model_name,
            "class_balanced_error": point,
            "ci_low": float(low),
            "ci_high": float(high),
            "classes": NUM_CLASSES,
            "signs": len(columns),
            "single_sign_classes": single_sign_classes,
        })

    for left, right in CONTRASTS:
        if left not in values or right not in values:
            continue
        # Differenced per sign and per condition before anything is averaged,
        # so the contrast is paired; the chain above is linear, so this equals
        # the difference of the two arms' class-balanced errors, and that is
        # checked rather than assumed.
        delta = values[left] - values[right]
        point = _class_balanced(delta, identity, starts, counts)
        if abs(point - (points[left] - points[right])) > 1e-12:
            raise RuntimeError(
                f"the paired class-balanced difference {left} - {right} is "
                f"{point!r} but the two arm estimates differ by "
                f"{points[left] - points[right]!r}")
        resampled = np.array([_class_balanced(delta, draw, starts, counts)
                              for draw in draws])
        low, high = np.percentile(resampled, INTERVAL)
        contrast_table.append({
            "suite": suite_name,
            "seed": seed,
            "contrast": f"{ARM_LABELS[left]}-{ARM_LABELS[right]}",
            "left": left,
            "right": right,
            "mean_difference": point,
            "ci_low": float(low),
            "ci_high": float(high),
            "classes": NUM_CLASSES,
            "single_sign_classes": single_sign_classes,
        })
    return arm_table, contrast_table


def contrast_rows(rates, seed, suite_name, available):
    """Every declared contrast between the arms present at one seed."""
    rows = []
    for left, right in CONTRASTS:
        if left not in available or right not in available:
            continue
        record = paired_contrast(rates, ["model"], left, right)
        rows.append({
            "suite": suite_name,
            "seed": seed,
            "contrast": f"{ARM_LABELS[left]}-{ARM_LABELS[right]}",
            "left": left,
            "right": right,
            "left_mean_error": record["left_mean"],
            "right_mean_error": record["right_mean"],
            "difference": record["mean"],
            "ci_low": record["ci_low"],
            "ci_high": record["ci_high"],
            "excludes_zero": bool(record["ci_low"] > 0 or record["ci_high"] < 0),
            "signs_worse_under_left": record["tracks_worse_under_left"],
            "signs_unchanged": record["tracks_unchanged"],
            "signs": record["tracks"],
            "draws": record["draws"],
        })
    return rows


def breakdown_rows(frames, seed, suite_name, grouping):
    """Per-family or per-severity contrasts, aggregated the same way.

    Same protocol as the primary metric with a narrower set of conditions, so
    these are decompositions of the headline number rather than a separate
    analysis. There are 5, 6, or 11 of them per suite per seed and their
    intervals carry no multiplicity adjustment, so they are read as structure in
    the result, not as a list of independent findings.
    """
    corrupted = frames[frames["corruption"] != CLEAN_CONDITION]
    per_condition = condition_track_rates(corrupted)
    rows = []
    for value, group in per_condition.groupby(grouping):
        rates = group.groupby(["model", "track"], as_index=False)["rate"].mean()
        available = set(rates["model"])
        for left, right in CONTRASTS:
            if left not in available or right not in available:
                continue
            record = paired_contrast(rates, ["model"], left, right)
            rows.append({
                "suite": suite_name,
                "seed": seed,
                grouping: value,
                "contrast": f"{ARM_LABELS[left]}-{ARM_LABELS[right]}",
                "left_mean_error": record["left_mean"],
                "right_mean_error": record["right_mean"],
                "difference": record["mean"],
                "ci_low": record["ci_low"],
                "ci_high": record["ci_high"],
                "excludes_zero": bool(record["ci_low"] > 0 or record["ci_high"] < 0),
                "signs_worse_under_left": record["tracks_worse_under_left"],
            })
    return rows


def arm_rows(rates, seed, suite_name):
    """The mean per-sign error rate of each arm, with its own interval."""
    rows = []
    for model, group in rates.groupby("model"):
        record = bootstrap_mean(group["rate"].to_numpy())
        rows.append({
            "suite": suite_name,
            "seed": seed,
            "arm": ARM_LABELS[model],
            "model": model,
            "mean_sign_error": record["mean"],
            "ci_low": record["ci_low"],
            "ci_high": record["ci_high"],
            "signs": record["tracks"],
            "draws": record["draws"],
        })
    return rows


def clean_rows(models):
    """Clean accuracy and macro F1 per arm, read from the evaluator's summary.

    Taken from the frozen suite's summary, where the clean condition is the
    unmodified validation pass; both suites evaluate the identical clean
    condition, so the choice of suite here does not affect the number.
    """
    rows = []
    for model in models:
        summary = pd.read_csv(model["dir"] / "frozen" / "summary.csv")
        clean = summary[summary["corruption"] == CLEAN_CONDITION]
        if len(clean) != 1:
            raise RuntimeError(f"{model['key']} has {len(clean)} clean rows")
        clean = clean.iloc[0]
        rows.append({
            "arm": ARM_LABELS[model["name"]],
            "model": model["name"],
            "seed": model["seed"],
            "clean_accuracy": float(clean["accuracy"]),
            "clean_macro_f1": float(clean["macro_f1"]),
            "clean_loss": float(clean["loss"]),
        })
    return rows


def clean_tradeoff_rows(clean, contrasts):
    """Pair each arm's clean cost with its corrupted gain, under the fixed rule.

    A policy that buys corrupted accuracy with more than `CLEAN_TRADEOFF_LIMIT`
    of clean accuracy is labelled a trade-off, not an improvement. The threshold
    was fixed before the results existed so that an arm cannot be promoted by
    choosing a looser one afterwards.
    """
    clean_by_key = {(row["model"], row["seed"]): row for row in clean}
    rows = []
    for row in contrasts:
        left, right, seed = row["left"], row["right"], row["seed"]
        if (left, seed) not in clean_by_key or (right, seed) not in clean_by_key:
            continue
        clean_cost = (clean_by_key[(right, seed)]["clean_accuracy"]
                      - clean_by_key[(left, seed)]["clean_accuracy"])
        rows.append({
            "suite": row["suite"],
            "seed": seed,
            "contrast": row["contrast"],
            "clean_accuracy_cost": clean_cost,
            "corrupted_error_change": row["difference"],
            "exceeds_clean_limit": bool(clean_cost > CLEAN_TRADEOFF_LIMIT),
            "verdict": _tradeoff_verdict(clean_cost, row["difference"]),
        })
    return rows


def _tradeoff_verdict(clean_cost, error_change):
    """Label one arm's clean cost against its corrupted gain."""
    if error_change >= 0:
        return "no corrupted gain"
    if clean_cost > CLEAN_TRADEOFF_LIMIT:
        return (f"trade-off: corrupted error falls but clean accuracy costs "
                f"{clean_cost:.4f}, over the {CLEAN_TRADEOFF_LIMIT:g} limit")
    return "improvement within the clean-accuracy limit"


def across_seeds(contrasts):
    """Summarise each contrast across training seeds.

    Three seeds support a direction and a range, and nothing more. The summary
    therefore reports the mean of the per-seed point estimates, their spread,
    how many seeds pointed each way, and a label - never a p-value, an interval
    across seeds, or a claim of significance, none of which three runs can
    carry.
    """
    grouped = defaultdict(list)
    for row in contrasts:
        grouped[(row["suite"], row["contrast"])].append(row)
    rows = []
    for (suite_name, contrast), group in sorted(grouped.items()):
        differences = np.array([row["difference"] for row in group], dtype=float)
        negative = int((differences < 0).sum())
        positive = int((differences > 0).sum())
        rows.append({
            "suite": suite_name,
            "contrast": contrast,
            "seeds": len(group),
            "seed_list": " ".join(str(row["seed"]) for row in sorted(
                group, key=lambda item: item["seed"])),
            "mean_difference": float(differences.mean()),
            "min_difference": float(differences.min()),
            "max_difference": float(differences.max()),
            "range": float(differences.max() - differences.min()),
            "seeds_favouring_left": negative,
            "seeds_favouring_right": positive,
            "paired_intervals_excluding_zero": sum(
                1 for row in group if row["excludes_zero"]),
            "direction": _direction(negative, positive, len(group)),
        })
    return rows


def _direction(negative, positive, seeds):
    """Label the directional consistency of one contrast across seeds."""
    if seeds == 0:
        return "no seeds"
    if negative == seeds:
        return "consistent: left lower at every seed"
    if positive == seeds:
        return "consistent: right lower at every seed"
    return (f"seed-sensitive: {negative} of {seeds} seeds favour the left arm, "
            "so the direction is inconclusive")


def targeted_verdict(summaries, seeds):
    """Apply the declared decision rule for C against B.

    The rule, fixed with the design: C is preferred over B only if its mean
    per-sign error is lower at every training seed, and - because the whole
    point of the held-out suite is that C's weights were derived from the frozen
    benchmark's own headroom - that has to hold on both suites. Winning on the
    frozen suite alone is the outcome the design predicted as a risk, and it is
    reported as such rather than as a win.
    """
    relevant = {row["suite"]: row for row in summaries
                if row["contrast"] == "C-B"}
    missing = sorted(set(SUITES) - set(relevant))
    if missing:
        return {"verdict": "incomplete",
                "detail": f"no C-B contrast for the {', '.join(missing)} suite"}
    consistent = {suite: row["seeds_favouring_left"] == row["seeds"]
                  for suite, row in relevant.items()}
    partial = [suite for suite, value in consistent.items() if not value]
    if not partial:
        return {
            "verdict": "targeted preferred",
            "detail": (f"C's mean per-sign error is lower than B's at all "
                       f"{seeds} seeds on both suites"),
        }
    if len(partial) == len(consistent):
        return {"verdict": "no advantage for the targeted weights",
                "detail": "C is not consistently lower than B on either suite"}
    won = sorted(set(consistent) - set(partial))
    return {
        "verdict": "frozen-suite only, not a general improvement",
        "detail": (f"C is consistently lower than B on the {', '.join(won)} "
                   f"suite but not on the {', '.join(partial)} suite. C's "
                   "family weights were derived from where the frozen benchmark "
                   "had the most headroom, so a gain confined to that suite is "
                   "consistent with having matched the benchmark rather than "
                   "having improved robustness."),
    }


def per_class_rows(models):
    """Mean per-class recall under corruption, per arm and suite.

    Recall is averaged over the corrupted conditions of a suite. The track count
    travels with every row because ten validation classes hold a single sign, so
    a recall change for one of those classes is an observation about that sign.
    """
    rows = []
    for model in models:
        for suite_name in sorted(SUITES):
            path = model["dir"] / suite_name / "per_class_metrics.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            corrupted = frame[frame["corruption"] != CLEAN_CONDITION]
            clean = frame[frame["corruption"] == CLEAN_CONDITION]
            grouped = corrupted.groupby("class", as_index=False).agg(
                corrupted_recall=("recall", "mean"),
                tracks=("tracks", "first"))
            clean_recall = clean.set_index("class")["recall"]
            for record in grouped.to_dict("records"):
                rows.append({
                    "suite": suite_name,
                    "arm": ARM_LABELS[model["name"]],
                    "model": model["name"],
                    "seed": model["seed"],
                    "class": int(record["class"]),
                    "validation_tracks": int(record["tracks"]),
                    "clean_recall": float(clean_recall.loc[record["class"]]),
                    "mean_corrupted_recall": float(record["corrupted_recall"]),
                })
    return rows


def format_report(arm_table, contrast_table, seed_table, clean_table,
                  tradeoff_table, verdict, balanced_table=(),
                  balanced_contrasts=()):
    """Render the tables the write-up quotes."""
    lines = []
    lines.append("Mean per-sign error rate under corruption, by arm")
    header = (f"{'suite':9s} {'seed':>5s} {'arm':>3s} {'model':10s} "
              f"{'error':>8s} {'95% CI':>18s}")
    lines += [header, "-" * len(header)]
    for row in arm_table:
        interval = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        lines.append(f"{row['suite']:9s} {row['seed']:>5d} {row['arm']:>3s} "
                     f"{row['model']:10s} {row['mean_sign_error']:>8.4f} "
                     f"{interval:>18s}")

    lines.append("")
    lines.append("Paired within-sign contrasts, per seed "
                 "(negative favours the left arm)")
    header = (f"{'suite':9s} {'seed':>5s} {'contrast':>8s} {'difference':>11s} "
              f"{'95% CI':>20s} {'signs worse':>12s}")
    lines += [header, "-" * len(header)]
    for row in contrast_table:
        interval = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        lines.append(f"{row['suite']:9s} {row['seed']:>5d} {row['contrast']:>8s} "
                     f"{row['difference']:>+11.4f} {interval:>20s} "
                     f"{row['signs_worse_under_left']:>12d}")
    lines.append("Intervals come from resampling the 178 validation signs. They "
                 "describe sign sampling only, not training-seed variation.")

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
                 "across training seeds is claimed.")

    lines.append("")
    lines.append("Clean performance and the trade-off")
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

    if balanced_table:
        lines.append("")
        lines.append("Secondary: class-balanced per-sign error, all 43 classes "
                     "weighted equally")
        header = (f"{'suite':9s} {'seed':>5s} {'arm':>3s} {'model':10s} "
                  f"{'error':>8s} {'95% CI':>18s}")
        lines += [header, "-" * len(header)]
        for row in balanced_table:
            interval = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            lines.append(f"{row['suite']:9s} {row['seed']:>5d} {row['arm']:>3s} "
                         f"{row['model']:10s} "
                         f"{row['class_balanced_error']:>8.4f} {interval:>18s}")
        for row in balanced_contrasts:
            lines.append(f"  {row['suite']:9s} seed {row['seed']} "
                         f"{row['contrast']}: "
                         f"{row['mean_difference']:+.4f} "
                         f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]")
        single = balanced_table[0]["single_sign_classes"]
        lines.append(f"This is a secondary metric. {single} of "
                     f"{balanced_table[0]['classes']} validation classes hold a "
                     "single physical sign, so resampling within those classes "
                     "returns the same sign and the interval understates the "
                     "sign-sampling uncertainty. It is not independent "
                     "confirmation from 43 well-sampled classes, and no "
                     "checkpoint or verdict depends on it.")

    lines.append("")
    lines.append(f"Targeted-versus-balanced rule: {verdict['verdict']}")
    lines.append(f"  {verdict['detail']}")
    return "\n".join(lines)


def compare(models, output_dir):
    """Run every declared comparison and write the tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Before any number is computed: every directory has to show that it holds
    # the arm and the seed it was supplied as.
    identities = verify_model_identities(models)
    print("Verified model identities, from each result set's own metadata:")
    for key, record in sorted(identities.items()):
        print(f"  {key:16s} {record['recorded_arm']:22s} epoch "
              f"{record['checkpoint_epoch']:>3d}  md5 "
              f"{record['checkpoint_md5']} ({record['checkpoint_path']})")
    print()

    track_index = validation_track_index()

    by_seed = defaultdict(list)
    for model in models:
        by_seed[model["seed"]].append(model)

    arm_table, contrast_table = [], []
    family_table, severity_table = [], []
    balanced_table, balanced_contrasts = [], []
    pairing = {}
    for suite_name in sorted(SUITES):
        # Checked over every arm and seed at once, not within a seed: the
        # corrupted realisations depend on the dataset index alone, so all nine
        # checkpoints must have been shown the same pixels, and that is the
        # claim worth verifying.
        pairing[suite_name] = check_identical_examples(models, suite_name)
        for seed, group in sorted(by_seed.items()):
            frames = pd.concat(
                [load_suite_frames(model, suite_name, track_index)
                 for model in group],
                ignore_index=True)
            rates = primary_rates(frames)
            arm_table += arm_rows(rates, seed, suite_name)
            rows = contrast_rows(rates, seed, suite_name,
                                 available=set(rates["model"]))
            contrast_table += rows
            family_table += breakdown_rows(frames, seed, suite_name, "corruption")
            severity_table += breakdown_rows(frames, seed, suite_name, "severity")
            balanced_arms, balanced_pairs = class_balanced_rows(
                frames, seed, suite_name)
            balanced_table += balanced_arms
            balanced_contrasts += balanced_pairs

    seed_table = across_seeds(contrast_table)
    clean_table = clean_rows(models)
    tradeoff_table = clean_tradeoff_rows(clean_table, contrast_table)
    verdict = targeted_verdict(seed_table, len(by_seed))
    class_table = per_class_rows(models)

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
    }
    written = []
    for name, rows in tables.items():
        if not rows:
            continue
        write_table(output_dir / name, rows)
        written.append(output_dir / name)

    report = format_report(arm_table, contrast_table, seed_table, clean_table,
                           tradeoff_table, verdict, balanced_table,
                           balanced_contrasts)
    (output_dir / "comparison.txt").write_text(report + "\n")
    (output_dir / "comparison.json").write_text(json.dumps({
        "models": [{"name": model["name"], "seed": model["seed"],
                    "dir": str(model["dir"])} for model in models],
        "model_identity": identities,
        "protocol": {
            "primary_metric": "mean per-sign error rate, conditions equally "
                              "weighted, clean excluded",
            "aggregation": ["frame error", "mean over 30 frames per sign",
                            "equal mean over conditions",
                            "bootstrap over whole signs"],
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interval": list(INTERVAL),
            "signs": EXPECTED_TRACKS,
            "frames_per_sign": FRAMES_PER_TRACK,
            "frames_resampled": False,
            "clean_accuracy_limit": CLEAN_TRADEOFF_LIMIT,
            "suites": sorted(SUITES),
            "suites_are_co_primary": True,
            "held_out_families": sorted(HELDOUT_CORRUPTIONS),
            "secondary_metric": {
                "name": "class-balanced mean per-sign error",
                "aggregation": ["frame error", "mean over 30 frames per sign",
                                "mean over the signs of a class",
                                "equal mean over the 43 classes",
                                "equal mean over conditions"],
                "bootstrap": "signs resampled within their class, class counts "
                             "fixed",
                "limitation": "classes holding a single validation sign "
                              "contribute no sign-sampling variation, so this "
                              "interval understates uncertainty and the metric "
                              "is not independent confirmation across 43 "
                              "well-sampled classes",
                "used_for_checkpoint_selection": False,
                "used_for_the_success_criterion": False,
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
        "targeted_versus_balanced": verdict,
    }, indent=2) + "\n")
    written += [output_dir / "comparison.txt", output_dir / "comparison.json"]

    print(report)
    print("\nWritten:")
    for path in written:
        print(f"  {path}")
    return {"report": report, "verdict": verdict, "paths": written}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True,
                        metavar="NAME:SEED:DIR",
                        help="one evaluated checkpoint; repeat for each arm "
                             "and seed")
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()

    models = [parse_model(specification) for specification in arguments.model]
    keys = [model["key"] for model in models]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate arm and seed: {sorted(keys)}")
    output_dir = Path(arguments.output_dir) if arguments.output_dir else ANALYSIS_DIR
    compare(models, output_dir)


if __name__ == "__main__":
    main()
