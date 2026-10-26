"""Paired, track-level analysis of the occlusion-location ablation.

    .venv/bin/python -m src.analyze_placement

The estimators live here rather than in a notebook so they are fixed before the
sweep runs, together with the geometry they consume. Nothing in this module
reads the model or the images: it works from `predictions.csv`, which carries
one row per frame per pass with the realised placement attached.

Units of analysis
-----------------
A validation frame is not an independent observation. Each of the 178 validation
tracks is 30 near-duplicate frames of one physical sign, so treating frames as
independent would overstate the evidence roughly thirtyfold, and treating the
nine uniform placements of a frame as independent would overstate it again. The
analysis therefore aggregates in a fixed order:

1. placements within a frame, by the protocol's weights, giving the frame's
   error rate under a policy;
2. frames within a track, equally weighted, giving the sign's error rate;
3. tracks, by resampling whole tracks, giving the uncertainty of a contrast.

Every pass covers all 5,340 frames, so contrasts are paired at frame level:
the same signs, frames, and preprocessing on both sides, differing only in
where the patch sits. The reported quantity is the mean over signs of the
within-sign difference, and its interval comes from a bootstrap over signs.

Two numbers describe variability, and they are not a decomposition into
independent parts. The bootstrap resamples the per-sign rates as observed, so
its interval covers everything those rates vary with: how much signs differ
from one another and, under the uniform policy, the placement draw each rate
was computed from. The placement-noise term bounds that second component on
its own - the uniform policy draws one position per stratified cell per frame
instead of evaluating all of them, so its estimate carries Monte-Carlo error
even with these signs held fixed. It says how much of the interval's width is
that error rather than adding to it, and the two must not be summed. The
centre, corner and ladder-rung conditions evaluate their full placement sets,
so for them the term is zero and the interval is sign sampling alone.

The reported contrasts are the ones fixed here before the sweep, and their
intervals are not adjusted for reporting several of them together.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate_placement import (
    EXPECTED_TRACKS,
    FRAMES_PER_TRACK,
    RESULTS_DIR,
    geometry_metadata,
)
from .occlusion_placement import (
    CLEAN_CONDITION,
    LADDER_DISTANCES,
    PATCH_SIDES,
    evaluation_conditions,
    ladder_table,
    uniform_weights,
)

ANALYSIS_DIR = RESULTS_DIR / "analysis"

# Policies whose per-frame rate is a weighted average over their own passes.
POLICIES = ("centre", "corner", "uniform")

# Paired contrasts reported per patch side, as (left, right): a positive mean
# difference means the left policy produces more errors.
CONTRASTS = (("centre", "corner"), ("centre", "uniform"), ("uniform", "corner"))

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 42
INTERVAL = (2.5, 97.5)

# Per-frame policy weights sum to 1 by construction; this allows only float
# rounding, not a missing or duplicated pass.
WEIGHT_TOLERANCE = 1e-9


def load_predictions(path):
    """Per-frame errors, split into the clean pass and the placement passes."""
    table = pd.read_csv(path)
    table["error"] = (table["prediction"] != table["label"]).astype(float)
    clean = table[table["policy"] == CLEAN_CONDITION].copy()
    placement = table[table["policy"] != CLEAN_CONDITION].copy()
    placement["side"] = placement["side"].astype(int)
    if len(clean) == 0:
        raise RuntimeError("predictions.csv holds no clean pass")
    names = set(table["condition"])
    expected = {condition.name for condition in evaluation_conditions()}
    if names != expected:
        raise RuntimeError(
            f"predictions.csv does not hold the protocol's {len(expected)} "
            f"conditions: missing {sorted(expected - names)}, "
            f"unexpected {sorted(names - expected)}")
    return clean, placement


def frame_rates(placement):
    """Weighted error rate per frame for each policy and patch side.

    The weights are the ones frozen in the protocol table: equal shares within
    the centre and corner sets, and the stratum weights within the uniform set.
    Summing `weight * error` over a policy's passes is therefore the policy's
    estimand for that frame, and the weight sum is checked rather than assumed.
    """
    selected = placement[placement["policy"].isin(POLICIES)].copy()
    selected["weighted"] = selected["weight"] * selected["error"]
    grouped = selected.groupby(["policy", "side", "track", "validation_position"],
                               as_index=False).agg(
        rate=("weighted", "sum"),
        weight=("weight", "sum"),
    )
    off = (grouped["weight"] - 1.0).abs().max()
    if off > WEIGHT_TOLERANCE:
        raise RuntimeError(f"per-frame weights sum to 1 only within {off:.2e}")
    return grouped


def ladder_assignment():
    """Which pass supplies each ladder rung, with its weight inside the rung.

    The extreme rungs are the centre and corner policies, so the ladder reads
    their passes instead of repeating them; the intermediate rungs have passes
    of their own. Each rung's placements are equally weighted because a rung is
    the set of diagonal positions at one distance, not a sample from a region.
    """
    lookup = {
        (condition.policy, condition.side, condition.coordinates): condition.name
        for condition in evaluation_conditions() if condition.coordinates is not None
    }
    rows = []
    for side, distance, rung, policy in ladder_table():
        for coordinates in rung:
            rows.append({
                "condition": lookup[(policy, side, coordinates)],
                "side": side,
                "centre_distance": distance,
                "rung_weight": 1 / len(rung),
            })
    return pd.DataFrame(rows)


def ladder_frame_rates(placement):
    """Weighted error rate per frame at each rung of the distance ladder."""
    assignment = ladder_assignment()
    merged = placement.merge(assignment, on=["condition", "side"], how="inner")
    if len(merged) != len(assignment) * placement["validation_position"].nunique():
        raise RuntimeError("ladder rungs do not cover every frame exactly once")
    merged["weighted"] = merged["rung_weight"] * merged["error"]
    grouped = merged.groupby(["side", "centre_distance", "track", "validation_position"],
                             as_index=False).agg(
        rate=("weighted", "sum"),
        weight=("rung_weight", "sum"),
    )
    off = (grouped["weight"] - 1.0).abs().max()
    if off > WEIGHT_TOLERANCE:
        raise RuntimeError(f"ladder rung weights sum to 1 only within {off:.2e}")
    return grouped


def track_rates(frames, keys):
    """Average a frame-level rate within each track, after checking the frames.

    Tracks are equally sized in this split, so a plain mean over a track's
    frames is also the mean over the split's frames; the check keeps that true
    rather than relying on it.
    """
    counts = frames.groupby(keys + ["track"], as_index=False).size()
    if set(counts["size"]) != {FRAMES_PER_TRACK}:
        raise RuntimeError(f"tracks hold {sorted(set(counts['size']))} frames, "
                           f"expected {FRAMES_PER_TRACK}")
    tracks = frames.groupby(keys + ["track"], as_index=False)["rate"].mean()
    per_condition = tracks.groupby(keys, as_index=False).size()
    if set(per_condition["size"]) != {EXPECTED_TRACKS}:
        raise RuntimeError(f"conditions cover {sorted(set(per_condition['size']))} "
                           f"tracks, expected {EXPECTED_TRACKS}")
    return tracks


def bootstrap_draws(count, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Resampling indices for the track bootstrap.

    Built once per call from a fixed seed, so every contrast of the same length
    is resampled on the same draws: the intervals of two contrasts then differ
    because the contrasts differ, not because their resampling did.
    """
    return np.random.default_rng(seed).integers(0, count, size=(draws, count))


def bootstrap_mean(values, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Mean of per-track values with a percentile interval over resampled tracks."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("expected a non-empty vector of per-track values")
    resampled = values[bootstrap_draws(values.size, draws, seed)].mean(axis=1)
    low, high = np.percentile(resampled, INTERVAL)
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "tracks": int(values.size),
        "draws": int(draws),
    }


def paired_contrast(tracks, keys, left, right):
    """Mean within-sign difference between two conditions, with its interval.

    The two sides are joined on the track, so the difference is taken within a
    sign before anything is averaged. A positive mean means `left` produces more
    errors than `right` on the same signs.
    """
    selected = tracks.set_index(keys + ["track"])["rate"]
    first = selected.xs(left, level=keys[0])
    second = selected.xs(right, level=keys[0])
    joined = pd.concat({"left": first, "right": second}, axis=1).dropna()
    if len(joined) != EXPECTED_TRACKS:
        raise RuntimeError(f"{left} vs {right} pairs on {len(joined)} tracks, "
                           f"expected {EXPECTED_TRACKS}")
    difference = joined["left"] - joined["right"]
    record = bootstrap_mean(difference.to_numpy())
    record.update({
        "left": left,
        "right": right,
        "left_mean": float(joined["left"].mean()),
        "right_mean": float(joined["right"].mean()),
        "tracks_worse_under_left": int((difference > 0).sum()),
        "tracks_unchanged": int((difference == 0).sum()),
    })
    return record


def placement_noise(side, cell_rates):
    """Bound the Monte-Carlo error the uniform policy's finite draws contribute.

    One position is drawn per stratified cell per frame, so a frame's uniform
    estimate is a weighted sum of nine Bernoulli outcomes, independent across
    cells and across frames. Conditional on the frame, that variance is
    `sum w^2 q(1 - q)` with `q` the share of the cell's positions that would
    misclassify this frame. Averaging 30 frames within a track and then 178
    tracks divides it by the number of independent draws, which is what keeps
    the term small next to the spread across signs - but it is reported rather
    than assumed away.

    Neither reported number is a plug-in estimate of the variance. The
    worst-case figure is a bound, from `q(1 - q) <= 1/4`. The pooled figure
    estimates a second bound from the observed cell rates: it substitutes the
    cell's error rate across frames for the per-frame `q`, and since `x(1 - x)`
    is concave the average of `q(1 - q)` cannot exceed the value at the pooled
    rate, so it overstates the noise by however much the frames differ. Because
    the pooled rate is itself measured from one draw per cell per frame it
    carries its own sampling error of about `sqrt(q(1 - q) / 5340)`, which is
    small next to that concavity slack but is why this is an estimated bound
    rather than a guaranteed one.
    Recovering the per-frame `q` would mean evaluating every position of every
    cell, which is the exhaustive sweep the stratified design replaces.
    """
    weights = uniform_weights(side)
    squared = sum(weight ** 2 for weight in weights.values())
    worst = 0.25 * squared
    pooled = sum(weights[cell] ** 2 * rate * (1 - rate)
                 for cell, rate in cell_rates.items())
    frames = EXPECTED_TRACKS * FRAMES_PER_TRACK
    return {
        "side": side,
        "sum_squared_weights": squared,
        "frame_sd_bound": worst ** 0.5,
        "frame_sd_pooled_bound": pooled ** 0.5,
        "track_sd_bound": (worst / FRAMES_PER_TRACK) ** 0.5,
        "track_sd_pooled_bound": (pooled / FRAMES_PER_TRACK) ** 0.5,
        "grand_mean_sd_bound": (worst / frames) ** 0.5,
        "grand_mean_sd_pooled_bound": (pooled / frames) ** 0.5,
    }


def uniform_cell_rates(placement, side):
    """Realised error rate of each stratified cell, for the noise estimate.

    The cell of a pass comes from the protocol table rather than from its name,
    so a renamed condition cannot quietly reassign a rate to the wrong cell.
    """
    cells = {condition.name: condition.cell for condition in evaluation_conditions()
             if condition.policy == "uniform" and condition.side == side}
    selected = placement[placement["condition"].isin(cells)]
    rates = {cells[name]: float(group["error"].mean())
             for name, group in selected.groupby("condition")}
    if set(rates) != set(uniform_weights(side)):
        raise RuntimeError(f"side {side} is missing uniform cells: "
                           f"{sorted(set(uniform_weights(side)) - set(rates))}")
    return rates


def write_table(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()

    results_dir = Path(arguments.results_dir) if arguments.results_dir else RESULTS_DIR
    output_dir = Path(arguments.output_dir) if arguments.output_dir else ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    clean, placement = load_predictions(results_dir / "predictions.csv")

    # The clean pass is the reference the placement rates sit on top of: a sign
    # that is already misclassified without a patch is not evidence about
    # placement, and the per-track clean rate is what makes that visible.
    clean_tracks = track_rates(clean.assign(rate=clean["error"]), ["policy"])
    clean_summary = bootstrap_mean(
        clean_tracks[clean_tracks["policy"] == CLEAN_CONDITION]["rate"].to_numpy())

    frames = frame_rates(placement)
    tracks = track_rates(frames, ["policy", "side"])
    ladder_tracks = track_rates(ladder_frame_rates(placement),
                                ["centre_distance", "side"])

    policy_rows = []
    for side in PATCH_SIDES:
        for policy in POLICIES:
            values = tracks[(tracks["policy"] == policy)
                            & (tracks["side"] == side)]["rate"].to_numpy()
            policy_rows.append({"side": side, "policy": policy,
                                **bootstrap_mean(values)})

    contrast_rows = []
    for side in PATCH_SIDES:
        side_tracks = tracks[tracks["side"] == side]
        for left, right in CONTRASTS:
            contrast_rows.append({
                "side": side,
                **paired_contrast(side_tracks, ["policy", "side"], left, right),
            })

    ladder_rows = []
    ladder_contrast_rows = []
    for side in PATCH_SIDES:
        side_tracks = ladder_tracks[ladder_tracks["side"] == side]
        distances = LADDER_DISTANCES[side]
        for distance in distances:
            values = side_tracks[side_tracks["centre_distance"] == distance]["rate"]
            ladder_rows.append({"side": side, "centre_distance": distance,
                                **bootstrap_mean(values.to_numpy())})
        # Each rung against the innermost one, so the ladder reads as a profile
        # from the centre outwards with paired intervals at every step.
        for distance in distances[1:]:
            contrast = paired_contrast(side_tracks, ["centre_distance", "side"],
                                       distances[0], distance)
            # The ladder's end rungs are the centre and corner policies, so the
            # outermost row restates the centre-minus-corner contrast rather
            # than adding evidence to it.
            contrast["repeats_policy_contrast"] = distance == distances[-1]
            contrast["inner_distance"] = contrast.pop("left")
            contrast["outer_distance"] = contrast.pop("right")
            contrast["inner_mean"] = contrast.pop("left_mean")
            contrast["outer_mean"] = contrast.pop("right_mean")
            contrast["tracks_worse_inner"] = contrast.pop("tracks_worse_under_left")
            ladder_contrast_rows.append({"side": side, **contrast})

    noise_rows = [placement_noise(side, uniform_cell_rates(placement, side))
                  for side in PATCH_SIDES]

    write_table(output_dir / "policy_rates.csv", policy_rows)
    write_table(output_dir / "paired_contrasts.csv", contrast_rows)
    write_table(output_dir / "ladder_rates.csv", ladder_rows)
    write_table(output_dir / "ladder_contrasts.csv", ladder_contrast_rows)
    write_table(output_dir / "placement_noise.csv", noise_rows)
    tracks.to_csv(output_dir / "track_rates.csv", index=False)
    ladder_tracks.to_csv(output_dir / "ladder_track_rates.csv", index=False)

    summary = {
        "estimands": {
            "frame": "weighted mean over a policy's placements of the error "
                     "indicator for that frame",
            "track": "unweighted mean over the sign's 30 frames",
            "contrast": "mean over the 178 validation signs of the within-sign "
                        "difference between two conditions",
            "interval": f"percentile {INTERVAL} over {BOOTSTRAP_DRAWS} bootstrap "
                        f"resamples of whole tracks, seed {BOOTSTRAP_SEED}",
            "placement_noise": "upper bound on the standard deviation "
                               "contributed by drawing one position per "
                               "stratified cell; a share of the interval's "
                               "width, not an addition to it",
            "multiplicity": "the contrasts were fixed before the sweep and the "
                            "intervals are not adjusted for reporting several "
                            "together",
        },
        "clean_reference": clean_summary,
        "geometry": geometry_metadata(),
        "policy_rates": policy_rows,
        "paired_contrasts": contrast_rows,
        "ladder_rates": ladder_rows,
        "ladder_contrasts": ladder_contrast_rows,
        "placement_noise": noise_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Clean reference error rate per sign: {clean_summary['mean']:.4f} "
          f"[{clean_summary['ci_low']:.4f}, {clean_summary['ci_high']:.4f}]")
    print("\nPolicy error rates (mean over signs, track bootstrap interval):")
    for row in policy_rows:
        print(f"  side {row['side']:>2d} {row['policy']:8s} {row['mean']:.4f} "
              f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
    print("\nPaired contrasts (mean within-sign difference):")
    for row in contrast_rows:
        print(f"  side {row['side']:>2d} {row['left']:8s} - {row['right']:8s} "
              f"{row['mean']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] "
              f"({row['tracks_worse_under_left']} of {row['tracks']} signs worse "
              f"under {row['left']}, {row['tracks_unchanged']} unchanged)")
    print("\nDistance ladder (Chebyshev distance of the patch centre):")
    for row in ladder_rows:
        print(f"  side {row['side']:>2d} d {row['centre_distance']:>4g} "
              f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
    print("\nFinite-placement noise of the uniform estimate:")
    for row in noise_rows:
        print(f"  side {row['side']:>2d} per-sign sd <= {row['track_sd_bound']:.4f} "
              f"(<= {row['track_sd_pooled_bound']:.4f} using the pooled cell "
              f"rates), grand mean sd <= {row['grand_mean_sd_bound']:.5f}")
    print(f"\nWritten to {output_dir}")


if __name__ == "__main__":
    main()
