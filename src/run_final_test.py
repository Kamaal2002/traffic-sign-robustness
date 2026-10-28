"""Run the eighteen official-test evaluations in a fixed order, one at a time.

    PYTHONPATH=. .venv/bin/python -m src.run_final_test
    PYTHONPATH=. .venv/bin/python -m src.run_final_test --execute

Without `--execute` this prints the plan and exits, having created nothing and
read nothing. With it, step 4 of Part 12 of the frozen protocol runs: nine
checkpoints by two suites, sequentially, each in its own process, each writing
its own result directory and appending one line to the access log.

Three properties the wrapper exists to provide.

**A predetermined order.** The sequence is the checkpoint inventory's own row
order - A42, A43, A44, B42, B43, B44, C42, C43, C44 - with the frozen suite
before the held-out one for each checkpoint. It is read from a committed file
rather than chosen here, and the whole plan is printed and validated before the
first run starts, so no decision about what to run next is taken while results
are arriving.

**No performance number on the console.** The evaluator prints progress and no
metric; this wrapper adds the run counter, the clock and an estimate, and reads
no metric from any result it produces. The measurements are in the artifact
sets, to be read once all eighteen runs have completed and been verified. A
sweep whose accuracy appeared on the terminal between runs would make stopping
early, rerunning a seed or substituting a checkpoint possible, which is the
degree of freedom the protocol closes by freezing the procedure first.

**A stop at the first technical failure.** A non-zero exit stops the sequence
where it happened. The failed run's log is kept, the access log keeps its
`failed` line, and nothing proceeds to the next checkpoint: a decision about
retrying is the operator's and is taken with the failure in front of them.
Once the failure is resolved, `--resume` runs the rest through the same checks
rather than by hand, skipping only the pairs the access log and the artifacts
agree are finished.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .evaluate_checkpoint import _repository_path
from .evaluate_robustness import file_md5
from .evaluate_test import (
    ACCESS_LOG,
    INVENTORY_PATH,
    PROTOCOL_READ_COUNTS,
    TEST_RESULTS_ROOT,
    check_protocol_identity,
    INCOMPLETE_MARKER,
    RESULT_FILES,
    file_sha256,
    format_duration,
    parse_arguments as parse_evaluator_arguments,
    results_dir,
)
from .train import select_device

# Each checkpoint's frozen suite before its held-out one. Both suites of one
# checkpoint run together so that a stop part-way leaves whole checkpoints
# done rather than a scatter of half-covered ones.
SUITE_ORDER = ("frozen", "heldout")

LOG_DIR = TEST_RESULTS_ROOT / "logs"

MODULE = "src.evaluate_test"


def inventory_models(inventory_path=None):
    """The nine approved checkpoints, in the order the inventory records them."""
    path = Path(inventory_path) if inventory_path else INVENTORY_PATH
    inventory = json.loads(path.read_text())
    models = inventory["models"]
    if len(models) != 9:
        raise RuntimeError(f"{path} records {len(models)} models, expected 9")
    return models


def plan_runs(device, output_root=None, log_dir=None, inventory_path=None):
    """The eighteen runs, as commands, derived from the inventory and the suites."""
    runs = []
    logs = Path(log_dir) if log_dir else LOG_DIR
    for model in inventory_models(inventory_path):
        for suite in SUITE_ORDER:
            arm = model["arm_label"]
            seed = int(model["seed"])
            target = results_dir(arm, seed, suite, output_root)
            run = {
                "position": len(runs) + 1,
                "model": f"{model['arm']}{seed}",
                "arm": model["arm"],
                "arm_label": arm,
                "seed": seed,
                "selected_epoch": int(model["selected_epoch"]),
                "suite": suite,
                "checkpoint": model["checkpoint_path"],
                "checkpoint_md5": model["checkpoint_md5"],
                "checkpoint_sha256": model["checkpoint_sha256"],
                "output_dir": target,
                "log": logs / f"{arm}_seed{seed}_{suite}.log",
                "device": device,
                "num_workers": 0,
            }
            command = [sys.executable, "-m", MODULE,
                       "--checkpoint", run["checkpoint"],
                       "--suite", suite,
                       "--model-name", arm,
                       "--device", device,
                       "--num-workers", "0"]
            if output_root is not None:
                # The evaluator's --output-dir is the root it derives
                # <arm>_seed<seed>/<suite> under, not the target directory, so
                # a redirected sequence passes the root and the default one
                # passes nothing at all.
                command += ["--output-dir", str(output_root)]
            run["command"] = command
            runs.append(run)
    return runs


def check_plan(runs, verify_digests=True, done=()):
    """Validate every command before the first one runs.

    Everything a command could get wrong that would not stop the run on its
    own: a checkpoint that is not one of the nine, an arm label that would file
    a result under the wrong arm, a repeated or missing (model, suite) pair, an
    output directory outside the test tree or one that already holds a result,
    a worker count that breaks the read accounting, or two runs sharing a
    directory or a log file.
    """
    if len(runs) != 18:
        raise RuntimeError(f"{len(runs)} runs planned, expected 18")

    records = {f"{model['arm']}{model['seed']}": model
               for model in inventory_models()}
    pairs = set()
    targets = set()
    logs = set()
    for run in runs:
        record = records.get(run["model"])
        if record is None:
            raise RuntimeError(f"run {run['position']} names {run['model']}, "
                               "which is not in the checkpoint inventory")
        if run["checkpoint"] != record["checkpoint_path"]:
            raise RuntimeError(
                f"run {run['position']} would read {run['checkpoint']} but "
                f"{run['model']} is {record['checkpoint_path']}")
        if run["arm_label"] != record["arm_label"]:
            raise RuntimeError(
                f"run {run['position']} would file {run['model']} under arm "
                f"{run['arm_label']!r}, not {record['arm_label']!r}")
        if run["seed"] != int(record["seed"]):
            raise RuntimeError(f"run {run['position']} seed {run['seed']} is "
                               f"not {record['seed']}")
        if run["suite"] not in PROTOCOL_READ_COUNTS:
            raise RuntimeError(f"run {run['position']} names suite "
                               f"{run['suite']!r}")
        if run["num_workers"] != 0:
            raise RuntimeError(f"run {run['position']} would use "
                               f"{run['num_workers']} workers, which breaks "
                               "the read accounting")

        checkpoint = Path(run["checkpoint"])
        if not checkpoint.is_file():
            raise RuntimeError(f"{checkpoint} does not exist")
        for algorithm in ("md5", "sha256"):
            if run[f"checkpoint_{algorithm}"] != \
                    record[f"checkpoint_{algorithm}"]:
                raise RuntimeError(
                    f"run {run['position']} carries a {algorithm} the "
                    f"inventory does not record for {run['model']}")
        if verify_digests:
            for algorithm, digest in (("md5", file_md5(checkpoint)),
                                      ("sha256", file_sha256(checkpoint))):
                if digest != run[f"checkpoint_{algorithm}"]:
                    raise RuntimeError(
                        f"{checkpoint} does not match the {algorithm} the "
                        f"inventory records for {run['model']}; it is not "
                        "the approved checkpoint")

        target = Path(run["output_dir"]).resolve()
        if run["position"] in done:
            # Already finished, and `completed_runs` is what established that:
            # a full artifact set and a completed access-log line. The run is
            # kept in the plan so the sequence still validates all eighteen
            # commands, and is not re-run.
            pass
        elif target.exists() and any(target.iterdir()):
            raise RuntimeError(
                f"{target} already holds results. The evaluator would refuse "
                "it; the sequence refuses it before the first run instead of "
                "at run "
                f"{run['position']}.")

        # The row above is what the plan prints and what step 5 of Part 12
        # will look for; the command is what actually runs. They are only the
        # same thing if the evaluator's own parser says so.
        parsed = parse_evaluator_arguments(run["command"][3:])
        derived = results_dir(parsed.model_name, run["seed"], parsed.suite,
                              parsed.output_dir)
        if derived != Path(run["output_dir"]):
            raise RuntimeError(
                f"run {run['position']} would write {derived} and the plan "
                f"says {run['output_dir']}")
        if (parsed.checkpoint, parsed.suite, parsed.model_name,
                parsed.device, parsed.num_workers, parsed.smoke_test) != (
                run["checkpoint"], run["suite"], run["arm_label"],
                run["device"], run["num_workers"], False):
            raise RuntimeError(f"run {run['position']}'s command does not "
                               "say what the plan's row says")

        pair = (run["model"], run["suite"])
        if pair in pairs:
            raise RuntimeError(f"{pair} is planned twice")
        pairs.add(pair)
        if target in targets:
            raise RuntimeError(f"{target} is the target of two runs")
        targets.add(target)
        log = Path(run["log"]).resolve()
        if log in logs:
            raise RuntimeError(f"{log} is the log of two runs")
        logs.add(log)

    expected = {(f"{model['arm']}{model['seed']}", suite)
                for model in inventory_models() for suite in SUITE_ORDER}
    if pairs != expected:
        raise RuntimeError(f"the plan covers {sorted(pairs)}, not the nine "
                           "checkpoints by two suites")
    devices = {run["device"] for run in runs}
    if len(devices) != 1:
        raise RuntimeError(f"the plan spans devices {devices}; all eighteen "
                           "runs report one device")
    return {"runs": len(runs), "models": len({run["model"] for run in runs}),
            "suites": sorted({run["suite"] for run in runs}),
            "device": devices.pop(), "digests_verified": verify_digests,
            "already_completed": len(done)}


def format_plan(runs, summary):
    """The plan as it is printed before anything runs."""
    lines = [f"{summary['runs']} runs: {summary['models']} checkpoints x "
             f"{len(summary['suites'])} suites "
             f"({', '.join(summary['suites'])}), sequentially, on "
             f"{summary['device']}",
             "Order: the checkpoint inventory's own row order, frozen suite "
             "before held-out for each checkpoint",
             ""]
    header = (f"{'#':>2s}  {'model':6s} {'arm':9s} {'seed':>4s} {'suite':8s} "
              f"{'checkpoint md5':32s} {'output directory':44s} "
              f"{'device':6s} {'workers':>7s}")
    lines += [header, "-" * len(header)]
    for run in runs:
        lines.append(
            f"{run['position']:2d}  {run['model']:6s} {run['arm_label']:9s} "
            f"{run['seed']:>4d} {run['suite']:8s} {run['checkpoint_md5']:32s} "
            f"{_repository_path(run['output_dir']):44s} {run['device']:6s} "
            f"{run['num_workers']:>7d}")
    lines.append("")
    for run in runs:
        lines.append(f"{run['position']:2d}  " + " ".join(run["command"][1:]))
    return "\n".join(lines)


def completed_log_entries(path=None):
    """The access log's completed lines, as entries."""
    path = Path(path) if path else ACCESS_LOG
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("status") == "completed":
            entries.append(entry)
    return entries


def completed_log_lines(path=None):
    """The access log's completed lines, as (arm, seed, suite) triples."""
    return [(entry.get("arm"), entry.get("seed"), entry.get("suite"))
            for entry in completed_log_entries(path)]


def _disagreement(run, found):
    """The fields of one record that do not say what the plan's row says.

    Returns the description of the disagreement, or None when the record
    agrees. The caller raises, so the message it builds stays a literal at
    the raise and nothing about it is assembled out of sight.
    """
    planned = {"checkpoint md5": run["checkpoint_md5"],
               "checkpoint sha256": run["checkpoint_sha256"],
               "arm": run["arm_label"], "seed": run["seed"],
               "suite": run["suite"],
               "output directory": _repository_path(run["output_dir"])}
    wrong = sorted(name for name, value in found.items()
                   if value != planned[name])
    if not wrong:
        return None
    return "; ".join(f"{name}: the plan says {planned[name]!r} and the "
                     f"record says {found[name]!r}" for name in wrong)


def completed_runs(runs, access_log=None):
    """The planned runs that are already finished, by records that all agree.

    A resumed sequence skips a run only when the access log holds exactly one
    completed line for its arm, seed and suite, the directory that line names
    is the one the plan derives, it holds the whole artifact set with no
    partial-write marker, and both the line and the result set's own metadata
    name the checkpoint the inventory approves for that model. Any record
    without its counterpart, or two records that disagree, is a state to
    resolve by hand rather than to skip: a logged read with no artifacts, an
    artifact set with no logged read and a result set written from a different
    checkpoint are different problems, and guessing which one happened is how
    an incomplete or misfiled sweep gets reported as a complete one.
    """
    entries = {}
    for entry in completed_log_entries(access_log):
        key = (entry.get("arm"), entry.get("seed"), entry.get("suite"))
        entries.setdefault(key, []).append(entry)
    finished = []
    for run in runs:
        logged = entries.get((run["arm_label"], run["seed"], run["suite"]))
        if not logged:
            continue
        if len(logged) > 1:
            raise RuntimeError(
                f"the access log records {run['model']} {run['suite']} as "
                f"completed {len(logged)} times, which is the repeated "
                "evaluation of one pair that Part 9 of the protocol counts; "
                "resolve that by hand before resuming")
        target = Path(run["output_dir"])
        missing = [name for name in RESULT_FILES
                   if not (target / name).exists()]
        if (target / INCOMPLETE_MARKER).exists():
            missing.append(INCOMPLETE_MARKER)
        if missing:
            raise RuntimeError(
                f"the access log records {run['model']} {run['suite']} as "
                f"completed but {_repository_path(target)} is missing "
                f"{missing}; resolve that by hand before resuming")

        entry = logged[0]
        detail = _disagreement(run, {
            "checkpoint md5": entry.get("checkpoint_md5"),
            "arm": entry.get("arm"), "seed": entry.get("seed"),
            "suite": entry.get("suite"),
            "output directory": entry.get("output_dir")})
        if detail:
            raise RuntimeError(
                f"the access log's completed line for {run['model']} "
                f"{run['suite']} does not agree with the plan's row on "
                f"{detail}. A resumed sequence skips a run only on records "
                "that agree, so resolve that by hand before resuming.")

        metadata = json.loads((target / "metadata.json").read_text())
        detail = _disagreement(run, {
            "checkpoint md5": metadata["checkpoint_digests"]["md5"],
            "checkpoint sha256": metadata["checkpoint_digests"]["sha256"],
            "arm": metadata["model"]["name"],
            "seed": metadata["model"]["seed"],
            "suite": metadata["evaluation"]["suite"]})
        if detail:
            raise RuntimeError(
                f"the metadata in {_repository_path(target)} does not agree "
                f"with the plan's row on {detail}. A resumed sequence skips "
                "a run only on records that agree, so resolve that by hand "
                "before resuming.")
        if metadata["evaluation"]["split"] != "official_test":
            raise RuntimeError(
                f"{_repository_path(target)} holds a result for the "
                f"{metadata['evaluation']['split']} split, not the official "
                "test split; resolve that by hand before resuming")
        finished.append(run["position"])
    return finished


def run_one(run):
    """One evaluation, in its own process, with its output kept.

    The child's output is written to the run's log and echoed as it arrives, so
    a long sweep is visible without the log being the only copy. The child
    prints no measurement, so echoing it cannot put one on the terminal.

    The log is appended to, never truncated. A retried pair's earlier attempt
    is the one artifact that explains why it failed, and Part 12 step 4 keeps
    logs per run; each attempt is headed by its own timestamp and command.
    """
    run["log"].parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(run["log"], "a") as handle:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        handle.write(f"{stamp} " + " ".join(run["command"]) + "\n\n")
        handle.flush()
        process = subprocess.Popen(run["command"], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(f"    | {line.rstrip()}")
        code = process.wait()
    return {"returncode": code, "seconds": time.time() - started}


def execute(runs, access_log=None, total=None):
    """The eighteen runs, stopping at the first that fails.

    After each run the access log is required to have gained exactly one
    completed line for that checkpoint and suite. A run that produced a result
    set without recording the read, or recorded a read without completing,
    stops the sequence in the same way a crash does.
    """
    started = time.time()
    before = len(completed_log_lines(access_log))
    total = total or len(runs)
    already = total - len(runs)
    durations = []
    for run in runs:
        print(f"\n[{run['position']}/{total}] {run['model']} "
              f"{run['suite']}: {_repository_path(run['checkpoint'])} -> "
              f"{_repository_path(run['output_dir'])}")
        outcome = run_one(run)
        if outcome["returncode"] != 0:
            print(f"\nRun {run['position']} ({run['model']} {run['suite']}) "
                  f"exited {outcome['returncode']} after "
                  f"{format_duration(outcome['seconds'])}. The sequence stops "
                  f"here with {already + len(durations)} of {total} runs "
                  "completed.")
            print(f"Its output is in {_repository_path(run['log'])} and its "
                  f"attempt is in {_repository_path(ACCESS_LOG)}. Nothing "
                  "proceeds to the next checkpoint; whether to retry is a "
                  "decision to take with the failure in front of you.")
            return {"completed": already + len(durations),
                    "failed": run["position"],
                    "returncode": outcome["returncode"],
                    "seconds": time.time() - started}

        triples = completed_log_lines(access_log)
        expected = (run["arm_label"], run["seed"], run["suite"])
        gained = triples[before + len(durations):]
        if gained != [expected]:
            raise RuntimeError(
                f"run {run['position']} exited 0 but the access log gained "
                f"{gained} rather than exactly one completed line for "
                f"{expected}; the sequence stops rather than treating it as "
                "a result")
        durations.append(outcome["seconds"])
        elapsed = time.time() - started
        remaining = (len(runs) - len(durations)) * elapsed / len(durations)

        print(f"    done in {format_duration(outcome['seconds'])} | elapsed "
              f"{format_duration(elapsed)} | ETA "
              f"{format_duration(remaining)}")

    print(f"\nAll {total} runs completed, {len(durations)} of them here, in "
          f"{format_duration(time.time() - started)}. No measurement was "
          "printed here; the results are in the eighteen directories under "
          f"{_repository_path(TEST_RESULTS_ROOT)} and are read once the "
          "post-run verification of Part 12 steps 5 and 6 has passed.")
    return {"completed": already + len(durations), "failed": None,
            "returncode": 0, "seconds": time.time() - started}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="run the eighteen evaluations; without it the "
                             "plan is printed and nothing is run or created")
    parser.add_argument("--device", default=None,
                        help="resolved once and passed to all eighteen runs")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="skip the pairs the access log and the artifacts "
                             "agree are finished, and run the rest; without "
                             "it a sequence that has already written a result "
                             "is refused")
    parser.add_argument("--skip-digests", action="store_true",
                        help="skip the checkpoint md5 recomputation in the "
                             "plan check; the evaluator verifies it per run")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    if arguments.resume and arguments.output_root:
        raise SystemExit(
            "--resume reads the real access log, whose completed lines name "
            "directories under the default results root, so it does not "
            "combine with --output-root.")
    device = arguments.device or str(select_device(None))
    runs = plan_runs(device, arguments.output_root, arguments.log_dir)
    done = completed_runs(runs) if arguments.resume else []
    summary = check_plan(runs, verify_digests=not arguments.skip_digests,
                         done=done)

    protocol = check_protocol_identity()
    print(f"Protocol {protocol['final_test_protocol']} sha256 "
          f"{protocol['final_test_protocol_sha256'][:16]} (frozen)")
    print(format_plan(runs, summary))

    if not arguments.execute:
        print(f"\nPlan only. {summary['runs']} commands validated"
              + (", every checkpoint md5 and sha256 recomputed and matched"
                 if summary["digests_verified"] else "")
              + ". Nothing was run, nothing was created, and no official test "
                "image was read. Pass --execute to run the sequence.")
        return 0

    remaining = [run for run in runs if run["position"] not in done]
    if not remaining:
        print(f"\nAll {summary['runs']} pairs are already recorded as "
              "completed with a full artifact set and agreeing records. "
              "Nothing was run. The results are read once the post-run "
              "verification of Part 12 steps 5 and 6 has passed.")
        return 0
    if done:
        print(f"\nResuming: {len(done)} of {summary['runs']} pairs are "
              "already recorded as completed with a full artifact set and are "
              f"not re-run. {len(remaining)} to go, from run "
              f"{remaining[0]['position']}.")
    outcome = execute(remaining, total=summary["runs"])
    return 0 if outcome["returncode"] == 0 and outcome["failed"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
