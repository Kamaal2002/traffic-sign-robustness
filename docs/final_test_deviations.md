# Deviations from the frozen final-test protocol

`docs/final_test_protocol.md` was frozen in commit `f16c8a5`
(sha256 `41df8eef10ae5d736d203245bfce1156c0ef57c51ee41e0574ba6a5e8eff07ef`) and
is not edited. Where the implementation departs from it, or where the document
contradicts itself, the departure is recorded here instead, with the date it
was recorded and the reasoning behind it. Every entry below was written before
the official test evaluation of Part 12 step 4 was run, so none of them was
chosen with a test number in view.

## 1. The access log records failed attempts, and carries a `status` field

**Recorded 2026-10-28, before step 4.**

Part 9 gives the access-log schema as "one JSON object per line, written once
per completed run", lists its fields, and gives its verification as: "the log
must hold exactly 18 lines, one per `(arm, seed, suite)` triple with no triple
repeated … Any extra line is a repeated test evaluation and is reported as
such rather than deleted."

`src/evaluate_test.py` adds a `status` field, which that field list does
not contain, and appends a line for an attempt that failed as well as for a
run that completed. So the log can hold more than 18 lines without any test
evaluation having been repeated.

The reason is that the read has already happened by the time most failures can
occur. A run that sweeps all 31 conditions of the frozen suite, reading the
404,581 images Part 9 budgets for it, and then fails while writing its outputs
has spent that look at the test set whether or not it produced a result,
and a log that recorded only completed runs would understate the number of
times the official split was read — which is the one thing the log exists to
state. Discarding the line would also lose the only record of the attempt,
since a failed run writes no artifact set.

The verification of step 5 therefore applies this amended rule:

* exactly 18 lines with `"status": "completed"`, one per `(arm, seed, suite)`
  triple, with no triple repeated;
* any number of lines with `"status": "failed"`, each corresponding to a run
  that stopped the sequence and none of them counted as a result. A failed
  line names a planned pair unless the run stopped before the inventory had
  named one, in which case it names no arm and no seed and must record
  `"test_images_read": 0`, since the pair is resolved from the inventory
  before the first image is read;
* a repeated `completed` triple is what Part 9 means by a repeated test
  evaluation, and is reported as such rather than deleted.

This is what `check_run_record` in `src/verify_test_path.py --post-run`
enforces. Both the amended rule and the code that applies it were committed
before step 4 ran.

## 2. The clean trade-off's reference arm

**Recorded 2026-10-28, before step 4.**

Part 7 describes the clean-accuracy trade-off twice and the two descriptions
disagree. The paragraph at lines 484–486 states that
`compare_models.clean_tradeoff_rows` measures each contrast's clean cost
against "that contrast's own right-hand arm" at the same seed, "not against arm
A", and spells out that the cost on the C−B row is B's clean accuracy minus
C's. The reporting list at lines 532–533 instead describes the same table as
holding "the cost against arm A at the same seed". The two agree on the B−A
and C−A rows, whose right-hand arm is A, and disagree only on C−B.

The implementation follows the right-hand arm. That is what the named function
does, the function is imported rather than reimplemented and produced the
committed validation table, the paragraph that names it is specific about the
C−B row, and Part 1's fourth question asks it the same way. Reading it the
other way would make the test table's trade-off column incomparable with
the validation one, which is the reason the paragraph gives for the
convention.

The limit of 0.01 is a reporting rule under either reading, and the decision
rule does not read this column, so nothing about the study's conclusion turns
on it. `CLEAN_COST_REFERENCE_NOTE` in `src/compare_test.py` travels with both
the rendered report and `comparison.json`, so the interpretation is stated
beside every number it affects.
