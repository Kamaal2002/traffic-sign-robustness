# Controlled augmentation protocol

Kamaal Alag

This document freezes the experiment that the modules `augmentation.py`,
`corruptions_heldout.py`, `train_augmented.py`, `train_clean.py`,
`evaluate_checkpoint.py` and `compare_models.py` implement. Every number below
is the value those modules actually use; nothing here is a plan or an intention.
No arm has been trained and no comparison has been run at the time of freezing,
so this document contains no results.

## Research question

How robust are traffic-sign recognition models to adverse visual conditions,
and can targeted augmentation improve their robustness?

The experiment answers the second half of that question by changing one thing
about the accepted baseline recipe — the distribution the training images are
drawn from — and measuring the effect on corrupted-input error.

## The three arms

| Arm | Policy name | Training distribution |
|---|---|---|
| A | none | the clean recipe, unchanged |
| B | `balanced` | uniform over the six benchmark corruption families |
| C | `targeted` | reweighted towards the families the baseline fails on |

Arm A at seed 42 is the accepted baseline checkpoint in `artifacts/baseline`,
reused exactly as it is rather than retrained. Its replicates at seeds 43 and
44 are trained by `train_clean.py`, which calls the same `run_experiment` with
no loaders so that `data.py` builds the original clean pipeline. B and C are
trained by `train_augmented.py`, which also calls `run_experiment` rather than
reimplementing it, so the only difference between an arm and the baseline is
that its training images pass through an augmentation policy.

Neither B nor C is warm-started from the baseline checkpoint: each run starts
from a fresh initialisation under its own seed. The comparison has nine
arm-and-seed cells, of which eight are trained here and the ninth is the
accepted seed-42 baseline.

## Training configuration

Inherited from `train.py` without modification, which is the recipe that
produced the accepted baseline:

| Setting | Value |
|---|---|
| Model | `BaselineCNN`, four conv-BN-ReLU-pool blocks of 32/64/128/256 channels, a 256-unit classifier, 1,448,779 parameters |
| Input | 64x64 RGB, `Normalize(mean=0.5, std=0.5)` per channel |
| Loss | cross entropy |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4 |
| Schedule | cosine annealing over the epoch budget |
| Dropout | 0.5 in the classifier |
| Epochs | 30 |
| Batch size | 64 |
| Split | track-level, 710 training tracks / 21,300 images, 178 validation tracks / 5,340 images, from 888 tracks of 30 frames |
| Classes | 43 |

The split is track-level, so no physical sign contributes frames to both sides.
`train_augmented.py` asserts the two image counts before training and refuses
to continue if either differs, because an arm trained on a different number of
images would not be comparable to the baseline.

## Augmentation policies

`P(augment) = 0.50` in both augmented arms. An augmented image receives exactly
one corruption family at one severity; the other half of each epoch's images are
left clean. The draws are per image and per epoch, so these are expected shares
rather than fixed counts.

Family weights, as implemented in `augmentation.py`:

| Family | B (`balanced`) | C (`targeted`) |
|---|---|---|
| `gaussian_blur` | 1/6 | 0.06 |
| `brightness_dark` | 1/6 | 0.06 |
| `brightness_bright` | 1/6 | 0.09 |
| `gaussian_noise` | 1/6 | 0.22 |
| `fog` | 1/6 | 0.15 |
| `occlusion` | 1/6 | 0.42 |

C's weights were derived from the committed baseline robustness curve by a rule
fixed before training: each family's mean clean-to-corrupted accuracy drop over
severities 1-5 (occlusion 0.4133, gaussian_noise 0.1897, fog 0.1051,
brightness_bright 0.0430, brightness_dark 0.0155, gaussian_blur 0.0088) is
normalised to a probability, a 5% floor is applied per family, the remainder is
distributed proportionally, and the result is rounded to two decimals with the
rounding residual taken by fog. The floor keeps C a reweighting of B's support
rather than a subset of it.

B's uniform weights give the illumination axis a third of the mass, because
underexposure and overexposure are two separate families in the frozen table.

### Severity distribution

Uniform over severities 1-5, weight 0.20 each, in **both** arms. Severity is
drawn before the family, and the severity distribution does not depend on the
family, so two arms sharing a training seed augment the same images at the same
severities and differ only in which family lands on each one. That is what
makes the B-versus-C contrast a contrast in family weights alone.

Severity parameters are read from `corruptions.py` at the same five values the
benchmark measures. The augmentation module defines no corruption of its own.

### Augmentation seed derivation

    base seed | "train" | epoch | stable dataset index

digested with BLAKE2b, exactly as `corruptions.corruption_seed` does, under a
namespace no evaluation condition uses. The payload holds no batch position, no
shuffle position and no worker id, so what an image receives is a function of
(seed, epoch, image) and nothing else. The base seed is the training seed.

### Training occlusion placement

The patch side and the mid-grey fill come from the frozen definitions: area
fractions (0.05, 0.1, 0.15, 0.2, 0.3), which at 64x64 round to sides of 14, 20,
25, 29 and 35 pixels, filled with `OCCLUSION_FILL = 0.5`, which the frozen
normalisation maps to exactly zero.

The position does not. Training placement is `uniform_whole_frame`: the
top-left corner is drawn uniformly over every position where the patch fits, so
the patch can reach the frame border at every severity. The benchmark instead
draws one anchor per image and nests all five patches around it, which confines
every patch to the central window that leaves room for the severity-5 patch.
Copying the benchmark's placement into training would make the training and
evaluation occlusion distributions identical and a gain on the benchmark's
occlusion conditions would then be unreadable. Uniform whole-frame placement is
a strict superset of the central window.

The cost of that choice is stated rather than hidden: a uniformly placed patch
lands on sign content less often than a centred one, so the share of genuinely
hard occlusion samples is lower than the raw count of occluded images.
Training patches are not nested across severities, because nesting exists in
the benchmark to isolate patch area from patch-position luck along a per-sample
severity curve, and training has no such curve to read.

## Training seeds

42, 43, 44 — three replicates per arm, nine arm-and-seed cells in total. The
seed sets the
model initialisation, the loader shuffle and the augmentation stream, so the
three replicates of an arm draw genuinely independent augmentation streams.
Three seeds measure the spread of training noise; they do not support a claim
of statistical significance across seeds, and none is made.

## Checkpoint selection

The epoch with the highest **clean** validation macro F1
(`selection_metric = "validation_macro_f1"`) is kept, which is the baseline's
rule unchanged. Corrupted validation performance is never computed during
training and therefore cannot influence which epoch is kept. The augmented
arms wrap the training split only; the validation loader carries the unmodified
clean pipeline. The class-balanced secondary metric defined below is not used
for selection either.

## Evaluation suites

Both suites are evaluated on the 5,340 clean-split validation frames (178
signs x 30 frames) and share one clean condition.

**Frozen suite** — the six training families from `corruptions.py` at five
severities, 30 corrupted conditions plus clean:

| Family | Parameter | Severities 1-5 |
|---|---|---|
| `gaussian_blur` | sigma | 0.5, 0.75, 1.0, 1.5, 2.0 |
| `brightness_dark` | gain | 0.8, 0.65, 0.5, 0.35, 0.25 |
| `brightness_bright` | gain | 1.25, 1.55, 1.9, 2.35, 2.9 |
| `gaussian_noise` | sigma | 0.02, 0.04, 0.07, 0.11, 0.15 |
| `fog` | veil | 0.15, 0.3, 0.45, 0.6, 0.75 |
| `occlusion` | area fraction | 0.05, 0.1, 0.15, 0.2, 0.3 |

**Held-out suite** — five families no arm is trained on, from
`corruptions_heldout.py`, 25 corrupted conditions plus clean:

| Family | Parameter | Severities 1-5 | Realisation |
|---|---|---|---|
| `motion_blur` | kernel length (px) | 3, 5, 7, 11, 15 | angle drawn once per image, reflected edges |
| `jpeg_compression` | libjpeg quality | 70, 50, 30, 15, 7 | deterministic |
| `contrast_reduction` | retained contrast | 0.7, 0.55, 0.4, 0.25, 0.15 | deterministic, towards mid grey |
| `salt_pepper` | fraction of pixels | 0.005, 0.01, 0.03, 0.06, 0.1 | whole pixels replaced, split evenly black/white |
| `rotation` | degrees | 3, 7, 12, 18, 25 | direction drawn per image, bilinear, mid-grey corner fill |

Held-out family names are disjoint from the six training families, and
`augmentation.py` assigns no weight to any of them: no held-out corruption is
ever sampled during training. One realisation seed is drawn per image and
shared across all five severities of a family, so a severity curve varies only
the strength of the corruption.

The two suites are **co-primary**, not primary and secondary. C's family
weights were derived from the frozen benchmark's own headroom, so a gain
confined to the frozen suite is consistent with having matched the benchmark
rather than having improved robustness.

Each condition's input tensors are hashed into an input fingerprint, and the
comparison refuses to proceed unless every arm's fingerprint for a condition is
identical. That establishes that the arms were shown the same pixels. It does
not establish which checkpoint produced a set of predictions; see the identity
checks below.

## Co-primary outcomes

For each suite, the outcome is the **mean per-sign error rate**, aggregated in
this fixed order:

1. error indicator per frame;
2. mean over the 30 frames of a physical sign, giving one rate per sign per
   condition (exactly 30 frames behind every rate, enforced);
3. equal-weight mean over the suite's corrupted conditions — 30 for the frozen
   suite, 25 for the held-out suite. The clean condition is excluded from the
   robustness outcome and reported separately.

Arms are contrasted as paired within-sign differences: B-A, C-A and C-B. The
same 178 signs appear in every arm, so the difference is taken per sign before
being averaged.

**Decision rule, fixed with the design.** C is preferred over B only if its
mean per-sign error is lower at every one of the three training seeds, on both
suites. A consistent win on the frozen suite alone is reported as
"frozen-suite only, not a general improvement", not as a win.

## Uncertainty: track-level bootstrap

Percentile bootstrap over **whole validation tracks**: 2,000 resamples of the
178 signs with replacement, seed 42, interval taken at the 2.5th and 97.5th
percentiles. A sign enters a resample with all 30 of its frames or none of
them. Individual frames are never resampled, because frames within a track are
near-duplicates and resampling them would treat 5,340 correlated observations
as independent and understate the interval.

The interval describes sampling variation over the 178 validation signs at a
fixed training seed. It says nothing about training-seed variation, which the
three replicates report separately as a direction-consistency count.

## Secondary analysis: class-balanced error

Reported alongside the co-primary outcomes, clearly labelled as secondary:

1. error indicator per frame;
2. mean within each physical sign;
3. mean of the sign rates within each class;
4. equal-weight mean of the 43 class means;
5. equal-weight mean over the suite's corrupted conditions, the clean condition
   excluded, as in the co-primary outcome.

This answers a different question from the co-primary metric — it weights a rare
class as heavily as a common one. Its confidence intervals bootstrap physical
signs **within** classes, so each class keeps its own number of signs.

**Limitation, stated in the output itself.** Ten of the 43 classes hold exactly
one physical sign in the validation split. For those classes the within-class
bootstrap has nothing to resample, and the class mean is one sign's rate. This
metric is therefore not independent confirmation from 43 well-sampled classes.
It is not used for checkpoint selection and it does not redefine the success
criterion; `comparison.json` records both facts explicitly.

## Clean-accuracy trade-off

Clean validation accuracy is reported for every arm. A drop of more than
**0.01** against arm A is reported as a trade-off rather than absorbed into the
robustness result (`CLEAN_TRADEOFF_LIMIT = 0.01`). The threshold is a reporting
rule, not a pass criterion: the clean cost and the robustness change are
reported together either way.

## Validation-reuse limitations

The 178-sign validation split carries a lot of load. It selected the baseline
checkpoint, it produced the robustness curve that C's family weights were
derived from, it selects every new arm's checkpoint, and it is the evaluation
set for both suites. Three consequences follow and are not argued away:

* C's weights are not independent of the data they are evaluated on. The
  held-out suite exists because of this, and is co-primary for that reason.
* Clean validation accuracy is a selection metric for these models, so it is
  optimistic as an estimate of clean generalisation.
* Ten classes hold a single validation sign, so per-class numbers from this
  split are weak for those classes, in both the primary and the secondary
  analysis.

A fully independent estimate would require the official test split, which this
protocol does not touch.

## Official test set

No official test image is read anywhere in this experiment. The split is
reserved for one final reported evaluation after all development decisions are
complete; nothing in training, checkpoint selection, policy derivation, suite
construction or comparison may access it.

Two mechanisms hold that rather than a promise. An augmented training run
refuses to start if either of its loaders is built on any split other than the
training archive. Every evaluation wraps the official test loader in an
instrumented dataset and records both its iteration count and the number of
test images read; those two counters are printed and written into the
evaluation metadata, so a run that touched the test split would say so in its
own output.

## Checkpoint provenance and model identity

Each arm's results are filed under an arm name and a seed, so the pipeline
verifies those labels rather than trusting the command line.

`evaluate_checkpoint.py` cross-checks each checkpoint against the records the
training run wrote beside it: the training seed, the selected epoch, the
selection metric, the validation metrics at that epoch, the checkpoint
filename, and every shared key of the recorded training configuration. The
metric comparison is the binding one — checkpoint and history are written by one
process in one pass, so they agree to the last digit, and a checkpoint moved
between two run directories is rejected instead of relabelled.

`compare_models.py` then requires every result directory to identify itself
before any number is computed. It rejects a recorded arm name or seed that
differs from the one supplied, a checkpoint seed that disagrees with the
recorded training seed, a recorded best epoch that differs from the evaluated
checkpoint's epoch, a model whose two suites were produced by different
checkpoints, and one checkpoint appearing under two model identities.

**Two limitations are reported rather than assumed away.**

* The checkpoint format carries no augmentation-policy field. Two runs that
  differ only in their policy write identical training configurations, so an
  augmented arm's name rests on the `policy.json` beside its checkpoint. Seed,
  configuration, epoch and metrics must all agree with that record, which rules
  out a checkpoint from a different recipe, seed or epoch — but not one from a
  different policy at the same seed.
* The protocol identity recorded in `protocol.json` is a sidecar written beside
  the checkpoint before training starts, not a field inside it. It shows which
  protocol version a run was started under, and a run whose files had drifted
  from the manifest says so in its own metadata. It is **not** cryptographic
  proof that these files produced that checkpoint.

The accepted baseline checkpoint predates both records: it has no `policy.json`,
which is what its clean recipe produces and is read as the clean arm, and no
`protocol.json`, which the evaluator reports as an unverifiable field.

## Protocol identity

`docs/augmentation_manifest.json` records the SHA-256 digest of this document
and of the two implementation modules that realise the distributions it
describes. Every training run writes a `protocol.json` beside its checkpoint
holding the digests it observed at startup and the names of any files that
differ from the manifest. `verify_augmentation.py` fails if the manifest does
not match the files as they stand.

## Protected artifacts

No run in this experiment can write into `artifacts/baseline` or
`artifacts/robustness`. Both training entry points claim their output directory
through a guard that refuses those directories, any directory inside or
containing them, and any directory that already holds a checkpoint, a history,
a policy record or a protocol record. `train_clean.py` refuses a seed-42
replicate outright unless an explicit output directory is given, because arm A
at seed 42 is the accepted checkpoint and is reused rather than retrained.
`train.py` itself is unmodified, because it is the code that trained the
baseline.
