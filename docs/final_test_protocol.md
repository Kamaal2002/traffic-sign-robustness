# Final confirmatory evaluation on the official GTSRB test split

Kamaal Alag

**Version 1.0 — frozen 2026-10-27.** The frozen version is identified by the
SHA-256 of this file as committed; a file cannot contain its own digest, so
that value is recorded in the freeze report and recoverable at any time with
`git show <commit>:docs/final_test_protocol.md | shasum -a 256`.

This document is the frozen protocol for the one confirmatory evaluation of the
nine trained checkpoints on the official GTSRB test split. Every methodological
choice in it — the models evaluated, the conditions, the estimator, the
aggregation order, the bootstrap constants, the contrast definitions, the
decision rule and what appears in which table — is settled here, before any
official test prediction exists. No official test prediction has been computed
at the time of freezing. Nothing below may be changed once step 4 of Part 12
has produced a number; Part 8 lists what that prohibits. A later amendment, if
one is ever warranted for a mechanical reason, must be a new version recorded
alongside this one, not an edit to it.

Every statement below is marked either **verified** — established by reading
the released dataset files, the dataset documentation, or this repository's own
code and artifacts — or **to confirm** — an assumption that a named check must
settle before the evaluation runs. Nothing is asserted on the strength of
resemblance to the training split.

---

## Part 1 — Purpose and research questions

The validation comparison in `artifacts/analysis/augmentation/` is complete. It
was computed on 178 physical signs that also selected every checkpoint, and
that produced the failure profile from which arm C's family weights were
derived. The purpose of this evaluation is to see whether its conclusions
survive on data that played no part in any development decision.

This is a confirmatory evaluation. The design is the existing three-arm,
three-seed design; the arms, the policies, the checkpoints and the corruption
definitions are the ones already fixed. No model, policy, seed or condition is
added, and no checkpoint-selection decision is revisited.

Four questions, each answerable from the artifacts this protocol produces:

1. **Does augmentation reduce corrupted-input error on independent data?**
   Are the B−A and C−A contrasts negative on the official test split, on both
   suites, at all three training seeds, as they are on validation?

2. **Does the targeted reweighting beat the balanced one?** Is C−B negative at
   all three seeds on both suites? The validation answer was no: C was lower
   than B on the frozen suite at all three seeds and higher than B on the
   held-out suite at all three seeds, which the declared rule records as
   "frozen-suite only, not a general improvement". The test question is whether
   that same split pattern reappears.

3. **Does the frozen-versus-held-out gap persist?** The validation result is
   consistent with C having matched the benchmark its weights were read from
   rather than having become more robust. If that reading is right, C should
   again lose to B on the five families no arm trains on.

4. **What does augmentation cost on clean inputs?** How does clean test
   accuracy compare across arms, and does any contrast's clean cost — measured
   against that contrast's own right-hand arm at the same seed, as Part 7
   specifies — exceed the declared 0.01 limit?

The validation values these questions are asked against are already committed
and will not be recomputed:

| Suite | Contrast | Mean difference over seeds | Range | Direction |
|---|---|---|---|---|
| frozen | B−A | −0.0885 | 0.0195 | left lower at every seed |
| frozen | C−A | −0.0919 | 0.0145 | left lower at every seed |
| frozen | C−B | −0.0033 | 0.0067 | left lower at every seed |
| held-out | B−A | −0.0524 | 0.0452 | left lower at every seed |
| held-out | C−A | −0.0472 | 0.0447 | left lower at every seed |
| held-out | C−B | +0.0053 | 0.0023 | right lower at every seed |

---

## Part 2 — Model inventory

**Verified.** Nine checkpoints, recorded in
`artifacts/robustness/checkpoint_inventory.json` (SHA-256
`a8b92ae009010721de768b85940a2d57c73ee8697a34347308da4fe5f3ad99d7`, file mode
`-r--r--r--`). These are the only checkpoints this evaluation may read.

| Arm | Policy | Seed | Epoch | Checkpoint MD5 | Run directory |
|---|---|---|---|---|---|
| A | clean | 42 | 21 | `7c6ea3f9773d7a644235463d6a02bf0f` | `artifacts/baseline` |
| A | clean | 43 | 27 | `0390dd3efb8792016955c5d0edc61500` | `artifacts/clean_seed43` |
| A | clean | 44 | 30 | `3e4f07defd52018279f98f7263acdbff` | `artifacts/clean_seed44` |
| B | balanced | 42 | 29 | `ae08ff97fbae0f6d70cc3636f6a33052` | `artifacts/augmented_balanced_seed42` |
| B | balanced | 43 | 16 | `c6249e311c927a27569179ee0ac610c4` | `artifacts/augmented_balanced_seed43` |
| B | balanced | 44 | 29 | `5e46b7de1324ba7f9e50dc87a51e16ca` | `artifacts/augmented_balanced_seed44` |
| C | targeted | 42 | 29 | `4eda6565ab4f640ecbd49d4ab4f93a30` | `artifacts/augmented_targeted_seed42` |
| C | targeted | 43 | 20 | `9fe27b5fec2bf8ba761957e1af04273d` | `artifacts/augmented_targeted_seed43` |
| C | targeted | 44 | 29 | `a00b5d79ac911be39898fd20e35a06f2` | `artifacts/augmented_targeted_seed44` |

Every checkpoint was selected on clean validation macro F1 at the epoch shown.
Those selections are final. Nothing in this protocol re-selects an epoch,
re-trains a seed, or substitutes a checkpoint.

---

## Part 3 — Official test dataset structure

### What the released data contains

**Verified** from `data/gtsrb/GTSRB/Readme-Images-Final-test.txt` and from the
files themselves:

* 12,630 images, `00000.ppm` to `12629.ppm`, in one flat directory
  `data/gtsrb/GTSRB/Final_Test/Images`. All 12,630 filenames are unique and
  match `^[0-9]{5}\.ppm$`; 12,630 `.ppm` files are present on disk.
* Labels are in `data/gtsrb/GT-final_test.csv`, semicolon-separated, header
  `Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId`, 12,630 data
  rows in ascending filename order. The copy shipped alongside the images,
  `Final_Test/Images/GT-final_test.test.csv`, has the same 12,630 rows without
  the `ClassId` column — it is the competition's unlabelled release and is not
  the file used here.
* `torchvision.datasets.GTSRB` with `split="test"` reads
  `GT-final_test.csv` through `csv.DictReader(delimiter=";")` and builds
  `(path, int(row["ClassId"]))` pairs into `_samples`. The labels are therefore
  reachable as integers without opening a single image.
* GTSRB has 43 classes. The number of distinct classes actually present in the
  test labels is **to confirm** by a check that counts distinct class ids and
  nothing else; the per-class label counts are deliberately not inspected
  before the evaluation runs, because no analysis choice in this protocol may
  depend on them.

### Physical-sign grouping is not available

**Verified, and this is the central structural difference from validation.**
The official readme states it directly:

> Images are in random order, ie. there is no track structure or class
> information as in the training set.

Three further facts confirm it:

* The test annotation schema has no track field. It carries `Filename`,
  `Width`, `Height`, the four ROI coordinates and `ClassId` — nothing else.
* Test filenames carry no track component. The training archive encodes the
  physical sign as `XXXXX_YYYYY.ppm`, where `XXXXX` is the track and `YYYYY`
  the frame within it; **zero** of the 12,630 test filenames contain an
  underscore.
* `data.read_training_index` recovers a track as
  `f"{path.parent.name}/{path.name.split('_')[0]}"` — the per-class directory
  plus the filename's track field. The test split has neither a per-class
  directory nor a track field, so this mechanism returns nothing usable and is
  not applied to it.

So reliable physical-sign identifiers **do not exist** for the official test
split, and this protocol does not construct any. The released image order is
documented as random, which rules out treating contiguous blocks of filenames
as tracks. Clustering on `Width`, `Height` or ROI geometry would be inventing
track identifiers from data that was never meant to carry them, and is
excluded.

### Whether the dependence is nonetheless present

It very likely is, and this must be stated rather than wished away. The GTSRB
acquisition recorded 30 frames per physical sign — the training readme says so
explicitly ("All images of one class with identical track numbers originate
from one single physical traffic sign") and the training archive contains
exactly 30 frames for each of its 888 tracks. The test split holds 12,630
images, which is exactly 421 × 30. That divisibility is **consistent with** the
same 30-frame structure surviving into the test release, unlabelled. It is not
proof: it is an arithmetic observation, and no released file confirms a track
count.

The operative conclusion is the same either way. Test images are almost
certainly not 12,630 independent observations, and the grouping that would let
me account for that is not recoverable. Part 6 says what is done about it.

### Overlap with train and validation

**Verified from documentation.** The final-test readme instructs that the
GTSRB final training set be used for training and describes it as "the joint
dataset of the online competition's training and test set" — a separate
collection from the final test set released for the competition session. The
local training archive holds 26,640 images in 888 tracks of 30, which is the
online-competition training set, a subset of that final training set. On the
documented provenance, no final-test image is a training image.

**To confirm, and only in part.** Documentation is not a byte-level
guarantee, and one cheap check is available: compare the SHA-256 of all 12,630
test image files against the SHA-256 of all 26,640 training image files and
require an empty intersection. This reads file bytes to hash them; it decodes
no image and computes no metric. It runs as part of Part 10, before any
inference.

That check settles less than it appears to. It detects **byte-identical
duplicates only**. The leakage mode this project works hardest to prevent —
other frames of the same physical sign appearing on both sides of a split — is
invisible to it, because two frames of one sign are different images with
different bytes. Ruling that out would need test track identifiers, which do
not exist. So the absence of cross-split physical-sign overlap is an
**assumption** resting on the benchmark's published construction, not something
the released test data permits verifying. It is carried as a limitation in
Part 11, not as a verified fact.

### Preprocessing compatibility

**Verified.** `data.create_data_loaders` builds the test dataset as
`GTSRB(root=data_root, split="test", transform=build_preprocessing())` — the
same `build_preprocessing()` object the training and validation datasets get:
`Resize((64, 64), bilinear, antialias=True)` → `ToTensor()` →
`Normalize(mean=0.5, std=0.5)` per channel. `torchvision`'s `GTSRB.__getitem__`
opens the file with PIL and calls `.convert("RGB")` for both splits. Neither
split is ROI-cropped; both are resized whole, border margin included, and both
readmes document that margin as 10 percent of the sign size with a 5-pixel
floor. Test images vary in size exactly as training images do, and the resize
absorbs that.

So the trained models can consume test images with no change to the input
pipeline, and the corruption wrappers can be inserted at the same point: the
`observation` half of `corruptions.split_preprocessing()` (resize and
`ToTensor`, normalisation held back) applies to the test split unchanged.

### Statistically meaningful independent units

There are none above the image. The analysis unit is therefore the test image,
with the consequences set out in Part 6 and Part 11.

---

## Part 4 — Evaluation conditions

Both existing suites, unchanged, on all 12,630 test images.

**Frozen suite** — the six training families from `src/corruptions.py` at the
five committed severities, 30 corrupted conditions plus clean:

| Family | Parameter | Severities 1–5 |
|---|---|---|
| `gaussian_blur` | sigma | 0.5, 0.75, 1.0, 1.5, 2.0 |
| `brightness_dark` | gain | 0.8, 0.65, 0.5, 0.35, 0.25 |
| `brightness_bright` | gain | 1.25, 1.55, 1.9, 2.35, 2.9 |
| `gaussian_noise` | sigma | 0.02, 0.04, 0.07, 0.11, 0.15 |
| `fog` | veil | 0.15, 0.3, 0.45, 0.6, 0.75 |
| `occlusion` | area fraction | 0.05, 0.1, 0.15, 0.2, 0.3 |

**Held-out suite** — the five families no arm trains on, from
`src/corruptions_heldout.py`, 25 corrupted conditions plus clean:

| Family | Parameter | Severities 1–5 |
|---|---|---|
| `motion_blur` | kernel length (px) | 3, 5, 7, 11, 15 |
| `jpeg_compression` | libjpeg quality | 70, 50, 30, 15, 7 |
| `contrast_reduction` | retained contrast | 0.7, 0.55, 0.4, 0.25, 0.15 |
| `salt_pepper` | fraction of pixels | 0.005, 0.01, 0.03, 0.06, 0.1 |
| `rotation` | degrees | 3, 7, 12, 18, 25 |

31 conditions on the frozen suite and 26 on the held-out suite, the clean
condition shared, 57 per model, 513 across nine models.

Nothing is retuned. The corruption parameters, the severity values, the
severity count, the preprocessing, the architecture, the checkpoint weights and
the batch size (64) are the committed ones. The suites stay **co-primary**: a
gain confined to the frozen suite is not a robustness result, for the reason
given in Part 1.

### Corruption seeds and the test namespace

**Verified.** `corruptions.corruption_seed` derives a per-sample seed as

```
blake2b(f"{base_seed}|{namespace}|{corruption}|{severity}|{sample_index}")
```

right-shifted by one, and `corruptions_heldout.heldout_sample_seed` does the
same with its own table. `sample_index` comes from
`corruptions.stable_index`, which unwraps nested `Subset` layers and returns
the underlying dataset position; the official test split is the whole dataset
rather than a `Subset`, so it returns the index directly and needs no change.

The `namespace` field exists for exactly this situation. Both
`CorruptedDataset.__init__` and `HeldoutDataset.__init__` already accept
`namespace`, and `CorruptedDataset`'s own docstring records the intent: the
namespace "names the split being corrupted and enters every seed, so the final
test-split evaluation cannot reuse validation's realisations."

**The one obstruction.** `evaluate_checkpoint.Suite.dataset` passes
`namespace=VALIDATION_NAMESPACE` as a literal in both of its branches, so a
suite object cannot currently realise anything but validation's corruptions.
The smallest safe change is therefore not an edit to that method but a
subclass in a new module that overrides it — Part 9 specifies it. Validation
behaviour stays byte-identical because no existing file changes.

**The test namespace.** `TEST_NAMESPACE = "final_test"`, fixed here, before any
test evaluation. It differs from `"validation"` and from the `"train"`
namespace the augmentation stream uses, so every test corruption realisation is
reproducible from the committed seed and is not coupled to any validation
sample's identity. Two consequences worth stating: the test realisations are
drawn independently of the validation ones, and they are identical across all
nine models, which is what makes the arms paired on the test split as they are
on validation.

---

## Part 5 — Primary outcomes

Per suite, the **mean per-image error rate**, aggregated in this fixed order:

1. error indicator per image per condition — 1 if the predicted class differs
   from `ClassId`, else 0;
2. equal-weight mean over the suite's corrupted conditions — 30 for the frozen
   suite, 25 for the held-out suite — giving one rate in [0, 1] per test image;
3. mean over the 12,630 test images.

The clean condition is excluded from this average and reported separately. Each
family contributes equally regardless of how many severities it spans, exactly
as on validation.

This differs from the validation outcome in one step and one step only: the
validation protocol averaged a sign's 30 frames into a per-sign rate before
averaging conditions, and there is no analogue of that step here because there
are no sign identifiers. The condition-weighting, the clean exclusion and the
error definition are unchanged.

Reported alongside, per model and suite:

* **clean test accuracy and clean test macro F1**, from the shared clean
  condition;
* **per-condition accuracy, macro F1, loss and drops from clean**, the same 57
  rows per model the validation evaluation produced;
* **class-balanced error** as a **descriptive diagnostic**, reported without
  confidence intervals and without per-class inference: error indicator per
  image, mean within each class, equal-weight mean over the classes present,
  equal-weight mean over the suite's corrupted conditions.

  The reason for reporting it without intervals is the same clustering argument
  the rest of this protocol rests on. A class is well supported here in
  *images*, but the independent unit is the physical sign, and the number of
  signs behind a class is unknown — plausibly single digits for the rare
  classes. A within-class image bootstrap would inherit the primary
  estimator's independence violation and concentrate it in exactly the classes
  where it does most damage, producing tight-looking intervals around numbers
  that may rest on one or two signs. Counting images does not fix that, so the
  metric is reported as a descriptive breakdown of where error sits across
  classes, carries no interval, supports no per-class claim, and enters no
  decision.

---

## Part 6 — Statistical analysis plan

All constants in this part are fixed now, before any test inference, and are
the committed validation constants: 2,000 resamples, seed 42, percentile
interval at the 2.5th and 97.5th percentiles.

### The resampling unit is the test image

The validation analysis resampled 178 whole physical signs, a sign entering a
resample with all 30 of its frames or none. That estimator cannot be
transferred, because the grouping it needs does not exist on the test split
(Part 3).

The replacement is a **paired bootstrap over test images**: resample the 12,630
test images with replacement, 2,000 times, from seed 42, and take the 2.5th and
97.5th percentiles of the resampled mean. Pairing is preserved by resampling
image identities once per draw and reading every arm's rate for the drawn
images, so a contrast's two sides always rest on the same images.

**The independence assumption, and that it is likely violated.** This
procedure assumes **independent image-level sampling**: that the 12,630
per-image rates are independent draws. That assumption is **likely violated**.
The evidence in Part 3 points to repeated images of the same physical signs —
on the order of 30 near-duplicate frames per sign — present in the test release
but unlabelled.

Three things follow, and the write-up must hold to all three:

* The interval **does not account for physical-sign dependence**. It is not
  corrected for it, it is not adjusted for it, and it will not be described as
  if it were.
* It is **not a definitive population-level confidence interval**. Its nominal
  95 percent coverage holds under independent image-level sampling and not
  otherwise, so it is reported as an interval under a stated and probably false
  assumption — a measure of how much the estimate moves when the measured
  images are resampled, nothing broader.
* Its width is **narrower than the dependence-aware width would be**. Where a
  conclusion depends on the interval's width, that is stated as a limitation
  of the conclusion, not resolved by widening the interval.

### The √30 design effect, as a sensitivity illustration only

To keep the understatement's possible size visible rather than buried in a
sentence, a design-effect calculation is reported in a separate sensitivity
subsection of the output. It is an illustration under an assumption, and it is
labelled as one everywhere it appears.

For a mean of n = m·k observations in m clusters of size k, with common
within-cluster correlation ρ,

```
Var(mean) = (sigma^2 / n) * [1 + (k - 1) * rho]
```

so the standard error is √[1 + (k − 1)ρ] times the independent-sampling value.
Setting k = 30 and ρ = 1 — the most extreme value, under which a sign's 30
frames would carry one image of information — gives a factor of √30 ≈ 5.477.
Scaling each percentile endpoint's distance from the point estimate by that
factor, separately above and below so the asymmetry survives, gives the
illustration.

**What this is not.** It is not a confidence bound, not a worst-case
confidence interval, and not a corrected interval. Both k = 30 and ρ = 1 are
assumptions the released test data cannot confirm: Part 3 establishes that no
track identifiers exist, so neither the cluster size nor the correlation is
observable. ρ = 1 is also implausible on its face, and the factor is loosest
exactly where it would be applied — to **paired between-model differences**,
where two arms' errors on the same image largely cancel, so a difference's
within-sign correlation is far below a raw error rate's. The number shows how
far an interval could move under a stated assumption. It licenses no inference
of its own.

**Where it may and may not appear.** It appears in the sensitivity subsection,
labelled as assumption-dependent, and nowhere else. It does **not** appear in
the headline contrast tables, it is **not** reported as a second interval
alongside the primary one, and it **does not enter the decision rule of
Part 7** in any form. The decision rule is a direction count across three
training seeds and does not reference any interval, primary or illustrative.

I considered and rejected a further option: drawing random subsamples of about
421 images to approximate one frame per sign. It assumes the same unverified
421 × 30 structure, discards 97 percent of the data, and adds Monte Carlo noise
of its own, while establishing nothing the illustration does not.

### Contrasts

The three declared contrasts, unchanged, reported as left minus right so that a
**negative** value means the first-named arm has the lower error:

* B − A, balanced against clean;
* C − A, targeted against clean;
* C − B, targeted against balanced.

Each is a paired within-image difference: the difference is taken per image
before anything is averaged. Suites are reported separately and never pooled;
the clean condition never enters a corruption average.

Secondary decompositions, as on validation and labelled as decompositions
rather than separate analyses: per-family and per-severity contrasts computed
by the same aggregation over the narrower condition set, each with the primary
image-level interval and the same caveat. The class-balanced breakdown is
reported as point estimates only, for the reason given in Part 5.

### Example uncertainty and seed variability are reported separately

Two kinds of variation are in play and they are not combined:

* **Image-level sampling uncertainty, at a fixed training seed.** This is
  what the bootstrap interval describes, under the assumption above and with
  its stated limits. Nine models give nine per-model intervals per suite. It
  says nothing about training-seed variation.
* **Across training seeds.** Three replicates support a direction, a mean of
  the three point estimates, and a min–max range. Nothing more. No interval is
  computed across seeds, no p-value is reported, and no claim of statistical
  significance across seeds is made — three runs cannot carry one. The
  across-seed summary reports how many seeds point each way and labels the
  contrast consistent or seed-sensitive, as `compare_models.across_seeds`
  already does.

A single interval merging the two would be misleading and is not produced.
Training-seed variability is reported only as a direction count, a mean of the
per-seed point estimates, and a min-max range; image-level sampling uncertainty
is reported only as the per-model interval above. Neither is substituted for
the other, and no quantity combines them.

---

## Part 7 — Decision criteria

The rule is the one fixed with the design and is not restated in friendlier
terms:

> **C is preferred over B only if C's mean per-image error is lower than B's at
> every one of the three training seeds, on both suites.** A consistent win on
> the frozen suite alone is reported as "frozen-suite only, not a general
> improvement", not as a win.

And the clean-accuracy rule, also unchanged, including its reference arm.
`compare_models.clean_tradeoff_rows` measures each contrast's clean cost
against **that contrast's own right-hand arm** at the same seed, not against
arm A: the cost on the C-B row is B's clean accuracy minus C's. A cost above
`CLEAN_TRADEOFF_LIMIT = 0.01` is reported as a trade-off rather than as an
improvement, whatever the corrupted error does. The test path uses the same
convention, so the two splits' trade-off columns are comparable. The threshold
is a reporting rule, not a pass criterion; the clean cost and the corrupted
change are reported together either way.

No new success rule is introduced, no threshold is loosened, and the decision
rests on the three-seed direction count on both suites, as before.

**The decision rule references no interval.** Neither the image-level bootstrap
interval nor the √30 sensitivity illustration enters it. The criterion is the
sign of the per-seed point estimate at each of the three seeds, on each of the
two suites, exactly as on validation. Intervals are reported alongside the
decision; they do not participate in it.

### What gets reported

* **Reproduction.** For each of the six contrast-and-suite cells, whether the
  test sign of the difference matches validation's and whether the
  three-seed direction is equally consistent. Reproduction means the same
  pattern, including validation's split verdict for C−B; it does not mean the
  same magnitudes.
* **What non-reproduction of C−B does and does not mean, pre-registered
  here.** B−A and C−A are large on validation relative to their intervals:
  −0.0885 and −0.0919 on the frozen suite, −0.0524 and −0.0472 on the held-out
  suite, with all twelve per-seed intervals excluding zero. C−B is not:
  −0.0033 and +0.0053, with only 2 of its 6 per-seed intervals excluding zero.
  An effect that small relative to its interval will not reliably reproduce
  across splits whether or not the underlying difference is real. So a C−B
  direction that flips on the test split is **weak evidence either way** and
  will be reported as such, not as evidence against the targeted weights; and a
  C−B direction that holds is not evidence for them. The existing verdict,
  "frozen-suite only, not a general improvement", stands unless the test result
  contradicts it at a magnitude comparable to B−A.
* **Contradiction.** Any cell whose test direction differs from validation's is
  reported as a contradiction, in the summary and not only in a limitations
  section, with both numbers side by side. It is not resolved by retuning
  anything. If the test split contradicts the validation conclusion, the test
  split is the independent evidence and the write-up says so.
* **Seed differences.** Per-seed point estimates, the three-seed mean, the
  min–max range, and the direction label. A contrast that flips sign between
  seeds is labelled seed-sensitive and no overall direction is claimed for it.
* **Family differences.** Per-family and per-severity contrasts for both
  suites, all families reported whether or not they favour any arm. The
  frozen-versus-held-out comparison in question 3 is read from these.
* **Clean trade-offs.** Clean test accuracy and macro F1 per model, the cost
  against arm A at the same seed, and whether it exceeds 0.01.
* **Statistical uncertainty.** For every contrast, in the headline table: the
  point estimate and the image-level percentile interval, carrying the
  statement that the interval assumes independent image-level sampling, that
  the assumption is likely violated by repeated images of the same physical
  signs, and that it is therefore not a definitive population-level confidence
  interval. An interval excluding zero is described as excluding zero under
  that stated assumption — never as establishing significance, and never as
  evidence about the population. The √30 design-effect illustration appears in
  its own sensitivity subsection, labelled assumption-dependent, and not in
  this table.

---

## Part 8 — Preventing test-set leakage

### The prior access, stated accurately

The official test images have been read before. The record must say so plainly,
and must not claim that this repository has never accessed test images. Two
distinct things are involved and the distinction is the whole point:

**The original experimental procedure did not read any official test image.**
Training, checkpoint selection, the derivation of arm C's family weights, the
construction of both corruption suites and the entire validation comparison ran
without touching the test split. That is not a claim about intent; it is
measured. Every augmented training run refuses to start unless each of its
loaders is built on the training archive, and all 18 committed evaluation
metadata files under `artifacts/robustness/` record
`test_loader_iterations: 0` and `test_dataset_reads: 0` through instrumented
loaders. `docs/augmentation_protocol.md` states "No official test image is read
anywhere in this experiment", and as a statement about the experiment it
describes, that is correct and remains correct.

**A later verification utility did read 64 official test images.**
`src/verify_data_pipeline.py` is a pipeline-checking script, not part of the
experiment. It iterates `next(iter(loader))` for every split
`create_data_loaders` returns, the test split included, to assert batch shape,
dtype, pixel range and label range. It therefore decodes one batch of 64
official test images on **every run**, and it has been run more than once. It
constructs no model, runs no forward pass, computes no prediction, calculates
no performance metric, and writes no file. Its entire output for the test split
is a shape, a dtype and two ranges.

So, precisely:

> The official test images have been accessed previously, by a verification
> utility that decoded 64 of them to check batch shape, dtype and pixel range.
> No official test performance information has been produced or used. No model
> selection, checkpoint selection, policy derivation, corruption definition,
> estimator choice or analysis decision has been informed by the test split.

The test split must not be described as completely untouched, and this
repository must not be described as never having accessed test images.

`docs/augmentation_protocol.md` and `docs/augmentation_manifest.json` are the
historical record of what was frozen before the nine runs were trained. They
are **not** modified — not to add this distinction, not for any other reason.
Amending that document would change its SHA-256, invalidate the manifest the
nine runs recorded at startup, and make every future evaluation report protocol
drift against a protocol that had not actually changed. The distinction above
is recorded here instead, which is where it belongs: this document covers the
test evaluation, and the test access is a fact about this stage of the work.

### Prohibited after test results exist

Once any official test metric has been computed, all of the following are
prohibited, and the protocol's value rests on them:

* retraining any model, at any seed, in any arm;
* selecting or re-selecting a checkpoint epoch on test performance;
* adjusting augmentation family weights, severity weights or `P(augment)`;
* changing any corruption parameter, severity value or suite membership;
* changing the estimator, the aggregation order, the bootstrap constants, the
  contrast definitions or the decision rule;
* dropping a corruption family, severity or condition from a reported average
  because it is unfavourable;
* reporting a subset of the three seeds, or promoting a per-seed result over
  the three-seed direction;
* evaluating any further model, variant or checkpoint on the test split until a
  desired result appears. The test evaluation is run once, over the nine
  inventoried checkpoints, and reported as it comes out;
* choosing, after seeing a test number, which interval, sensitivity
  illustration or secondary metric appears in the headline tables. That is a
  reporting degree of freedom that can change how strong a conclusion looks
  without any other prohibited step being taken. Every such choice is settled
  in this document, before the protocol is frozen and before step 4 of Part 12
  produces a number;
* re-deriving the √30 illustration with a different k or ρ after seeing the
  results, or promoting it from the sensitivity subsection into a headline
  table or into the decision rule.

Three mechanical supports, not promises:

* The output root `artifacts/final_test/` is new and separate, and a run
  refuses to write into a directory that already holds results, so a second
  evaluation of the same model and suite cannot quietly overwrite the first.
* Every test run appends one line to
  `artifacts/final_test/test_access_log.jsonl` recording the timestamp, the
  module, the repository HEAD, the checkpoint MD5, the suite and the number of
  test images read. A repeated evaluation is visible in that log whether or not
  its outputs are kept.
* The statistical plan in Part 6 is fixed in this document, under version
  control, before the first test prediction exists.

---

## Part 9 — Implementation requirements

All new behaviour goes into two new modules, so `src/corruptions.py`,
`src/corruptions_heldout.py`, `src/evaluate_checkpoint.py`,
`src/evaluate_robustness.py`, `src/data.py` and `src/compare_models.py` stay
byte-identical and the committed validation results stay reproducible from
them. No evaluator or corruption behaviour changes.

One existing file does change, and it is the only one: the frozen-module tuple
in `src/verify_augmentation.py` is widened to cover `src/evaluate_checkpoint.py`
and `src/compare_models.py`, for the reason given in Part 10 check 2. That adds
a check; it changes no measured quantity, and it must land before the new
modules are written rather than after.

### `src/evaluate_test.py` — one checkpoint, one suite, on the test split

* **`TEST_NAMESPACE = "final_test"`** and `EXPECTED_TEST_SIZE = 12630` defined
  here, so no frozen module changes to carry them.
* **`class TestSuite(Suite)`** overriding `dataset()` alone, to pass
  `namespace=TEST_NAMESPACE` on both the clean and the corrupted branch. A
  helper builds the two test suites from `FROZEN_SUITE` and `HELDOUT_SUITE` by
  reading their own `table`, `dataset_class`, `parameter`, `specification`,
  `description` and `notes` attributes, so the condition definitions are shared
  by reference and cannot drift from the validation ones.
* **`build_test_observations(data_root=None, download=False)`**, the counterpart
  of `corruptions.build_validation_observations`: construct
  `GTSRB(root=data_root, split="test", transform=observation)` with the
  `observation` half of `split_preprocessing()`, and return the dataset,
  `numpy.arange(12630)` as the index vector, and the label vector. Labels come
  from the dataset's own `_samples`, the same route
  `data.read_training_index` already uses for the training split, so they are
  exactly the integers the loader will yield. They are cross-checked against an
  independent parse of `GT-final_test.csv` — two readings of the same file by
  different code, required to agree on all 12,630 entries.
* **Checkpoint binding, from existing artifacts rather than new inference.**
  `check_identity`, `discover_run_metadata`, `resolve_model_name` and
  `check_provenance` are imported and called as they are; they read the
  checkpoint and its `history.json`, `policy.json` and `protocol.json`
  sidecars and touch no image data.
  The purpose of the binding step is to establish that the file being evaluated
  is the checkpoint that the arm and seed it is filed under actually produced.
  That is already established for all nine checkpoints, on byte-identical
  files, by the committed validation evaluations, so no validation inference is
  rerun here. Instead the run requires, per checkpoint, that: its recomputed
  MD5 and SHA-256 equal `artifacts/robustness/checkpoint_inventory.json`; the
  four provenance calls above pass; the arm and seed named on the command line
  equal the recorded ones; and both committed validation metadata files for
  that checkpoint — `artifacts/robustness/<arm>_seed<seed>/frozen/metadata.json`
  and `.../heldout/metadata.json` — carry `checkpoint.md5` equal to the
  recomputed MD5, a `clean_reproduction` block, a
  `checkpoint.recorded_validation_metrics` block, and
  `integrity.clean_metrics_match_checkpoint_record: true`. All four fields are
  present in all 18 committed metadata files (**verified**), so this is a
  comparison against existing artifacts, not a new measurement. A missing or
  disagreeing field stops the run. The arm-and-seed directory name is the
  evaluation run's, which for A42 is `clean_seed42`; the separate
  `artifacts/robustness/baseline/` tree holds the earlier accepted baseline
  evaluation and is read by neither module.
* **One clean reproduction, on the test split, binding.** A pass over
  `data.py`'s own test loader, whose predictions the suite's clean row must
  match byte for byte, so every test drop is measured against this model's own
  clean test score through the frozen pipeline. This is the only forward pass
  outside the suite sweep, and it is on the split being evaluated.
* **Split accounting, enforced rather than recorded.** `create_data_loaders` is
  called once and `instrument_loaders` wraps all three loaders. The run then
  requires exact counts and raises otherwise: train 0 iterations and 0 reads;
  validation 0 iterations and 0 reads; test 1 iteration and 12,630 reads. The
  existing evaluator records these numbers without raising; here they are
  assertions, which is what rules out the training split — and, now, the
  validation split — entering a test evaluation.
* **Positive read accounting on the sweep.** The test observation dataset is
  wrapped in `CountingDataset` — safe because `CorruptedDataset` and
  `HeldoutDataset` read only `dataset[i]` and take normalisation from
  `split_preprocessing()`, never from `dataset.transform`. The run then
  asserts the exact expected read count: 12,630 × 31 + 421 × 31 = 404,581 for
  the frozen suite and 12,630 × 26 + 421 × 26 = 339,326 for the held-out suite,
  the second term being the input fingerprints. A sweep that read the wrong
  number of images fails instead of reporting.
* **Per-example predictions recorded.** A `predictions.csv` with
  `corruption, severity, test_position, dataset_index, label, prediction` for
  all 12,630 images in every condition. No filename column: `GT-final_test.csv`
  holds its 12,630 rows in ascending filename order over the contiguous names
  `00000.ppm` to `12629.ppm`, and `torchvision` builds `_samples` in that row
  order, so `dataset_index` *is* the filename — row `i` is `f"{i:05d}.ppm"`.
  The run asserts that identity for all 12,630 entries and records the rule in
  the metadata, which makes every row traceable to an official image without
  storing a redundant column. Storing it would add about 10 bytes per row,
  roughly 7 MB per model and 65 MB across the nine, for no information.
* **Fingerprints preserved.** `input_fingerprint` is reused unchanged at
  `FINGERPRINT_STRIDE = 30`, so each condition gets one fingerprint computed
  over 421 strided image positions: **31 fingerprints for the frozen suite and
  26 for the held-out suite**, each over 421 positions. A verification step
  must assert those cardinalities, not 421, which is the position count.
  The stride's original rationale — one frame per track — does not apply here;
  it is now simply a 1-in-30 systematic sample, and the metadata says so. The
  model fingerprint check is also reused unchanged: all 28 state tensors hashed
  before and after, gradients asserted absent, `eval()` asserted.
* **Separate outputs.** `TEST_RESULTS_ROOT = artifacts/final_test`, with one
  directory per model and suite, `artifacts/final_test/<arm>_seed<seed>/<suite>/`.
  A module-level guard refuses any output directory inside or containing
  `artifacts/baseline`, `artifacts/robustness` or `artifacts/analysis`, and the
  existing refusal of a non-empty output directory is kept. No validation
  artifact is readable-and-writable from this module.
* **Access log.** One appended line per run to
  `artifacts/final_test/test_access_log.jsonl`, as Part 8 describes.
* **Reproducibility.** The metadata records the test namespace, the base seed,
  the full condition specification, the per-condition fingerprints and their
  421 strided positions, the repository HEAD
  and dirty-file list, the torch, numpy, Python and platform versions, the
  device, the batch size and the worker count — the same fields the validation
  metadata carries, with `"split": "official_test"` in place of
  `"split": "validation"` and the track-per-class block replaced by a
  class-support block, since the test split has no tracks.
* **Failure recovery.** Each of the 18 runs is independent and writes only its
  own directory. A failed run leaves a partial directory and a non-zero exit;
  it is rerun only after the partial directory is removed or renamed, never by
  writing over it. The analysis in the next module refuses to start unless all
  18 directories exist with complete condition sets, so a partial run cannot
  silently enter a result.

#### Exact interface

Module-level constants, all defined in `src/evaluate_test.py`:

```
TEST_NAMESPACE      = "final_test"
EXPECTED_TEST_SIZE  = 12630
TEST_RESULTS_ROOT   = PROJECT_ROOT / "artifacts" / "final_test"
ACCESS_LOG          = TEST_RESULTS_ROOT / "test_access_log.jsonl"
INVENTORY_PATH      = PROJECT_ROOT / "artifacts" / "robustness" /
                      "checkpoint_inventory.json"
VALIDATION_ROOT     = PROJECT_ROOT / "artifacts" / "robustness"
PROTECTED_ROOTS     = (PROJECT_ROOT / "artifacts" / "baseline",
                       PROJECT_ROOT / "artifacts" / "robustness",
                       PROJECT_ROOT / "artifacts" / "analysis")
TEST_SUITES         = {"frozen": …, "heldout": …}   # built from the two
                                                   # validation Suite objects
```

`BATCH_SIZE`, `SEED`, `FINGERPRINT_STRIDE`, `SMOKE_CONDITIONS`, `SMOKE_STRIDE`
and `CLEAN_CONDITION` are imported, not redefined.

Functions, with their responsibilities:

| Signature | Responsibility |
|---|---|
| `class TestSuite(Suite)` | Overrides `dataset(self, observations, corruption, severity)` and nothing else, passing `namespace=TEST_NAMESPACE` on both the clean and the corrupted branch. Inherits `table`, `parameter`, `specification`, `description`, `notes` and `conditions` untouched. |
| `build_test_suite(suite)` | Returns a `TestSuite` constructed from `suite`'s own attributes, so the condition tables are shared by reference with `FROZEN_SUITE` / `HELDOUT_SUITE`. |
| `build_test_observations(data_root=None, download=False)` | Returns `(dataset, indices, labels)`: a `GTSRB(split="test")` carrying the observation half of `split_preprocessing()`, `numpy.arange(EXPECTED_TEST_SIZE)`, and the label vector read from the dataset's own `_samples`. Raises unless `len(dataset) == EXPECTED_TEST_SIZE`. |
| `parse_test_ground_truth(data_root=None)` | Independent `;`-separated parse of `GT-final_test.csv` into an ordered `(filename, label)` list. Does not use `torchvision`. |
| `check_label_agreement(dataset, parsed)` | Requires the two readings to agree on all 12,630 `(filename, label)` pairs; raises on the first disagreement. |
| `check_filename_identity(dataset)` | Requires `Path(dataset._samples[i][0]).name == f"{i:05d}.ppm"` for all `i`, which is what licenses omitting the filename column. Returns the rule as a string for the metadata. |
| `inventory_record(checkpoint_path)` | Returns the matching entry from `checkpoint_inventory.json`, keyed on the resolved repository-relative path; raises if absent. |
| `check_inventory_digests(checkpoint_path, record)` | Recomputes MD5 and SHA-256 and requires equality with the record. |
| `check_validation_binding(record)` | Reads `artifacts/robustness/<arm>_seed<seed>/frozen/metadata.json` and `.../heldout/metadata.json` and requires, in each: `checkpoint.md5` equal to the recomputed MD5, a `clean_reproduction` block, a `checkpoint.recorded_validation_metrics` block, and `integrity.clean_metrics_match_checkpoint_record is True`. Reads JSON only; runs no inference. |
| `class_support(labels)` | Per-class image counts over the 12,630 test labels, the metadata block that replaces validation's tracks-per-class. |
| `results_dir(arm, seed, suite_name, output_root=None)` | `TEST_RESULTS_ROOT / f"{arm}_seed{seed}" / suite_name`. |
| `check_output_target(path)` | Raises if the resolved path is inside, equal to, or a parent of any entry in `PROTECTED_ROOTS`, and raises if it exists and is non-empty. Called before any forward pass. |
| `append_access_log(entry)` | Appends one JSON object and a newline to `ACCESS_LOG`, opened `"a"`, flushed and `os.fsync`ed. Never rewrites an existing line. |
| `write_test_outputs(suite, results, conditions, metadata, indices, labels, support, results_dir)` | The counterpart of `evaluate_checkpoint.write_outputs`, differing only in the two split-specific arguments and the prediction column name. Writes the artifact set below. |
| `main()` | The sequence of Part 12 for one `(checkpoint, suite)` pair. |

Command line, mirroring the existing evaluator so the two are operated the same
way:

```
python -m src.evaluate_test --checkpoint PATH --suite {frozen,heldout}
       [--model-name ARM] [--output-dir DIR] [--device DEV]
       [--batch-size N] [--num-workers N] [--smoke-test]
```

`--model-name` is resolved through the existing `resolve_model_name`, so it is
needed only when a run records no policy. `--output-dir` is still passed
through `check_output_target`; it cannot be used to write into a protected
tree. `--smoke-test` computes and reports no performance metric, as in the
existing evaluator, and writes nothing — including no access-log line, since
its stride subset is covered by the line the real run writes.

Output set, per `(model, suite)`, under
`artifacts/final_test/<arm>_seed<seed>/<suite>/`:

| File | Contents |
|---|---|
| `summary.csv` | one row per condition: corruption, severity, accuracy, macro-F1, mean loss, error rate, and the drop against this model's clean test row |
| `summary.json` | the same numbers plus the suite specification |
| `per_class_metrics.csv` | per-condition, per-class precision, recall, F1 and support |
| `confusion_matrices.npz` | one 43 × 43 integer matrix per condition |
| `predictions.csv` | `corruption, severity, test_position, dataset_index, label, prediction`, 12,630 rows per condition |
| `metadata.json` | the reproducibility block described above, with `"split": "official_test"`, `"namespace": "final_test"`, the class-support block, the per-condition fingerprints and the read counts |

Overwrite protection is three-layered: `PROTECTED_ROOTS` makes the validation
and baseline trees unwritable from this module; a non-empty target directory
raises before the model is built; and the access log is append-only, so a
second run of the same pair is recorded even if its outputs are discarded.

Access-log line schema — one JSON object per line, written once per completed
run, after the sweep and before the process exits:

```json
{"timestamp": "2026-10-27T14:03:11Z", "module": "src/evaluate_test.py",
 "git_head": "<40-hex>", "git_dirty": ["…"], "arm": "targeted", "seed": 44,
 "checkpoint_md5": "a00b5d79ac911be39898fd20e35a06f2",
 "suite": "heldout", "conditions": 26,
 "test_images_read": 339326, "test_loader_iterations": 1,
 "test_loader_reads": 12630,
 "train_loader_iterations": 0, "train_loader_reads": 0,
 "validation_loader_iterations": 0, "validation_loader_reads": 0,
 "output_dir": "artifacts/final_test/targeted_seed44/heldout",
 "elapsed_seconds": 0.0}
```

Its verification, in step 5 of Part 12: the log must hold exactly 18 lines,
one per `(arm, seed, suite)` triple with no triple repeated; every
`checkpoint_md5` must appear in the inventory; every `test_images_read` must
equal 404,581 for a frozen line and 339,326 for a held-out line; every
`train_loader_*` and `validation_loader_*` field must be 0; and each line's
`output_dir` must exist and hold the full artifact set. Any extra line is a
repeated test evaluation and is reported as such rather than deleted.

### `src/compare_test.py` — the nine-model test comparison

Separate from `src/compare_models.py` for the same reason the test evaluator is
separate: that file produced the committed validation comparison and stays
untouched.

* Reuses `ARMS`, `ARM_LABELS`, `CONTRASTS`, `CLEAN_TRADEOFF_LIMIT`,
  `BOOTSTRAP_DRAWS`, `BOOTSTRAP_SEED`, `INTERVAL`, `across_seeds`,
  `_direction` and `targeted_verdict` by import, so the contrast definitions,
  the bootstrap constants and the decision rule are literally the same objects,
  not restatements.
* Replaces the sign-level aggregation with the image-level aggregation of
  Part 5, and `track_rates`/`paired_contrast` with image-level equivalents that
  enforce 12,630 rows per condition instead of 178 signs of 30 frames.
* Reuses the identity checks: every result directory must record its arm, seed,
  best epoch and checkpoint MD5 and must agree with the command line and with
  `checkpoint_inventory.json`; the input fingerprints must be identical across
  all nine arms for every condition, or the comparison refuses to produce an
  unpaired number.
* Adds the √30 sensitivity illustration of Part 6, in its own file and
   nowhere else, and the validation-versus-test reproduction table of Part 7.
* Memory note: the draw matrix `bootstrap_draws` produces for this split is
  2,000 × 12,630 — about 202 MB as int64 and **about 101 MB as int32**. It is
  built once per suite from seed 42, which also keeps every contrast resampled
  on identical draws as on validation, held as int32 for the duration of that
  suite, with the resampled means accumulated over it in chunks of 200 draws.
  The chunk buffers add roughly 20 MB on top, so peak working set is about
  120 MB — affordable, but dominated by the index matrix rather than by the
  chunking, which is the figure to size the run on.
* Writes to `artifacts/analysis/final_test/`, a new directory, leaving
  `artifacts/analysis/augmentation/` untouched.

#### Exact interface

```
python -m src.compare_test --model ARM:SEED:DIR [--model …] [--output-dir DIR]
```

`--model` is repeatable and uses the existing `compare_models.parse_model`
format, so the same specification string works for both comparisons. The module
refuses to start unless all nine arms × seeds are supplied and each directory
holds both suites' complete condition sets.

| Signature | Responsibility |
|---|---|
| `load_suite_images(model, suite_name)` | Reads one run's `predictions.csv` and returns per-image correctness, requiring exactly 12,630 rows per condition and the full condition set. The image-level counterpart of `compare_models.load_suite_frames`. |
| `image_rates(images, keys)` | Per-condition per-image error, then the equal-weight mean over conditions. |
| `paired_image_contrast(images, keys, left, right)` | The 12,630-vector of paired per-image differences for one contrast, left − right. |
| `bootstrap_image_mean(differences, draws, seed)` | Percentile interval over `bootstrap_draws(12630, 2000, 42)`, held as int32 and consumed in chunks of 200 draws. |
| `design_effect_illustration(interval)` | The √30 sensitivity width of Part 6, written only into the sensitivity table. |
| `validation_reproduction_rows(models)` | The validation-versus-test table of Part 7, reading `artifacts/analysis/augmentation/` read-only. |
| `compare(models, output_dir)` | Runs every declared comparison and writes the tables. |

Imported unchanged from `src/compare_models.py`, so the definitions are the
same objects rather than restatements: `ARMS`, `ARM_LABELS`, `CONTRASTS`,
`CLEAN_TRADEOFF_LIMIT`, `BOOTSTRAP_DRAWS`, `BOOTSTRAP_SEED`, `INTERVAL`,
`REQUIRED_IDENTITY`, `parse_model`, `_recorded`, `read_recorded_identity`,
`verify_model_identities`, `check_identical_examples`, `clean_rows`,
`clean_tradeoff_rows`, `_tradeoff_verdict`, `across_seeds`, `_direction`,
`targeted_verdict` and `write_table`.

Output set under `artifacts/analysis/final_test/`, the same file names as the
validation comparison so the two are read side by side, plus two files the
test stage adds:

```
arm_error_rates.csv          clean_performance.csv
paired_contrasts.csv         clean_tradeoff.csv
across_seeds.csv             per_family_contrasts.csv
per_class_recall.csv         per_severity_contrasts.csv
class_balanced_error.csv     class_balanced_contrasts.csv
comparison.txt               comparison.json
design_effect_sensitivity.csv   validation_reproduction.csv
```

The module refuses to write if `artifacts/analysis/final_test/` already holds
any of those files, and it opens `artifacts/analysis/augmentation/` and
`artifacts/robustness/` for reading only. `design_effect_sensitivity.csv` is
the only place the √30 illustration appears.

### Runtime and storage

Measured validation runtimes, excluding two machine-level stalls that added
about 26 and 27 minutes to two runs and are not extra work: 117–128 s for a
31-condition frozen sweep and 108–142 s for a 26-condition held-out sweep over
5,340 images, on `mps`, batch size 64, zero workers.

The test split is 12,630 / 5,340 = 2.37 times larger, so a frozen sweep should
take about 280–305 s and a held-out sweep about 255–335 s: roughly 9–11 minutes
per model, **80–100 minutes for all nine**. No validation inference runs, so
nothing is added to that figure beyond the clean test reproduction, which is
one of the 31 and 26 conditions already counted. The two observed stalls were
unpredictable, so the wall-clock budget should allow for more.

Storage, scaled from the committed artifacts at about 31–32 bytes per
prediction row: 391,530 frozen rows and 328,380 held-out rows per model, about
12.3 MB and 10.5 MB of `predictions.csv`, roughly **23 MB per model and 210 MB
for all nine**, with summaries, per-class metrics, confusion matrices and
metadata adding under 1 MB in total. `artifacts/` is git-ignored, so none of it
is committed.

---

## Part 10 — Verification plan

Checks 1–8, 11 and 12 run **before** any test inference and settle without
it; checks 9 and 10 are the post-run halves of snapshots taken before it. None
has been executed.

1. **Checkpoint identity and validation binding.** Recompute the MD5 and
   SHA-256 of all nine checkpoint files and require them to equal
   `checkpoint_inventory.json`. Then, for each, require both committed
   validation metadata files —
   `artifacts/robustness/<arm>_seed<seed>/frozen/metadata.json` and
   `.../heldout/metadata.json` — to carry `checkpoint.md5` equal to the
   recomputed MD5, a `clean_reproduction` block, a
   `checkpoint.recorded_validation_metrics` block and
   `integrity.clean_metrics_match_checkpoint_record: true`. That is what ties
   each file to the arm and seed it is filed under, and it was established on
   these byte-identical files by the committed validation evaluations, so no
   validation inference is rerun. All four fields are present in all 18
   committed metadata files (**verified**). Requires reading checkpoint files
   and committed JSON only; constructs no model.
2. **Protocol integrity.** Recompute the SHA-256 of
   `docs/augmentation_protocol.md`, `src/augmentation.py` and
   `src/corruptions_heldout.py` against `docs/augmentation_manifest.json`, and
   confirm the frozen modules are identical to git HEAD.
   `verify_augmentation.check_frozen_modules` currently covers
   `src/corruptions.py`, `src/data.py`, `src/model.py`, `src/train.py`,
   `src/evaluate_robustness.py` and `src/analyze_placement.py` — six modules,
   and **two short of what this protocol needs**. It must be extended to cover
   `src/evaluate_checkpoint.py`, whose `Suite` the new `TestSuite` subclasses,
   and `src/compare_models.py`, from which the new comparison imports
   `CONTRASTS`, `CLEAN_TRADEOFF_LIMIT`, `across_seeds` and
   `targeted_verdict`. Without those two, an edit to `Suite.dataset` or to the
   decision rule while implementing the test path would change the comparison
   without tripping any gate. Widening that tuple is the one change to an
   existing file this protocol asks for; it adds a check rather than altering
   behaviour, and it must land before implementation begins. Runs on no data.
3. **Dataset integrity.** Recompute the MD5 of the three downloaded GTSRB
   archives against the digests `torchvision.datasets.GTSRB` records
   (training `513f3c79a4c5141765e10e952eaa2478`, test images
   `c7e4e6327067d32654124b0fe9e82185`, test ground truth
   `fe31e9c9270bbcd7b84b7f21a9d9d9e5`) where the archives are still present,
   and otherwise verify the extracted trees by file count and by hashing file
   bytes. Hashing bytes decodes no image.
4. **Label correctness.** Parse `GT-final_test.csv` independently of
   `torchvision`, require 12,630 rows, require the filename set to equal the
   12,630 `.ppm` files on disk exactly, and require every `ClassId` in
   `[0, 43)`. Then require the independent parse to agree with the dataset's
   own `_samples` on all 12,630 `(filename, label)` pairs. Reads one CSV and
   one directory listing; opens no image.
5. **Preprocessing correctness.** Assert that `build_preprocessing()` is still
   `Resize → ToTensor → Normalize` through the existing
   `corruptions.check_stage_layout`, that the test dataset built by
   `create_data_loaders` carries that exact pipeline, and that the observation
   dataset built by `build_test_observations` carries its first two stages and
   not the third. Inspects transform objects, not images.
6. **Condition coverage.** Assert 31 frozen and 26 held-out conditions, assert
   the realised severity tables equal the committed ones value by value, and
   assert the held-out family names are disjoint from the six training
   families.
7. **Deterministic corruption realisations.** Re-derive
   `corruption_seed("final_test", ...)` and
   `heldout_sample_seed("final_test", ...)` independently of the module, for a
   fixed set of indices across all 57 conditions, and require agreement; then
   require every test seed to differ from the validation seed and from the
   training-augmentation seed at the same index and condition. Pure seed
   arithmetic on integers — no image is read.
8. **No byte-identical overlap with train or validation.** The SHA-256
   intersection check of Part 3, over 12,630 test files and 26,640 training
   files, required to be empty. Reads bytes to hash them; decodes nothing. It
   establishes the absence of duplicate files and nothing more: cross-split
   physical-sign overlap leaves no identical bytes and is not testable from the
   released test metadata, as Part 3 and Part 11 state.
9. **Checkpoints not modified.** Snapshot all nine checkpoint digests and
   file modes before the evaluation and re-verify them after. The checkpoint
   files carry mode `0644` (**verified**), not read-only, so the digest
   snapshot — not the filesystem — is what establishes they were not modified.
   The evaluator opens them with `torch.load` only, constructs no optimizer and
   populates no gradient, the last two already asserted in the code being
   reused. The inventory itself is `-r--r--r--` (**verified**).
10. **Neither the training nor the validation split is used during test
    evaluation.** The exact-count assertions of Part 9 — train 0 iterations and
    0 reads, validation 0 iterations and 0 reads — turn this into a measured
    quantity per run rather than a property of the control flow. The
    limits of the instrumentation are the ones `CountingDataset` already
    documents: it covers reads through the loaders
    `create_data_loaders` returned on that call, in that process, and a
    separately constructed dataset or a worker's copy is outside it. The runs
    use zero workers, and the module constructs no second training dataset.
11. **Validation results unchanged.** Snapshot the SHA-256 of every file under
    `artifacts/robustness/` and `artifacts/analysis/augmentation/` before the
    test evaluation and re-verify them after, and confirm the new modules
    reference no path inside either tree for writing.
12. **Suite definitions unchanged.** Confirm the test suites' `table`,
    `parameter` and `specification` attributes are the same objects as the
    validation suites', so the only difference between a validation condition
    and its test counterpart is the split and the namespace.

Checks 1–8, 11 and 12 can run as one script with no model constructed at all.
Checks 9 and 10 need the evaluation to have run and are its own exit
conditions; their pre-run halves are the snapshots.

---

## Part 11 — Known limitations

* **No physical-sign grouping.** The primary interval resamples images and
  therefore assumes an independence the data probably does not have. It is
  reported as an image-level sampling interval under that assumption, not as a
  population-level confidence interval, and it does not account for
  physical-sign dependence. The √30 calculation illustrates how far the width
  could be understated if the unobserved clusters were 30 frames and perfectly
  correlated; it is an assumption-dependent sensitivity figure, not a bound,
  and it enters no decision. Neither quantity substitutes for the sign-level
  resampling the validation analysis could do, and the test intervals are not
  directly comparable in width to the validation ones.
* **The two splits' point estimates are on the same scale; their intervals
  are not.** The per-sign averaging step is absent from the test estimator
  because the identifiers are absent, but on validation that step is a no-op:
  every condition covers every one of the 178 tracks with exactly 30 frames —
  `analyze_placement.track_rates` enforces it — so averaging within a sign and
  then over signs is arithmetically the plain per-image mean.
  `src/compare_models.py` records the same thing: steps 2 and 3 commute, and
  are applied in the declared order so the code states the protocol rather than
  an equivalent shortcut. Validation and test point estimates may therefore be
  compared in magnitude as well as in direction. What is not comparable is
  interval *width*: validation resamples 178 independent signs, the test split
  resamples 12,630 correlated images, and that gap is the whole reason the
  sensitivity illustration is computed at all.
* **Cross-split physical-sign overlap is assumed absent, not verified.** The
  hash check in Part 10 excludes byte-identical duplicates only. Whether a
  physical sign contributes frames to both the training archive and the
  official test split cannot be checked, because the test release carries no
  track identifiers. The assumption rests on the benchmark's published
  construction — the final training set is documented as the joint online
  training and test sets, a separate collection from the final test release —
  and the track-aware split this project otherwise enforces does not and cannot
  extend across that boundary. This is the one leakage mode in the study that
  is argued rather than measured.
* **Three seeds.** Three replicates give a direction and a range. They do not
  support a significance claim across seeds, and none is made on either split.
* **The test images have been read before.** One batch of 64 images per run of
  the pipeline-verification utility, producing tensor-shape and value-range
  assertions only. Part 8 records the scope exactly. No test performance
  information was produced.
* **The training archive is the online-competition training set.** 26,640
  images, not the 39,209-image final training set the benchmark intends for
  final testing. The absolute test numbers this protocol produces are therefore
  not directly comparable to published GTSRB results trained on the larger set.
  The arms are affected identically, so the contrasts — which is what the
  research question asks about — are unaffected.
* **Arm C's weights were derived from the frozen suite.** The held-out suite
  exists for that reason and stays co-primary. A test gain confined to the
  frozen suite carries the same reading on the test split as on validation.
* **Single evaluation.** The protocol is run once. If it fails for a mechanical
  reason it can be rerun, and the access log will record both attempts; it is
  not rerun because of what the numbers say.
* **Clean test accuracy is an honest estimate here; clean validation accuracy
  is not.** Validation clean accuracy selected these checkpoints. The test
  clean numbers are the first unselected clean estimates in the study, and the
  two should not be read as the same quantity.

---

## Part 12 — Execution sequence

None of the steps below has been run. The protocol is frozen first; execution
is authorised separately and begins at step 1. Steps 1–3 compute no model
output of any kind.

1. **Pre-run verification.** Widen the frozen-module tuple in
   `src/verify_augmentation.py` as Part 10 check 2 requires, then run the
   combined script covering checks 1–8, 11 and 12 of Part 10, plus
   `verify_augmentation.py` and `verify_corruptions.py`. Record the artifact
   and checkpoint digest snapshots. If any check fails, stop and report; do not
   proceed to step 2.
2. **Unit checks on the new modules.** Exercise `build_test_observations`,
   `TestSuite.dataset`, the namespace derivation and the read-count assertions
   on a small stride subset, in the manner of the existing `--smoke-test` path:
   shapes, counts, label order, fingerprint reproducibility and namespace
   separation only. No accuracy or loss is computed, printed or written.
3. **Checkpoint provenance and binding.** Run the identity and provenance
   checks for all nine checkpoints without evaluating anything: confirm each
   file still agrees with its `history.json`, `policy.json` and
   `protocol.json`, that its recomputed digests equal the inventory, that the
   arm and seed each result will be filed under are the recorded ones, and that
   both of its committed validation metadata files carry the matching
   `checkpoint.md5`, the `clean_reproduction` block, the
   `checkpoint.recorded_validation_metrics` block and
   `integrity.clean_metrics_match_checkpoint_record: true`. This step reads
   checkpoint files, sidecars and committed JSON; it constructs no model and
   runs no inference on any split.
4. **Test evaluation.** 18 runs — nine checkpoints × two suites — sequentially,
   one at a time, because the runtime is a reported number and concurrent `mps`
   runs contend. Each writes its own directory under `artifacts/final_test/`
   and appends to the access log. Logs are preserved per run. A failure stops
   the sequence and is reported before any decision about retrying.
5. **Post-run integrity.** Re-verify the checkpoint digests and file modes
   (check 9), confirm every run's recorded split counts (check 10), and
   re-verify the validation artifact digests (check 11). Confirm all 18
   directories hold complete condition sets, and verify the access log against
   the schema and counts given in Part 9.
6. **Independent re-derivation.** Recount each run's confusion matrices and
   summary accuracy from its own `predictions.csv`, and confirm the input
   fingerprints are identical across all nine arms for every condition, so the
   comparison is paired on the pixels and not merely on the seeding scheme.
7. **Statistical comparison.** Run `src/compare_test.py` over the 18 result
   directories, producing the primary rates, the three contrasts per suite per
   seed with the image-level interval, the across-seed summaries, the
   per-family and per-severity decompositions, the class-balanced secondary,
   the clean performance and trade-off tables, the √30 sensitivity table and
   the validation-versus-test reproduction table.
8. **Report.** Apply the Part 7 decision criteria as written, report
   contradictions in the summary, and state what the study concludes. No
   parameter, weight, checkpoint or statistical procedure changes after step 4
   has produced a number.

---

## Reporting choices settled here

Four reporting choices were open in the first draft. Leaving any of them open
past the first test number would be a post-hoc degree of freedom: choosing
afterwards between a wide interval and a narrow one, or whether a secondary
metric appears at all, can change how strong a conclusion looks without any
prohibited step being taken. All four are closed here, and Part 8 prohibits
reopening them.

* **The headline tables carry the point estimate and the image-level
  percentile interval, and nothing else.** The interval is labelled with its
  independence assumption wherever it appears.
* **The √30 design effect appears only in
  `design_effect_sensitivity.csv`**, as an assumption-dependent sensitivity
  illustration with k = 30 and ρ = 1 stated on the face of it. It is not a
  second interval, not a bound, and not part of the decision rule, which
  references no interval at all.
* **No filename column in `predictions.csv`.** `dataset_index` *is* the
  filename, as Part 9 establishes and the run asserts, so the column carries no
  information.
* **The class-balanced breakdown is reported, as a descriptive diagnostic
  without intervals.** Part 5 gives the reasoning. It is not a second primary
  result and cannot be read as one, because it carries no uncertainty
  quantification at all.

Two further decisions were open and are now closed the same way.

* **`docs/augmentation_protocol.md` and `docs/augmentation_manifest.json` are
  not modified.** That document's sentence "No official test image is read
  anywhere in this experiment" stays as the historical record of what was
  frozen before the nine runs were trained, and it is accurate about the
  experiment it describes. The later verification utility's access to 64 test
  images is recorded in Part 8 of this document instead, which is where the
  test stage is documented. Amending the frozen protocol would change its
  SHA-256, invalidate the manifest digest each of the nine runs recorded at
  startup, and make every future evaluation report protocol drift against a
  protocol that had not changed.
* **No validation inference is rerun.** The weights-to-recorded-metrics
  binding is already established for all nine checkpoints, on byte-identical
  files, by the committed validation evaluations; Part 9 verifies it from
  digests, provenance sidecars and those committed metadata files instead. The
  only forward passes this protocol runs are on the official test split.

No methodological decision in this document is left to be made later. Every
open question identified during the design is resolved above or in the part
that owns it, and the only quantities not yet known are the test measurements
themselves.
