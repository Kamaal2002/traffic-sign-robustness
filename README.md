# Traffic sign recognition under image corruption

Kamaal Alag

A controlled study of how a convolutional traffic-sign classifier behaves when
its inputs are degraded, and of whether the training augmentation can be
reweighted towards the model's measured weaknesses to do better than
augmenting uniformly. Everything is run on GTSRB (43 classes) with a single
architecture, a frozen evaluation protocol, and three training seeds per
condition.

The full findings, with every measured number and its limitations, are in
[docs/final_test_findings.md](docs/final_test_findings.md).

## Research question

1. How much does training-time corruption augmentation reduce error on
   corrupted inputs, including on corruption types never seen in training?
2. Does concentrating that augmentation on the families the clean baseline
   handles worst beat spreading it uniformly across families?

The second question is the one the design is built around. The answer measured
here is no, for the specific reweighting tested, and that negative result is
reported in full below.

## Approach

Three training policies, three seeds each (42, 43, 44), giving nine
checkpoints. Each policy differs from the others only in the distribution its
training images are drawn from; the model, optimiser, schedule, epoch budget
and checkpoint-selection rule are identical across all nine runs.

| Arm | Policy | Training distribution |
|---|---|---|
| A | `clean` | no corruption augmentation |
| B | `balanced` | uniform over six corruption families |
| C | `targeted` | reweighted towards the families A handles worst |

Arm A at seed 42 is the accepted baseline checkpoint, reused rather than
retrained. Neither augmented arm is warm-started from it.

B and C corrupt each training image with probability 0.5, applying one family
at one severity drawn uniformly from levels 1 to 5. C's weights come from the
baseline's per-family accuracy drop, normalised with a 5 percent floor and
fixed before any augmented training ran: occlusion 0.42, gaussian_noise 0.22,
fog 0.15, brightness_bright 0.09, brightness_dark 0.06, gaussian_blur 0.06.
B uses 1/6 for each.

Severity is drawn before the family and from the same table in both arms, and
the per-image seed does not depend on the policy, so B and C at the same
training seed augment the same images at the same severities and differ only in
which family lands. That property is checked in
`src/verify_augmentation.py` and was re-confirmed over 36,000 enumerated plans
while writing up the findings.

## Dataset and preprocessing

The torchvision GTSRB training archive used here holds 26,640 images in 888
tracks of exactly 30 frames, one track per physical sign. It is split at the
**track level**, so no sign contributes frames to both sides.

| Split | Tracks | Images |
|---|---:|---:|
| Training | 710 | 21,300 |
| Validation | 178 | 5,340 |
| Official test | — | 12,630 |

An image-level split was the pipeline's earlier behaviour and put 887 of the
888 tracks on both sides, which made validation close to a memorisation check.
The reasoning behind the track-level replacement is in
[src/README.md](src/README.md).

Preprocessing is identical for every split and every arm: resize to 64x64
bilinear with antialiasing, convert to float32 in `[0, 1]`, then
`Normalize(mean=0.5, std=0.5)` per channel. No test-time augmentation.

## Experimental setup

| Setting | Value |
|---|---|
| Model | `BaselineCNN`, four conv-BN-ReLU-pool blocks of 32/64/128/256 channels, 256-unit classifier, 1,448,779 parameters |
| Loss | cross entropy, unweighted |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4 |
| Schedule | cosine annealing over 30 epochs |
| Dropout | 0.5 in the classifier |
| Batch size | 64 |
| Checkpoint selection | highest validation macro F1 |
| Device used | MPS (Apple Silicon), torch 2.14.1 |

None of these values came from a search. They are conventional defaults fixed
once for the baseline, so any difference between the arms is not confounded
with per-arm tuning. Checkpoints are selected on clean validation macro F1;
corrupted performance never influences selection.

## Corruption suites

Corruptions are synthetic transformations with five discrete severity levels
each, defined in `src/corruptions.py` and `src/corruptions_heldout.py`. They
stand in for adverse capture conditions and are not photographs of real rain,
fog or night driving.

| Suite | Families | Conditions | Role |
|---|---|---:|---|
| original | brightness_bright, brightness_dark, fog, gaussian_blur, gaussian_noise, occlusion | 31 | the families B and C train on |
| held-out | contrast_reduction, jpeg_compression, motion_blur, rotation, salt_pepper | 26 | corruption types never sampled during training |

The code and the result files name these suites `frozen` and `heldout`. Each
suite's conditions are its families at severities 1 to 5 plus one clean
condition. The clean condition is excluded from every corruption-error average
and reported separately.

**What "held-out" means here.** The held-out suite holds out *corruption
types*, not images. Both suites are applied to the same 12,630 official test
images, so the held-out suite measures transfer to unseen perturbations, not
generalisation to an independent population of physical signs, cameras or
capture conditions.

## Main results

Official GTSRB test split, 12,630 images. Primary outcome is mean per-image
error under corruption: error per image on each condition, then an equal-weight
mean over that suite's corrupted conditions. Values are averaged over the three
training seeds.

| Suite | A `clean` | B `balanced` | C `targeted` |
|---|---:|---:|---:|
| original | 0.1628 | 0.0701 | 0.0657 |
| held-out | 0.1372 | 0.0817 | 0.0867 |

Paired within-image contrasts, negative favouring the left arm:

| Suite | Contrast | Mean | Per seed (42 / 43 / 44) | Seeds favouring left |
|---|---|---:|---|---|
| original | B−A | −0.0926 | −0.0921 / −0.0801 / −0.1057 | 3 of 3 |
| original | C−B | −0.0045 | +0.0003 / −0.0128 / −0.0009 | 2 of 3 |
| held-out | B−A | −0.0555 | −0.0737 / −0.0302 / −0.0624 | 3 of 3 |
| held-out | C−B | +0.0050 | +0.0079 / +0.0002 / +0.0069 | 0 of 3 |

Clean test accuracy is 0.9820 for A, 0.9825 for B and 0.9822 for C averaged
across seeds, a span of 0.00053. No arm pays a meaningful clean-accuracy cost.

**The predefined decision rule and its verdict.** The rule, fixed before any
test image was classified, required C's mean per-image error to be lower than
B's at every one of the three seeds on both suites. C is lower at two of three
seeds on the original suite and at none of the three on the held-out suite, so
**the targeted weighting is not preferred over balanced augmentation.** The
protocol's 0.01 clean-accuracy limit is a reporting rule alongside that
verdict, not a pass condition: a cost above it would have been reported as a
trade-off rather than an improvement. No arm came near it, so nothing here was
reported as a trade-off.

## Interpretation

The large and seed-consistent effect came from augmenting with corruptions at
all. B−A is negative at every seed on both suites. On the across-seed mean, B
cuts the clean-trained arm's corruption error by 57 percent on the original
suite and by 40 percent on the held-out suite, but the per-seed reductions
range from 49 to 62 percent and from 25 to 50 percent, so it is the direction
rather than the size of the effect that repeats. There is no meaningful
clean-accuracy cost, and the gain grows with severity.

The targeted reweighting helped on the family it was aimed at and lost
elsewhere. On the original suite C improves over B on occlusion at all three
seeds, by 0.0380 on the across-seed mean, and is worse on the four families it
down-weighted; the gain is partly offset by those losses, leaving a modest
average advantage of 0.0045 that two of the three seeds support. On the
held-out suite C is worse than B at every seed. The alignment between the
weights and the family-level outcomes is an association observed in this
experiment rather than an established rule, since no intermediate weighting was
trained and a weight raised on one family is a weight removed from the others.

This is evidence against the particular weighting tested, not evidence that
augmentation weighting is unimportant in general. Neither augmented arm helped
on held-out `rotation`, a geometric perturbation unlike any of the six training
families.

The per-family, per-severity, per-class and class-balanced breakdowns, the
validation-against-test comparison and the full statistical limitations are in
[docs/final_test_findings.md](docs/final_test_findings.md).

## Reproducing the study

Python 3.13 with the pinned versions in `requirements.txt`:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`data/` is Git-ignored, so the dataset is not shipped with the repository, and
nothing in `src/` downloads it: every dataset call there passes
`download=False`. The archives were fetched once through
`notebooks/01_data_exploration.ipynb`, and the same two lines populate `data/`
in a fresh clone:

```sh
.venv/bin/python -c "from torchvision.datasets import GTSRB; \
GTSRB(root='data', split='train', download=True); \
GTSRB(root='data', split='test', download=True)"
```

Run every command from the project root. These four modules train nothing and
read no official test image:

```sh
.venv/bin/python -m src.verify_data_pipeline
.venv/bin/python -m src.verify_model
.venv/bin/python -m src.verify_corruptions
.venv/bin/python -m src.verify_augmentation
```

The first three run on a fresh clone as soon as `data/` is populated.
`src.verify_augmentation` does not: its provenance check requires
`artifacts/baseline/best_checkpoint.pt` at the MD5 recorded in
`src/evaluate_robustness.py`, so it passes against the accepted baseline and
not against a retrained one.

Training the nine checkpoints, 30 epochs each. Each command writes its own
directory under `artifacts/`:

```sh
.venv/bin/python -m src.train                                      # A, seed 42
.venv/bin/python -m src.train_clean --seed 43                      # A, seed 43
.venv/bin/python -m src.train_clean --seed 44                      # A, seed 44
.venv/bin/python -m src.train_augmented --policy balanced --seed 42
.venv/bin/python -m src.train_augmented --policy balanced --seed 43
.venv/bin/python -m src.train_augmented --policy balanced --seed 44
.venv/bin/python -m src.train_augmented --policy targeted --seed 42
.venv/bin/python -m src.train_augmented --policy targeted --seed 43
.venv/bin/python -m src.train_augmented --policy targeted --seed 44
```

`--policy` accepts `balanced` or `targeted`; `--seed` is a plain integer, and
the three seeds used in this study are 42, 43 and 44. `--smoke-test` runs two
short epochs into a separate directory. Output directories are
`artifacts/baseline`, `artifacts/clean_seed43`, `artifacts/clean_seed44` and
`artifacts/augmented_<policy>_seed<seed>`.

`src.train_clean` and `src.train_augmented` refuse to write into a directory
that already holds a run, and refuse `artifacts/baseline` outright. `src.train`
has no such guard: it writes `artifacts/baseline` with `exist_ok=True` and
would silently overwrite an existing accepted baseline, so run it only when
that directory is absent or pass an explicit `--output-dir`.

Validation-stage corruption evaluation, one call per checkpoint and suite, 18
calls in total. Results land in `artifacts/robustness/<arm>_seed<seed>/<suite>`
and the module refuses to evaluate into a directory that already holds results:

```sh
.venv/bin/python -m src.evaluate_checkpoint \
    --checkpoint artifacts/baseline/best_checkpoint.pt \
    --model-name clean --suite frozen
.venv/bin/python -m src.evaluate_checkpoint \
    --checkpoint artifacts/augmented_balanced_seed42/best_checkpoint.pt \
    --suite frozen
```

`--model-name` labels the arm. The augmented runs take it from the
`policy.json` beside the checkpoint; the arm-A runs have no `policy.json` and
fall back to `clean`, so passing it there is an explicit override rather than a
requirement. Repeat both calls with `--suite heldout`.

The validation comparison takes all nine cells in one call:

```sh
.venv/bin/python -m src.compare_models \
    --model clean:42:artifacts/robustness/clean_seed42 \
    --model clean:43:artifacts/robustness/clean_seed43 \
    --model clean:44:artifacts/robustness/clean_seed44 \
    --model balanced:42:artifacts/robustness/balanced_seed42 \
    --model balanced:43:artifacts/robustness/balanced_seed43 \
    --model balanced:44:artifacts/robustness/balanced_seed44 \
    --model targeted:42:artifacts/robustness/targeted_seed42 \
    --model targeted:43:artifacts/robustness/targeted_seed43 \
    --model targeted:44:artifacts/robustness/targeted_seed44
```

### The official test stage

This stage follows `docs/final_test_protocol.md`, frozen in commit `f16c8a5`
before it was run, and it is a one-shot path. Step 1 records an integrity
snapshot of the validation artifacts and the nine checkpoints; step 5 requires
that snapshot back.

```sh
PYTHONPATH=. .venv/bin/python -m src.verify_test_path \
    --record-snapshot artifacts/final_test_integrity/pre_run_snapshot.json
PYTHONPATH=. .venv/bin/python -m src.run_final_test
PYTHONPATH=. .venv/bin/python -m src.run_final_test --execute
PYTHONPATH=. .venv/bin/python -m src.verify_test_path --post-run \
    --verify-snapshot artifacts/final_test_integrity/pre_run_snapshot.json
PYTHONPATH=. .venv/bin/python -m src.compare_test \
    --model clean:42:artifacts/final_test/clean_seed42 \
    --model clean:43:artifacts/final_test/clean_seed43 \
    --model clean:44:artifacts/final_test/clean_seed44 \
    --model balanced:42:artifacts/final_test/balanced_seed42 \
    --model balanced:43:artifacts/final_test/balanced_seed43 \
    --model balanced:44:artifacts/final_test/balanced_seed44 \
    --model targeted:42:artifacts/final_test/targeted_seed42 \
    --model targeted:43:artifacts/final_test/targeted_seed43 \
    --model targeted:44:artifacts/final_test/targeted_seed44
```

`src.run_final_test` without `--execute` prints the plan and exits: nothing is
run, nothing is created, and no official test image is read. It does read the
checkpoint inventory and recompute the nine checkpoint digests, which is how it
validates the plan. `--record-snapshot` refuses an existing file, and
`--post-run` refuses to run without `--verify-snapshot`. The default pre-run
mode also checks that `artifacts/final_test/` does not exist yet, which is why
it cannot be rerun after the sequence. The eighteen runs took 2 h 29 min on the
machine used here.

`src.compare_test` also reads the validation comparison from
`artifacts/analysis/augmentation/across_seeds.csv` to build
`validation_reproduction.csv`, and has no flag pointing elsewhere, so the
earlier `src.compare_models` call has to have written that file first.

**One prerequisite this repository cannot supply.**
`src/evaluate_test.py` reads `artifacts/robustness/checkpoint_inventory.json`
and will only evaluate a checkpoint that is listed there with matching MD5 and
SHA-256 digests. That inventory is under the ignored `artifacts/` tree, no
module in the repository generates it, and retrained checkpoints will not carry
the recorded digests. A fresh clone therefore has to write its own inventory
for its own nine checkpoints, in the same schema, before this stage will run.
That reproduces the methodology, not the accepted evaluation.

### Figures

The four figures under `figures/` are drawn from the tables
`src.compare_test` wrote. No error rate and no interval is recomputed; the only
aggregation the figures add is the mean across the three seeds, which
reproduces `across_seeds.csv`.

```sh
.venv/bin/python -m src.plot_paper_figures
```

The script reads `artifacts/analysis/final_test/` and writes a PDF and a PNG
for each figure; `--analysis-dir` and `--figure-dir` move either end. It stops
if that directory does not record the official test split, since the same table
names are written for validation.

* [Figure 1](figures/fig1_overall_corruption_error.pdf) — mean per-image error
  for the three policies on each suite, with the three seeds also marked
  individually.
* [Figure 2](figures/fig2_per_family_contrasts.pdf) — the B−A and C−B
  differences family by family, both columns ordered by the B−A effect so the
  rows line up. The four panels carry different horizontal scales, as the
  figure states.
* [Figure 3](figures/fig3_severity.pdf) — error against severity levels 1 to 5,
  and the C−B difference at each level.
* [Figure 4](figures/fig4_clean_versus_corruption.pdf) — clean accuracy against
  corruption error for all nine checkpoints.

### What the repository does and does not contain

`artifacts/` is Git-ignored in its entirety, so checkpoints, per-image
prediction files, result tables and the comparison outputs are not in the
repository. What is committed is everything needed to regenerate them: the
source, the frozen protocols in `docs/`, the pinned dependency versions, and
the fixed seeds. Reproducing the reported numbers means retraining the nine
checkpoints and rerunning the evaluations with the commands above. Exact
bit-level agreement should only be expected on the same device type and library
versions, since CPU, MPS and CUDA use different kernels and reduction orders.

## Repository structure

```
src/
  data.py                  track-level GTSRB split and preprocessing
  model.py                 BaselineCNN
  train.py                 clean baseline training and checkpoint selection
  train_clean.py           arm A replicates at seeds 43 and 44
  train_augmented.py       arms B and C
  augmentation.py          the balanced and targeted training policies
  corruptions.py           the six original-suite corruption families
  corruptions_heldout.py   the five held-out families
  evaluate_robustness.py   baseline degradation curve on validation
  evaluate_checkpoint.py   any checkpoint under either suite, on validation
  compare_models.py        paired sign-level validation comparison
  evaluate_test.py         one checkpoint, one suite, official test split
  run_final_test.py        the eighteen official test runs in a fixed order
  compare_test.py          paired image-level official test comparison
  verify_*.py              checks for each stage; they train nothing
  occlusion_placement.py   occlusion patch placement, with its own study in
                           analyze_placement.py and evaluate_placement.py
  analyze_failures.py      per-class and per-condition failure breakdown,
                           plotted by plot_failures.py
  plot_paper_figures.py    the four paper figures, from the test analysis
  plot_robustness.py       degradation curves
  visualize_corruptions.py severity grids for each family
  README.md                data pipeline and baseline, with the design reasoning
docs/
  augmentation_protocol.md the augmentation design, frozen before training
  final_test_protocol.md   the official test protocol, frozen before the runs
  final_test_deviations.md departures from the frozen protocol, with dates
  final_test_findings.md   the official test results and their limitations
notebooks/
  01_data_exploration.ipynb
figures/
  fig1 .. fig4             the paper figures, PDF and PNG
```

## Limitations

* **Synthetic corruptions.** Severity levels are parameter settings, not
  calibrated physical conditions. Nothing here measures real adverse-weather
  driving performance.
* **One architecture, one dataset.** All results are for the 1.45M-parameter
  `BaselineCNN` on GTSRB at 64x64.
* **Bootstrap intervals assume independent test images.** GTSRB is built from
  roughly 30-frame sequences per physical sign and the official test release
  carries no track identifiers, so the clustering cannot be measured and the
  intervals are narrower than a dependence-aware interval would be. They are
  not definitive population-level confidence intervals, and no interval enters
  the decision rule.
* **Three seeds give a direction and a spread, not significance.** No test
  across seeds was computed, and seed variability is not pooled with
  image-level sampling uncertainty.
* **The held-out suite is held out in corruption type only.** Both suites use
  the same 12,630 test images.
* **The validation split carries a lot of load.** It selected the baseline
  checkpoint, produced the robustness curve that C's weights were derived from,
  and selects every arm's checkpoint. The held-out suite exists because of
  that, and the official test split was evaluated once, after the protocol was
  frozen.
* **Not comparable with published GTSRB numbers.** The torchvision training
  archive used here holds 26,640 images, not the commonly cited 39,209-image
  variant.
