# Traffic sign recognition under image corruption

Kamaal Alag

**Research paper:** [Evaluating Weakness-Targeted Corruption Augmentation for
Robust Traffic Sign Recognition](paper/Traffic_Sign_Robustness.pdf)

This project tests whether training a traffic-sign classifier with corruption
augmentation aimed at its measured weaknesses improves robustness more than
sampling corruption families uniformly. I compare clean training, balanced
augmentation and weakness-targeted augmentation across nine CNN checkpoints on
GTSRB (43 classes), with the two augmented policies matched on augmentation
rate and severity distribution so that only the family weighting differs. All
nine are evaluated on the official test split, against both the corruption
families used in training and a set of held-out synthetic perturbations. The
final evaluation protocol was frozen before the official test split was used to
measure model performance.

**Main result.** Balanced augmentation reduced mean corruption error by about
57 percent on the original suite and about 40 percent on the held-out suite
relative to clean training, and it was lower at every training seed on both
suites. Targeted augmentation improved the original-suite average slightly but
did not beat balanced augmentation consistently across seeds, and it was worse
than balanced augmentation on the held-out suite at all three seeds. The
predefined success criterion for the targeted policy was therefore not met. No
statistical test across seeds was performed, so these are observed directions
and spreads rather than statistically established effects.

## Research question

Does allocating more augmentation probability to the corruption families that
cause the largest baseline performance degradation improve robustness compared
with balanced sampling?

A second question runs alongside it: how far does either form of augmentation
carry to corruption types never sampled during training? A policy that only
helps on the families it trained on is a narrower result than it first appears,
so both are also evaluated on a held-out suite.

## Experimental design

Three training policies, three seeds each (42, 43, 44), giving nine
checkpoints.

| Arm | Policy | Training distribution |
|---|---|---|
| A | `clean` | no corruption augmentation |
| B | `balanced` | uniform over the six corruption families |
| C | `targeted` | reweighted towards the families A handles worst |

The architecture, optimiser, schedule, epoch budget and batch size are
identical across all nine runs, and every checkpoint is selected on clean
validation macro F1, so corrupted performance never influences selection. B and
C corrupt each training image with probability 0.5 and draw one severity
uniformly from levels 1 to 5. The only difference is the family distribution: B
gives each family weight 1/6, while C's weights come from the clean baseline's
per-family accuracy drop measured on the validation split, normalised with a 5
percent floor and fixed before any augmented training ran — occlusion 0.42,
gaussian_noise 0.22, fog 0.15, brightness_bright 0.09, brightness_dark 0.06,
gaussian_blur 0.06.

Severity is drawn before the family, from the same table in both arms, and the
per-image seed does not depend on the policy, so B and C at the same seed
augment the same images at the same severities and differ only in which family
lands. That is what makes the B-versus-C contrast a test of the weights rather
than of the augmentation rate. Arm A at seed 42 is the accepted baseline
checkpoint, reused rather than retrained.

**Training.** Images are resized to 64x64 and normalised to [-1, 1]. The
1.45M-parameter `BaselineCNN` is trained for exactly 30 epochs with AdamW at
learning rate 1e-3 and weight decay 1e-4, cosine-annealed over the 30 epochs,
batch size 64 and unweighted cross-entropy. The reported checkpoint is the
epoch with the highest clean validation macro F1, with ties broken on
validation accuracy, then loss, then the earlier epoch.

**Evaluation.** Each checkpoint is scored twice, once per suite, on every
condition in that suite. Arm C's weights come from an earlier stage:
`src.evaluate_robustness` measured the clean baseline's per-family drops on the
validation split before any augmented arm existed, and the resulting weights
are hard-coded constants in `src/augmentation.py`. The validation stage
(`src.evaluate_checkpoint`, `src.compare_models`) then scores all nine trained
checkpoints on the validation split. The official-test stage
(`src.run_final_test`, `src.compare_test`) is the frozen one: it runs each of
the nine checkpoints over both suites on the 12,630 official test images, under
a protocol pinned by SHA-256 inside the evaluation script, and writes the
per-family, per-severity, paired-contrast and class-balanced tables reported
here and in [docs/final_test_findings.md](docs/final_test_findings.md).

**Data.** The torchvision GTSRB training archive holds 26,640 images in 888
tracks of 30 frames, one track per physical sign, split at the track level so
no sign contributes frames to both sides. An image-level split of this archive
put 887 of the 888 tracks on both sides, which would leave validation close to
a memorisation check.

| Split | Tracks | Images |
|---|---:|---:|
| Training | 710 | 21,300 |
| Validation | 178 | 5,340 |
| Official test | — | 12,630 |

**Corruption suites.** Corruptions are synthetic transformations with five
discrete severity levels each.

| Suite | Families | Corrupted conditions | Total evaluated |
|---|---|---:|---:|
| original | brightness_bright, brightness_dark, fog, gaussian_blur, gaussian_noise, occlusion | 30 | 31 |
| held-out | contrast_reduction, jpeg_compression, motion_blur, rotation, salt_pepper | 25 | 26 |

A suite's corrupted conditions are its families at severities 1 to 5: six
families give 30, five give 25. Each suite also evaluates one clean
reference condition, which is where the "total evaluated" column's extra
entry comes from. The clean condition is excluded from every
corruption-error average and reported separately, so the primary metric
averages over 30 conditions on the original suite and 25 on the held-out
suite. The code and result files name them `frozen` and `heldout`. The
held-out suite holds out corruption *types*, not images: both suites are
applied to the same 12,630 official test images, so it measures transfer to
unseen perturbations rather than generalisation to new signs or cameras.

Full details are in the frozen
[augmentation](docs/augmentation_protocol.md) and
[final test](docs/final_test_protocol.md) protocols, with recorded departures
in [docs/final_test_deviations.md](docs/final_test_deviations.md).

## Results

Mean per-image error under corruption on the official test split: error per
image on each condition, then a mean over that suite's corrupted conditions
with every condition weighted equally. Clean conditions are excluded. Lower is
better, and all values are averaged over the three training seeds. Both suites
use the same official test images.

| Suite | A: clean | B: balanced | C: targeted |
|---|---:|---:|---:|
| original | 0.1628 | 0.0701 | 0.0657 |
| held-out | 0.1372 | 0.0817 | 0.0867 |

Paired contrasts at matched training seeds, negative favouring the left
arm:

| Suite | Contrast | Mean | Seed 42 | Seed 43 | Seed 44 | Favouring left |
|---|---|---:|---:|---:|---:|---|
| original | B−A | −0.0926 | −0.0921 | −0.0801 | −0.1057 | 3 of 3 |
| original | C−A | −0.0971 | −0.0918 | −0.0928 | −0.1066 | 3 of 3 |
| original | C−B | −0.0045 | +0.0003 | −0.0128 | −0.0009 | 2 of 3 |
| held-out | B−A | −0.0555 | −0.0737 | −0.0302 | −0.0624 | 3 of 3 |
| held-out | C−A | −0.0505 | −0.0658 | −0.0300 | −0.0556 | 3 of 3 |
| held-out | C−B | +0.0050 | +0.0079 | +0.0002 | +0.0069 | 0 of 3 |

Augmenting at all is the large effect: both augmented arms are lower than clean
training at every seed on both suites. Balanced augmentation's per-seed
reductions range from 49 to 62 percent on the original suite and from about 25
to 50 percent on the held-out suite, so the direction repeats more reliably
than the size. Only one contrast fails to repeat across seeds, and only on the
original suite: the targeted policy's 0.0045 advantage there is an average over
seeds that disagree, two supporting it and one reversing it. On the held-out
suite balanced augmentation is lower at all three seeds.

**The predefined success criterion and its outcome.** The rule was fixed with
the design, before any test image was classified: C is preferred over B only if
C's mean per-image error is lower than B's at every one of the three training
seeds, on both suites. C is lower at two of three seeds on the original suite
and at none of the three on the held-out suite, so the criterion is not met and
the targeted weighting is not preferred over balanced augmentation. The
protocol's 0.01 clean-accuracy limit is a reporting rule that sits alongside
the verdict rather than a pass condition; no arm came near it.

Clean test accuracy is about 98.2 percent for all three policies: 0.9820,
0.9825 and 0.9822 for A, B and C averaged across seeds, a span of 0.00053. No
arm pays a meaningful clean-accuracy cost for its robustness. The per-family
and per-severity breakdowns, the bootstrap procedure and its assumptions, the
validation-against-test comparison and the class-balanced sensitivity analysis
are in [docs/final_test_findings.md](docs/final_test_findings.md). The
bootstrap interval endpoints are in
[results/paired_contrasts.csv](results/paired_contrasts.csv) and the full
per-class tables in
[results/per_class_recall.csv](results/per_class_recall.csv).

## Published experimental results

[results/](results/) holds the aggregate analysis tables written by
`src.compare_test` at the end of the official-test evaluation, the provenance
record written beside them, and the baseline validation summary the targeted
weights were derived from. Every aggregate result quoted in this README, in
the findings and in the manuscript is read from these files, so the reported
findings can be inspected, and the four figures rebuilt, without retraining
the nine checkpoints. The one table the run wrote that is not here as its own
file, the validation-against-test comparison, is embedded in
`comparison.json`.

| File | Contents |
|---|---|
| [arm_error_rates.csv](results/arm_error_rates.csv) | Mean per-image corruption error for each arm, seed and suite, with bootstrap interval endpoints |
| [across_seeds.csv](results/across_seeds.csv) | Each contrast summarised over the three seeds: mean, range, and the count of seeds favouring each arm |
| [paired_contrasts.csv](results/paired_contrasts.csv) | The 18 paired differences (B−A, C−A, C−B) at matched seeds, with interval endpoints |
| [per_family_contrasts.csv](results/per_family_contrasts.csv) | The same contrasts split by corruption family |
| [per_severity_contrasts.csv](results/per_severity_contrasts.csv) | The same contrasts split by severity level |
| [per_class_recall.csv](results/per_class_recall.csv) | Clean recall and mean corrupted recall for each of the 43 classes, per checkpoint and suite, with each class's test support |
| [class_balanced_error.csv](results/class_balanced_error.csv) | Arm error under the class-balanced metric |
| [class_balanced_contrasts.csv](results/class_balanced_contrasts.csv) | Paired contrasts under the class-balanced metric |
| [clean_performance.csv](results/clean_performance.csv) | Clean test accuracy, macro F1 and loss for each of the nine checkpoints |
| [clean_tradeoff.csv](results/clean_tradeoff.csv) | Clean-accuracy cost against corrupted-error change per contrast, with the protocol's 0.01 reporting limit applied |
| [design_effect_sensitivity.csv](results/design_effect_sensitivity.csv) | What the paired intervals would become if the test images were clustered in groups of 30 at perfect within-cluster correlation. An assumption-dependent illustration, not a corrected interval and not an input to the decision rule |
| [comparison.json](results/comparison.json) | Provenance for the run: split and namespace, per-checkpoint digests and selected epochs, the frozen protocol settings and the decision rule. It also embeds seven of the tables above, all but the per-family, per-severity and per-class breakdowns and the design-effect illustration, plus the validation-against-test rows. `src.plot_paper_figures` reads it to establish which split the directory came from |
| [baseline_validation_summary.csv](results/baseline_validation_summary.csv) | The clean baseline's loss, accuracy and macro F1 on the validation split at every family and severity, over the 5,340 validation images. Its per-family accuracy drops are what the targeted probabilities were derived from. Measured on checkpoint MD5 `7c6ea3f9773d7a644235463d6a02bf0f` at epoch 21, the same checkpoint `comparison.json` records for arm A at seed 42. Written by `src.evaluate_robustness` as `artifacts/robustness/baseline/summary.csv` and renamed here |

These are derived results, not model checkpoints and not per-image
predictions. All but the last are copied unmodified from the completed
official-test run, which evaluated nine checkpoints across three training
strategies over both suites; `baseline_validation_summary.csv` is copied
unmodified from the earlier validation-split measurement of the clean baseline
that fixed the targeted weights. They are the recorded output of those runs
rather than a regeneration of them. Rerunning the training code does not
guarantee the same numbers: the checkpoints are not published, and backends
differ in kernels and reduction orders.

## Figures

![Mean per-image error under corruption for the three training policies on each suite, with the three seeds marked individually](figures/fig1_overall_corruption_error.png)

Figure 1. Mean per-image error under corruption for each policy on each suite,
averaged over seeds 42, 43 and 44. Bars are the across-seed mean; the open
markers are the three individual seeds. The markers show seed spread and are
not uncertainty intervals.

* [Figure 2](figures/fig2_per_family_contrasts.png) — the B−A and C−B
  differences family by family, both columns ordered by the B−A effect so the
  rows line up. Each contrast column shares a horizontal scale across the
  original and held-out suites; the B−A and C−B columns have different scales.
* [Figure 3](figures/fig3_severity.png) — error against severity levels 1 to 5
  for each policy, and the C−B difference at each level.
* [Figure 4](figures/fig4_clean_versus_corruption.png) — clean accuracy against
  corruption error for all nine checkpoints.

## Interpretation

The improvement from augmenting at all grows with severity, from 0.021 at
severity 1 to 0.186 at severity 5 on the original suite, and it carries to
corruption types the model never trained on: held-out error is lower than A's
at every seed, and four of that suite's five families improve on the
across-seed mean.

Within the augmented arms, the targeted policy improved occlusion, the family
it weighted most heavily, at all three seeds, and gave mixed or worse results
on the other families. That pattern is consistent with a trade-off from
concentrating augmentation on occlusion, but this experiment cannot isolate the
effect of raising one family's weight from that of lowering the others, since
no intermediate weighting was trained. These are observations about where the
differences fall, not a demonstrated mechanism.

The targeted policy did not meet its predefined success criterion. That is
evidence against the particular weighting tested, under this architecture, this
dataset and these synthetic corruption settings. It does not show that targeted
augmentation fails in general, and it does not rule out a different weighting
rule or weights derived from a broader set of families.

## Reproducing the study

[docs/reproduction.md](docs/reproduction.md) is the full workflow behind the
reported results, from a fresh clone through the official test stage, with
every prerequisite and one-shot restriction. Three things worth separating
first.

*Rerunning the methodology* is supported. The official-test stage needs a
checkpoint inventory that is not committed and that no module here generates;
it comes with the published checkpoints, under
[Trained checkpoints](#trained-checkpoints) below. *Reproducing the reported
numbers* is not expected to match bit for bit, since different backends use
different kernels and reduction orders.
*Inspecting the reported results* needs neither: the aggregate tables are in
[results/](results/), together with the provenance record the plotting script
checks. On a fresh clone, with nothing trained and no image decoded,

```sh
.venv/bin/python -m src.plot_paper_figures \
    --analysis-dir results --figure-dir /tmp/figures
```

rebuilds all four figures from the published tables. On the pinned
`matplotlib==3.11.2` the PNGs it writes are byte-for-byte identical to the
ones committed under `figures/`. Section 1 of the guide has the detail.

Quick start, from the project root:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "from torchvision.datasets import GTSRB; \
GTSRB(root='data', split='train', download=True); \
GTSRB(root='data', split='test', download=True)"
```

Nothing in `src/` enables torchvision's `download` flag, so the dataset has to
be fetched explicitly as above. Once `data/` is populated, these three run on a
fresh clone and train nothing:

```sh
.venv/bin/python -m src.verify_data_pipeline
.venv/bin/python -m src.verify_model
.venv/bin/python -m src.verify_corruptions
```

`src.verify_data_pipeline` decodes 64 official test images on every run, to
check batch shape, dtype and pixel range across all three splits. It builds no
model and makes no prediction, so no test performance information comes out of
it, but the test split is not completely untouched; Part 8 of
[docs/final_test_protocol.md](docs/final_test_protocol.md) records that access.
A fourth check, `src.verify_augmentation`, needs the uncommitted baseline
checkpoint, so on a clone it runs only once the published checkpoints are in
place. Section 4 of the guide has the detail.

The nine training runs use three entry points. `src.train` produces the
accepted arm-A baseline at seed 42; `src.train_clean` produces the other two
arm-A seeds; `src.train_augmented` produces both augmented arms.

```sh
.venv/bin/python -m src.train
.venv/bin/python -m src.train_clean --seed 43
.venv/bin/python -m src.train_clean --seed 44
for policy in balanced targeted; do
  for seed in 42 43 44; do
    .venv/bin/python -m src.train_augmented --policy "$policy" --seed "$seed"
  done
done
```

Validation-stage evaluation is one call per checkpoint and suite, eighteen in
total, followed by one comparison call that takes all nine cells at once:

```sh
.venv/bin/python -m src.evaluate_checkpoint \
    --checkpoint artifacts/augmented_balanced_seed42/best_checkpoint.pt \
    --suite frozen

models() { for arm in clean balanced targeted; do for seed in 42 43 44; do
    printf -- '--model %s:%s:%s/%s_seed%s ' "$arm" "$seed" "$1" "$arm" "$seed"
  done; done; }

.venv/bin/python -m src.compare_models $(models artifacts/robustness)
```

The official-test stage stops here on a fresh clone. `src/evaluate_test.py`
only evaluates checkpoints listed in
`artifacts/robustness/checkpoint_inventory.json` with matching MD5 and SHA-256,
that file is not committed, and no module in this repository writes one. It
comes with the published checkpoints; without them the inventory has to be
authored by hand, and section 8 of the guide gives the required fields.

```sh
PYTHONPATH=. .venv/bin/python -m src.verify_test_path \
    --record-snapshot artifacts/final_test_integrity/pre_run_snapshot.json
PYTHONPATH=. .venv/bin/python -m src.run_final_test
PYTHONPATH=. .venv/bin/python -m src.run_final_test --execute
PYTHONPATH=. .venv/bin/python -m src.verify_test_path --post-run \
    --verify-snapshot artifacts/final_test_integrity/pre_run_snapshot.json

.venv/bin/python -m src.compare_test $(models artifacts/final_test)
.venv/bin/python -m src.plot_paper_figures
```

Without `--execute`, `src.run_final_test` prints the plan and exits without
reading an official test image. The sequence is one-shot by construction: the
pre-run check refuses to run if `artifacts/final_test/` already exists, and the
evaluator refuses any non-empty output directory. `src.compare_test` requires
all nine arms and seeds and reads the validation comparison's
`across_seeds.csv`, so the validation stage is a prerequisite rather than an
optional diagnostic. `src.plot_paper_figures` defaults to
`artifacts/analysis/final_test/` and refuses an analysis directory that does
not record the official test split.

## Trained checkpoints

`artifacts/` is Git-ignored, so the nine accepted checkpoints and the
evaluation records that bind each one to its arm and seed are not in the
repository. They are published instead on this repository's Releases page,
under the tag `v1.0-checkpoints`:

* `traffic-sign-robustness-checkpoints-v1.0.tar.gz`
* `SHA256SUMS.txt` — one SHA-256 per extracted file
* `SHA256SUMS.archive.txt` — the digest of the archive itself

Download all three into the project root and extract there. The archive
carries repository-relative `artifacts/` paths, so nothing needs rearranging,
and `.gitignore` keeps the extracted tree out of `git status`. Extraction
writes over a file of the same name, so unpack somewhere else first if the
working copy already holds results:

```sh
shasum -a 256 -c SHA256SUMS.archive.txt
tar -xzf traffic-sign-robustness-checkpoints-v1.0.tar.gz
shasum -a 256 -c SHA256SUMS.txt
```

The archive holds the nine `best_checkpoint.pt` files with whichever of the
`history.json`, `policy.json` and `protocol.json` sidecars each run produced:
all nine have a history, the accepted baseline predates both of the others,
and the clean arm has no augmentation policy to record. With them come
`checkpoint_inventory.json`, the validation-stage records for all nine arms
and seeds on both suites, the clean baseline's per-family validation sweep
that arm C's probabilities were derived from, and the validation comparison
tables.

The per-image validation predictions are included, for all nine checkpoints
on both suites, so the validation tables the archive ships are re-derivable
from it and not merely readable: `src.compare_models` rebuilds all twelve of
them byte for byte. What the archive leaves out is the dataset, the
official-test output and the earlier occlusion-placement stage, which is why
`src.analyze_placement` is the one analysis entry point a download cannot
run — it reads that stage's predictions, not the robustness sweeps'.

The digests do not have to be taken on trust. Every checkpoint's MD5 and
SHA-256 were fixed in `checkpoint_inventory.json` before the official-test
stage ran, and `src/evaluate_test.py` recomputed both and refused any
mismatch at every one of the eighteen runs. They are carried into the
repository by [results/comparison.json](results/comparison.json): the MD5
under `model_identity`, both digests under `test_provenance`. A download can
therefore be checked against a record that predates the results rather than
against whoever uploaded it. That holds for the nine weight files. Of the other
150 entries, four are MD5-pinned in committed source — `ACCEPTED_ARTIFACTS` in
`src/verify_placement.py` covers the baseline sweep's `predictions.csv`,
`summary.csv`, `per_class_metrics.csv` and `confusion_matrices.npz` — and the
validation across-seed means are carried in `results/comparison.json`. The
rest are fixed only by `SHA256SUMS.txt`, which is published alongside them
rather than independently of them.

With the archive in place, the steps a clone cannot complete on its own run:

```sh
.venv/bin/python -m src.verify_augmentation
PYTHONPATH=. .venv/bin/python -m src.verify_placement
.venv/bin/python -m src.compare_models $(models artifacts/robustness)
PYTHONPATH=. .venv/bin/python -m src.analyze_failures
PYTHONPATH=. .venv/bin/python -m src.verify_test_path \
    --record-snapshot artifacts/final_test_integrity/pre_run_snapshot.json
PYTHONPATH=. .venv/bin/python -m src.run_final_test
```

The first needs the baseline checkpoint at a pinned MD5. The second re-derives
the occlusion placement rule on synthetic frames and checks five accepted
artifacts against pinned digests, among them the baseline checkpoint and its
validation predictions. The next two rebuild the validation comparison and the
baseline failure analysis from the per-image predictions, the first using the
`models()` helper defined with the validation stage above. The fifth is the
pre-run integrity battery, which among other things ties each checkpoint to
the arm and seed its results will be filed under. The last prints and
validates the eighteen-run plan without reading an official test image; adding
`--execute` runs the evaluation, which the protocol budgets at 80 to 100
minutes for all nine checkpoints on an Apple M-series GPU. The accepted run
took 2:28:49, with one of the eighteen sweeps taking 47:51 against about five
minutes for each of the others. After it, `src.compare_test` rebuilds the
comparison, and the rest of the sequence is as set out above.

Two limits. The checkpoints carry model weights, the training configuration and
the recorded validation metrics, but no optimizer state, so they support
inference rather than resumed training. And a rerun cannot promise the same
numbers. Each corrupted input is a function of its image index and condition,
so the inputs are reproduced exactly on any machine, with the one exception
the held-out suite documents: `jpeg_compression` decodes through whichever
libjpeg build `torchvision` is linked against, which is stable within an
environment but not guaranteed across libjpeg versions. The forward pass is
not reproduced exactly either, since CPU, MPS and CUDA use different kernels
and reduction orders. The accepted numbers were produced on MPS with torch
2.14.1 and Python 3.13.1.

## Repository structure

```
src/        data pipeline, model, the three training entry points, the two
            corruption suites, validation and official-test evaluation and
            comparison, verification utilities and plotting scripts
docs/       the frozen protocols, the reproduction guide, the findings
results/    the aggregate result tables, the run's provenance record and the
            baseline validation summary
figures/    the four figures, PNG
paper/      the typeset manuscript, PDF
notebooks/  01_data_exploration.ipynb
```

* [docs/reproduction.md](docs/reproduction.md) — the full workflow, with the
  data pipeline and the clean baseline's configuration in section 14,
  including the track-level split and checkpoint selection using macro F1
* [docs/augmentation_protocol.md](docs/augmentation_protocol.md) and
  [docs/final_test_protocol.md](docs/final_test_protocol.md) — the two frozen
  protocols, with departures recorded in
  [docs/final_test_deviations.md](docs/final_test_deviations.md)
* [docs/final_test_findings.md](docs/final_test_findings.md) — the results and
  their detailed limitations

**What is and is not committed.** The repository carries the source, the
pinned `requirements.txt`, the two frozen protocols and the digests that pin
them, the reproduction guide, the recorded deviations, the findings write-up,
the exploration notebook, the four figures, the result files in
[results/](results/) and the typeset manuscript. It does not carry the GTSRB
data, the per-image predictions, the remaining analysis outputs under
`artifacts/analysis/final_test/`, or the manuscript's LaTeX source;
`artifacts/` and `data/` are both Git-ignored. The nine checkpoints, their
validation records and predictions, and the checkpoint inventory are outside
Git too, and are distributed as a release asset instead; see
[Trained checkpoints](#trained-checkpoints). Every aggregate reported above
can therefore be checked against a committed file, and regenerating the
official-test aggregates means rerunning the evaluation stage from the
published weights rather than retraining.

## Limitations

* **Synthetic corruptions.** Severity levels are parameter settings, not
  calibrated physical conditions. Nothing here measures performance on real
  adverse-weather or night-time photographs.
* **One architecture, one dataset.** All results are for the 1.45M-parameter
  `BaselineCNN` on GTSRB at 64x64.
* **Three training seeds give a limited view of run-to-run variability.** No
  statistical test across seeds was performed, and seed variability is not
  pooled with image-level sampling uncertainty.
* **Bootstrap intervals treat test images as independent.** Because multiple
  frames may come from the same physical sign, these intervals may
  underestimate uncertainty. The official test release does not provide track
  identifiers needed to account for this clustering. The intervals do not
  establish significance, and none enters the success criterion.
* **The held-out suite is held out in corruption type only.** Both suites use
  the same 12,630 test images, and cross-split physical-sign overlap is assumed
  absent rather than verified, again for want of track identifiers.
* **The headline metric is image-weighted.** Official test support runs from 60
  to 750 images per class, and class-balanced error is higher for every arm on
  both suites. Class-balanced evaluation flips the sign of two of the eighteen
  per-seed contrasts, both C−B, at seed 44 on the original suite and seed 43 on
  the held-out suite. Neither flip changes the predefined verdict, which needs
  C lower at all three seeds on both suites.
* **Not everything behind the results is distributed.** `artifacts/` is
  Git-ignored entirely. The nine checkpoints, their validation records and
  predictions, and the checkpoint inventory are published as a release asset,
  so the official-test stage can be rerun without retraining and the
  validation comparison and the baseline failure analysis can be rebuilt. The
  official-test per-image output and the earlier occlusion-placement stage are
  not distributed. A rerun on a different backend is not expected to reproduce
  the reported numbers exactly.
* **Not comparable with published GTSRB numbers.** The torchvision training
  archive used here holds 26,640 images, not the commonly cited 39,209-image
  variant.

## Dataset

The experiments use the German Traffic Sign Recognition Benchmark, downloaded
through `torchvision.datasets.GTSRB`. The benchmark was assembled and released
by its authors, and is described in:

> J. Stallkamp, M. Schlipsing, J. Salmen and C. Igel. Man vs. computer:
> Benchmarking machine learning algorithms for traffic sign recognition.
> *Neural Networks*, 32:323-332, 2012.
> <https://doi.org/10.1016/j.neunet.2012.02.016>

No dataset archive is redistributed here or in the release asset. `data/` is
Git-ignored, nothing in `src/` enables torchvision's `download` flag, and the
archives are fetched explicitly by the snippet in *Reproducing the study*. The
exploration notebook displays 53 sign crops as figures: one per class, eight
samples of class 14, and the smallest and largest training images.

## License

The source, the documentation, the figures and the notebook code are released
under the MIT License, as are the trained weights published under the
`v1.0-checkpoints` tag; see [LICENSE](LICENSE).

Two things sit outside that grant, both stated in the same file. The
manuscript PDF remains my copyright and is not under the MIT License. GTSRB
and material derived from its images, including the sign crops shown as
figures in the exploration notebook, are governed by the dataset authors'
terms: training on GTSRB is what produced the weights, but no ownership of the
dataset is claimed here and no rights over it are granted. No third-party
source is vendored into this repository; dependencies are installed from their
own distributions rather than redistributed.
