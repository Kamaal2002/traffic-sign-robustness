# Clean GTSRB data pipeline

Kamaal Alag

`data.py` loads the torchvision GTSRB training and test splits. The training
archive is divided 80/20 into training and validation **at the track level**:
whole tracks are assigned to one side, so frames of the same physical sign
never appear in both. The official test dataset remains independent and is
reserved for final evaluation, not training or model selection.

## Why the split is track-level

GTSRB training files are named `<track>_<frame>.ppm` inside a per-class
directory. A track is one physical sign filmed over consecutive frames, so
frames within a track are near-duplicates that differ mainly in scale,
brightness, and apparent sharpness — the same axes this project later perturbs
deliberately to measure robustness.

An image-level stratified split (the pipeline's earlier behaviour, seed 42) put
**887 of the 888 tracks on both sides**, so **100% of the 5,328 validation
images came from a track that also had frames in training**. That makes
validation close to a memorisation check and would bias both clean validation
accuracy and the measured benefit of targeted augmentation. The track-level
split removes this.

Track numbers restart in every class directory, so the group key is the class
directory together with the track number, not the track number alone.

## Splitting method

Each track lies in exactly one class directory and contains exactly 30 frames,
so tracks are equally sized, single-label units. The 888 tracks are therefore
split directly with `train_test_split(test_size=0.2, random_state=42,
stratify=<track class>)`: the track is the unit being split, which makes track
integrity structural rather than a constraint applied afterwards, and
stratifying on track class preserves class proportions.

A grouped splitter was considered and rejected. `GroupShuffleSplit` does not
stratify and left 6 of the 43 classes with no validation track at all.
`StratifiedGroupKFold(n_splits=5)` produced a valid split, but it applies a
greedy label-balancing heuristic built for groups that span several labels and
have unequal sizes — neither of which is true here — and on the measured split
it tracked the official class distribution slightly less closely (maximum
absolute class-proportion shift 0.004857 against 0.003265).

## Preprocessing

Every split uses the same deterministic sequence, built by
`build_preprocessing()`:

1. `Resize((64, 64), interpolation=InterpolationMode.BILINEAR, antialias=True)`.
2. `ToTensor()` produces RGB float32 tensors in `[0, 1]`.
3. `Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))` maps values to `[-1, 1]`.

The normalization constants are fixed baseline values, not measured GTSRB
statistics. Resizing directly to a square changes the aspect ratio of nonsquare
images; padding and cropping are not included. There are no augmentations or
robustness corruptions.

Training and validation are built on **two independent dataset instances** over
the same training archive, each with its own `Compose`. `build_preprocessing()`
returns a new object on every call, so no transform is shared. Training
augmentation added later can be attached to the training dataset without
silently reaching validation.

## Usage

From the project root:

```python
from src.data import create_data_loaders

loaders, split_info = create_data_loaders(batch_size=64, download=False)
train_loader = loaders["train"]
validation_loader = loaders["validation"]
test_loader = loaders["test"]
```

`split_info` holds `labels`, `tracks`, `train_indices`, and
`validation_indices`. The `labels` and `tracks` arrays follow the original
torchvision training order, so the returned indices can be used to audit class
counts, track counts, and split membership.
The default data location is the project's `data/` directory, independent of
the current working directory; `data_root` can override it. `download=True`
is the function default and allows torchvision to fetch missing datasets.

Batch size defaults to 64 and `num_workers` to 0. Only training is shuffled,
using a separate PyTorch generator seeded with 42. Validation and test preserve
their dataset order. All loaders retain the final partial batch. Recreating
the loaders reproduces the split and initial training order; successive epochs
advance the training generator.

## Executed verification

```sh
.venv/bin/python -m src.verify_data_pipeline
```

The script also accepts `--batch-size`. It uses cached data and checks the
split at both image and track level, class coverage, reproducibility, transform
ownership, and one batch from each loader.

The executed default configuration produced:

| Split | Tracks | Images |
| --- | ---: | ---: |
| Training | 710 | 21,300 |
| Validation | 178 | 5,340 |
| Official test | — | 12,630 |

The downloaded torchvision training archive contains 26,640 images in 888
tracks of exactly 30 frames each. Validation holds 0.2005 of the training
images. Verified in this run: no shared image index and **no shared track**
between training and validation; every track assigned whole, with all 30 frames
on one side; the two subsets covering all 26,640 images exactly once; all 43
classes present in both; identical indices across two independent pipeline
builds; and separate dataset objects whose transforms are independent —
replacing the training transform changed training output while validation
output stayed at `(3, 64, 64)`.

Grouping shifts class proportions slightly, because tracks move as whole blocks
and per-class track counts are not all divisible by 5. The maximum absolute
shift from the official training distribution was 0.000818 for training and
0.003265 for validation, both on class 5. Per-class track counts range from 4
train / 1 validation for the smallest classes to 40 train / 10 validation for
classes 1 and 2. Ten classes — 0, 19, 24, 27, 29, 32, 37, 39, 41, 42 — have a
single validation track, so their 30 validation images all come from one
physical sign and their per-class validation accuracy is correspondingly noisy.

All three inspected batches had image shape `(64, 3, 64, 64)` and label shape
`(64,)`. Image tensors were finite float32 values within `[-1, 1]`, and labels
were int64 class IDs in 0–42. Repeated access to a training image produced an
identical tensor. All verification checks passed; no model was created or
trained.

Baseline choices recorded for review are fixed normalization to `[-1, 1]`,
direct square resizing, and track-level grouped splitting with its reduced
effective validation size.

# Clean CNN baseline

`model.py` defines the baseline: a four-block CNN (Conv-BatchNorm-ReLU-MaxPool
per block, widening 32-64-128-256) over the 64x64 preprocessed images, followed
by a 256-unit fully connected layer with Dropout 0.5 and a 43-way linear
output. It has **1,448,779 trainable parameters** and returns raw logits.

`train.py` trains it on the track-aware split and selects a checkpoint on the
validation split:

| | |
| --- | --- |
| Training data | 21,300 images / 710 tracks |
| Validation data | 5,340 images / 178 tracks |
| Track overlap | 0 |
| Preprocessing | resize to 64x64, normalize to `[-1, 1]` (no augmentation) |
| Loss | `CrossEntropyLoss`, unweighted |
| Optimizer | AdamW, learning rate 1e-3, weight decay 1e-4 |
| Schedule | `CosineAnnealingLR`, `T_max` 30, over 30 epochs |
| Batch size | 64 |
| Seed | 42 |
| Checkpoint selection | highest validation macro F1 |

AdamW rather than Adam because Adam's `weight_decay` adds an L2 term to the
gradient, which then passes through the adaptive per-parameter scaling, so the
effective decay depends on each weight's gradient history. AdamW applies the
decay to the weights directly. Cosine annealing is used because it introduces
no tuned parameter of its own once the epoch budget is fixed, and ends at a
near-zero learning rate so the final model is settled rather than oscillating.
None of these values came from a search; they are conventional defaults chosen
once, so later augmentation gains are not confounded with baseline tuning.

## Why macro F1 selects the checkpoint

Macro F1 is always averaged over all 43 labels, with a class that has neither
predictions nor true samples scoring 0. This matters for the corruption stage:
a class that fails completely still contributes 0 to the mean instead of
dropping out of it.

Measured on the actual validation labels, the smallest class holds 30 of the
5,340 images and the largest 300, a roughly 10:1 imbalance with no dominant
class. Sending every image of the smallest class to the largest one costs
0.005618 accuracy but 0.024363 macro F1 — accuracy barely registers a rare
class being lost, which is the behaviour a robustness study cannot afford to
miss. For reference, always predicting the most frequent class scores 0.056180
accuracy and 0.002474 macro F1 on this split. Ties in selection are broken by
validation accuracy, then lower validation loss, then the earlier epoch.

## Executed baseline result

```sh
.venv/bin/python -m src.verify_training    # pre-flight checks
.venv/bin/python -m src.train              # full run, writes artifacts/baseline/
```

Trained for 30 epochs on MPS (Apple Silicon, torch 2.14.1). The selected
checkpoint is **epoch 21**:

| Metric | Value |
| --- | ---: |
| Validation loss | 0.047733 |
| Validation accuracy | 0.987640 |
| Validation macro F1 | 0.974451 |

**These are validation results, not official test results.** The official test
loader was not iterated or evaluated at any point during training or model
selection: `train.py` never references the test loader, no test metric appears
in the saved history, and the instrumented pre-flight run — which wraps all
three loaders in a counter — recorded test iterations 0. Reloading the saved
checkpoint and re-evaluating reproduces all three numbers exactly, so the
artifact matches the reported epoch.

The configuration was run twice on this machine. All 210 recorded per-epoch
metric values agree bit-for-bit between the two runs; the earlier run's history
is kept at `artifacts/baseline_first_run/history.json` for comparison. That
level of agreement holds for the same device and library versions only — CPU,
MPS, and CUDA use different kernels and reduction orders.

Two caveats on reading the clean accuracy. First, validation holds 178 tracks,
and ten classes have a single validation track, so per-class validation
accuracy for those is based on 30 frames of one physical sign and is
correspondingly noisy. For scale, losing one of those classes entirely moves
macro F1 by 0.024363, so differences below roughly 0.01 are finer than a single
class and should not be read as improvements — that is a granularity argument,
not a measured confidence interval. Second, the torchvision training archive
used here contains **26,640 training images**, not the commonly cited
39,209-image GTSRB training set, so this clean accuracy is not an
apples-to-apples comparison with published numbers trained on that variant.

Checkpoints and histories are written to `artifacts/`, which is Git-ignored;
model weights are not committed. The accepted checkpoint predates the change
that made `train.py` record the loss, optimizer, and scheduler by reading them
off the constructed objects, so its metadata is in the older layout: it has no
`loss` or `scheduler_config` entry, and its `optimizer_config` uses
`learning_rate` rather than `lr`. The loss and schedule rows in the table above
are therefore traceable to `train.py` and the recorded epoch count, not to
dedicated fields in that artifact. Checkpoints written now carry both.
