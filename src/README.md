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
