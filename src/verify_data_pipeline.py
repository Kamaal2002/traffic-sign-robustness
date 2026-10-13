"""Verify clean GTSRB preprocessing with `python -m src.verify_data_pipeline`.

Run from the project root with the project virtual environment. Batch size is
configurable with --batch-size. This script uses the cached dataset and does
not download data, train a model, or modify the dataset.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms

from .data import IMAGE_SIZE, NUM_CLASSES, create_data_loaders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    loaders, info = create_data_loaders(batch_size=args.batch_size, download=False)
    repeated_loaders, repeated_info = create_data_loaders(
        batch_size=args.batch_size, download=False
    )

    labels = info["labels"]
    tracks = info["tracks"]
    train_indices = info["train_indices"]
    validation_indices = info["validation_indices"]
    expected_classes = np.arange(NUM_CLASSES)

    # Image-level split integrity: disjoint, and together exactly the full
    # torchvision training split with no sample used twice.
    assert np.intersect1d(train_indices, validation_indices).size == 0
    assert np.array_equal(
        np.sort(np.concatenate([train_indices, validation_indices])),
        np.arange(len(labels)),
    )
    assert len(labels) == 26640

    # Track-level split integrity: no physical sign is shared across the split,
    # and every track contributes all of its frames to exactly one side.
    train_tracks = set(tracks[train_indices])
    validation_tracks = set(tracks[validation_indices])
    assert not (train_tracks & validation_tracks)
    assert len(train_tracks) + len(validation_tracks) == len(set(tracks))
    assignment = np.where(np.isin(tracks, list(train_tracks)), "train", "validation")
    frames_by_track = pd.crosstab(pd.Series(tracks, name="track"), assignment)
    assert ((frames_by_track["train"] == 0) | (frames_by_track["validation"] == 0)).all()
    assert (frames_by_track.sum(axis="columns") == 30).all()

    # Class coverage is a weaker guarantee than with an image-level split,
    # because whole tracks move together, so it is checked explicitly.
    assert np.array_equal(np.unique(labels), expected_classes)
    assert np.array_equal(np.unique(labels[train_indices]), expected_classes)
    assert np.array_equal(np.unique(labels[validation_indices]), expected_classes)

    # Reproducibility of the seed-42 split across independent pipeline builds.
    assert np.array_equal(labels, repeated_info["labels"])
    assert np.array_equal(tracks, repeated_info["tracks"])
    assert np.array_equal(train_indices, repeated_info["train_indices"])
    assert np.array_equal(validation_indices, repeated_info["validation_indices"])

    train_base = loaders["train"].dataset.dataset
    validation_base = loaders["validation"].dataset.dataset
    test_base = loaders["test"].dataset
    assert train_base._split == "train" and validation_base._split == "train"
    assert test_base._split == "test"
    assert train_base is not validation_base
    assert train_base.transform is not validation_base.transform
    assert isinstance(loaders["train"].sampler, RandomSampler)
    assert isinstance(loaders["validation"].sampler, SequentialSampler)
    assert isinstance(loaders["test"].sampler, SequentialSampler)

    # Transform ownership is checked by behaviour, not just object identity:
    # swapping the training transform must leave validation output unchanged.
    original_transform = train_base.transform
    train_base.transform = transforms.Compose([
        transforms.Resize((8, 8)), transforms.ToTensor()
    ])
    probe_image, _ = loaders["train"].dataset[0]
    validation_image, _ = loaders["validation"].dataset[0]
    assert probe_image.shape == (3, 8, 8)
    assert validation_image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    train_base.transform = original_transform
    restored_image, _ = loaders["train"].dataset[0]
    assert restored_image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    print("Sample counts:")
    for split, loader in loaders.items():
        print(f"  {split}: {len(loader.dataset):,}")
    print(f"Tracks: train {len(train_tracks):,}; validation {len(validation_tracks):,}; "
          f"total {len(set(tracks)):,}")
    print("No track and no image index is shared between train and validation.")
    print("Every track is assigned whole; all 30 frames stay on one side.")
    print("Train and validation together cover all 26,640 training images once.")
    print("Train and validation both contain all 43 classes.")
    print("Seed-42 split reproducibility verified across two pipeline builds.")
    print("Train and validation own separate datasets and separate transforms; "
          "replacing the training transform left validation output unchanged.")

    track_labels = pd.Series(labels).groupby(pd.Series(tracks)).first()
    track_counts = pd.DataFrame({
        "official_tracks": track_labels.value_counts().sort_index(),
        "train_tracks": track_labels[track_labels.index.isin(train_tracks)]
            .value_counts().sort_index(),
        "validation_tracks": track_labels[track_labels.index.isin(validation_tracks)]
            .value_counts().sort_index(),
    }).reindex(expected_classes).fillna(0).astype(int)
    image_counts = pd.DataFrame({
        "official_images": np.bincount(labels, minlength=NUM_CLASSES),
        "train_images": np.bincount(labels[train_indices], minlength=NUM_CLASSES),
        "validation_images": np.bincount(labels[validation_indices], minlength=NUM_CLASSES),
    })
    counts = track_counts.join(image_counts)
    counts.index.name = "class_id"
    proportions = image_counts.div(image_counts.sum(), axis="columns")
    # Grouping shifts class proportions because tracks move as whole blocks and
    # the per-class track counts are not all divisible by 5.
    deviations = proportions[["train_images", "validation_images"]].sub(
        proportions["official_images"], axis="index"
    ).abs()
    print("\nPer-class track and image counts:")
    print(counts.to_string())
    print("\nMaximum absolute class-proportion shift from the official training split:")
    for split in ("train_images", "validation_images"):
        worst = deviations[split].idxmax()
        print(f"  {split}: {deviations[split].max():.6f} (class {worst})")
    print(f"Validation share of images: {len(validation_indices) / len(labels):.4f}")
    print(f"Classes with a single validation track: "
          f"{counts.index[counts['validation_tracks'] == 1].tolist()}")

    print()
    for split, loader in loaders.items():
        images, batch_labels = next(iter(loader))
        assert images.shape == (
            min(args.batch_size, len(loader.dataset)), 3, IMAGE_SIZE, IMAGE_SIZE
        )
        assert batch_labels.shape == (images.shape[0],)
        assert images.dtype == torch.float32
        assert batch_labels.dtype == torch.int64
        assert torch.isfinite(images).all()
        assert images.min() >= -1 and images.max() <= 1
        assert torch.all((batch_labels >= 0) & (batch_labels < NUM_CLASSES))
        if split == "train":
            repeated_images, repeated_batch_labels = next(iter(repeated_loaders[split]))
            assert torch.equal(images, repeated_images)
            assert torch.equal(batch_labels, repeated_batch_labels)
        print(
            f"{split} batch: images {tuple(images.shape)}, labels {tuple(batch_labels.shape)}, "
            f"dtype {images.dtype}, range [{images.min().item():.4f}, {images.max().item():.4f}]"
        )

    first_image, first_label = loaders["train"].dataset[0]
    repeated_image, repeated_label = loaders["train"].dataset[0]
    assert torch.equal(first_image, repeated_image) and first_label == repeated_label
    print("Deterministic preprocessing and initial training shuffle verified.")
    print("All clean data pipeline checks passed.")


if __name__ == "__main__":
    main()
