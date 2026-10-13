"""Clean GTSRB splits and preprocessing for the baseline experiments."""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB
from torchvision.transforms import InterpolationMode

SEED = 42
NUM_CLASSES = 43
IMAGE_SIZE = 64
VALIDATION_FRACTION = 0.2


def build_preprocessing():
    """Return the clean preprocessing pipeline used by every split.

    A new Compose is built on each call so that no two datasets share one
    transform object. Resizing straight to a square changes the aspect ratio of
    nonsquare images. The mean/std of 0.5 map [0, 1] tensors to [-1, 1] and are
    fixed baseline constants, not measured GTSRB statistics. No augmentation and
    no robustness corruptions are included here.
    """
    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def read_training_index(dataset):
    """Return per-sample class labels and track identifiers for the train split.

    Labels and paths are read from the dataset's sample list instead of by
    iterating the dataset, which avoids decoding all 26,640 images just to
    recover metadata.

    GTSRB training frames are named ``<track>_<frame>.ppm`` inside a per-class
    directory, where a track is one physical sign filmed over consecutive
    frames. Track numbers restart in every class directory, so a track is only
    uniquely identified by the class directory together with the track number.
    """
    labels = []
    tracks = []
    for path, label in dataset._samples:
        path = Path(path)
        labels.append(label)
        tracks.append(f"{path.parent.name}/{path.name.split('_')[0]}")
    return np.array(labels), np.array(tracks)


def split_tracks(labels, tracks):
    """Split whole tracks into train and validation, stratified by class.

    Frames of the same physical sign are near-duplicates, so an image-level
    split would place closely related frames on both sides and make validation
    accuracy optimistic. Splitting at the track level removes that.

    Each track sits in exactly one class directory and holds exactly 30 frames,
    so tracks are equally sized, single-label units. That makes an ordinary
    stratified split over the list of tracks sufficient: the track is the unit
    being split, so it cannot be divided, and stratifying on track class keeps
    class proportions. A grouped splitter would add a greedy label-balancing
    heuristic that this dataset's structure does not require.
    """
    unique_tracks, first_occurrence = np.unique(tracks, return_index=True)
    train_tracks, validation_tracks = train_test_split(
        unique_tracks,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=labels[first_occurrence],
    )
    train_indices = np.flatnonzero(np.isin(tracks, train_tracks))
    validation_indices = np.flatnonzero(np.isin(tracks, validation_tracks))
    return train_indices, validation_indices


def create_data_loaders(data_root=None, batch_size=64, num_workers=0, download=True):
    """Return a dictionary of loaders and the split metadata behind them.

    Training and validation are built on two independent dataset instances over
    the same torchvision training archive, each owning its own transform. Later
    training augmentation can therefore be attached to the training dataset
    without silently reaching validation.

    The split keeps every track entirely on one side and uses seed 42. The
    training shuffle uses a separate seed-42 generator. Validation and test
    loaders preserve dataset order, and every loader keeps its last batch.

    The official test split is loaded independently and never enters the split
    calculation. It is reserved for final evaluation, not model selection.
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / "data"

    train_dataset = GTSRB(
        root=data_root, split="train", transform=build_preprocessing(), download=download
    )
    validation_dataset = GTSRB(
        root=data_root, split="train", transform=build_preprocessing(), download=download
    )
    test_dataset = GTSRB(
        root=data_root, split="test", transform=build_preprocessing(), download=download
    )

    labels, tracks = read_training_index(train_dataset)
    train_indices, validation_indices = split_tracks(labels, tracks)

    shuffle_generator = torch.Generator().manual_seed(SEED)
    loaders = {
        "train": DataLoader(
            Subset(train_dataset, train_indices.tolist()),
            batch_size=batch_size, shuffle=True, generator=shuffle_generator,
            num_workers=num_workers, drop_last=False,
        ),
        "validation": DataLoader(
            Subset(validation_dataset, validation_indices.tolist()),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
        ),
        "test": DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
        ),
    }
    # Labels and tracks follow the original torchvision training order, so the
    # returned indices can be used to audit class counts and split membership.
    split_info = {
        "labels": labels,
        "tracks": tracks,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    return loaders, split_info
