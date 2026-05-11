"""
Unit tests for SpanishMNISTDataset.
Run: pytest tests/test_dataset.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from datasets.spanish_mnist_dataset import SpanishMNISTDataset, get_train_transform, get_eval_transform

CSV = "data/raw/spanish_mnist.csv"
IMG = "data/raw"


@pytest.fixture(scope="module")
def full_dataset():
    return SpanishMNISTDataset(CSV, IMG, split="all")


@pytest.fixture(scope="module")
def train_dataset():
    return SpanishMNISTDataset(CSV, IMG, split="train", transform=get_eval_transform())


def test_dataset_loads(full_dataset):
    assert len(full_dataset) > 0, "Dataset should not be empty"


def test_num_classes(full_dataset):
    assert full_dataset.num_classes == 80, f"Expected 80 classes, got {full_dataset.num_classes}"


def test_getitem_returns_tensor(train_dataset):
    img, label = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Image should be a tensor"
    assert isinstance(label, int), "Label should be an int"


def test_image_shape(train_dataset):
    img, _ = train_dataset[0]
    assert img.shape == (1, 28, 28), f"Expected (1, 28, 28), got {img.shape}"


def test_label_range(full_dataset):
    for i in range(min(50, len(full_dataset))):
        _, label = full_dataset[i]
        assert 0 <= label < 80, f"Label {label} out of range [0, 80)"


def test_split_sizes():
    all_ds    = SpanishMNISTDataset(CSV, IMG, split="all")
    train_ds  = SpanishMNISTDataset(CSV, IMG, split="train")
    val_ds    = SpanishMNISTDataset(CSV, IMG, split="val")
    test_ds   = SpanishMNISTDataset(CSV, IMG, split="test")

    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total == len(all_ds), "Split sizes should sum to total"


def test_get_class_name(full_dataset):
    name = full_dataset.get_class_name(0)
    assert name == "0", f"Class 0 should map to '0', got '{name}'"


def test_class_distribution(train_dataset):
    dist = train_dataset.class_distribution()
    assert isinstance(dist, dict)
    assert len(dist) == train_dataset.num_classes


def test_repr(train_dataset):
    r = repr(train_dataset)
    assert "train" in r


def test_train_transform():
    ds = SpanishMNISTDataset(CSV, IMG, split="train", transform=get_train_transform())
    img, _ = ds[0]
    assert img.shape == (1, 28, 28)


def test_no_transform():
    ds = SpanishMNISTDataset(CSV, IMG, split="all")
    img, _ = ds[0]
    assert isinstance(img, torch.Tensor)


def test_invalid_split():
    with pytest.raises(AssertionError):
        SpanishMNISTDataset(CSV, IMG, split="invalid")
