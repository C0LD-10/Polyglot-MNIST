"""
Spanish MNIST Dataset — PyTorch Dataset class
Supports train / val / test splits and custom transforms.
"""

import os
import csv
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SpanishMNISTDataset(Dataset):
    """
    PyTorch Dataset for the Spanish MNIST character recognition dataset.

    Args:
        csv_file  : Path to the metadata CSV file.
        img_dir   : Root directory containing the images/ folder.
        split     : One of 'train', 'val', 'test', or 'all'.
        transform : Optional torchvision transforms applied to each image.
        split_ratios : Tuple (train, val, test) must sum to 1.0.
        seed      : Random seed for reproducible splits.

    Example:
        >>> dataset = SpanishMNISTDataset(
        ...     csv_file="data/raw/spanish_mnist.csv",
        ...     img_dir="data/raw",
        ...     split="train"
        ... )
        >>> img, label = dataset[0]
        >>> print(img.shape, label)
    """

    CATEGORIES = [
        "digit",
        "uppercase",
        "accented_upper",
        "lowercase",
        "accented_lower",
        "special",
    ]

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        split: str = "all",
        transform=None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ):
        assert split in ("train", "val", "test", "all"), \
            f"split must be one of 'train', 'val', 'test', 'all'. Got: {split}"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, \
            "split_ratios must sum to 1.0"

        self.img_dir = Path(img_dir)
        self.transform = transform
        self.split = split

        # Load metadata
        self.records: List[Dict] = []
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)

        # Build class mappings
        self.class_ids = sorted(set(int(r["class_id"]) for r in self.records))
        self.num_classes = len(self.class_ids)
        self.id_to_char = {int(r["class_id"]): r["character"] for r in self.records}
        self.id_to_label = {int(r["class_id"]): r["label"] for r in self.records}
        self.char_to_id = {v: k for k, v in self.id_to_char.items()}

        # Split per class (stratified)
        if split != "all":
            self.records = self._stratified_split(self.records, split, split_ratios, seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stratified_split(
        self,
        records: List[Dict],
        split: str,
        ratios: Tuple[float, float, float],
        seed: int,
    ) -> List[Dict]:
        """Perform stratified split grouped by class_id."""
        from collections import defaultdict

        rng = random.Random(seed)
        buckets: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            buckets[r["class_id"]].append(r)

        train_r, val_r, test_r = ratios
        chosen = []
        for cid, items in buckets.items():
            items = items[:]
            rng.shuffle(items)
            n = len(items)
            n_train = max(1, int(n * train_r))
            n_val = max(1, int(n * val_r))
            if split == "train":
                chosen.extend(items[:n_train])
            elif split == "val":
                chosen.extend(items[n_train: n_train + n_val])
            else:  # test
                chosen.extend(items[n_train + n_val:])
        return chosen

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        record = self.records[idx]
        img_path = self.img_dir / record["filename"]

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = int(record["class_id"])
        return image, label

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_class_name(self, class_id: int) -> str:
        """Return the character string for a given class ID."""
        return self.id_to_char.get(class_id, "?")

    def get_label_name(self, class_id: int) -> str:
        """Return the string label (e.g. 'upper_A') for a given class ID."""
        return self.id_to_label.get(class_id, "unknown")

    def class_distribution(self) -> Dict[int, int]:
        """Return a dict of {class_id: count} for the current split."""
        from collections import Counter
        return dict(Counter(int(r["class_id"]) for r in self.records))

    def sample_images(self, n: int = 5, class_id: Optional[int] = None) -> List[Tuple[Image.Image, int]]:
        """Return n random (PIL image, class_id) tuples, optionally filtered by class."""
        pool = self.records
        if class_id is not None:
            pool = [r for r in pool if int(r["class_id"]) == class_id]
        chosen = random.sample(pool, min(n, len(pool)))
        result = []
        for r in chosen:
            img = Image.open(self.img_dir / r["filename"]).convert("L")
            result.append((img, int(r["class_id"])))
        return result

    def __repr__(self) -> str:
        return (
            f"SpanishMNISTDataset("
            f"split='{self.split}', "
            f"samples={len(self)}, "
            f"classes={self.num_classes})"
        )


# ------------------------------------------------------------------
# Default transforms
# ------------------------------------------------------------------

def get_train_transform() -> transforms.Compose:
    """Standard training augmentation pipeline."""
    return transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_eval_transform() -> transforms.Compose:
    """Standard eval/test transform — no augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
