"""
Prepare the Spanish MNIST dataset: create train/val/test split CSV files.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --split 0.7 0.15 0.15
    python scripts/prepare_data.py --csv data/raw/spanish_mnist.csv --out data/processed
"""

import csv
import argparse
import random
from pathlib import Path
from collections import defaultdict


def split_dataset(
    csv_file: str,
    output_dir: str,
    split: tuple = (0.7, 0.15, 0.15),
    seed: int = 42,
):
    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys()) + ["split"]

    # Group by class
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["class_id"]].append(row)

    rng = random.Random(seed)
    train_rows, val_rows, test_rows = [], [], []
    train_r, val_r, _ = split

    for cid, items in buckets.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_r))
        n_val   = max(1, int(n * val_r))

        for i, row in enumerate(items):
            if i < n_train:
                row["split"] = "train"
                train_rows.append(row)
            elif i < n_train + n_val:
                row["split"] = "val"
                val_rows.append(row)
            else:
                row["split"] = "test"
                test_rows.append(row)

    def write(path, data):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    write(output_dir / "train.csv", train_rows)
    write(output_dir / "val.csv",   val_rows)
    write(output_dir / "test.csv",  test_rows)
    write(output_dir / "all_splits.csv", train_rows + val_rows + test_rows)

    total = len(train_rows) + len(val_rows) + len(test_rows)
    print(f"\n✅ Split complete ({total} total samples)")
    print(f"   Train : {len(train_rows):4d}  ({len(train_rows)/total*100:.1f}%)")
    print(f"   Val   : {len(val_rows):4d}  ({len(val_rows)/total*100:.1f}%)")
    print(f"   Test  : {len(test_rows):4d}  ({len(test_rows)/total*100:.1f}%)")
    print(f"   Output: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare Spanish MNIST train/val/test splits")
    parser.add_argument("--csv",   type=str, default="data/raw/spanish_mnist.csv")
    parser.add_argument("--out",   type=str, default="data/processed")
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.csv, args.out, tuple(args.split), args.seed)


if __name__ == "__main__":
    main()
