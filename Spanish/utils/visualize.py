"""
Visualization utilities for the Spanish MNIST dataset.
"""

import random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


def show_sample_grid(
    csv_file: str,
    img_dir: str,
    n_cols: int = 10,
    n_rows: int = 8,
    figsize: Tuple[int, int] = (20, 16),
    save_path: Optional[str] = None,
    seed: int = 42,
):
    """
    Display a grid of random sample images from the dataset.

    Args:
        csv_file  : Path to spanish_mnist.csv.
        img_dir   : Root directory containing images/.
        n_cols    : Number of columns in the grid.
        n_rows    : Number of rows in the grid.
        figsize   : Figure size.
        save_path : If provided, save figure to this path.
        seed      : Random seed.
    """
    import csv

    rng = random.Random(seed)
    img_dir = Path(img_dir)

    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sampled = rng.sample(rows, min(n_cols * n_rows, len(rows)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("Spanish MNIST — Sample Images", fontsize=16, fontweight="bold", y=1.01)

    for ax, row in zip(axes.flat, sampled):
        img = Image.open(img_dir / row["filename"]).convert("L")
        ax.imshow(np.array(img), cmap="gray", interpolation="nearest")
        ax.set_title(row["character"], fontsize=9, pad=2)
        ax.axis("off")

    for ax in axes.flat[len(sampled):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def show_class_samples(
    csv_file: str,
    img_dir: str,
    class_id: int,
    n: int = 10,
    save_path: Optional[str] = None,
):
    """Show all samples for a specific class_id."""
    import csv

    img_dir = Path(img_dir)
    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["class_id"]) == class_id]

    rows = rows[:n]
    fig, axes = plt.subplots(1, len(rows), figsize=(len(rows) * 1.5, 2))
    if len(rows) == 1:
        axes = [axes]

    char = rows[0]["character"]
    fig.suptitle(f"Class {class_id} — '{char}' ({rows[0]['category']})", fontsize=13)

    for ax, row in zip(axes, rows):
        img = Image.open(img_dir / row["filename"]).convert("L")
        ax.imshow(np.array(img), cmap="gray")
        ax.set_title(f"#{row['sample_index']}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_distribution(
    csv_file: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 5),
):
    """Bar chart of class distribution."""
    import csv
    from collections import Counter

    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    counts = Counter(r["category"] for r in rows)
    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
    bars = ax.bar(labels, values, color=colors[: len(labels)], edgecolor="white", linewidth=0.7)

    ax.set_title("Spanish MNIST — Samples per Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Sample Count")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 18),
):
    """Plot a confusion matrix from predictions."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        ax=ax, linewidths=0.3, linecolor="lightgray"
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
):
    """Plot training and validation loss/accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train", color="#4e79a7")
    ax1.plot(epochs, val_losses, label="Val", color="#e15759")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_accs, label="Train", color="#4e79a7")
    ax2.plot(epochs, val_accs, label="Val", color="#e15759")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Training Curves — Spanish MNIST", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
