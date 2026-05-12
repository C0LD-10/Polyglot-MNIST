"""
Evaluation metrics for Spanish MNIST classification.
"""

from typing import Dict, List, Optional
import numpy as np


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Overall top-1 accuracy."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(y_true == y_pred)) * 100.0


def top_k_accuracy(logits: np.ndarray, y_true: List[int], k: int = 5) -> float:
    """Top-K accuracy."""
    y_true = np.array(y_true)
    top_k = np.argsort(logits, axis=1)[:, -k:]
    correct = np.any(top_k == y_true[:, None], axis=1)
    return float(np.mean(correct)) * 100.0


def per_class_accuracy(y_true: List[int], y_pred: List[int]) -> Dict[int, float]:
    """Return per-class accuracy dict {class_id: accuracy_pct}."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    result = {}
    for c in classes:
        mask = y_true == c
        result[int(c)] = float(np.mean(y_pred[mask] == c)) * 100.0
    return result


def classification_report_dict(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[Dict[int, str]] = None,
) -> Dict:
    """Return a classification report as a dict."""
    from sklearn.metrics import classification_report
    target_names = None
    if class_names:
        unique_ids = sorted(set(y_true))
        target_names = [class_names.get(i, str(i)) for i in unique_ids]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    return report


def print_summary(y_true: List[int], y_pred: List[int], class_names: Optional[Dict[int, str]] = None):
    """Print a summary of evaluation metrics."""
    acc = accuracy(y_true, y_pred)
    pca = per_class_accuracy(y_true, y_pred)
    worst = sorted(pca.items(), key=lambda x: x[1])[:5]
    best  = sorted(pca.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\n{'='*40}")
    print(f"  Overall Accuracy : {acc:.2f}%")
    print(f"  Num Samples      : {len(y_true)}")
    print(f"  Num Classes      : {len(pca)}")
    print(f"\n  Best 5 classes:")
    for cid, acc_c in best:
        name = class_names.get(cid, str(cid)) if class_names else str(cid)
        print(f"    {name:20s}: {acc_c:.1f}%")
    print(f"\n  Worst 5 classes:")
    for cid, acc_c in worst:
        name = class_names.get(cid, str(cid)) if class_names else str(cid)
        print(f"    {name:20s}: {acc_c:.1f}%")
    print(f"{'='*40}\n")
